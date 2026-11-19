#!/usr/bin/env python3
"""Reproduce selected validation, freeze artifacts, then compare fixed heads on test; no fitting."""
import argparse
import gc
import math
from pathlib import Path
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_train_bridge import LABELS
from calibrate_bridge_chorus import metrics, predict
from train_section_context import (PROTOCOL, COVERAGE_POLICY, SectionContextHead,
                                   prepare_samples, evaluate_context, source_sections)


def validate_inventory(records, details, split):
    if (split not in ('val', 'test') or not records
            or any(r['split'] != split or not r['approved'] for r in records)
            or len({r['track_id'] for r in records}) != len(records)):
        raise ValueError('Expected distinct approved records from ' + split)
    targets = {(r['track_id'], s['start'], s['end']): s['label'] for r in records for s in r['sections']}
    observed = set()
    for item in details:
        key = item['track_id'], item['start'], item['end']
        if key in observed or targets.get(key) != item['label_manual']:
            raise ValueError('Prediction inventory differs from frozen manual targets.')
        p = item['probabilities']
        if (set(p) != set(LABELS) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values())
                or abs(sum(p.values()) - 1) > 1e-4
                or item['prediction'] != max(LABELS, key=lambda name: p[name])):
            raise ValueError('Invalid original pooled probabilities/prediction.')
        observed.add(key)
    if observed != set(targets):
        raise ValueError('Incomplete frozen prediction inventory.')


def split_records(data):
    groups = {s: [r for r in data['records'] if r['approved'] and r['split'] == s]
              for s in ('train', 'val', 'test')}
    if any(not rows for rows in groups.values()):
        raise ValueError('Approved train, val and test inventories are required.')
    for key in ('track_id', 'group_id', 'audio_sha256'):
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if {r[key] for r in groups[left]} & {r[key] for r in groups[right]}:
                raise ValueError('Frozen split leakage: ' + key)
    return groups


def verify_scores(actual, reference):
    for key in ('sections', 'label_order', 'confusion_matrix'):
        if actual[key] != reference[key]:
            raise ValueError('Selected validation does not reproduce: ' + key)
    for key in ('section_accuracy', 'macro_f1_present'):
        if abs(actual[key] - reference[key]) > 1e-12:
            raise ValueError('Selected validation does not reproduce: ' + key)
    if 'section_nll' in reference and abs(actual['section_nll'] - reference['section_nll']) > 1e-6:
        raise ValueError('Selected validation does not reproduce: section_nll')


def changes(before, after):
    if [(i['track_id'], i['start'], i['end']) for i in before] != [(i['track_id'], i['start'], i['end']) for i in after]:
        raise ValueError('Context prediction order/inventory differs from baseline.')
    return [dict(track_id=a['track_id'], start=a['start'], end=a['end'], label_manual=a['label_manual'],
        baseline_prediction=a['prediction'], candidate_prediction=b['prediction'],
        baseline_correct=a['prediction'] == a['label_manual'], candidate_correct=b['prediction'] == a['label_manual'])
        for a, b in zip(before, after) if a['prediction'] != b['prediction']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    parser.add_argument('--context-run', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True, help='Existing pooled section-context cache')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], help='Default: same device as the selected context run')
    parser.add_argument('--expected-epoch', type=int)
    parser.add_argument('--limit', type=int, default=0, help='0=all frozen test songs; positive values are smoke checks only')
    args = parser.parse_args()
    if args.limit < 0: raise ValueError('limit must be >=0')
    # Resolve ALL paths before upstream load_head changes the process working directory.
    root = args.songstruct_root.expanduser().resolve()
    selected_run = args.context_run.expanduser().resolve()
    pooled_cache = args.cache_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    sys.path.insert(0, str(root))
    from finetuning.common import config, read, write, fingerprint, sha_file, iter_features, device, load_head
    from finetuning.engine import manifest, evaluate
    torch.set_num_threads(2)
    cfg = config(root / 'finetuning/config.json'); data, provenance = manifest(cfg)
    splits = split_records(data)
    for row in splits['train'] + splits['val'] + splits['test']:
        if row['annotation_sha256'] != fingerprint(row['sections']):
            raise ValueError('Manifest manual-annotation digest is stale.')
    run = read(selected_run / 'run.json')
    selected = read(selected_run / 'comparison.json')
    state_path = selected_run / 'best.pt'
    state_sha256 = sha_file(state_path)
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    if (run['protocol'] != PROTOCOL or run['identity'] != fingerprint({k: v for k, v in run.items() if k != 'identity'})
            or run['torch'] != str(torch.__version__) or run.get('coverage_policy') != COVERAGE_POLICY
            or run['plan_id'] != data['plan_id'] or run['provenance_id'] != provenance
            or selected.get('split') != 'val' or not selected.get('full_validation') or selected.get('test_used') is not False
            or selected['tracks'] != len(splits['val']) or selected['protocol'] != PROTOCOL
            or selected['experiment_identity'] != run['identity'] or selected['plan_id'] != data['plan_id']
            or selected['provenance_id'] != provenance or state['identity'] != run['identity']
            or state['architecture'] != PROTOCOL or state['settings'] != run['settings']
            or state['plan_id'] != data['plan_id'] or state['provenance_id'] != provenance
            or state['best_epoch'] != selected['best_epoch']):
        raise ValueError('Selected context checkpoint/report differs from its frozen training run.')
    if args.expected_epoch is not None and selected['best_epoch'] != args.expected_epoch:
        raise ValueError('Unexpected selected epoch; test evaluation did not run.')
    known_sample_ids = dict(run['sample_identities'])
    if set(known_sample_ids) != {r['track_id'] for r in splits['train'] + splits['val']}:
        raise ValueError('Selected run training/validation inventory changed.')
    calibration_path = args.calibration.expanduser().resolve() if args.calibration else None
    if calibration_path is None:
        candidates = (Path(__file__).resolve().parents[1] / 'outputs.local').glob('bridge-calibration-*/comparison.json')
        calibration_path = next((p.resolve() for p in candidates if sha_file(p) == run['calibration_report_sha256']), None)
    if calibration_path is None or sha_file(calibration_path) != run['calibration_report_sha256']:
        raise ValueError('The exact original calibration report is missing/changed; supply --calibration.')
    calibration = read(calibration_path); adapter_path = calibration_path.parent / 'adapter.json'
    adapter = read(adapter_path)['adapter']
    checkpoint = Path(calibration['checkpoint']).expanduser().resolve()
    checkpoint_sha = sha_file(checkpoint)
    if (checkpoint_sha != calibration['checkpoint_sha256'] or checkpoint_sha != run['checkpoint_sha256']
            or calibration['plan_id'] != data['plan_id'] or calibration['provenance_id'] != provenance):
        raise ValueError('Original SongFormer checkpoint/plan changed.')
    audit_path = Path(calibration['train_audit']).expanduser().resolve(); audit = read(audit_path)
    for original_device in ('mps', 'cpu', 'cuda'):
        audit_identity = fingerprint({'protocol': 'train-bridge-audit-v1', 'device': original_device,
            'torch': str(torch.__version__), 'plan_id': data['plan_id'], 'provenance_id': provenance,
            'checkpoint_sha256': checkpoint_sha,
            'train_annotations': [(r['id'], r['annotation_sha256']) for r in splits['train']]})
        if audit_identity == audit['audit_identity']: break
    else:
        raise ValueError('Original train annotation/runtime identity changed.')
    for row in splits['train']:
        stored = read(audit_path.parent / 'per-track' / (row['id'] + '.json'))
        if stored['identity'] != audit_identity or stored['audio_sha256'] != row['audio_sha256']:
            raise ValueError('Original training audio identity changed.')
    expected_val = fingerprint({'protocol': 'pair-calibration-v1', 'device': original_device,
        'torch': str(torch.__version__), 'train_audit_sha256': sha_file(audit_path),
        'checkpoint_sha256': checkpoint_sha, 'plan_id': data['plan_id'], 'provenance_id': provenance,
        'val_annotations': [(r['id'], r['annotation_sha256']) for r in splits['val']], 'selected_adapter': adapter})
    if expected_val != calibration['experiment_identity']:
        raise ValueError('Calibration or validation annotation identity changed.')
    val_details = []
    for row in splits['val']:
        stored = read(calibration_path.parent / 'val-predictions' / (row['id'] + '.json'))
        if stored['identity'] != expected_val: raise ValueError('Validation probability cache changed.')
        val_details.extend(stored['sections'])
    validate_inventory(splits['val'], val_details, 'val')
    val_metadata = {r['track_id']: r for r in splits['val']}
    verify_scores(metrics(val_details, val_metadata, {'kind': 'identity'}), selected['baseline'])
    verify_scores(metrics(val_details, val_metadata, adapter), selected['previous_calibration'])
    target = device(args.device or run['device']); baseline_target = device(args.baseline_device)
    model = SectionContextHead(run['settings']['hidden'], run['settings']['dropout']).to(target)
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False); del state
    pooled_cache.mkdir(parents=True, exist_ok=True)
    helpers = read, write, fingerprint, sha_file, iter_features
    print(f'Frozen context epoch {selected["best_epoch"]}; reproducing {len(splits["val"])} validation songs...', flush=True)
    val_samples = prepare_samples(splits['val'], val_details, cfg, provenance, pooled_cache, helpers)
    if any(s['identity'] != known_sample_ids[s['track_id']] for s in val_samples):
        raise ValueError('Selected validation audio features changed.')
    reproduced, _ = evaluate_context(model, val_samples, splits['val'], target)
    verify_scores(reproduced, selected['candidate'])
    if sha_file(state_path) != state_sha256:
        raise ValueError('Selected checkpoint changed during validation; test did not run.')
    print(f'Validation reproduced: F1={reproduced["macro_f1_present"]:.6f}. No fitting.', flush=True)
    inventory = []
    for row in splits['test']:
        boundaries, _ = source_sections(read(row['source_json']), row)
        feature_meta = Path(cfg['work_dir']) / 'features' / row['id'] / 'metadata.json'
        inventory.append({k: row[k] for k in ('id', 'track_id', 'group_id', 'audio_sha256', 'annotation_sha256')})
        inventory[-1].update(boundaries_sha256=fingerprint(boundaries), feature_metadata_sha256=sha_file(feature_meta))
    lock = {'protocol': 'frozen-section-context-test-v1', 'context_epoch': selected['best_epoch'],
        'context_checkpoint_sha256': state_sha256, 'selected_validation_report_sha256': sha_file(selected_run / 'comparison.json'),
        'training_run_sha256': sha_file(selected_run / 'run.json'), 'checkpoint_sha256': checkpoint_sha,
        'calibration_report_sha256': sha_file(calibration_path), 'calibration_adapter_sha256': sha_file(adapter_path),
        'fixed_calibration_parameters': adapter,
        'plan_id': data['plan_id'], 'provenance_id': provenance, 'feature_settings': cfg['features'],
        'torch': str(torch.__version__), 'device': str(target), 'baseline_device': str(baseline_target),
        'coverage_policy': COVERAGE_POLICY, 'test_inventory': inventory,
        'selection_source': 'Previously selected training-run validation; no test-based epoch/parameter selection.'}
    identity = fingerprint(lock)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists() and read(output / 'selection.json') != {'identity': identity, **lock}:
        raise ValueError('Frozen selection or test data changed; existing test results were preserved.')
    # Lock all models, settings and the FULL test inventory before any test inference/scoring.
    write(output / 'selection.json', {'identity': identity, **lock})
    snapshot = output / 'frozen-context.pt'
    if not snapshot.exists():
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(state_path, temporary)
            if sha_file(temporary) != state_sha256:
                raise ValueError('Selected checkpoint changed while creating its snapshot.')
            temporary.replace(snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    if sha_file(snapshot) != state_sha256:
        raise ValueError('Frozen checkpoint snapshot changed; test did not run.')
    write(output / 'validation-reproduction.json', {'passed': True, 'selected_epoch': selected['best_epoch'],
        'reproduced': reproduced, 'reference': selected['candidate'], 'selection_identity': identity})
    rows = splits['test'][:args.limit] if args.limit else splits['test']
    predictions_dir = output / 'baseline-predictions'; predictions_dir.mkdir(exist_ok=True)
    all_details = {}; pending = []
    for row in rows:
        path = predictions_dir / (row['id'] + '.json')
        if path.exists():
            cached = read(path)
            if cached['identity'] != identity: raise ValueError('Test prediction cache is stale.')
            validate_inventory([row], cached['sections'], 'test')
            all_details[row['track_id']] = cached['sections']
        else: pending.append(row)
    if pending:
        original, _ = load_head(cfg, baseline_target, checkpoint)
        original.requires_grad_(False)
        for i, row in enumerate(pending, 1):
            print(f'Frozen SongFormer test head [{i}/{len(pending)}]: {row["track_id"]}', flush=True)
            details = []; evaluate(original, [row], cfg, baseline_target, provenance, details=details)
            validate_inventory([row], details, 'test')
            write(predictions_dir / (row['id'] + '.json'), {'identity': identity, 'sections': details})
            all_details[row['track_id']] = details
        del original; gc.collect()
        if baseline_target.type == 'mps': torch.mps.empty_cache()
        if baseline_target.type == 'cuda': torch.cuda.empty_cache()
    details = [s for r in rows for s in all_details[r['track_id']]]
    validate_inventory(rows, details, 'test')
    samples = prepare_samples(rows, details, cfg, provenance, pooled_cache, helpers)
    context_score, context_details = evaluate_context(model, samples, rows, target, split='test')
    test_metadata = {r['track_id']: r for r in rows}
    baseline_score = metrics(details, test_metadata, {'kind': 'identity'})
    calibration_score = metrics(details, test_metadata, adapter)
    calibrated_details = [{**item, 'prediction': predict(item, test_metadata, adapter)[0]} for item in details]
    diagnostics = [c for s in samples for c in s['coverage_diagnostics']]
    write(output / 'test-coverage.json', {'policy': COVERAGE_POLICY, 'sections': diagnostics})
    report = {'protocol': lock['protocol'], 'split': 'test', 'tracks': len(rows),
        'full_test': len(rows) == len(splits['test']), 'selection_identity': identity,
        'selected_epoch': selected['best_epoch'], 'plan_id': data['plan_id'], 'provenance_id': provenance,
        'frozen_context_checkpoint': str(snapshot),
        'worship_v3': baseline_score, 'fixed_calibration': calibration_score, 'section_context': context_score,
        'macro_f1_delta_from_v3': context_score['macro_f1_present'] - baseline_score['macro_f1_present'],
        'macro_f1_delta_from_fixed_calibration': context_score['macro_f1_present'] - calibration_score['macro_f1_present'],
        'changed_sections_vs_v3': changes(details, context_details),
        'changed_sections_vs_fixed_calibration': changes(calibrated_details, context_details),
        'validation_reproduction_passed': True, 'weights_updated': False, 'parameters_selected_on_test': False,
        'encoder_executed': False, 'manual_labels_changed': False, 'boundaries_changed': False,
        'ready_for_app_integration': False, 'audio_coverage_reviews': len(diagnostics),
        'limitations': ['Original frozen PyTorch encoder features; compressed Core ML app pipeline is not evaluated.',
            'The fixed test split was evaluated for earlier models in this project; this is not a newly unseen dataset.',
            'This evaluation fits no weights or calibration parameters and does not choose a new epoch from test.',
            'Additional context residual is applied to original v3 pooled probabilities; the binary calibrator is a separate comparator.']}
    destination = output / ('test-all.json' if report['full_test'] else f'test-first-{len(rows)}.json')
    write(destination, report)
    for label, score in (('worship-v3', baseline_score), ('fixed calibration', calibration_score), ('section context', context_score)):
        print(f'{label}: accuracy={score["section_accuracy"]:.4f}; macro F1={score["macro_f1_present"]:.4f}', flush=True)
    print('Report:', destination, flush=True)


if __name__ == '__main__': main()
