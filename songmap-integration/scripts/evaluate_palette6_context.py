#!/usr/bin/env python3
"""Evaluate the locked context head on saved palette6 encoder features/Core ML probabilities; no fitting."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_train_bridge import LABELS
from calibrate_bridge_chorus import metrics
from evaluate_frozen_context import validate_inventory, verify_scores, changes
from train_section_context import PROTOCOL, COVERAGE_POLICY, SectionContextHead, prepare_samples, evaluate_context


def ordered_details(rows, details):
    validate_inventory(rows, details, 'val')
    lookup = {(s['track_id'], s['start'], s['end']): s for s in details}
    return [lookup[(r['track_id'], s['start'], s['end'])] for r in rows for s in r['sections']]


def probability_metrics(details):
    result = metrics(details, {}, {'kind': 'identity'})
    result['section_nll'] = sum(-math.log(max(s['probabilities'][s['label_manual']], 1e-8)) for s in details) / len(details)
    return result


def validate_palette_report(proof, rows, plan, provenance, checkpoint_sha, fingerprint):
    if (not proof.get('full_validation') or proof.get('muq_scheme') != 'palette6'
            or proof.get('head_scheme') != 'palette6' or proof.get('reused_musicfm') is not True
            or proof['tracks'] != len(rows) or proof['plan_id'] != plan or proof['provenance_id'] != provenance
            or proof['checkpoint_sha256'] != checkpoint_sha
            or proof['annotations_id'] != fingerprint([(r['track_id'], r['sections']) for r in rows])):
        raise ValueError('Expected full palette6 validation on the same frozen labels/checkpoint.')
    details = ordered_details(rows, proof['sections'])
    verify_scores(probability_metrics(details), proof['metrics'])
    return details


def palette_features(root, row, candidate_id, read):
    folder = (root / row['id']).resolve()
    meta = read(folder / 'metadata.json')
    if (meta.get('candidate_encoder_id') != candidate_id or meta['audio_sha256'] != row['audio_sha256']
            or not meta.get('chunks')):
        raise ValueError('Palette6 cache identity changed or chunks missing.')
    for chunk in meta['chunks']:
        path = (folder / chunk['file']).resolve(); path.relative_to(folder)
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if (values.ndim != 2 or values.shape[1] != 4096 or not 3 <= len(values) <= 10500
                or len(values) % 3 or len(values) != chunk['frames']):
            raise ValueError('Palette6 feature shape/frame metadata mismatch.')
        yield values, chunk['offset'], candidate_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    parser.add_argument('--frozen-evaluation', type=Path, required=True)
    parser.add_argument('--context-run', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--palette6-report', type=Path)
    parser.add_argument('--original-cache-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True, help='Separate pooled palette6 cache, not the original section cache')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0=all validation; positive=smoke subset')
    args = parser.parse_args()
    if args.limit < 0: raise ValueError('limit must be >=0')
    root = args.songstruct_root.expanduser().resolve()
    frozen = args.frozen_evaluation.expanduser().resolve()
    original_cache = args.original_cache_dir.expanduser().resolve()
    candidate_cache = args.cache_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if original_cache == candidate_cache: raise ValueError('Use a separate pooled palette6 cache.')
    sys.path.insert(0, str(root))
    from finetuning.common import config, read, write, fingerprint, sha_file, iter_features, device
    from finetuning.engine import manifest
    cfg = config(root / 'finetuning/config.json'); data, provenance = manifest(cfg)
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    if not rows: raise ValueError('No approved validation tracks.')
    lock = read(frozen / 'selection.json')
    payload = {k: v for k, v in lock.items() if k != 'identity'}
    snapshot = frozen / 'frozen-context.pt'
    if (lock['protocol'] != 'frozen-section-context-test-v1' or lock['identity'] != fingerprint(payload)
            or lock['context_epoch'] != 5 or sha_file(snapshot) != lock['context_checkpoint_sha256']
            or lock['plan_id'] != data['plan_id'] or lock['provenance_id'] != provenance
            or lock['feature_settings'] != cfg['features'] or lock['torch'] != str(torch.__version__)
            or lock['coverage_policy'] != COVERAGE_POLICY):
        raise ValueError('Frozen epoch-5 model/dataset/runtime identity changed.')
    # Resolve artifacts by recorded hashes, never by test scores.
    songmap = Path(__file__).resolve().parents[1]
    context = args.context_run.expanduser().resolve() if args.context_run else None
    if context is None:
        context = next((p.parent.resolve() for p in (songmap / 'outputs.local').glob('section-context-*/run.json')
                        if sha_file(p) == lock['training_run_sha256']), None)
    if (context is None or sha_file(context / 'run.json') != lock['training_run_sha256']
            or sha_file(context / 'comparison.json') != lock['selected_validation_report_sha256']):
        raise ValueError('Original selected context run is missing/changed; supply --context-run.')
    run = read(context / 'run.json'); selected = read(context / 'comparison.json')
    if (run['identity'] != fingerprint({k: v for k, v in run.items() if k != 'identity'})
            or run['identity'] != selected['experiment_identity'] or selected['best_epoch'] != 5
            or selected['protocol'] != PROTOCOL or selected.get('split') != 'val' or not selected.get('full_validation')
            or selected['tracks'] != len(rows) or run['coverage_policy'] != COVERAGE_POLICY):
        raise ValueError('Selected context run metadata changed.')
    calibration_path = args.calibration.expanduser().resolve() if args.calibration else None
    if calibration_path is None:
        calibration_path = next((p.resolve() for p in (songmap / 'outputs.local').glob('bridge-calibration-*/comparison.json')
                                 if sha_file(p) == lock['calibration_report_sha256']), None)
    if calibration_path is None or sha_file(calibration_path) != lock['calibration_report_sha256']:
        raise ValueError('Original bridge calibration report missing/changed; supply --calibration.')
    calibration = read(calibration_path)
    if sha_file(calibration_path.parent / 'adapter.json') != lock['calibration_adapter_sha256']:
        raise ValueError('Original calibration artifact changed.')
    val_details = []
    for row in rows:
        if row['annotation_sha256'] != fingerprint(row['sections']): raise ValueError('Annotation digest changed.')
        stored = read(calibration_path.parent / 'val-predictions' / (row['id'] + '.json'))
        if stored['identity'] != calibration['experiment_identity']: raise ValueError('Original validation probabilities changed.')
        val_details.extend(stored['sections'])
    val_details = ordered_details(rows, val_details)
    verify_scores(probability_metrics(val_details), selected['baseline'])
    state = torch.load(snapshot, map_location='cpu', weights_only=True)
    if (state['identity'] != run['identity'] or state['architecture'] != PROTOCOL or state['best_epoch'] != 5
            or state['settings'] != run['settings'] or state['plan_id'] != data['plan_id'] or state['provenance_id'] != provenance):
        raise ValueError('Frozen context checkpoint metadata changed.')
    torch.set_num_threads(2); target = device(run['device'])
    model = SectionContextHead(state['settings']['hidden'], state['settings']['dropout']).to(target)
    model.load_state_dict(state['model']); model.requires_grad_(False); del state
    original_cache.mkdir(parents=True, exist_ok=True)
    helpers = read, write, fingerprint, sha_file, iter_features
    print('Reproducing locked context validation with original cached features...', flush=True)
    original_samples = prepare_samples(rows, val_details, cfg, provenance, original_cache, helpers)
    expected_samples = dict(run['sample_identities'])
    if any(s['identity'] != expected_samples[s['track_id']] for s in original_samples):
        raise ValueError('Original validation audio features/targets changed.')
    reference, reference_details = evaluate_context(model, original_samples, rows, target)
    verify_scores(reference, selected['candidate'])
    proof_path = args.palette6_report.expanduser().resolve() if args.palette6_report else None
    if proof_path is None:
        found = list((root / 'outputs.local').glob('muq-compression-*/palette6-all-validation.json'))
        if not found: raise ValueError('Full palette6 report missing; supply --palette6-report.')
        proof_path = max(found, key=lambda p: p.stat().st_mtime).resolve()
    proof = read(proof_path)
    palette_details = validate_palette_report(proof, rows, data['plan_id'], provenance, lock['checkpoint_sha256'], fingerprint)
    def artifact(value):
        p = Path(value).expanduser()
        return (root / p).resolve() if not p.is_absolute() else p.resolve()
    base_path = artifact(proof['baseline_combination'])
    export_path = artifact(proof['muq_export_report'])
    if sha_file(base_path) != proof['baseline_combination_sha256'] or sha_file(export_path) != proof['muq_export_report_sha256']:
        raise ValueError('Palette6 source report chain changed.')
    base, export = read(base_path), read(export_path)
    if (not base.get('full_validation') or base.get('head_scheme') != 'palette6'
            or base.get('musicfm_override', {}).get('compression_scheme') != 'palette6'
            or export.get('compression_scheme') != 'palette6' or not export.get('source_full_validation')
            or not export.get('window_finite_passed')):
        raise ValueError('Expected validated palette6 MusicFM, MuQ and head source chain.')
    for key in ('plan_id', 'provenance_id', 'annotations_id', 'checkpoint_sha256', 'tracks'):
        if base[key] != proof[key]: raise ValueError('Palette6 source inventory mismatch: ' + key)
    source = artifact(proof['diagnostic_cache']); header_path = source / 'encoder_provenance.json'; header = read(header_path)
    encoder_id = proof['candidate_encoder_id']
    if (header['id'] != encoder_id or fingerprint({k: v for k, v in header.items() if k != 'id'}) != encoder_id
            or header['baseline_encoder_id'] != base['candidate_encoder_id'] or header.get('reused_musicfm') is not True
            or header['muq_export_report_sha256'] != proof['muq_export_report_sha256']
            or header['muq_package_digest'] != export['package_digest']
            or header['environment']['feature_settings'] != cfg['features']):
        raise ValueError('Palette6 encoder cache identity/profile mismatch.')
    pipeline_path = artifact(base['encoder_pipeline_report'])
    if sha_file(pipeline_path) != base['encoder_pipeline_report_sha256']:
        raise ValueError('MusicFM palette6 pipeline source report changed.')
    pipeline = read(pipeline_path)
    if (pipeline['candidate_encoder_id'] != header['baseline_encoder_id']
            or pipeline.get('musicfm_override', {}).get('compression_scheme') != 'palette6'):
        raise ValueError('Reused MusicFM encoder is not the verified palette6 pipeline.')
    input_identity = {'protocol': 'palette6-context-validation-v1', 'frozen_selection_identity': lock['identity'],
        'context_checkpoint_sha256': sha_file(snapshot), 'palette6_report_sha256': sha_file(proof_path),
        'encoder_header_sha256': sha_file(header_path), 'candidate_encoder_id': encoder_id,
        'plan_id': data['plan_id'], 'provenance_id': provenance,
        'annotations_id': proof['annotations_id'], 'torch': str(torch.__version__), 'device': str(target),
        'coverage_policy': COVERAGE_POLICY}
    identity = fingerprint(input_identity)
    output.mkdir(parents=True, exist_ok=True); candidate_cache.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists() and read(output / 'selection.json') != {'identity': identity, **input_identity}:
        raise ValueError('Palette6 evaluation inputs changed; existing results were preserved.')
    write(output / 'selection.json', {'identity': identity, **input_identity})
    chosen = rows[:args.limit] if args.limit else rows
    chosen_ids = {r['track_id'] for r in chosen}
    chosen_details = [s for s in palette_details if s['track_id'] in chosen_ids]
    readers = read, write, fingerprint, sha_file, lambda _cfg, row: palette_features(source, row, encoder_id, read)
    print(f'Reusing saved palette6 probabilities and features for {len(chosen)} validation songs; no Core ML inference.', flush=True)
    samples = prepare_samples(chosen, chosen_details, cfg, encoder_id, candidate_cache, readers, feature_cache_root=source)
    candidate, predictions = evaluate_context(model, samples, chosen, target)
    baseline = probability_metrics(chosen_details)
    reference_subset = [s for s in reference_details if s['track_id'] in chosen_ids]
    reference_metrics = probability_metrics(reference_subset)
    diagnostics = [d for s in samples for d in s['coverage_diagnostics']]
    write(output / 'coverage_diagnostics.json', {'policy': COVERAGE_POLICY, 'sections': diagnostics})
    report = {'protocol': input_identity['protocol'], 'split': 'val', 'tracks': len(chosen),
        'full_validation': len(chosen) == len(rows), 'selected_epoch': 5, 'selection_identity': identity,
        'plan_id': data['plan_id'], 'provenance_id': provenance, 'candidate_encoder_id': encoder_id,
        'original_context_validation_reproduced': True, 'saved_palette6_baseline_metrics_reproduced': True,
        'original_features_with_context': reference_metrics, 'palette6_baseline': baseline,
        'palette6_features_with_context': candidate,
        'macro_f1_delta_from_palette6_baseline': candidate['macro_f1_present'] - baseline['macro_f1_present'],
        'macro_f1_delta_from_original_context': candidate['macro_f1_present'] - reference_metrics['macro_f1_present'],
        'section_agreement_with_original_context': sum(a['prediction'] == b['prediction'] for a, b in zip(reference_subset, predictions)) / len(predictions),
        'changed_sections_vs_palette6_baseline': changes(chosen_details, predictions),
        'changed_sections_vs_original_context': changes(reference_subset, predictions),
        'audio_coverage_reviews': len(diagnostics), 'encoder_executed': False, 'coreml_head_executed': False,
        'context_runtime': 'PyTorch', 'weights_updated': False, 'test_metrics_used': False,
        'manual_labels_changed': False, 'boundaries_changed': False, 'ready_for_app_integration': False,
        'limitations': ['Context head is still PyTorch; it has not been converted to Core ML in this experiment.',
            'Inputs are saved palette6 encoder arrays and saved Core ML head probability outputs from the verified validation report.',
            'Encoder and old Core ML head inference are not re-executed; this is not end-to-end iPhone validation.',
            'All parameters and epoch remain fixed. Original validation reproduction and historical palette6 metrics are checked before comparison.']}
    destination = output / ('validation-all.json' if report['full_validation'] else f'validation-first-{len(chosen)}.json')
    write(destination, report)
    for name, score in (('Original + context', reference_metrics), ('Palette6 baseline', baseline), ('Palette6 + context', candidate)):
        print(f'{name}: accuracy={score["section_accuracy"]:.4f}; macro F1={score["macro_f1_present"]:.4f}', flush=True)
    print('Report:', destination, flush=True)


if __name__ == '__main__': main()
