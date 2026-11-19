#!/usr/bin/env python3
"""One locked test-split evaluation: production vs all4 vs MuQ-only label pipelines.

Pipelines (head + section-context residual), all PyTorch on the existing feature caches:
  production  worship-v3 head + original context (selected epoch 5)
  all4        120-second-window all4 head + its retrained context
  muq         120-second-window MuQ-only head + its retrained context

Before any test inference every pipeline must reproduce its recorded validation
result, and selection.json locks all checkpoint hashes, the decision rule and the
complete test inventory. Re-running is allowed only with an identical lock (same
deterministic results); nothing is fitted or selected on test.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_stream_ablation as ablation
from train_section_context import LABELS, SectionContextHead, evaluate_context, prepare_samples, source_sections
from train_variant_context import head_predictions, slice_samples
from calibrate_bridge_chorus import metrics

PROTOCOL = 'variant-pipeline-test-v1'
NAMES = ('production', 'all4', 'muq')
# Fixed before test inference: MuQ pipeline vs production pipeline on identical test sections.
TEST_RULE = {'candidate': 'muq', 'reference': 'production', 'max_section_accuracy_drop': 0.01,
             'max_macro_f1_drop': 0.01, 'bootstrap': ablation.DECISION_RULE['bootstrap']}
KEYS = ('sections', 'section_accuracy', 'macro_f1_present')


def same_scores(actual, expected, tolerance=1e-6):
    return all(abs(actual[k] - expected[k]) <= tolerance for k in KEYS)


def decide_test(results, rule=TEST_RULE):
    reference = results[rule['reference']]['sections']
    report = {}
    for name, entry in results.items():
        if name == rule['reference']:
            continue
        paired = ablation.paired_comparison(reference, entry['sections'], LABELS,
                                            rule={**ablation.DECISION_RULE, 'max_section_accuracy_drop': rule['max_section_accuracy_drop'],
                                                  'max_macro_f1_drop': rule['max_macro_f1_drop'], 'bootstrap': rule['bootstrap']})
        report[name] = {k: paired[k] for k in ('delta_section_accuracy', 'delta_macro_f1_present', 'bootstrap_interval',
                                               'only_control_correct', 'only_variant_correct', 'verdict', 'changed_sections')}
        report[name]['note'] = '"control" means the production pipeline here.'
    report['decision'] = report[rule['candidate']]['verdict']
    return report


def lock(output, payload, fingerprint, read, write):
    identity = fingerprint(payload)
    path = output / 'selection.json'
    if path.exists() and read(path) != {'identity': identity, **payload}:
        raise ValueError('A different test selection already exists here; test results were preserved.')
    write(path, {'identity': identity, **payload})
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    parser.add_argument('--production-head', type=Path, required=True, help='worship-v3 best.pt')
    parser.add_argument('--production-context-run', type=Path, required=True, help='Selected original context run (epoch 5)')
    parser.add_argument('--expected-epoch', type=int, default=5)
    parser.add_argument('--variant-context-dir', type=Path, required=True, help='.../stream-ablation-w120-*/context')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    args = parser.parse_args()
    paths = {k: getattr(args, k).expanduser().resolve() for k in
             ('songstruct_root', 'production_head', 'production_context_run', 'variant_context_dir', 'cache_dir', 'output_dir')}
    sys.path.insert(0, str(paths['songstruct_root']))
    from finetuning.common import config, device, fingerprint, iter_features, load_head, read, sha_file, write
    from finetuning.engine import evaluate, manifest
    cfg = config(paths['songstruct_root'] / 'finetuning/config.json')
    data, provenance = manifest(cfg)
    rows = {s: [r for r in data['records'] if r['approved'] and r['split'] == s] for s in ('train', 'val', 'test')}
    for a, b in (('train', 'test'), ('val', 'test'), ('train', 'val')):
        for key in ('track_id', 'group_id', 'audio_sha256'):
            if {r[key] for r in rows[a]} & {r[key] for r in rows[b]}:
                raise ValueError(f'{a}/{b} leakage: {key}')
    if not rows['test']:
        raise ValueError('No approved test songs.')
    target = device(args.device)

    production_run = paths['production_context_run']
    selected = read(production_run / 'comparison.json')
    if selected.get('best_epoch') != args.expected_epoch or selected.get('test_used') is not False:
        raise ValueError('Production context run is not the selected validation-only epoch.')
    # Each context runs on the device that produced its recorded validation (production: usually cpu).
    specs = {'production': {'head': paths['production_head'], 'context': production_run / 'best.pt',
                            'expected_val': selected['candidate'],
                            'context_device': read(production_run / 'run.json')['device']}}
    for name in ('all4', 'muq'):
        folder = paths['variant_context_dir'] / name
        summary = read(folder / 'summary.json')
        if summary['variant'] != name or summary.get('test_used') is not False:
            raise ValueError(f'{name} context summary is not a validation-only {name} run.')
        specs[name] = {'head': Path(summary['head_checkpoint']).resolve(), 'context': folder / 'best.pt',
                       'expected_val': summary['pipeline'], 'context_device': summary['device']}

    def load_pipeline(name):
        spec = specs[name]
        if name == 'production':
            meta = torch.load(spec['head'], map_location='cpu', weights_only=False)
            if meta.get('plan_id') != data['plan_id'] or meta.get('provenance_id') != provenance:
                raise ValueError('Production head belongs to another split/cache.')
            head, _ = load_head(cfg, target, spec['head']); cols = ablation.columns('all4')
            state = torch.load(spec['context'], map_location='cpu', weights_only=True)
            width = 4096
        else:
            head, saved = ablation.load_variant(cfg, target, spec['head'], load_head); cols = saved['columns']
            state = torch.load(spec['context'], map_location='cpu', weights_only=False)
            if state['head_sha256'] != sha_file(spec['head']) or state['columns'] != cols:
                raise ValueError(f'{name} context was not trained on this head.')
            width = state['width']
        if state['plan_id'] != data['plan_id'] or state['provenance_id'] != provenance:
            raise ValueError(f'{name} context belongs to another split/cache.')
        context_target = device(str(torch.device(spec['context_device']).type))
        model = SectionContextHead(state['settings']['hidden'], state['settings']['dropout'], width=width).to(context_target)
        model.load_state_dict(state['model'], strict=True); model.requires_grad_(False)
        head.eval()
        return head, cols, (model, context_target)

    helpers = read, write, fingerprint, sha_file, iter_features
    output = paths['output_dir']; output.mkdir(parents=True, exist_ok=True)

    def run_split(name, head, cols, contextual, split):
        model, context_target = contextual
        head_sha = sha_file(specs[name]['head'])
        identity = fingerprint({'protocol': PROTOCOL, 'head_sha256': head_sha, 'split': split, 'device': str(target),
                                'plan_id': data['plan_id'], 'provenance_id': provenance,
                                'annotations': [(r['id'], r['annotation_sha256']) for r in rows[split]]})
        _, details = head_predictions(head, rows[split], cfg, target, provenance, cols, evaluate, iter_features,
                                      output / f'{name}-head-{split}.json', identity, write, read)
        cache = paths['cache_dir'] / name; cache.mkdir(parents=True, exist_ok=True)
        samples = slice_samples(prepare_samples(rows[split], details, cfg, provenance, cache, helpers), cols)
        head_only = metrics(details, {r['track_id']: r for r in rows[split]}, {'kind': 'identity'})
        result, predictions = evaluate_context(model, samples, rows[split], context_target, split=split)
        return head_only, result, predictions

    # 1) Every pipeline must reproduce its recorded validation before the lock.
    loaded = {}
    for name in NAMES:
        print(f'[{name}] reproducing validation ({len(rows["val"])} songs)...', flush=True)
        head, cols, model = load_pipeline(name)
        _, result, _ = run_split(name, head, cols, model, 'val')
        if not same_scores(result, specs[name]['expected_val']):
            raise ValueError(f'{name} does not reproduce its recorded validation; test did not run.')
        print(f'[{name}] validation reproduced: acc={result["section_accuracy"]:.4f} F1={result["macro_f1_present"]:.4f}', flush=True)
        loaded[name] = (head, cols, model)
    # 2) Lock models, rule and the full test inventory before any test inference.
    inventory = []
    for row in rows['test']:
        boundaries, _ = source_sections(read(row['source_json']), row)
        inventory.append({**{k: row[k] for k in ('id', 'track_id', 'group_id', 'audio_sha256', 'annotation_sha256')},
                          'boundaries_sha256': fingerprint(boundaries)})
    payload = {'protocol': PROTOCOL, 'rule': TEST_RULE, 'plan_id': data['plan_id'], 'provenance_id': provenance,
               'device': str(target), 'torch': str(torch.__version__), 'test_inventory': inventory,
               'pipelines': {n: {'head_sha256': sha_file(specs[n]['head']), 'context_sha256': sha_file(specs[n]['context']),
                                 'context_device': specs[n]['context_device'],
                                 'validation': {k: specs[n]['expected_val'][k] for k in KEYS}} for n in NAMES}}
    lock(output, payload, fingerprint, read, write)
    # 3) Test inference, no fitting.
    results = {}
    for name in NAMES:
        print(f'[{name}] test ({len(rows["test"])} songs)...', flush=True)
        head_only, result, predictions = run_split(name, *loaded[name], 'test')
        results[name] = {'head_only': {k: head_only[k] for k in KEYS}, 'pipeline': {k: result[k] for k in KEYS},
                         'per_class': result['per_class'], 'confusion_matrix': result['confusion_matrix'],
                         'sections': [{**p, 'correct': p['prediction'] == p['label_manual']} for p in predictions]}
    decision = decide_test(results)
    report = {'protocol': PROTOCOL, 'split': 'test', 'tracks': len(rows['test']), 'rule': TEST_RULE,
              'results': {n: {k: v for k, v in r.items() if k != 'sections'} for n, r in results.items()},
              'comparisons': decision, 'decision': decision['decision'],
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'limitations': ['PyTorch on original feature caches; Core ML/iPhone runtime still to be measured.',
                              'This test split was used once before for the production pipeline; do not retune on it.',
                              'Boundaries are the fixed Apple boundaries; boundary accuracy is not evaluated.']}
    write(output / 'test-comparison.json', report)
    write(output / 'test-sections.json', {n: r['sections'] for n, r in results.items()})
    print(f"{'pipeline':<11} {'head acc':>8} {'head F1':>8} {'acc':>7} {'F1':>7} {'bridge F1':>9}")
    for name in NAMES:
        r = report['results'][name]
        print(f"{name:<11} {r['head_only']['section_accuracy']:>8.4f} {r['head_only']['macro_f1_present']:>8.4f} "
              f"{r['pipeline']['section_accuracy']:>7.4f} {r['pipeline']['macro_f1_present']:>7.4f} "
              f"{r['per_class']['bridge']['f1']:>9.4f}")
    for name in ('all4', 'muq'):
        c = decision[name]
        print(f"{name} vs production: dAcc={c['delta_section_accuracy']:+.4f} dF1={c['delta_macro_f1_present']:+.4f} -> {c['verdict']}")
    print('Decision (muq vs production):', report['decision'], '| report:', output / 'test-comparison.json')


if __name__ == '__main__':
    main()
