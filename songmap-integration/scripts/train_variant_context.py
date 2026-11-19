#!/usr/bin/env python3
"""Retrain the section-context residual on top of an encoder-stream-ablation head.

The current app pipeline is worship-v3 + a context residual trained on worship-v3
section probabilities and 4096-wide pooled audio. A stream variant (e.g. MuQ only)
changes both inputs, so its context must be retrained before comparing whole
pipelines. This script:

1. loads an `encoder_stream_ablation.py` best.pt (reduced input projection);
2. computes that head's section probabilities on approved train and validation
   targets from the existing caches (retained columns only; no encoder runs);
3. pools section audio means and keeps only the retained stream columns
   (exactly equal to pooling the sliced features, since pooling is linear);
4. trains the unchanged context recipe (`train_section_context.SETTINGS`) with a
   zero-initialized residual and validation early stopping.

`compare` checks the variant pipeline against the all4 pipeline trained the same
way AND against the original-feature production context on validation. The test
split is never read here.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_stream_ablation as ablation
from train_section_context import (LABELS, SETTINGS, SectionContextHead, atomic_torch_save, evaluate_context,
                                   prepare_samples, supervised_loss, validate_splits)
from calibrate_bridge_chorus import metrics

PROTOCOL = 'variant-section-context-v1'
# Original-feature worship-v3 + context epoch 5 on the same 32 validation songs / 451 sections.
PRODUCTION_VALIDATION = {'section_accuracy': 0.793792, 'macro_f1_present': 0.775538,
                         'source': 'original-feature worship-v3 head + section context (selected epoch 5)'}
DECISION_RULE = {'max_section_accuracy_drop': 0.01, 'max_macro_f1_drop': 0.01,
                 'paired_reference': 'all4 head + context retrained with this script',
                 'production_reference': PRODUCTION_VALIDATION, 'split': 'val',
                 'bootstrap': ablation.DECISION_RULE['bootstrap']}


def slice_samples(samples, cols):
    """Keep retained stream columns of 4096-wide pooled section means."""
    if len(cols) == len(ablation.STREAMS) * ablation.WIDTH:
        return samples
    index = torch.as_tensor(cols, dtype=torch.long)
    result = []
    for sample in samples:
        if sample['audio'].ndim != 2 or sample['audio'].shape[1] != len(ablation.STREAMS) * ablation.WIDTH:
            raise ValueError('Expected 4096-wide pooled section audio.')
        result.append({**sample, 'audio': sample['audio'].index_select(1, index).contiguous()})
    return result


def fit_context(model, train_samples, val_samples, val_rows, target, settings, log=print):
    """Same loop as train_section_context.main: song-level accumulation, val macro-F1 early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    initial, _ = evaluate_context(model, val_samples, val_rows, target)
    best_score, best_epoch, bad, history = initial['macro_f1_present'], 0, 0, [{'epoch': 0, 'val': initial}]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    generator = random.Random(settings['seed'])
    for epoch in range(1, settings['epochs'] + 1):
        if bad >= settings['patience']:
            break
        started = time.perf_counter(); model.train(); order = list(train_samples); generator.shuffle(order)
        optimizer.zero_grad(set_to_none=True); accumulated = targets = 0; loss_sum = 0.
        for i, sample in enumerate(order, 1):
            loss = supervised_loss(model, sample, target, settings['residual_penalty'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite context training loss.')
            loss.backward(); accumulated += len(sample['labels']); targets += len(sample['labels']); loss_sum += loss.item()
            if i % settings['accumulation'] == 0 or i == len(order):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
        result, _ = evaluate_context(model, val_samples, val_rows, target)
        improved = result['macro_f1_present'] > best_score + 1e-12
        if improved:
            best_score, best_epoch, bad = result['macro_f1_present'], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append({'epoch': epoch, 'train_loss': loss_sum / targets, 'val': result,
                        'seconds': time.perf_counter() - started})
        log(f'Epoch {epoch}: loss={loss_sum/targets:.4f} acc={result["section_accuracy"]:.4f} '
            f'F1={result["macro_f1_present"]:.4f}' + (' * best' if improved else ''))
    model.load_state_dict(best_state)
    return best_state, best_epoch, history


def head_predictions(model, rows, cfg, target, provenance, cols, evaluate, iter_features, cache_path, identity, write, read):
    """Section probabilities of the variant head for approved rows, cached per head identity."""
    if cache_path.exists():
        cached = read(cache_path)
        if cached['identity'] != identity:
            raise ValueError(f'{cache_path.name} belongs to another head/inventory; use a new output directory.')
        return cached['metrics'], cached['sections']
    details = []
    result = evaluate(model, rows, cfg, target, provenance, details=details,
                      feature_reader=ablation.sliced(iter_features, cols))
    write(cache_path, {'identity': identity, 'metrics': result, 'sections': details})
    return result, details


def run_train(args):
    # Resolve every path first: SongStruct's load_head changes the working directory.
    for name in ('songstruct_root', 'head_checkpoint', 'output_dir', 'cache_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    root = args.songstruct_root
    sys.path.insert(0, str(root))
    from finetuning.common import config, device, fingerprint, iter_features, load_head, read, sha_file, write
    from finetuning.engine import evaluate, manifest
    cfg = config(root / 'finetuning/config.json')
    data, provenance = manifest(cfg)
    train = [r for r in data['records'] if r['approved'] and r['split'] == 'train']
    val = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    validate_splits(train, val)
    target = device(args.device)
    head_path = args.head_checkpoint.expanduser().resolve()
    head, saved = ablation.load_variant(cfg, target, head_path, load_head)
    if saved['plan_id'] != data['plan_id'] or saved['provenance_id'] != provenance:
        raise ValueError('Head checkpoint belongs to another split/cache.')
    head.eval()
    cols, variant = saved['columns'], saved['variant']
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    head_sha = sha_file(head_path)
    identity = lambda split, rows: fingerprint({'protocol': PROTOCOL, 'head_sha256': head_sha, 'split': split,
                                                'device': str(target), 'plan_id': data['plan_id'], 'provenance_id': provenance,
                                                'annotations': [(r['id'], r['annotation_sha256']) for r in rows]})
    rows_train = train[:1] if args.smoke_test else train
    rows_val = val[:1] if args.smoke_test else val
    tag = 'smoke-' if args.smoke_test else ''
    print(f'{variant}: head predictions for {len(rows_val)} val / {len(rows_train)} train songs (cached features only)', flush=True)
    val_metrics, val_details = head_predictions(head, rows_val, cfg, target, provenance, cols, evaluate, iter_features,
                                                out / f'{tag}head_val.json', identity('val', rows_val), write, read)
    if not args.smoke_test and not ablation.reproduces(val_metrics, saved['validation']):
        raise ValueError('Head does not reproduce its saved validation; check device/checkpoint before training context.')
    _, train_details = head_predictions(head, rows_train, cfg, target, provenance, cols, evaluate, iter_features,
                                        out / f'{tag}head_train.json', identity('train', rows_train), write, read)
    del head
    if target.type == 'mps':
        torch.mps.empty_cache()
    random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed']); torch.manual_seed(SETTINGS['seed'])
    cache = args.cache_dir.expanduser().resolve() / variant
    cache.mkdir(parents=True, exist_ok=True)
    helpers = read, write, fingerprint, sha_file, iter_features
    train_samples = slice_samples(prepare_samples(rows_train, train_details, cfg, provenance, cache, helpers), cols)
    val_samples = slice_samples(prepare_samples(rows_val, val_details, cfg, provenance, cache, helpers), cols)
    model = SectionContextHead(SETTINGS['hidden'], SETTINGS['dropout'], width=len(cols)).to(target)
    baseline = metrics(val_details, {r['track_id']: r for r in rows_val}, {'kind': 'identity'})
    initial, _ = evaluate_context(model, val_samples, rows_val, target)
    if initial['confusion_matrix'] != baseline['confusion_matrix']:
        raise ValueError('Zero residual did not reproduce the variant head predictions.')
    if args.smoke_test:
        model.train()
        loss = supervised_loss(model, train_samples[0], target, SETTINGS['residual_penalty'])
        loss.backward()
        print(f'Smoke OK: context width {len(cols)}, loss={loss.item():.4f}. Nothing else saved.', flush=True)
        return
    print(f'Context: width={len(cols)} trainable={sum(p.numel() for p in model.parameters()):,}; '
          f'head-only val acc={baseline["section_accuracy"]:.4f} F1={baseline["macro_f1_present"]:.4f}', flush=True)
    best_state, best_epoch, history = fit_context(model, train_samples, val_samples, rows_val, target, SETTINGS,
                                                  log=lambda text: print(text, flush=True))
    candidate, predictions = evaluate_context(model, val_samples, rows_val, target)
    atomic_torch_save(out / 'best.pt', {'protocol': PROTOCOL, 'architecture': 'section-context-v1', 'settings': SETTINGS,
                                        'width': len(cols), 'columns': cols, 'variant': variant, 'model': best_state,
                                        'best_epoch': best_epoch, 'head_sha256': head_sha,
                                        'plan_id': data['plan_id'], 'provenance_id': provenance})
    write(out / 'history.json', history)
    write(out / 'val_sections.json', {'sections': [{**p, 'correct': p['prediction'] == p['label_manual']} for p in predictions]})
    write(out / 'summary.json', {'protocol': PROTOCOL, 'variant': variant, 'columns_count': len(cols),
                                 'head_checkpoint': str(head_path), 'head_sha256': head_sha, 'settings': SETTINGS,
                                 'device': str(target), 'best_epoch': best_epoch,
                                 'head_only': {k: baseline[k] for k in ('sections', 'section_accuracy', 'macro_f1_present')},
                                 'pipeline': {k: candidate[k] for k in ('sections', 'section_accuracy', 'macro_f1_present')},
                                 'per_class': candidate['per_class'], 'confusion_matrix': candidate['confusion_matrix'],
                                 'test_used': False, 'encoder_executed': False,
                                 'finished_utc': datetime.now(timezone.utc).isoformat()})
    print(f'Best epoch {best_epoch}: pipeline val acc={candidate["section_accuracy"]:.4f} '
          f'F1={candidate["macro_f1_present"]:.4f} -> {out}', flush=True)


def decide(control, variant, labels=LABELS, rule=DECISION_RULE):
    paired = ablation.paired_comparison(control, variant, labels)
    production = rule['production_reference']
    against_production = {k: paired['variant'][k] - production[k] for k in ('section_accuracy', 'macro_f1_present')}
    production_ok = (against_production['section_accuracy'] >= -rule['max_section_accuracy_drop'] - 1e-12
                     and against_production['macro_f1_present'] >= -rule['max_macro_f1_drop'] - 1e-12)
    verdict = 'candidate' if paired['verdict'] == 'candidate' and production_ok else 'rejected'
    return {**paired, 'delta_vs_production_validation': against_production, 'production_reference': production,
            'paired_verdict': paired['verdict'], 'production_verdict': 'pass' if production_ok else 'fail',
            'verdict': verdict,
            'verdict_scope': 'Validation only (already used for selection). Next: one test-split evaluation of the '
                             'selected pipeline, then Core ML export and iPhone measurement. Not production approval.'}


def run_compare(args):
    def load(folder):
        folder = folder.expanduser().resolve()
        summary = json.loads((folder / 'summary.json').read_text())
        return summary, json.loads((folder / 'val_sections.json').read_text())['sections']
    (control_summary, control), (variant_summary, variant) = load(args.control), load(args.variant)
    if control_summary['variant'] != ablation.CONTROL:
        raise ValueError('--control must be the all4 context run.')
    if control_summary['settings'] != variant_summary['settings']:
        raise ValueError('Control and variant context recipes differ.')
    report = {'protocol': PROTOCOL, 'decision_rule': DECISION_RULE,
              'control': control_summary, 'variant': variant_summary, 'comparison': decide(control, variant)}
    output = args.variant.expanduser().resolve() / 'comparison.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    c = report['comparison']
    print(f"control all4 pipeline: acc={c['control']['section_accuracy']:.4f} F1={c['control']['macro_f1_present']:.4f}")
    print(f"{variant_summary['variant']} pipeline: acc={c['variant']['section_accuracy']:.4f} F1={c['variant']['macro_f1_present']:.4f}")
    print(f"vs all4: dAcc={c['delta_section_accuracy']:+.4f} dF1={c['delta_macro_f1_present']:+.4f} ({c['paired_verdict']})")
    print(f"vs production {PRODUCTION_VALIDATION['section_accuracy']:.4f}/{PRODUCTION_VALIDATION['macro_f1_present']:.4f}: "
          f"dAcc={c['delta_vs_production_validation']['section_accuracy']:+.4f} "
          f"dF1={c['delta_vs_production_validation']['macro_f1_present']:+.4f} ({c['production_verdict']})")
    print('Verdict:', c['verdict'], '| report:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--head-checkpoint', type=Path, required=True, help='encoder_stream_ablation best.pt')
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--cache-dir', type=Path, required=True, help='Pooled section cache root (one subfolder per variant)')
    train.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    train.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    train.add_argument('--smoke-test', action='store_true')
    compare = sub.add_parser('compare')
    compare.add_argument('--control', type=Path, required=True, help='all4 context output directory')
    compare.add_argument('--variant', type=Path, required=True, help='variant context output directory')
    args = parser.parse_args()
    run_train(args) if args.command == 'train' else run_compare(args)


if __name__ == '__main__':
    main()
