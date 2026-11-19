#!/usr/bin/env python3
"""Encoder-stream ablation on frozen worship feature caches.

Question: can SongFormer's label head keep worship accuracy when one encoder
(MusicFM or MuQ) or the 120-second streams are removed? Removing one encoder
removes its compiled Core ML bank (about half of the 351.9-MiB canonical encoder
bytes) and roughly half of encoder/frontend inference. Removing the 120-second
streams keeps both banks but removes about half of encoder inference.

Only cached layer-10 features are read. No encoder runs, no audio is decoded,
annotations/boundaries/splits are unchanged and the test split is never read.
Each variant starts from the same fine-tuned checkpoint (worship-v3). The input
projection `mixed_win_downsample` (Linear 4096 -> 2048) keeps only the weight
columns of the retained streams; the following LayerNorm makes the start
scale-invariant. The `all4` control is trained with the identical recipe so a
variant is compared against equal extra training, not against an older run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

PROTOCOL = 'encoder-stream-ablation-v1'
WIDTH = 1024
# Column blocks of the cached 4096-wide feature, in extraction order.
STREAMS = ('musicfm30', 'muq30', 'musicfm120', 'muq120')
VARIANTS = {
    'all4': STREAMS,
    'musicfm': ('musicfm30', 'musicfm120'),
    'muq': ('muq30', 'muq120'),
    'short30': ('musicfm30', 'muq30'),
    'musicfm30': ('musicfm30',),
    'muq30': ('muq30',),
}
CONTROL = 'all4'
# Fixed BEFORE any variant result exists. A variant passing this rule is only a
# candidate for the next stage (context retraining, Core ML export, iPhone).
DECISION_RULE = {'max_section_accuracy_drop': 0.01, 'max_macro_f1_drop': 0.01,
                 'reference': CONTROL, 'split': 'val',
                 'bootstrap': {'unit': 'track', 'replicates': 2000, 'seed': 42, 'interval': 0.95}}
STAGES = {
    'projection_head': ('mixed_win_downsample.', 'input_norm.', 'function_head.'),
    'label_backbone': None,  # everything except boundary_head
}
# 1e-4 diverged on the all4 control (train loss rose from epoch 2); the earlier
# label_backbone run was stable at 1e-5.
DEFAULT_LR = {'projection_head': 1e-5, 'label_backbone': 1e-5}
SETTINGS = dict(epochs=20, patience=4, weight_decay=1e-4, max_grad_norm=1.0)
# Same window as the earlier label_backbone run that fit the 24-GB Mac.
DEFAULT_WINDOW_SECONDS = 30.
HEAD_WINDOW_SECONDS = 420.


def columns(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}; choose from {", ".join(VARIANTS)}.')
    return [STREAMS.index(s) * WIDTH + i for s in VARIANTS[variant] for i in range(WIDTH)]


def implications(variant, encoder_bytes=None):
    """Static deployment consequences. Compute fractions are approximate counts of
    full-song encoder passes (each stream encodes the whole song once)."""
    kept = VARIANTS[variant]
    encoders = sorted({s.rstrip('0123456789') for s in kept})
    sizes = encoder_bytes or {'musicfm': 1, 'muq': 1}
    if set(sizes) != {'musicfm', 'muq'} or any(v <= 0 for v in sizes.values()):
        raise ValueError('encoder_bytes needs positive musicfm and muq values.')
    kept_bytes = sum(sizes[e] for e in encoders)
    result = {'streams': list(kept), 'encoders': encoders,
              'encoder_bytes_fraction': kept_bytes / sum(sizes.values()),
              'encoder_passes_fraction': len(kept) / len(STREAMS),
              'input_dim_raw': len(kept) * WIDTH}
    if encoder_bytes:
        result['encoder_bytes'] = kept_bytes
    return result


def resolve_settings(recipe, stage, seed, learning_rate=None, epochs=None, patience=None, window_seconds=None,
                     gradient_accumulation=None):
    """Inherit the start checkpoint's audio per update, but never full 420-second training chunks.

    worship-v3 trained only the top function head, so full chunks were cheap. Here the
    gradient reaches the input projection through the whole transformer; on a 24-GB Mac a
    420-second chunk exhausts MPS memory. Train on fixed windows (inference is unchanged)
    and scale accumulation so each optimizer step still covers about the same audio as
    the start recipe (4 x 420 s -> 56 x 30 s), instead of ~2 minutes of one song.
    """
    window = window_seconds or recipe.get('window_seconds') or DEFAULT_WINDOW_SECONDS
    if not math.isfinite(window) or not 0.12 <= window <= HEAD_WINDOW_SECONDS:
        raise ValueError('window_seconds must be within 0.12-420 seconds.')
    source_window = recipe.get('window_seconds') or HEAD_WINDOW_SECONDS
    accumulation = gradient_accumulation or max(1, round(recipe['gradient_accumulation'] * source_window / window))
    return dict(SETTINGS, seed=seed, epochs=epochs or SETTINGS['epochs'],
                patience=patience or SETTINGS['patience'],
                learning_rate=learning_rate or DEFAULT_LR[stage],
                gradient_accumulation=int(accumulation), window_seconds=float(window),
                gradient_checkpointing=recipe.get('gradient_checkpointing', False))


def reduce_projection(model, cols):
    """Replace Linear(4096, d) with Linear(len(cols), d) holding the kept columns."""
    import torch
    layer = model.mixed_win_downsample
    if layer.in_features != len(STREAMS) * WIDTH:
        raise ValueError('Expected the original 4096-wide fused input projection.')
    if len(cols) == layer.in_features:
        return model
    index = torch.as_tensor(cols, dtype=torch.long, device=layer.weight.device)
    reduced = torch.nn.Linear(len(cols), layer.out_features, bias=layer.bias is not None,
                              device=layer.weight.device, dtype=layer.weight.dtype)
    with torch.no_grad():
        reduced.weight.copy_(layer.weight.index_select(1, index))
        if layer.bias is not None:
            reduced.bias.copy_(layer.bias)
    model.mixed_win_downsample = reduced
    return model


def sliced(reader, cols):
    """Wrap a cache reader so every chunk yields only the kept columns (owned float32)."""
    count = len(STREAMS) * WIDTH
    spans = []
    for c in cols:
        if spans and spans[-1][1] == c:
            spans[-1][1] = c + 1
        else:
            spans.append([c, c + 1])

    def read(cfg, row):
        for values, offset, signature in reader(cfg, row):
            if values.ndim != 2 or values.shape[1] != count:
                raise ValueError('Expected 4096-wide cached features.')
            if len(cols) == count:
                yield values, offset, signature
                continue
            yield (np.ascontiguousarray(np.concatenate([values[:, a:b] for a, b in spans], axis=1),
                                        dtype=np.float32), offset, signature)
    return read


def configure(model, stage):
    if stage not in STAGES:
        raise ValueError('trainable must be ' + ' or '.join(STAGES))
    prefixes = STAGES[stage]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes) if prefixes else not name.startswith('boundary_head.'))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters.')
    return params


def train_mode(model, stage):
    if stage == 'projection_head':
        model.eval()  # frozen dropout paths stay deterministic
        for name in ('mixed_win_downsample', 'input_norm', 'function_head'):
            getattr(model, name).train()
    else:
        model.train()
        model.boundary_head.eval()


def section_scores(items, labels):
    """Same formulas as finetuning.engine.metrics_from_sections."""
    matrix = [[0] * len(labels) for _ in labels]
    for item in items:
        matrix[labels.index(item['label_manual'])][labels.index(item['prediction'])] += 1
    f1s = []
    for i in range(len(labels)):
        support, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
        if support:
            f1s.append(2 * matrix[i][i] / (support + predicted))
    count = len(items)
    return {'sections': count,
            'section_accuracy': sum(matrix[i][i] for i in range(len(labels))) / count if count else 0.,
            'macro_f1_present': sum(f1s) / len(f1s) if f1s else 0.}


def paired_comparison(control, variant, labels, rule=DECISION_RULE):
    """Compare per-section predictions on IDENTICAL eligible validation targets."""
    def keyed(items):
        result = {}
        for item in items:
            key = (item['track_id'], round(float(item['start']), 6), round(float(item['end']), 6))
            if key in result:
                raise ValueError('Duplicate section in predictions.')
            result[key] = item
        return result
    a, b = keyed(control), keyed(variant)
    if not a or set(a) != set(b):
        raise ValueError('Control and variant were not evaluated on the same sections.')
    if any(a[k]['label_manual'] != b[k]['label_manual'] for k in a):
        raise ValueError('Manual targets differ between control and variant.')
    keys = sorted(a)
    base, cand = section_scores([a[k] for k in keys], labels), section_scores([b[k] for k in keys], labels)
    delta_acc = cand['section_accuracy'] - base['section_accuracy']
    delta_f1 = cand['macro_f1_present'] - base['macro_f1_present']
    only_control = sum(a[k]['correct'] and not b[k]['correct'] for k in keys)
    only_variant = sum(b[k]['correct'] and not a[k]['correct'] for k in keys)
    tracks = sorted({k[0] for k in keys})
    by_track = {t: [k for k in keys if k[0] == t] for t in tracks}
    settings = rule['bootstrap']
    generator = np.random.default_rng(settings['seed'])
    draws = {'section_accuracy': [], 'macro_f1_present': []}
    for _ in range(settings['replicates']):
        sample = [k for t in generator.choice(tracks, size=len(tracks), replace=True) for k in by_track[t]]
        x, y = section_scores([a[k] for k in sample], labels), section_scores([b[k] for k in sample], labels)
        for name in draws:
            draws[name].append(y[name] - x[name])
    tail = (1 - settings['interval']) / 2
    interval = {name: [float(np.quantile(v, tail)), float(np.quantile(v, 1 - tail))] for name, v in draws.items()}
    passed = (delta_acc >= -rule['max_section_accuracy_drop'] - 1e-12
              and delta_f1 >= -rule['max_macro_f1_drop'] - 1e-12)
    return {'sections': len(keys), 'tracks': len(tracks), 'control': base, 'variant': cand,
            'delta_section_accuracy': delta_acc, 'delta_macro_f1_present': delta_f1,
            'bootstrap_interval': interval, 'only_control_correct': only_control,
            'only_variant_correct': only_variant,
            'changed_sections': [dict(track_id=k[0], start=k[1], end=k[2], label_manual=a[k]['label_manual'],
                                      control=a[k]['prediction'], variant=b[k]['prediction'])
                                 for k in keys if a[k]['prediction'] != b[k]['prediction']],
            'verdict': 'candidate' if passed else 'rejected',
            'verdict_scope': 'Validation non-inferiority against the equally trained all4 control only. '
                             'Not a test-split result, context-model result, Core ML/iPhone result or production approval.'}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def train_variant(model, stage, settings, train_rows, read, step_loss, validate, save, log=print):
    """Shared epoch loop: gradient accumulation, clipping, early stopping on val macro F1.

    `step_loss(model, values, offset, row)` yields per-window losses; `validate(model)`
    returns (metrics, details); `save(tag, epoch, metrics, details)` persists results.
    Epoch 0 is the untrained (column-dropped) start and is always saved as best first.
    """
    import torch
    params = configure(model, stage)
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    metrics, details = validate(model)
    best, best_epoch, bad, history = metrics['macro_f1_present'], 0, 0, []
    save('best', 0, metrics, details)
    history.append({'epoch': 0, 'val': metrics})
    log(f"Epoch 0 (start, no training): acc={metrics['section_accuracy']:.4f} F1={metrics['macro_f1_present']:.4f}")
    generator = random.Random(settings['seed'])

    def update(count):
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(params, settings['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, settings['epochs'] + 1):
        train_mode(model, stage)
        order = list(train_rows)
        generator.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        pending, losses = 0, []
        for row in order:
            for values, offset, _ in read(row):
                for loss in step_loss(model, values, offset, row):
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite label loss.')
                    loss.backward()
                    losses.append(float(loss.detach()))
                    pending += 1
                    if pending == settings['gradient_accumulation']:
                        update(pending)
                        pending = 0
        if pending:
            update(pending)
        if not losses:
            raise ValueError('No supervised train section.')
        metrics, details = validate(model)
        history.append({'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'val': metrics})
        log(f"Epoch {epoch}: loss={history[-1]['train_loss']:.4f} acc={metrics['section_accuracy']:.4f} "
            f"F1={metrics['macro_f1_present']:.4f}")
        if metrics['macro_f1_present'] > best:
            best, best_epoch, bad = metrics['macro_f1_present'], epoch, 0
            save('best', epoch, metrics, details)
        else:
            bad += 1
            if bad >= settings['patience']:
                log('Early stopping on validation macro F1.')
                break
    return {'best_epoch': best_epoch, 'best_macro_f1_present': best, 'history': history}


def reproduces(metrics, stored):
    return bool(stored) and all(abs(metrics[k] - stored[k]) <= 1e-9
                                for k in ('sections', 'section_accuracy', 'macro_f1_present'))


def load_variant(cfg, target, path, load_head):
    """Load a best.pt written by this script (reduced projection) for later stages."""
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('protocol') != PROTOCOL or saved.get('columns') != columns(saved.get('variant')):
        raise ValueError('Not an encoder-stream-ablation checkpoint.')
    model, _ = load_head(cfg, 'cpu')
    reduce_projection(model, saved['columns'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(target), saved


def run_train(args):
    root = args.songstruct_root.expanduser().resolve()
    if not (root / 'finetuning/config.json').is_file():
        raise ValueError(f'SongStruct root missing: {root}')
    sys.path.insert(0, str(root))
    from finetuning.common import CLASS_IDS, LABELS, config, device, fingerprint, iter_features, load_head
    from finetuning.engine import evaluate, label_loss, logits, manifest, training_windows
    import torch
    cfg = config(root / 'finetuning/config.json')
    data, provenance = manifest(cfg)
    train = [r for r in data['records'] if r['approved'] and r['split'] == 'train']
    val = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    if not train or not val:
        raise ValueError('Approved train and validation records are required.')
    if {r['group_id'] for r in train} & {r['group_id'] for r in val}:
        raise ValueError('Train/validation group leakage.')
    checkpoint = args.init_checkpoint.expanduser().resolve()
    init = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (init.get('plan_id') != data['plan_id'] or init.get('provenance_id') != provenance
            or init.get('config', {}).get('features') != cfg['features']):
        raise ValueError('Initial checkpoint does not match this split/cache.')
    recipe = init['config']['training']
    stage = args.trainable
    settings = resolve_settings(recipe, stage, cfg['seed'], learning_rate=args.learning_rate, epochs=args.epochs,
                                patience=args.patience, window_seconds=args.window_seconds,
                                gradient_accumulation=args.gradient_accumulation)
    class_weights = init.get('class_weights')
    cols = columns(args.variant)
    out = args.output_dir.expanduser().resolve() / args.variant
    if not args.smoke_test:
        out.mkdir(parents=True, exist_ok=False)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    target = device(args.device)
    model, _ = load_head(cfg, target, checkpoint)
    reduce_projection(model, cols)
    reader = sliced(iter_features, cols)
    fractional = settings['window_seconds'] is not None

    def read(row):
        for values, offset, signature in reader(cfg, row):
            if signature != provenance:
                raise ValueError('Mixed encoder caches.')
            yield values, offset, signature

    def step_loss(model, values, offset, row):
        for piece, start in training_windows(values, offset, settings['window_seconds']):
            scores = logits(model, piece, target, checkpoint=settings['gradient_checkpointing'])
            loss = label_loss(scores, row['sections'], start, fractional=fractional, class_weights=class_weights)
            if loss is not None:
                yield loss

    def validate(model):
        details = []
        metrics = evaluate(model, val, cfg, target, provenance, details=details, feature_reader=reader)
        return metrics, details

    meta = {'protocol': PROTOCOL, 'variant': args.variant, 'columns': cols, 'implications': implications(args.variant),
            'decision_rule': DECISION_RULE, 'trainable': stage, 'settings': settings, 'class_weights': class_weights,
            'plan_id': data['plan_id'], 'provenance_id': provenance, 'config': cfg,
            'init_checkpoint': str(checkpoint), 'init_checkpoint_sha256': sha_file(checkpoint),
            'init_validation': init.get('validation'),
            'annotations_id': fingerprint([(r['id'], r['annotation_sha256']) for r in train + val]),
            'labels': list(LABELS), 'class_ids': list(CLASS_IDS),
            'train_tracks': len(train), 'val_tracks': len(val), 'test_split_read': False}
    trainable = sum(p.numel() for p in configure(model, stage))
    print(f'{args.variant}: streams={"+".join(VARIANTS[args.variant])} input={len(cols)} '
          f'trainable={trainable:,} train={len(train)} val={len(val)} device={target}', flush=True)
    if args.smoke_test:
        train_mode(model, stage)
        for row in train:
            for values, offset, _ in read(row):
                for loss in step_loss(model, values, offset, row):
                    loss.backward()
                    print(f'Smoke OK: loss={loss.item():.4f}, input width {values.shape[1]}. Nothing saved.')
                    return
        raise ValueError('No supervised window for smoke test.')
    write_json(out / 'run.json', {**meta, 'trainable_parameters': trainable,
                                  'started_utc': datetime.now(timezone.utc).isoformat()})

    def save(tag, epoch, metrics, details):
        torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    **meta, 'epoch': epoch, 'validation': metrics}, out / f'{tag}.pt')
        write_json(out / f'{tag}_val.json', {'epoch': epoch, **metrics})
        write_json(out / f'{tag}_val_sections.json', {'epoch': epoch, 'sections': details})

    summary = train_variant(model, stage, settings, train, read, step_loss, validate, save,
                            log=lambda text: print(text, flush=True))
    history = summary.pop('history')
    if args.variant == CONTROL:
        # The all4 start IS the unchanged initial checkpoint; record whether it reproduces
        # the validation stored in that checkpoint (differs if validation targets changed since).
        summary['init_reproduced'] = reproduces(history[0]['val'], init.get('validation'))
        if not summary['init_reproduced']:
            print('WARNING: all4 epoch 0 differs from the validation stored in the initial checkpoint. '
                  'Validation targets or features changed since it was trained; compare variants only '
                  'against this all4 control.', flush=True)
    with (out / 'history.jsonl').open('w', encoding='utf-8') as handle:
        for item in history:
            handle.write(json.dumps(item, allow_nan=False) + '\n')
    write_json(out / 'summary.json', {**summary, 'variant': args.variant,
                                      'finished_utc': datetime.now(timezone.utc).isoformat()})
    print(f'Best epoch {summary["best_epoch"]}: val macro F1 {summary["best_macro_f1_present"]:.4f} -> {out}')


def parse_bytes(text):
    if not text:
        return None
    values = dict(part.split('=', 1) for part in text.split(','))
    return {k.strip(): int(v) for k, v in values.items()}


def run_compare(args):
    root = args.output_dir.expanduser().resolve()
    encoder_bytes = parse_bytes(args.encoder_bytes)
    runs = {}
    for variant in VARIANTS:
        folder = root / variant
        if (folder / 'summary.json').is_file():
            runs[variant] = (json.loads((folder / 'run.json').read_text()),
                             json.loads((folder / 'best_val_sections.json').read_text())['sections'])
    if CONTROL not in runs:
        raise ValueError('The completed all4 control is required before comparing.')
    control_meta, control = runs[CONTROL]
    identity = ('plan_id', 'provenance_id', 'annotations_id', 'init_checkpoint_sha256', 'trainable', 'settings')
    labels = control_meta['labels']
    report = {'protocol': PROTOCOL, 'decision_rule': DECISION_RULE, 'variants': {}}
    for variant, (meta, details) in runs.items():
        if any(meta[k] != control_meta[k] for k in identity):
            raise ValueError(f'{variant} used a different split, cache, start checkpoint or recipe.')
        entry = {'implications': implications(variant, encoder_bytes),
                 'best_epoch': json.loads((root / variant / 'summary.json').read_text())['best_epoch']}
        if variant == CONTROL:
            entry['validation'] = section_scores(details, labels)
        else:
            entry['comparison'] = paired_comparison(control, details, labels)
        report['variants'][variant] = entry
    write_json(root / 'comparison.json', report)
    print(f"{'variant':<10} {'encoders':<12} {'bytes':>6} {'passes':>6} {'acc':>7} {'F1':>7} {'dAcc':>7} {'dF1':>7}  verdict")
    for variant, entry in report['variants'].items():
        imp = entry['implications']
        score = entry.get('validation') or entry['comparison']['variant']
        delta = entry.get('comparison', {})
        print(f"{variant:<10} {'+'.join(imp['encoders']):<12} {imp['encoder_bytes_fraction']:>6.0%} "
              f"{imp['encoder_passes_fraction']:>6.0%} {score['section_accuracy']:>7.4f} {score['macro_f1_present']:>7.4f} "
              f"{delta.get('delta_section_accuracy', 0):>+7.4f} {delta.get('delta_macro_f1_present', 0):>+7.4f}  "
              f"{delta.get('verdict', 'control')}")
    print('Report:', root / 'comparison.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--variant', choices=list(VARIANTS), required=True)
    train.add_argument('--init-checkpoint', type=Path, required=True, help='Fine-tuned best.pt, e.g. worship-v3')
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    train.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    train.add_argument('--trainable', choices=list(STAGES), default='projection_head')
    train.add_argument('--learning-rate', type=float)
    train.add_argument('--epochs', type=int)
    train.add_argument('--patience', type=int)
    train.add_argument('--gradient-accumulation', type=int, help='Windows per optimizer step (default: same audio per step as the start recipe)')
    train.add_argument('--window-seconds', type=float, help=f'Training window (default {DEFAULT_WINDOW_SECONDS:g}s; validation always uses full chunks)')
    train.add_argument('--smoke-test', action='store_true')
    compare = sub.add_parser('compare')
    compare.add_argument('--output-dir', type=Path, required=True)
    compare.add_argument('--encoder-bytes', help='Compiled bank bytes, e.g. musicfm=184000000,muq=185000000')
    args = parser.parse_args()
    if args.command == 'train':
        for name in ('learning_rate', 'epochs', 'patience', 'window_seconds', 'gradient_accumulation'):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                parser.error(f'--{name.replace("_", "-")} must be positive')
        run_train(args)
    else:
        run_compare(args)


if __name__ == '__main__':
    main()
