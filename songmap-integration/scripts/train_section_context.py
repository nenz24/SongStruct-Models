#!/usr/bin/env python3
"""Train a small section-context residual head on frozen audio features and manual train targets."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_train_bridge import LABELS, summarize
from calibrate_bridge_chorus import evaluate_validation, metrics

PROTOCOL = 'section-context-v1'
COVERAGE_POLICY = {'name': 'bounded-cache-tail-v1', 'review_below_fraction': .98,
                   'max_tail_seconds_per_chunk': 3.0 / 25}
SETTINGS = dict(hidden=64, dropout=0.25, epochs=40, learning_rate=0.0003,
                weight_decay=0.01, residual_penalty=0.01, accumulation=8,
                max_grad_norm=1.0, patience=6, seed=42)


def validate_splits(train, val):
    for rows, split in ((train, 'train'), (val, 'val')):
        if not rows or any(not r['approved'] or r['split'] != split for r in rows):
            raise ValueError('Only approved train and validation records are allowed.')
        if len({r['track_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate track.')
    for key in ('track_id', 'group_id', 'audio_sha256'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError('Train/validation leakage: ' + key)


def source_sections(document, record):
    """Use EVERY original boundary as input; only eligible manual targets receive loss."""
    duration = record['duration']
    if (not math.isfinite(duration) or duration <= 0 or
            abs(document['duration'] - duration) > 0.001 or document.get('track_id') != record['track_id']):
        raise ValueError('Source song identity/duration changed.')
    result = []
    for i, section in enumerate(document['sections']):
        start, end = section['start'], section['end']
        allowance = 0.020 + 1e-9 if i == len(document['sections']) - 1 else 0.
        if (not math.isfinite(start + end) or not 0 <= start < min(end, duration)
                or end > duration + allowance or (result and start < result[-1]['end'] - 1e-6)):
            raise ValueError('Invalid original section timing.')
        result.append({'start': start, 'end': min(end, duration)})
    if not result:
        raise ValueError('No source boundaries.')
    index = {(s['start'], s['end']): i for i, s in enumerate(result)}
    if len(index) != len(result):
        raise ValueError('Duplicate original boundaries.')
    try:
        targets = [index[(s['start'], s['end'])] for s in record['sections']]
    except KeyError as exc:
        raise ValueError('Manual target boundary differs from original source.') from exc
    return result, targets


def pool_audio(chunks, sections, provenance, width=4096):
    """Exact 25-Hz overlap pooling. Stream existing .npy chunks; no encoder inference."""
    sums = np.zeros((len(sections), width), dtype=np.float64)
    coverage = np.zeros(len(sections), dtype=np.float64)
    previous_end = -math.inf
    for values, offset, signature in chunks:
        if signature != provenance:
            raise ValueError('Mixed encoder provenance.')
        if (values.ndim != 2 or values.shape[1] != width or len(values) == 0
                or not math.isfinite(offset) or offset < previous_end - 1e-6):
            raise ValueError('Invalid or overlapping feature chunks.')
        previous_end = offset + len(values) / 25.
        for i, section in enumerate(sections):
            first = max(0, int(math.floor((section['start'] - offset) * 25)))
            last = min(len(values), int(math.ceil((section['end'] - offset) * 25)))
            for begin in range(first, last, 1024):
                stop = min(last, begin + 1024)
                left = offset + np.arange(begin, stop, dtype=np.float64) / 25.
                weights = np.maximum(0, np.minimum(left + .04, section['end']) - np.maximum(left, section['start']))
                block = np.asarray(values[begin:stop])
                if not np.isfinite(block).all():
                    raise ValueError('Non-finite audio feature.')
                sums[i] += np.einsum('n,nd->d', weights, block, dtype=np.float64, optimize=False)
                coverage[i] += weights.sum()
    if np.any(coverage <= 0):
        raise ValueError('Original section has no encoder feature coverage.')
    return (sums / coverage[:, None]).astype(np.float32), coverage


def geometry(sections, duration):
    return np.asarray([[s['start'] / duration, s['end'] / duration,
        math.log1p(s['end'] - s['start']) / math.log(601),
        max(0., s['start'] - sections[i-1]['end']) / duration if i else s['start'] / duration,
        max(0., sections[i+1]['start'] - s['end']) / duration if i+1 < len(sections) else (duration-s['end']) / duration,
        math.log1p(duration) / math.log(1201)] for i, s in enumerate(sections)], dtype=np.float32)


def coverage_review(record, sections, indices, coverage, feature_meta):
    """Allow <98% only if all missing time is explained by bounded cache-end truncation.

    The original extractor trims each 420s chunk to a multiple of three 25-Hz
    frames. Reuse its covered-time mean; do not invent features or trim targets.
    """
    diagnostics = []
    tolerance = 1e-6
    for i in indices:
        section = sections[i]
        duration = section['end'] - section['start']
        fraction = coverage[i] / duration
        error = (f'Manual target audio coverage invalid: {record["track_id"]}, '
                 f'{section["start"]:.3f}–{section["end"]:.3f}s, {fraction:.4%}.')
        if not math.isfinite(fraction) or fraction <= 0 or fraction > 1.0001:
            raise ValueError(error)
        if fraction >= COVERAGE_POLICY['review_below_fraction']:
            continue
        covered_by_metadata = 0.
        allowed_missing = 0.
        tails = []
        previous_end = -math.inf
        def overlap(start, end):
            return max(0., min(section['end'], end) - max(section['start'], start))
        for chunk in feature_meta.get('chunks', []):
            offset, frames = chunk['offset'], chunk['frames']
            if (not math.isfinite(offset) or offset < 0 or offset >= record['duration']
                    or not isinstance(frames, int) or frames <= 0):
                raise ValueError(error + ' Invalid feature metadata.')
            cache_end = offset + frames / 25.
            if offset < previous_end - tolerance:
                raise ValueError(error + ' Overlapping metadata chunks.')
            previous_end = cache_end
            covered_by_metadata += overlap(offset, cache_end)
            logical_end = min(offset + 420., record['duration'])
            gap = logical_end - cache_end
            if 0 < gap <= COVERAGE_POLICY['max_tail_seconds_per_chunk'] + tolerance:
                missing_here = overlap(cache_end, logical_end)
                if missing_here:
                    allowed_missing += missing_here
                    tails.append({'chunk_offset': offset, 'cache_end': cache_end,
                                  'logical_end': logical_end, 'missing_in_section_seconds': missing_here})
        missing = duration - float(coverage[i])
        if (abs(covered_by_metadata - coverage[i]) > tolerance or not tails
                or missing > allowed_missing + tolerance):
            raise ValueError(error + f' Missing {missing:.6f}s is not explained by bounded cache tails; '
                             'inspect feature metadata. Existing cache was preserved.')
        diagnostics.append({'track_id': record['track_id'], 'split': record['split'],
            'original_section_index': i + 1, **section, 'coverage_fraction': float(fraction),
            'covered_seconds': float(coverage[i]), 'missing_seconds': missing,
            'reason': 'bounded_cache_tail', 'tails': tails,
            'handling': 'Use existing covered-time mean; retain manual label and boundary.'})
    return diagnostics


class SectionContextHead(nn.Module):
    def __init__(self, hidden=64, dropout=.25, width=4096):
        super().__init__()
        # width < 4096 only for encoder-stream variants (e.g. 2048 MuQ30+MuQ120 pooled means).
        self.audio = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout))
        self.position = nn.Linear(6, hidden, bias=False)
        self.context = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout,
                                                  activation='gelu', batch_first=True)
        self.output = nn.Linear(hidden, 8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, audio, timing):
        tokens = self.audio(audio) + self.position(timing)
        return self.output(self.context(tokens.unsqueeze(0))[0])


def tensors(sample, target):
    return {k: sample[k].to(target) for k in ('audio', 'timing', 'indices', 'probabilities', 'labels')}


def forward_targets(model, sample, target):
    data = tensors(sample, target)
    residual = model(data['audio'], data['timing'])[data['indices']]
    scores = data['probabilities'].clamp_min(1e-8).log() + residual
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite context logits.')
    return scores, residual, data['labels']


def supervised_loss(model, sample, target, penalty):
    if sample['split'] != 'train':
        raise ValueError('Only manual TRAIN targets may contribute gradients.')
    scores, residual, labels = forward_targets(model, sample, target)
    return (F.cross_entropy(scores, labels, reduction='sum') +
            penalty * residual.square().mean(-1).sum())


def evaluate_context(model, samples, records, target, split='val'):
    if (split not in ('val', 'test') or not samples or not records
            or any(s['split'] != split for s in samples)
            or any(r['split'] != split or not r['approved'] for r in records)
            or [s['track_id'] for s in samples] != [r['track_id'] for r in records]):
        raise ValueError('Context evaluation requires matching approved records from the requested val/test split.')
    model.eval()
    details = []
    nll = 0.
    with torch.inference_mode():
        for sample in samples:
            scores, _, labels = forward_targets(model, sample, target)
            nll += F.cross_entropy(scores, labels, reduction='sum').item()
            probabilities = scores.softmax(-1).cpu().tolist()
            for item, p in zip(sample['details'], probabilities):
                prediction = LABELS[max(range(8), key=lambda j: p[j])]
                details.append({**item, 'prediction': prediction, 'probabilities': dict(zip(LABELS, p))})
    result = metrics(details, {r['track_id']: r for r in records}, {'kind': 'identity'})
    result['section_nll'] = nll / len(details)
    return result, details


def atomic_torch_save(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_samples(rows, details, cfg, provenance, cache, helpers, feature_cache_root=None):
    read, write, fingerprint, sha_file, iter_features = helpers
    candidate_header_sha = None
    if feature_cache_root is not None:
        feature_cache_root = Path(feature_cache_root).expanduser().resolve()
        header_path = feature_cache_root / 'encoder_provenance.json'
        header = read(header_path)
        if (header.get('id') != provenance or fingerprint({k: v for k, v in header.items() if k != 'id'}) != provenance
                or header.get('environment', {}).get('feature_settings') != cfg['features']):
            raise ValueError('Candidate encoder cache identity/profile changed.')
        candidate_header_sha = sha_file(header_path)
    by_track = {}
    for item in details:
        by_track.setdefault(item['track_id'], {})[(item['start'], item['end'])] = item
    samples = []
    for n, row in enumerate(rows, 1):
        sections, indices = source_sections(read(row['source_json']), row)
        ordered = [by_track[row['track_id']][(s['start'], s['end'])] for s in row['sections']]
        feature_meta = ((Path(cfg['work_dir']) / 'features') if feature_cache_root is None else feature_cache_root) / row['id'] / 'metadata.json'
        meta = read(feature_meta)
        if (meta['audio_sha256'] != row['audio_sha256']
                or (feature_cache_root is None and (meta['features'] != cfg['features'] or meta['provenance_id'] != provenance))
                or (feature_cache_root is not None and meta.get('candidate_encoder_id') != provenance)):
            raise ValueError('Frozen encoder cache differs from current manifest/config.')
        identity_fields = {'protocol': PROTOCOL, 'audio_sha256': row['audio_sha256'],
            'boundaries': sections, 'annotation_sha256': row['annotation_sha256'],
            'prediction_details': ordered, 'feature_metadata_sha256': sha_file(feature_meta),
            'provenance_id': provenance, 'feature_settings': cfg['features']}
        if candidate_header_sha is not None:
            identity_fields['candidate_header_sha256'] = candidate_header_sha
        identity = fingerprint(identity_fields)
        path, info = cache / (row['id'] + '.npz'), cache / (row['id'] + '.json')
        if info.exists():
            if (read(info)['identity'] != identity or not path.exists()
                    or read(info).get('array_sha256') != sha_file(path)):
                raise ValueError('Section cache changed/incomplete; choose a fresh --cache-dir.')
            with np.load(path, allow_pickle=False) as stored:
                audio = stored['audio'].copy()
                coverage = stored['coverage'].copy()
        else:
            print(f'Pooling cache [{n}/{len(rows)}]: {row["track_id"]}', flush=True)
            audio, coverage = pool_audio(iter_features(cfg, row), sections, provenance)
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, audio=audio, coverage=coverage)
            temporary.replace(path)
            write(info, {'identity': identity, 'array_sha256': sha_file(path),
                         'source_feature_metadata_sha256': sha_file(feature_meta)})
        if (audio.shape != (len(sections), 4096) or coverage.shape != (len(sections),)
                or not np.isfinite(audio).all() or not np.isfinite(coverage).all() or np.any(coverage <= 0)):
            raise ValueError('Invalid pooled cache.')
        coverage_diagnostics = coverage_review(row, sections, indices, coverage, meta)
        for item in coverage_diagnostics:
            print(f'Bounded cache tail: {row["track_id"]}, {item["start"]:.3f}–{item["end"]:.3f}s; '
                  f'coverage={item["coverage_fraction"]:.2%}, missing={item["missing_seconds"]:.6f}s. '
                  'Keeping covered-time mean and manual target.', flush=True)
        samples.append({'track_id': row['track_id'], 'split': row['split'], 'identity': identity,
            'audio': torch.from_numpy(audio), 'timing': torch.from_numpy(geometry(sections, row['duration'])),
            'indices': torch.tensor(indices, dtype=torch.long),
            'probabilities': torch.tensor([[s['probabilities'][label] for label in LABELS] for s in ordered]),
            'labels': torch.tensor([LABELS.index(s['label_manual']) for s in ordered], dtype=torch.long),
            'details': ordered, 'coverage_diagnostics': coverage_diagnostics})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    parser.add_argument('--calibration', type=Path, required=True, help='Completed original-feature bridge calibration comparison.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.songstruct_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from finetuning.common import config, read, write, fingerprint, sha_file, iter_features, device
    from finetuning.engine import manifest
    cfg = config(root / 'finetuning/config.json')
    data, provenance = manifest(cfg)
    train = [r for r in data['records'] if r['approved'] and r['split'] == 'train']
    val = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    validate_splits(train, val)
    calibration_path = args.calibration.expanduser().resolve()
    calibration = read(calibration_path)
    if (calibration.get('split') != 'val' or not calibration.get('full_validation')
            or calibration['tracks'] != len(val) or calibration['plan_id'] != data['plan_id']
            or calibration['provenance_id'] != provenance or calibration.get('test_used') is not False):
        raise ValueError('Expected completed original-feature validation calibration on this frozen plan.')
    checkpoint = Path(calibration['checkpoint']).resolve()
    if sha_file(checkpoint) != calibration['checkpoint_sha256']:
        raise ValueError('Original SongFormer checkpoint changed.')
    audit_path = Path(calibration['train_audit']).resolve()
    audit = read(audit_path)
    if (not audit.get('full_train') or audit.get('split') != 'train' or audit['tracks'] != len(train)
            or audit['plan_id'] != data['plan_id'] or audit['provenance_id'] != provenance
            or audit['checkpoint_sha256'] != calibration['checkpoint_sha256']):
        raise ValueError('Train audit identity changed.')
    validation_payload = {'protocol': 'pair-calibration-v1',
        'train_audit_sha256': sha_file(audit_path), 'checkpoint_sha256': sha_file(checkpoint),
        'plan_id': data['plan_id'], 'provenance_id': provenance,
        'val_annotations': [(r['id'], r['annotation_sha256']) for r in val],
        'selected_adapter': read(calibration_path.parent / 'adapter.json')['adapter'],
        'torch': str(torch.__version__)}
    # Accept the original CPU/CUDA audit device too; this new head can run on any supported device.
    for original_device in ('mps', 'cpu', 'cuda'):
        expected_validation_identity = fingerprint({**validation_payload, 'device': original_device})
        if expected_validation_identity == calibration['experiment_identity']:
            break
    else:
        raise ValueError('Calibration runtime, annotations or source files changed.')
    train_details, val_details = [], []
    for row in train:
        cached = read(audit_path.parent / 'per-track' / (row['id'] + '.json'))
        if cached['identity'] != audit['audit_identity'] or cached['audio_sha256'] != row['audio_sha256']:
            raise ValueError('Train prediction cache changed.')
        train_details.extend(cached['sections'])
    for row in val:
        cached = read(calibration_path.parent / 'val-predictions' / (row['id'] + '.json'))
        if cached['identity'] != expected_validation_identity:
            raise ValueError('Validation prediction cache changed.')
        val_details.extend(cached['sections'])
    summarize(train, train_details)
    baseline = evaluate_validation(val, val_details, {'kind': 'identity'})
    if baseline != calibration['baseline']:
        raise ValueError('Saved baseline predictions do not reproduce the calibration baseline.')
    target = device(args.device)
    torch.set_num_threads(2)
    random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed']); torch.manual_seed(SETTINGS['seed'])
    cache = args.cache_dir.expanduser().resolve(); cache.mkdir(parents=True, exist_ok=True)
    helpers = read, write, fingerprint, sha_file, iter_features
    train_samples = prepare_samples(train[:1] if args.smoke_test else train, train_details, cfg, provenance, cache, helpers)
    val_samples = prepare_samples(val[:1] if args.smoke_test else val, val_details, cfg, provenance, cache, helpers)
    model = SectionContextHead(SETTINGS['hidden'], SETTINGS['dropout']).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'], weight_decay=SETTINGS['weight_decay'])
    if args.smoke_test:
        model.train(); loss = supervised_loss(model, train_samples[0], target, SETTINGS['residual_penalty']) / len(train_samples[0]['labels'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        result, _ = evaluate_context(model, val_samples, val[:1], target)
        print(f'Smoke OK: full-song context, {sum(p.numel() for p in model.parameters()):,} parameters; loss={loss.item():.4f}; val F1={result["macro_f1_present"]:.4f}. No run/checkpoint saved.', flush=True)
        return
    run = args.output_dir.expanduser().resolve()
    metadata = {'protocol': PROTOCOL, 'settings': SETTINGS, 'device': str(target), 'torch': str(torch.__version__),
        'coverage_policy': COVERAGE_POLICY,
        'plan_id': data['plan_id'], 'provenance_id': provenance, 'checkpoint_sha256': sha_file(checkpoint),
        'calibration_report_sha256': sha_file(calibration_path),
        'sample_identities': [(s['track_id'], s['identity']) for s in train_samples + val_samples]}
    identity = fingerprint(metadata)
    if (run / 'run.json').exists():
        if not args.resume or read(run / 'run.json')['identity'] != identity:
            raise ValueError('Run exists or changed; use --resume with identical inputs/settings or a new output directory.')
    else:
        if args.resume: raise ValueError('No run to resume.')
        run.mkdir(parents=True, exist_ok=False)
        write(run / 'run.json', {'identity': identity, **metadata})
    coverage_diagnostics = [item for s in train_samples + val_samples for item in s['coverage_diagnostics']]
    write(run / 'coverage_diagnostics.json', {'policy': COVERAGE_POLICY, 'sections': coverage_diagnostics,
        'count': len(coverage_diagnostics), 'manual_labels_changed': False,
        'boundaries_changed': False, 'targets_excluded': False, 'missing_features_filled': False})
    start_epoch = 0; best_epoch = 0; bad = 0; history = []; best_score = baseline['macro_f1_present']
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.resume and (run / 'latest.pt').exists():
        state = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
        if state['identity'] != identity: raise ValueError('Resume checkpoint identity differs.')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_epoch, best_epoch, bad = state['epoch'], state['best_epoch'], state['bad_epochs']
        best_score, best_state, history = state['best_score'], state['best_model'], state['history']
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
        if target.type == 'mps': torch.mps.set_rng_state(state['device_rng'])
        if target.type == 'cuda': torch.cuda.set_rng_state(state['device_rng'], target)
    else:
        initial, _ = evaluate_context(model, val_samples, val, target)
        if initial['confusion_matrix'] != baseline['confusion_matrix']:
            raise ValueError('Zero residual did not reproduce frozen baseline predictions.')
    print(f'Train {len(train)}, val {len(val)}; full-song section context; device={target}; trainable={sum(p.numel() for p in model.parameters()):,}', flush=True)
    print(f'Baseline val F1={baseline["macro_f1_present"]:.4f}; previous calibration F1={calibration["candidate"]["macro_f1_present"]:.4f}', flush=True)

    def save_latest(epoch):
        device_rng = torch.mps.get_rng_state() if target.type == 'mps' else torch.cuda.get_rng_state(target) if target.type == 'cuda' else None
        atomic_torch_save(run / 'latest.pt', {'identity': identity, 'epoch': epoch,
            'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'best_model': best_state, 'best_score': best_score,
            'best_epoch': best_epoch, 'bad_epochs': bad, 'history': history,
            'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(), 'device_rng': device_rng})
        write(run / 'history.json', history)

    save_latest(start_epoch)
    for epoch in range(start_epoch + 1, SETTINGS['epochs'] + 1):
        if bad >= SETTINGS['patience']: break
        started = time.perf_counter(); model.train(); order = list(train_samples); random.shuffle(order)
        optimizer.zero_grad(set_to_none=True); accumulated = 0; targets = 0; loss_sum = 0.
        for i, sample in enumerate(order, 1):
            loss = supervised_loss(model, sample, target, SETTINGS['residual_penalty'])
            if not torch.isfinite(loss): raise ValueError('Non-finite context training loss.')
            loss.backward(); accumulated += len(sample['labels']); targets += len(sample['labels']); loss_sum += loss.item()
            if i % SETTINGS['accumulation'] == 0 or i == len(order):
                for parameter in model.parameters():
                    if parameter.grad is not None: parameter.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
        result, _ = evaluate_context(model, val_samples, val, target)
        improved = result['macro_f1_present'] > best_score + 1e-12
        if improved:
            best_score, best_epoch, bad = result['macro_f1_present'], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        history.append({'epoch': epoch, 'train_loss': loss_sum / targets, 'val': result, 'seconds': time.perf_counter() - started})
        save_latest(epoch)
        print(f'Epoch {epoch}: loss={loss_sum/targets:.4f}, val F1={result["macro_f1_present"]:.4f}, acc={result["section_accuracy"]:.4f}, {history[-1]["seconds"]:.1f}s' + (' * best' if improved else ''), flush=True)
    model.load_state_dict(best_state)
    candidate, predictions = evaluate_context(model, val_samples, val, target)
    atomic_torch_save(run / 'best.pt', {'identity': identity, 'architecture': PROTOCOL, 'settings': SETTINGS,
        'model': best_state, 'best_epoch': best_epoch, 'plan_id': data['plan_id'], 'provenance_id': provenance})
    changes = []
    for before, after in zip(val_details, predictions):
        if before['prediction'] != after['prediction']:
            changes.append({'track_id': before['track_id'], 'start': before['start'], 'end': before['end'],
                'label_manual': before['label_manual'], 'baseline_prediction': before['prediction'],
                'candidate_prediction': after['prediction'], 'baseline_correct': before['prediction'] == before['label_manual'],
                'candidate_correct': after['prediction'] == after['label_manual']})
    report = {'protocol': PROTOCOL, 'split': 'val', 'tracks': len(val), 'full_validation': True,
        'plan_id': data['plan_id'], 'provenance_id': provenance, 'experiment_identity': identity,
        'best_epoch': best_epoch, 'baseline': baseline, 'previous_calibration': calibration['candidate'], 'candidate': candidate,
        'macro_f1_delta_from_baseline': candidate['macro_f1_present'] - baseline['macro_f1_present'],
        'macro_f1_delta_from_previous_calibration': candidate['macro_f1_present'] - calibration['candidate']['macro_f1_present'],
        'changed_sections': changes, 'test_used': False, 'encoder_executed': False, 'manual_labels_changed': False,
        'boundary_changed': False, 'songformer_checkpoint_changed': False, 'ready_for_app_integration': False,
        'audio_coverage': {'policy': COVERAGE_POLICY, 'reviewed_sections_below_98_percent': len(coverage_diagnostics),
                           'details_file': str(run / 'coverage_diagnostics.json')},
        'created_at': datetime.now(timezone.utc).isoformat(),
        'limitations': ['Additional section-context residual model, not an update of the original SongFormer backbone.',
            'All original sections provide AUDIO context; only eligible manual train labels receive loss. Held annotations are never targets.',
            'Validation selects early stopping and best epoch; it is not an independent final accuracy estimate.',
            'The same validation set has already guided previous experiments. This experiment does not use test.',
            'Original frozen encoder features and PyTorch priors; compressed Core ML runtime and boundary accuracy are not evaluated.']}
    write(run / 'comparison.json', report)
    print('Best epoch:', best_epoch, '| report:', run / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
