#!/usr/bin/env python3
"""Full validation of the MuQ-only pipeline with Core ML MuQ encoder features (Mac CPU).

For every approved validation song this decodes the MP3 exactly like the feature cache
(librosa, 24 kHz mono), runs the app log-mel and MuQ convolution frontends and the
existing canonical MuQ bank (short_30 / long_120 / partial functions, unchanged weights)
on every 30-second and 120-second window, fuses MuQ30+MuQ120 with the same window and
frame rules as the training cache (`pcm_chunks`), then applies the verified palette6
MuQ head and Core ML MuQ context.

Reference: the same palette6 head + Core ML context on the ORIGINAL cached MuQ features
(the configuration that passed `export_muq_pipeline.py verify`). Gate fixed before
running: Core ML-encoder pipeline accuracy and macro F1 each within 0.01 of that
reference on identical sections. MusicFM is never loaded. Test split is not read.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_stream_ablation as ablation
from context_coreml import native_inputs
from export_muq_pipeline import CONTEXT, HEAD_PALETTE6, MAX_SECTIONS, CoreMLHead, compare_pipelines
from install_section_context import package_digest
from prepare_encoder_pipeline_iphone import pcm_chunks
from encoder_acceleration import candidate_inputs
from train_section_context import LABELS, geometry, pool_audio, source_sections
from calibrate_bridge_chorus import metrics

PROTOCOL = 'muq-encoder-quality-v1'
GATE = {'max_section_accuracy_drop': 0.01, 'max_macro_f1_drop': 0.01,
        'reference': 'palette6 MuQ head + Core ML context on original cached MuQ features'}
KEYS = ('sections', 'section_accuracy', 'macro_f1_present')
FUNCTIONS = {'short_30', 'long_120', 'partial'}


def validate_bank(report, compiled):
    if (report.get('encoder') != 'muq' or not report.get('native_cpu_passed') or report.get('error')
            or not report.get('candidate_for_quality_validation') or report.get('weights_requantized') is not False
            or report.get('retrained') is not False):
        raise ValueError('Expected the reviewed canonical MuQ bank report.')
    if not compiled.is_dir() or package_digest(compiled) != report['compiled_digest']:
        raise ValueError('Compiled canonical MuQ bank is missing or changed.')


def rotary(frames, frequencies):
    """Same rotary tensors as SongStruct deployment.verify_encoders.rotary."""
    inv = np.asarray(frequencies, dtype=np.float32)
    if inv.shape != (32,):
        raise ValueError('Expected 32 rotary inverse frequencies.')
    angles = np.arange(frames, dtype=np.float32)[:, None] * inv[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    return np.stack([np.cos(angles), np.sin(angles)]).reshape(2, frames, 1, 1, 64).astype(np.float32)


def function_for(samples, seconds, rate=24000):
    if seconds not in (30, 120):
        raise ValueError('Only 30- and 120-second streams exist.')
    return ('short_30' if seconds == 30 else 'long_120') if samples == seconds * rate else 'partial'


class MuQEncoder:
    """App log-mel + MuQ convolution frontends and the canonical MuQ bank, CPU only."""
    def __init__(self, ct, frontends, compiled, frequencies):
        self.ct, self.frontends, self.compiled, self.frequencies = ct, frontends, compiled, frequencies
        self.front, self.active, self.calls = {}, None, []

    def frontend(self, role, tensor):
        key = tuple(tensor.shape)
        if role not in self.front or self.front[role][0] != key:
            self.front.pop(role, None); gc.collect()
            self.front[role] = (key, self.ct.models.MLModel(str(self.frontends[role]), compute_units=self.ct.ComputeUnit.CPU_ONLY))
        model = self.front[role][1]
        name = sorted(f.name for f in model.get_spec().description.input)[0]
        values = model.predict({name: np.array(tensor, dtype=np.float32, order='C', copy=True)})
        if len(values) != 1:
            raise ValueError('Unexpected frontend outputs.')
        output = np.array(next(iter(values.values())), dtype=np.float32, order='C', copy=True)
        if not np.isfinite(output).all():
            raise ValueError('Non-finite frontend output.')
        return output

    def encode(self, audio, seconds):
        mel = self.frontend('log_mel', np.pad(audio, (1024, 1024), mode='reflect')[None])
        conv = self.frontend('muq_conv', mel)
        frames = conv.shape[1]
        if conv.shape != (1, frames, 1024):
            raise ValueError('MuQ frontend shape changed.')
        function = function_for(len(audio), seconds)
        positions = rotary(frames, self.frequencies)
        inputs = candidate_inputs(conv, positions, frames) if function != 'partial' else dict(conv_in=conv, rotary=positions)
        identity = (function, frames)
        if self.active is None or self.active[0] != identity:
            self.active = None; gc.collect()
            self.active = (identity, self.ct.models.CompiledMLModel(str(self.compiled), function_name=function,
                                                                    compute_units=self.ct.ComputeUnit.CPU_ONLY))
        started = time.perf_counter()
        result = self.active[1].predict(inputs)
        if len(result) != 1:
            raise ValueError('Unexpected encoder output.')
        output = np.array(next(iter(result.values())), dtype=np.float32, order='C', copy=True)
        if output.shape != conv.shape or not np.isfinite(output).all():
            raise ValueError('Canonical MuQ shape/non-finite failure.')
        self.calls.append({'function': function, 'frames': frames, 'seconds': time.perf_counter() - started})
        return output[0]

    def release(self):
        self.front.clear(); self.active = None; gc.collect()


def muq_chunks(waveform, encode, settings):
    """Training-cache window/frame rules via pcm_chunks; MuQ stands in for both pair slots."""
    cols = ablation.columns('muq')

    def pair(audio, seconds):
        values = encode(audio, seconds)
        return [values, values]
    for offset, fused in pcm_chunks(waveform, pair, settings):
        yield offset, np.ascontiguousarray(fused[:, cols])


def song_pipeline(row, chunks, head, context, offsets=None, arrays=None):
    """Palette6 head priors on eligible targets + Core ML context over ALL original sections.

    If `arrays` is a dict it receives the section audio means, eligible-target priors,
    target indices and original intervals (the iPhone probe's Mac reference)."""
    from finetuning.engine import logits, pooled_sections
    intervals, indices = source_sections(context['source'], row)
    sums = np.zeros((len(row['sections']), 8)); covered = np.zeros(len(row['sections']))
    kept = []
    for values, offset in chunks:
        kept.append((values, offset))
        with torch.inference_mode():
            for i, p, seconds in pooled_sections(logits(head, values, torch.device('cpu')), row['sections'], offset):
                sums[i] += np.asarray(p.tolist()) * seconds; covered[i] += seconds
    if offsets is not None and [round(o, 6) for _, o in kept] != [round(o, 6) for o in offsets]:
        raise ValueError('Head-chunk timing differs from the reference; boundaries are never adjusted.')
    if np.any(covered <= 0):
        raise ValueError('A target lost probability coverage.')
    audio, coverage = pool_audio(((v, o, 'x') for v, o in kept), intervals, 'x', width=kept[0][0].shape[1])
    timing = geometry(intervals, row['duration'])
    residual = np.asarray(context['native'].predict(native_inputs(audio, timing, MAX_SECTIONS, audio.shape[1]))['context_residual'])
    if residual.dtype != np.float32 or residual.shape != (len(intervals), 8) or not np.isfinite(residual).all():
        raise ValueError('Invalid Core ML context residual.')
    priors = torch.from_numpy((sums / covered[:, None]).astype(np.float32))
    if arrays is not None:
        arrays.update(audio=audio, priors=priors.numpy().copy(), indices=list(indices), intervals=intervals)
    scores = priors.clamp_min(1e-8).log() + torch.from_numpy(residual)[torch.as_tensor(indices)]
    details = []
    for section, p in zip(row['sections'], scores.softmax(-1).tolist()):
        prediction = LABELS[max(range(8), key=lambda j: p[j])]
        details.append({'track_id': row['track_id'], 'start': section['start'], 'end': section['end'],
                        'label_manual': section['label'], 'prediction': prediction,
                        'correct': prediction == section['label'], 'probabilities': dict(zip(LABELS, p))})
    return details, [o for _, o in kept], float((coverage / np.array([s['end'] - s['start'] for s in intervals])).min())


def feature_drift(reference, candidate):
    errors = [np.abs(a.astype(np.float64) - b.astype(np.float64)) for (a, _), (b, _) in zip(reference, candidate)]
    if len(reference) != len(candidate) or any(a.shape != b.shape for (a, _), (b, _) in zip(reference, candidate)):
        raise ValueError('Fused feature shapes differ from the reference cache.')
    return {'max_abs': float(max(e.max() for e in errors)), 'mean_abs': float(np.mean([e.mean() for e in errors]))}


def decide(reference, candidate, rule=GATE):
    return (candidate['section_accuracy'] >= reference['section_accuracy'] - rule['max_section_accuracy_drop'] - 1e-12
            and candidate['macro_f1_present'] >= reference['macro_f1_present'] - rule['max_macro_f1_drop'] - 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    p.add_argument('--songmap-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--export-dir', type=Path, required=True, help='muq-coreml-export-* folder')
    p.add_argument('--verification', type=Path, required=True, help='passed muq-coreml-verify-*/verification.json')
    p.add_argument('--muq-report', type=Path, required=True, help='canonical MuQ bank report (its Compiled/ folder is used)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='0 = all validation songs')
    a = p.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Run on macOS: Core ML encoders and models execute here.')
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.expanduser().resolve())
    sys.path.insert(0, str(a.songstruct_root))
    from finetuning.common import config, fingerprint, iter_features, read, sha_file, write
    from finetuning.engine import manifest
    import coremltools as ct
    import librosa
    cfg = config(a.songstruct_root / 'finetuning/config.json'); data, provenance = manifest(cfg)
    torch.set_num_threads(2)
    verification = read(a.verification)
    if not verification.get('passed') or verification.get('split') != 'val' or verification.get('test_used') is not False:
        raise ValueError('Run export_muq_pipeline.py verify first; it must pass.')
    export = read(a.export_dir / 'export_report.json')
    if export['package_bytes'] != verification['package_bytes'] or export['plan_id'] != data['plan_id']:
        raise ValueError('Verification does not belong to this export/split.')
    bank = read(a.muq_report)
    compiled = a.muq_report.parent / 'Compiled' / 'muq_Canonical.mlmodelc'
    validate_bank(bank, compiled)
    app_root = a.songmap_root / 'SongMap/Resources/MLModels/SongFormer'
    app = read(app_root / 'songformer_coreml_manifest.json')
    frontends = {role: app_root / app['models'][role] for role in ('log_mel', 'muq_conv')}
    out = a.output_dir; out.mkdir(parents=True, exist_ok=True); per_track = out / 'per-track'; per_track.mkdir(exist_ok=True)
    lock = {'protocol': PROTOCOL, 'gate': GATE, 'plan_id': data['plan_id'], 'provenance_id': provenance,
            'head_digest': package_digest(a.export_dir / HEAD_PALETTE6), 'context_digest': package_digest(a.export_dir / CONTEXT),
            'muq_bank_compiled_digest': bank['compiled_digest'], 'muq_bank_compiled_bytes': bank['compiled_bytes'],
            'frontends': {k: package_digest(v) for k, v in frontends.items()}, 'verification_sha256': sha_file(a.verification),
            'script_sha256': sha_file(Path(__file__)), 'coremltools': ct.__version__, 'librosa': librosa.__version__,
            'compute_units': 'CPU_ONLY', 'musicfm_loaded': False}
    identity = fingerprint(lock)
    if (out / 'selection.json').exists() and read(out / 'selection.json') != {'identity': identity, **lock}:
        raise ValueError('Different selection in this output folder; choose a new one.')
    write(out / 'selection.json', {'identity': identity, **lock})
    head = CoreMLHead(ct.models.MLModel(str(a.export_dir / HEAD_PALETTE6), compute_units=ct.ComputeUnit.CPU_ONLY))
    context = {'native': ct.models.MLModel(str(a.export_dir / CONTEXT), compute_units=ct.ComputeUnit.CPU_ONLY)}
    encoder = MuQEncoder(ct, frontends, compiled, app['rotary_inv_freq']['muq'])
    reader = ablation.sliced(iter_features, ablation.columns('muq'))
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    rows = rows[:a.limit] if a.limit else rows
    reference_details, candidate_details, drift, calls, timings = [], [], [], [], []
    for number, row in enumerate(rows, 1):
        info = per_track / (row['id'] + '.json')
        song_identity = fingerprint({'run': identity, 'track': row['id'], 'audio': row['audio_sha256'],
                                     'annotation': row['annotation_sha256']})
        if info.exists() and read(info).get('identity') == song_identity:
            item = read(info); print(f'[{number}/{len(rows)}] {row["track_id"]}: verified result reused', flush=True)
        else:
            print(f'[{number}/{len(rows)}] {row["track_id"]}', flush=True)
            if sha_file(row['audio_path']) != row['audio_sha256']:
                raise ValueError('Audio changed since the feature cache was built.')
            context['source'] = read(row['source_json'])
            reference = []
            for values, offset, signature in reader(cfg, row):
                if signature != provenance:
                    raise ValueError('Mixed feature-cache provenance.')
                reference.append((np.asarray(values, dtype=np.float32), offset))
            ref_details, offsets, _ = song_pipeline(row, reference, head, context)
            waveform, _ = librosa.load(row['audio_path'], sr=24000, mono=True)
            started = time.perf_counter(); before = len(encoder.calls)
            candidate = [(v, o) for o, v in muq_chunks(waveform, encoder.encode, cfg['features'])]
            encode_seconds = time.perf_counter() - started
            cand_details, _, min_coverage = song_pipeline(row, candidate, head, context, offsets)
            item = {'identity': song_identity, 'reference': ref_details, 'candidate': cand_details,
                    'feature_drift': feature_drift(reference, candidate), 'min_section_audio_coverage': min_coverage,
                    'encoder_calls': encoder.calls[before:], 'encode_seconds': encode_seconds,
                    'audio_seconds': len(waveform) / 24000}
            write(info, item); encoder.release(); del waveform, reference, candidate
            print(f"  drift max={item['feature_drift']['max_abs']:.2e}; encode {encode_seconds:.1f}s for "
                  f"{item['audio_seconds']:.0f}s audio", flush=True)
        reference_details.extend(item['reference']); candidate_details.extend(item['candidate'])
        drift.append({'track_id': row['track_id'], **item['feature_drift']}); calls.extend(item['encoder_calls'])
        timings.append({'track_id': row['track_id'], 'encode_seconds': item['encode_seconds'], 'audio_seconds': item['audio_seconds']})
    by_track = {r['track_id']: r for r in rows}
    ref_scores = metrics(reference_details, by_track, {'kind': 'identity'})
    cand_scores = metrics(candidate_details, by_track, {'kind': 'identity'})
    full = len(rows) == len([r for r in data['records'] if r['approved'] and r['split'] == 'val'])
    expected = verification['results']['palette6']['pipeline']
    result = {'protocol': PROTOCOL, 'split': 'val', 'tracks': len(rows), 'full_validation': full, 'selection_identity': identity,
              'gate': GATE, 'reference_original_features': {k: ref_scores[k] for k in KEYS},
              'reference_reproduces_verification': (not full) or all(abs(ref_scores[k] - expected[k]) <= 1e-6 for k in KEYS),
              'candidate_coreml_encoder': {k: cand_scores[k] for k in KEYS},
              'candidate_per_class': cand_scores['per_class'], 'candidate_confusion_matrix': cand_scores['confusion_matrix'],
              'against_reference': compare_pipelines(reference_details, candidate_details),
              'feature_drift': drift, 'encoder_function_calls': {f: sum(c['function'] == f for c in calls) for f in sorted(FUNCTIONS)},
              'mac_cpu_encode_seconds': timings, 'muq_bank_compiled_bytes': bank['compiled_bytes'],
              'head_palette6_bytes': export['package_bytes']['head_palette6'], 'context_bytes': export['package_bytes']['context_fp32'],
              'musicfm_loaded': False, 'test_used': False, 'ready_for_app_integration': False,
              'created_utc': datetime.now(timezone.utc).isoformat()}
    result['passed'] = full and result['reference_reproduces_verification'] and decide(ref_scores, cand_scores)
    write(out / ('validation-all.json' if full else 'validation-partial.json'), result)
    print(f"Original MuQ features: acc={ref_scores['section_accuracy']:.4f} F1={ref_scores['macro_f1_present']:.4f}")
    print(f"Core ML MuQ encoder:   acc={cand_scores['section_accuracy']:.4f} F1={cand_scores['macro_f1_present']:.4f}")
    print(f"Changed sections: {len(result['against_reference']['changed_sections'])}; "
          f"MuQ bank {bank['compiled_bytes']/2**20:.1f} MiB; head {export['package_bytes']['head_palette6']/2**20:.1f} MiB")
    print('Passed' if result['passed'] else ('Partial run (not a verdict)' if not full else 'FAILED'), flush=True)


if __name__ == '__main__':
    main()
