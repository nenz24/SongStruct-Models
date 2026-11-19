#!/usr/bin/env python3
"""Export the fixed section-context residual to FP32 Core ML and verify saved palette6 validation inputs."""
import argparse
from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_train_bridge import LABELS
from evaluate_palette6_context import validate_palette_report, palette_features, probability_metrics
from evaluate_frozen_context import verify_scores, changes, validate_inventory
from train_section_context import PROTOCOL, COVERAGE_POLICY, SectionContextHead, prepare_samples, evaluate_context

EXPORT_PROTOCOL = 'section-context-coreml-fp32-v1'
ATOL = RTOL = 1e-4
PROBABILITY_ATOL = 1e-5


class ExportableContext(nn.Module):
    """Same frozen weights; explicit unmasked self-attention avoids fused PyTorch-only ops."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        block = model.context
        if (block.norm_first or not block.self_attn.batch_first or block.self_attn.bias_k is not None
                or block.self_attn.bias_v is not None or block.self_attn.add_zero_attn
                or block.self_attn.in_proj_weight is None):
            raise ValueError('Unsupported context attention architecture.')
        self.heads = block.self_attn.num_heads
        self.hidden = block.self_attn.embed_dim
        self.head_size = self.hidden // self.heads

    def forward(self, section_audio, section_timing):
        m = self.model; block = m.context
        tokens = m.audio(section_audio) + m.position(section_timing)
        count = tokens.shape[0]
        q, k, v = F.linear(tokens, block.self_attn.in_proj_weight, block.self_attn.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(count, self.heads, self.head_size).permute(1, 0, 2)
        k = k.reshape(count, self.heads, self.head_size).permute(1, 0, 2)
        v = v.reshape(count, self.heads, self.head_size).permute(1, 0, 2)
        attention = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_size ** -.5), dim=-1)
        attended = (attention @ v).permute(1, 0, 2).reshape(count, self.hidden)
        attended = F.linear(attended, block.self_attn.out_proj.weight, block.self_attn.out_proj.bias)
        tokens = block.norm1(tokens + attended)
        feed = block.linear2(block.activation(block.linear1(tokens)))
        return m.output(block.norm2(tokens + feed))


def numerical(reference, actual, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(reference), np.asarray(actual)
    if a.shape != b.shape:
        return {'passed': False, 'reason': 'shape_mismatch', 'reference_shape': list(a.shape), 'candidate_shape': list(b.shape)}
    if a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'passed': False, 'reason': 'empty_or_nonfinite'}
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'passed': bool(np.allclose(a, b, atol=atol, rtol=rtol)), 'max_abs_error': float(error.max()),
            'mean_abs_error': float(error.mean()), 'atol': atol, 'rtol': rtol}


def native_inputs(audio, timing, max_sections, width=4096):
    audio = np.array(audio, dtype=np.float32, order='C', copy=True)
    timing = np.array(timing, dtype=np.float32, order='C', copy=True)
    if (audio.ndim != 2 or audio.shape[1] != width or not 1 <= len(audio) <= max_sections
            or timing.shape != (len(audio), 6) or not np.isfinite(audio).all() or not np.isfinite(timing).all()):
        raise ValueError(f'Expected finite Float32 audio [N,{width}] and timing [N,6] with shared N in range.')
    return {'section_audio': audio, 'section_timing': timing}


def precision_counts(spec):
    counts = Counter()
    def visit(block):
        for op in block.operations:
            for out in op.outputs:
                if out.type.HasField('tensorType'):
                    t = out.type.tensorType
                    enum = t.DESCRIPTOR.fields_by_name['dataType'].enum_type
                    counts[enum.values_by_number[t.dataType].name] += 1
            for inner in op.blocks: visit(inner)
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values(): visit(block)
    return dict(counts)


def check_contract(spec, max_sections, width=4096):
    observed = {f.name: f.type.multiArrayType for f in spec.description.input}
    if set(observed) != {'section_audio', 'section_timing'}: raise ValueError('Core ML input names differ.')
    for name, width in (('section_audio', width), ('section_timing', 6)):
        array = observed[name]; ranges = array.shapeRange.sizeRanges
        if (array.dataType != 65568 or len(ranges) != 2 or ranges[0].lowerBound != 1
                or ranges[0].upperBound != max_sections or ranges[1].lowerBound != width
                or ranges[1].upperBound != width):
            raise ValueError('Core ML Float32/flexible-shape contract differs: ' + name)
    if len(spec.description.output) != 1 or spec.description.output[0].name != 'context_residual':
        raise ValueError('Core ML output differs.')
    if spec.description.output[0].type.multiArrayType.dataType != 65568:
        raise ValueError('Context residual output must be Float32.')
    types = precision_counts(spec)
    if types.get('FLOAT16', 0): raise ValueError('Unexpected FLOAT16 declarations in FP32 context export.')
    return types


def convert_context(ct, traced, max_sections=512, skip_model_load=False, width=4096):
    n = ct.RangeDim(lower_bound=1, upper_bound=max_sections, default=min(16, max_sections))
    model = ct.convert(traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name='section_audio', shape=(n, width), dtype=np.float32),
                ct.TensorType(name='section_timing', shape=(n, 6), dtype=np.float32)],
        outputs=[ct.TensorType(name='context_residual', dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=skip_model_load)
    check_contract(model.get_spec(), max_sections, width)
    return model


def package_digest(path, fingerprint, sha_file):
    path = Path(path)
    return fingerprint([(str(p.relative_to(path)), sha_file(p)) for p in sorted(path.rglob('*')) if p.is_file()])


def load_inputs(args):
    root = args.songstruct_root.expanduser().resolve(); sys.path.insert(0, str(root))
    from finetuning.common import config, read, write, fingerprint, sha_file
    from finetuning.engine import manifest
    cfg = config(root / 'finetuning/config.json'); data, provenance = manifest(cfg)
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    if not rows or any(r['annotation_sha256'] != fingerprint(r['sections']) for r in rows):
        raise ValueError('Validation annotation identity changed.')
    path = args.palette_validation.expanduser().resolve(); report = read(path); selected = read(path.parent / 'selection.json')
    frozen = args.frozen_evaluation.expanduser().resolve(); lock = read(frozen / 'selection.json'); snapshot = frozen / 'frozen-context.pt'
    if (selected['identity'] != fingerprint({k: v for k, v in selected.items() if k != 'identity'})
            or selected['protocol'] != 'palette6-context-validation-v1' or report.get('split') != 'val'
            or report.get('protocol') != selected['protocol']
            or not report.get('full_validation') or report['tracks'] != len(rows) or report['selected_epoch'] != 5
            or report['selection_identity'] != selected['identity'] or report.get('weights_updated') is not False
            or report['context_runtime'] != 'PyTorch' or not report['original_context_validation_reproduced']
            or not report['saved_palette6_baseline_metrics_reproduced']
            or selected['plan_id'] != data['plan_id'] or selected['provenance_id'] != provenance
            or report['plan_id'] != data['plan_id'] or report['provenance_id'] != provenance
            or selected['annotations_id'] != fingerprint([(r['track_id'], r['sections']) for r in rows])
            or lock['identity'] != fingerprint({k: v for k, v in lock.items() if k != 'identity'})
            or lock.get('protocol') != 'frozen-section-context-test-v1'
            or selected['frozen_selection_identity'] != lock['identity'] or lock['context_epoch'] != 5
            or lock['plan_id'] != data['plan_id'] or lock['provenance_id'] != provenance
            or selected['context_checkpoint_sha256'] != sha_file(snapshot) or lock['context_checkpoint_sha256'] != sha_file(snapshot)
            or selected['torch'] != str(torch.__version__) or selected['coverage_policy'] != COVERAGE_POLICY
            or lock['feature_settings'] != cfg['features']):
        raise ValueError('Completed palette6 validation or frozen model/data identity changed.')
    proof_path = args.palette6_report.expanduser().resolve() if args.palette6_report else None
    if proof_path is None:
        proof_path = next((p.resolve() for p in (root / 'outputs.local').glob('muq-compression-*/palette6-all-validation.json')
                           if sha_file(p) == selected['palette6_report_sha256']), None)
    if proof_path is None or sha_file(proof_path) != selected['palette6_report_sha256']:
        raise ValueError('The exact saved palette6 source report is missing/changed.')
    proof = read(proof_path)
    details = validate_palette_report(proof, rows, data['plan_id'], provenance, lock['checkpoint_sha256'], fingerprint)
    source = Path(proof['diagnostic_cache']).expanduser()
    if not source.is_absolute(): source = root / source
    source = source.resolve()
    if (sha_file(source / 'encoder_provenance.json') != selected['encoder_header_sha256']
            or proof['candidate_encoder_id'] != selected['candidate_encoder_id']):
        raise ValueError('Palette6 encoder cache changed.')
    cache = args.cache_dir.expanduser().resolve(); cache.mkdir(parents=True, exist_ok=True)
    helpers = read, write, fingerprint, sha_file, lambda _cfg, row: palette_features(source, row, proof['candidate_encoder_id'], read)
    samples = prepare_samples(rows, details, cfg, proof['candidate_encoder_id'], cache, helpers, feature_cache_root=source)
    state = torch.load(snapshot, map_location='cpu', weights_only=True)
    if (state['architecture'] != PROTOCOL or state['best_epoch'] != 5 or state['plan_id'] != data['plan_id']
            or state['provenance_id'] != provenance): raise ValueError('Frozen context architecture differs.')
    torch.set_num_threads(2)
    model = SectionContextHead(state['settings']['hidden'], state['settings']['dropout']).cpu()
    model.load_state_dict(state['model']); model.eval().requires_grad_(False)
    reference, _ = evaluate_context(model, samples, rows, torch.device('cpu'))
    verify_scores(reference, report['palette6_features_with_context'])
    identity = {'protocol': EXPORT_PROTOCOL, 'context_checkpoint_sha256': sha_file(snapshot),
        'palette_validation_sha256': sha_file(path), 'palette_selection_identity': selected['identity'],
        'plan_id': data['plan_id'], 'provenance_id': provenance, 'selected_epoch': 5,
        'samples': [[s['track_id'], s['identity']] for s in samples], 'torch': str(torch.__version__)}
    return model, samples, rows, identity, (read, write, fingerprint, sha_file)


def input_contract(max_sections):
    return {'inputs': {'section_audio': ['N', 4096], 'section_timing': ['N', 6]},
        'output': {'context_residual': ['N', 8]}, 'dtype': 'float32', 'min_sections': 1, 'max_sections': max_sections,
        'all_original_sections_required': True, 'label_order': list(LABELS),
        'stream_order': ['musicfm_30s', 'muq_30s', 'musicfm_120s', 'muq_120s'],
        'audio_pooling': '25-Hz exact overlap-weighted sum / covered seconds for each original section; no zero-filling.',
        'timing_features': ['start/song_duration', 'end/song_duration', 'log1p(section_duration)/log(601)',
            'gap_from_previous/song_duration (first: start/song_duration)',
            'gap_to_next/song_duration (last: remaining/song_duration)', 'log1p(song_duration)/log(1201)'],
        'final_scores': 'log(max(original_allowed_8_class_pooled_probabilities,1e-8)) + context_residual',
        'final_prediction': 'argmax of final_scores; softmax only across these eight labels',
        'boundaries_changed': False, 'coreml_compute_units_for_verification': 'CPU_ONLY'}


def export_model(args):
    model, samples, rows, identity, helpers = load_inputs(args)
    read, write, fingerprint, sha_file = helpers
    out = args.output_dir.expanduser().resolve()
    if out.exists(): raise ValueError('Export directory already exists; use a fresh output directory.')
    if args.max_sections < max(len(s['audio']) for s in samples): raise ValueError('max-sections is smaller than real inputs; sections cannot be truncated.')
    import coremltools as ct
    out.mkdir(parents=True)
    wrapper = ExportableContext(model).eval(); checks = []
    with torch.inference_mode():
        for sample in samples:
            checks.append({'track_id': sample['track_id'], **numerical(model(sample['audio'], sample['timing']).numpy(),
                wrapper(sample['audio'], sample['timing']).numpy(), atol=1e-5, rtol=1e-5)})
    write(out / 'attention-parity.json', checks)
    if not all(c['passed'] for c in checks): raise RuntimeError('Explicit attention differs; inspect attention-parity.json.')
    generator = torch.Generator().manual_seed(42)
    example = torch.randn(min(16, args.max_sections), 4096, generator=generator), torch.rand(min(16, args.max_sections), 6, generator=generator)
    traced = torch.jit.trace(wrapper, example, check_trace=False, strict=True)
    counts = sorted({n for n in (1, 2, 7, 16, 65, 129, args.max_sections) if n <= args.max_sections})
    trace_checks = []
    with torch.inference_mode():
        for n in counts:
            audio = torch.randn(n, 4096, generator=generator); timing = torch.rand(n, 6, generator=generator)
            trace_checks.append({'sections': n, **numerical(wrapper(audio, timing).numpy(), traced(audio, timing).numpy(), atol=1e-5, rtol=1e-5)})
    write(out / 'trace-parity.json', trace_checks)
    if not all(c['passed'] for c in trace_checks): raise RuntimeError('Variable-length trace parity failed.')
    print('Converting fixed epoch-5 context residual to Core ML FLOAT32...', flush=True)
    converted = convert_context(ct, traced, args.max_sections)
    metadata = converted.user_defined_metadata
    metadata.update(export_protocol=EXPORT_PROTOCOL, precision='fp32', selected_epoch='5',
        context_checkpoint_sha256=identity['context_checkpoint_sha256'], label_order=','.join(LABELS),
        max_sections=str(args.max_sections), kind='section_context_residual')
    converted.short_description = 'Fixed worship epoch-5 section-context residual; all original section means and timing required.'
    package = out / 'WorshipSectionContextFP32.mlpackage'; converted.save(str(package))
    native_checks = []
    with torch.inference_mode():
        for n in counts:
            audio = torch.randn(n, 4096, generator=generator); timing = torch.rand(n, 6, generator=generator)
            actual = converted.predict(native_inputs(audio.numpy(), timing.numpy(), args.max_sections))['context_residual']
            native_checks.append({'sections': n, **numerical(wrapper(audio, timing).numpy(), actual)})
    passed = all(c['passed'] for c in native_checks)
    report = {**identity, 'package': str(package), 'package_digest': package_digest(package, fingerprint, sha_file),
        'precision': 'fp32', 'max_sections': args.max_sections, 'input_contract': input_contract(args.max_sections),
        'attention_parity_passed': True, 'trace_parity_passed': True, 'native_synthetic_parity_passed': passed,
        'native_checks': native_checks, 'declared_tensor_dtypes': check_contract(converted.get_spec(), args.max_sections),
        'coremltools': ct.__version__, 'package_bytes': sum(p.stat().st_size for p in package.rglob('*') if p.is_file()),
        'platform': platform.platform(),
        'ready_for_app_integration': False, 'weights_updated': False, 'full_validation_pending': True}
    write(out / 'export_report.json', report)
    if not passed: raise RuntimeError('Native Core ML synthetic parity failed; package remains diagnostic only.')
    print('FP32 export smoke passed:', package, flush=True)


def verify_track(model, sample, native_predict, max_sections):
    inputs = native_inputs(sample['audio'].numpy(), sample['timing'].numpy(), max_sections)
    actual = np.asarray(native_predict(inputs)['context_residual'])
    if actual.dtype != np.float32: raise ValueError('Core ML residual must have Float32 dtype.')
    with torch.inference_mode():
        reference = model(sample['audio'], sample['timing'])
        residual_check = numerical(reference.numpy(), actual)
        if actual.shape != (len(sample['audio']), 8) or not np.isfinite(actual).all():
            raise ValueError('Invalid native residual shape or NaN/Inf.')
        prior = sample['probabilities'].clamp_min(1e-8).log()
        ref_scores = prior + reference[sample['indices']]
        native_scores = prior + torch.from_numpy(np.array(actual, copy=True))[sample['indices']]
        ref_p, native_p = ref_scores.softmax(-1), native_scores.softmax(-1)
        checks = {'residual': residual_check, 'final_logits': numerical(ref_scores.numpy(), native_scores.numpy()),
            'probabilities': numerical(ref_p.numpy(), native_p.numpy(), atol=PROBABILITY_ATOL, rtol=0),
            'section_argmax_agreement': float((ref_scores.argmax(-1) == native_scores.argmax(-1)).float().mean())}
        details = [{**item, 'prediction': LABELS[int(torch.argmax(p))], 'probabilities': dict(zip(LABELS, p.tolist()))}
                   for item, p in zip(sample['details'], native_p)]
    checks['passed'] = all(checks[k]['passed'] for k in ('residual', 'final_logits', 'probabilities')) and checks['section_argmax_agreement'] == 1.
    return checks, details


def verify_model(args):
    model, samples, rows, identity, helpers = load_inputs(args)
    read, write, fingerprint, sha_file = helpers
    export_dir = args.export_dir.expanduser().resolve(); exported = read(export_dir / 'export_report.json')
    for key, value in identity.items():
        if exported.get(key) != value: raise ValueError('Export source identity changed: ' + key)
    package = Path(exported['package']).resolve()
    if (not exported['native_synthetic_parity_passed'] or exported['precision'] != 'fp32'
            or package_digest(package, fingerprint, sha_file) != exported['package_digest']):
        raise ValueError('Export is incomplete, changed or failed native synthetic parity.')
    import coremltools as ct
    native = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    if (native.user_defined_metadata.get('context_checkpoint_sha256') != identity['context_checkpoint_sha256']
            or native.user_defined_metadata.get('export_protocol') != EXPORT_PROTOCOL
            or native.user_defined_metadata.get('precision') != 'fp32'):
        raise ValueError('Core ML model metadata differs from export.')
    check_contract(native.get_spec(), exported['max_sections'])
    selected = list(zip(rows, samples)); selected = selected[:args.limit] if args.limit else selected
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    evaluation_id = fingerprint({'export_report_sha256': sha_file(export_dir / 'export_report.json'),
        'identity': identity, 'tolerances': [ATOL, RTOL, PROBABILITY_ATOL], 'coremltools': ct.__version__,
        'platform': platform.platform()})
    cache = out / 'per-track'; cache.mkdir(exist_ok=True); checks = []; details = []
    for i, (row, sample) in enumerate(selected, 1):
        print(f'Core ML context [{i}/{len(selected)}]: {row["track_id"]}', flush=True)
        path = cache / (row['id'] + '.json')
        if path.exists():
            stored = read(path)
            if stored['identity'] != evaluation_id: raise ValueError('Native parity cache changed.')
            check, produced = stored['check'], stored['sections']
        else:
            check, produced = verify_track(model, sample, native.predict, exported['max_sections'])
            write(path, {'identity': evaluation_id, 'check': check, 'sections': produced})
        validate_inventory([row], produced, 'val')
        checks.append({'track_id': row['track_id'], 'original_sections': len(sample['audio']), **check}); details.extend(produced)
    selected_rows = [r for r, _ in selected]; selected_samples = [s for _, s in selected]
    reference, reference_details = evaluate_context(model, selected_samples, selected_rows, torch.device('cpu'))
    candidate = probability_metrics(details)
    agreement = sum(a['prediction'] == b['prediction'] for a, b in zip(reference_details, details)) / len(details)
    passed = all(c['passed'] for c in checks) and agreement == 1. and abs(candidate['macro_f1_present'] - reference['macro_f1_present']) < 1e-12
    report = {**identity, 'tracks': len(selected), 'split': 'val', 'full_validation': len(selected) == len(rows),
        'precision': 'fp32', 'package_digest': exported['package_digest'], 'pytorch': reference, 'coreml': candidate,
        'section_agreement': agreement, 'macro_f1_delta': candidate['macro_f1_present'] - reference['macro_f1_present'],
        'checks': checks, 'changed_sections': changes(reference_details, details),
        'status': 'numerically_consistent' if passed else 'parity_failed_requires_review', 'passed': passed,
        'platform': platform.platform(), 'coremltools': ct.__version__, 'compute_units': 'CPU_ONLY',
        'encoder_executed': False, 'old_coreml_head_executed': False, 'weights_updated': False,
        'test_metrics_used': False, 'ready_for_app_integration': False,
        'limitations': ['CPU Core ML context inference on saved palette6 validation means/probabilities; not an iPhone audio-to-label test.',
            'Residual, final logits, probabilities and labels are checked; boundary timing is unchanged.']}
    destination = out / ('validation-all.json' if report['full_validation'] else f'validation-first-{len(selected)}.json')
    write(destination, report)
    print(f'Section agreement={agreement:.4%}; F1 delta={report["macro_f1_delta"]:+.8f}; report={destination}', flush=True)
    if not passed: raise RuntimeError('Core ML FP32 context parity failed; complete diagnostics were preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'verify'])
    parser.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
    parser.add_argument('--palette-validation', type=Path, required=True)
    parser.add_argument('--frozen-evaluation', type=Path, required=True)
    parser.add_argument('--palette6-report', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--max-sections', type=int, default=512)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if platform.system() != 'Darwin': raise RuntimeError('Native Core ML export/verification must run on the Mac.')
    if args.max_sections < 1 or args.limit < 0: raise ValueError('Invalid section/track bounds.')
    if args.action == 'verify' and args.export_dir is None: parser.error('--export-dir is required for verify')
    if args.action == 'export': export_model(args)
    else: verify_model(args)


if __name__ == '__main__': main()
