#!/usr/bin/env python3
"""Export and verify the MuQ-only label pipeline in Core ML (no encoders).

`export` (macOS): from the selected MuQ head (encoder_stream_ablation best.pt) and its
retrained context (train_variant_context best.pt) write
  MuQFunctionHeadFP32.mlpackage      input_embeddings [1,T,2048] -> function_logits [1,T/3,128]
  MuQFunctionHeadPalette6.mlpackage  same graph, palette6 weights (same config as the current head)
  MuQSectionContextFP32.mlpackage    section_audio [N,2048], section_timing [N,6] -> context_residual [N,8]
with synthetic native parity checks.

`verify` (macOS): on all approved validation songs, from the existing caches (MuQ30+MuQ120
columns only), compare PyTorch, Core ML FP32 and Core ML palette6 heads, each followed by
its context (PyTorch for the reference, Core ML otherwise). Gates fixed before running:
  FP32 pipeline: identical eligible labels and max final probability error <= 1e-4;
  palette6 pipeline: accuracy and macro F1 each within 0.01 of the PyTorch pipeline.
Encoders are not run here; Core ML MuQ features are the next (canonical-bank) stage.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_stream_ablation as ablation
from context_coreml import ExportableContext, convert_context, native_inputs, numerical
from train_section_context import LABELS, SectionContextHead, evaluate_context, prepare_samples
from train_variant_context import slice_samples
from calibrate_bridge_chorus import metrics

PROTOCOL = 'muq-pipeline-coreml-v1'
HEAD_FP32, HEAD_PALETTE6, CONTEXT = ('MuQFunctionHeadFP32.mlpackage', 'MuQFunctionHeadPalette6.mlpackage',
                                     'MuQSectionContextFP32.mlpackage')
MAX_FRAMES, MAX_SECTIONS = 10500, 512
GATES = {'fp32': {'identical_labels': True, 'max_final_probability_error': 1e-4},
         'palette6': {'max_section_accuracy_drop': 0.01, 'max_macro_f1_drop': 0.01}}
KEYS = ('sections', 'section_accuracy', 'macro_f1_present')


def package_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def convert_head(ct, wrapper, width, skip_model_load=False):
    """Trace with variable-length checks, then FP32 ML Program with a flexible [1,T,width] input."""
    from deployment.export_coreml import parity
    example = torch.randn(1, 750, width) * .2
    traced = torch.jit.trace(wrapper, example, check_trace=False, strict=True)
    checks = []
    with torch.inference_mode():
        for frames in (3, 93, 750, 1500):
            x = torch.randn(1, frames, width) * .2
            checks.append({'frames': frames, **parity(wrapper(x).numpy(), traced(x).numpy(), atol=1e-4, rtol=1e-4)})
    if not all(c['passed'] for c in checks):
        raise RuntimeError('Head trace failed variable-length checks.')
    model = ct.convert(traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name='input_embeddings',
                              shape=(1, ct.RangeDim(lower_bound=3, upper_bound=MAX_FRAMES, default=750), width),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name='function_logits', dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=skip_model_load)
    return model, checks


def palettize_head(model):
    """Same weights-only palette6 configuration as the current app head."""
    import coremltools.optimize.coreml as optimization
    from deployment.compress_coreml import compression_config, operation_counts
    before = operation_counts(model.get_spec())
    candidate = optimization.palettize_weights(model, config=compression_config(optimization, 'palette6'))
    added = operation_counts(candidate.get_spec()).get('constexpr_lut_to_dense', 0) - before.get('constexpr_lut_to_dense', 0)
    if added <= 0:
        raise RuntimeError('Palettization added no constexpr_lut_to_dense operations.')
    candidate.user_defined_metadata.update(dict(model.user_defined_metadata))
    candidate.user_defined_metadata['compression_scheme'] = 'palette6'
    candidate.user_defined_metadata['compression_scope'] = 'weights_only; float32 compute; activations not quantized'
    return candidate, added


def export_context(ct, model, width, max_sections=MAX_SECTIONS, skip_model_load=False):
    wrapper = ExportableContext(model).eval()
    generator = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        audio, timing = torch.randn(23, width, generator=generator), torch.rand(23, 6, generator=generator)
        attention = numerical(model(audio, timing).numpy(), wrapper(audio, timing).numpy(), atol=1e-5, rtol=1e-5)
    if not attention['passed']:
        raise RuntimeError('Explicit context attention differs from PyTorch.')
    example = torch.randn(16, width, generator=generator), torch.rand(16, 6, generator=generator)
    traced = torch.jit.trace(wrapper, example, check_trace=False, strict=True)
    with torch.inference_mode():
        for n in (1, 2, 7, 65):
            a, t = torch.randn(n, width, generator=generator), torch.rand(n, 6, generator=generator)
            if not numerical(wrapper(a, t).numpy(), traced(a, t).numpy(), atol=1e-5, rtol=1e-5)['passed']:
                raise RuntimeError('Context trace failed variable-length checks.')
    return convert_context(ct, traced, max_sections, skip_model_load=skip_model_load, width=width), wrapper


class CoreMLHead:
    """Duck-types SongFormer.forward_func for finetuning.engine.evaluate."""
    def __init__(self, model):
        self.model = model

    def eval(self):
        return self

    def forward_func(self, batch):
        x = batch['input_embeddings'].detach().cpu().numpy().astype(np.float32, copy=False)
        out = np.asarray(self.model.predict({'input_embeddings': x})['function_logits'], dtype=np.float32)
        if out.shape != (1, x.shape[1] // 3, 128) or not np.isfinite(out).all():
            raise ValueError(f'Invalid Core ML head output {out.shape}.')
        return {'function_logits': torch.from_numpy(out)}


def coreml_context(native, samples, rows, width, split='val'):
    """Same scoring as train_section_context.evaluate_context, with a Core ML residual."""
    details, residuals = [], []
    for sample in samples:
        residual = np.asarray(native.predict(native_inputs(sample['audio'].numpy(), sample['timing'].numpy(),
                                                           MAX_SECTIONS, width))['context_residual'])
        if residual.dtype != np.float32 or residual.shape != (len(sample['audio']), 8) or not np.isfinite(residual).all():
            raise ValueError('Invalid Core ML context residual.')
        residuals.append(residual)
        scores = sample['probabilities'].clamp_min(1e-8).log() + torch.from_numpy(residual)[sample['indices']]
        for item, p in zip(sample['details'], scores.softmax(-1).tolist()):
            details.append({**item, 'prediction': LABELS[max(range(8), key=lambda j: p[j])],
                            'probabilities': dict(zip(LABELS, p))})
    return metrics(details, {r['track_id']: r for r in rows}, {'kind': 'identity'}), details, residuals


def compare_pipelines(reference, candidate):
    """Label and probability agreement on identical sections."""
    if [(d['track_id'], d['start'], d['end']) for d in reference] != [(d['track_id'], d['start'], d['end']) for d in candidate]:
        raise ValueError('Pipelines were not evaluated on identical sections.')
    errors = [max(abs(a['probabilities'][k] - b['probabilities'][k]) for k in LABELS) for a, b in zip(reference, candidate)]
    changed = [{'track_id': a['track_id'], 'start': a['start'], 'end': a['end'], 'label_manual': a['label_manual'],
                'reference': a['prediction'], 'candidate': b['prediction']}
               for a, b in zip(reference, candidate) if a['prediction'] != b['prediction']]
    return {'label_agreement': 1 - len(changed) / len(reference), 'changed_sections': changed,
            'max_final_probability_error': max(errors), 'mean_final_probability_error': float(np.mean(errors))}


def gate(name, reference_scores, candidate_scores, agreement):
    if name == 'fp32':
        return (not agreement['changed_sections']
                and agreement['max_final_probability_error'] <= GATES['fp32']['max_final_probability_error'])
    rule = GATES['palette6']
    return (candidate_scores['section_accuracy'] >= reference_scores['section_accuracy'] - rule['max_section_accuracy_drop'] - 1e-12
            and candidate_scores['macro_f1_present'] >= reference_scores['macro_f1_present'] - rule['max_macro_f1_drop'] - 1e-12)


def load_sources(args):
    # Resolve every path first: SongStruct's load_head changes the working directory.
    for name in ('songstruct_root', 'context_dir', 'output_dir', 'export_dir', 'cache_dir'):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    root = args.songstruct_root
    sys.path.insert(0, str(root))
    from finetuning.common import config, fingerprint, iter_features, load_head, read, sha_file, write
    from finetuning.engine import evaluate, manifest
    cfg = config(root / 'finetuning/config.json')
    data, provenance = manifest(cfg)
    context_dir = args.context_dir.expanduser().resolve()
    summary = read(context_dir / 'summary.json')
    if summary['variant'] != 'muq' or summary.get('test_used') is not False:
        raise ValueError('Expected the validation-only MuQ context run.')
    head_path = Path(summary['head_checkpoint']).resolve()
    head, saved = ablation.load_variant(cfg, torch.device('cpu'), head_path, load_head)
    state = torch.load(context_dir / 'best.pt', map_location='cpu', weights_only=False)
    if (state['head_sha256'] != sha_file(head_path) or state['columns'] != saved['columns']
            or state['plan_id'] != data['plan_id'] or state['provenance_id'] != provenance):
        raise ValueError('MuQ context does not belong to this head/split/cache.')
    context = SectionContextHead(state['settings']['hidden'], state['settings']['dropout'], width=state['width'])
    context.load_state_dict(state['model'], strict=True)
    head.eval().requires_grad_(False); context.eval().requires_grad_(False)
    tools = dict(cfg=cfg, data=data, provenance=provenance, read=read, write=write, sha_file=sha_file,
                 fingerprint=fingerprint, iter_features=iter_features, evaluate=evaluate)
    return head, saved, head_path, context, context_dir, summary, tools


def run_export(args):
    if platform.system() != 'Darwin':
        raise RuntimeError('Run on macOS: export includes native Core ML parity checks.')
    import coremltools as ct
    head, saved, head_path, context, context_dir, summary, t = load_sources(args)
    # SongStruct's `deployment` package is importable only after load_sources adds its root.
    from deployment.export_coreml import FunctionLogitsWrapper, disable_flash, parity
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    width = len(saved['columns'])
    torch.manual_seed(42); torch.set_num_threads(2)
    wrapper = FunctionLogitsWrapper(head).eval()
    example = torch.randn(1, 750, width) * .2
    with torch.inference_mode():
        before = wrapper(example).numpy(); changed = disable_flash(head); after = wrapper(example).numpy()
    flash = parity(before, after, atol=1e-4, rtol=1e-4)
    if not flash['passed']:
        raise RuntimeError('Non-flash attention path differs from the reference.')
    print('Converting MuQ head to Core ML FP32...', flush=True)
    fp32, trace_checks = convert_head(ct, wrapper, width)
    meta = dict(protocol=PROTOCOL, precision='fp32', variant='muq', input_width=str(width),
                stream_order='muq_30s,muq_120s', checkpoint_sha256=t['sha_file'](head_path),
                allowed_label_ids='0,1,2,3,4,5,6,26', input_contract=f'float32 [1,T,{width}], T divisible by 3, 3<=T<={MAX_FRAMES}')
    fp32.user_defined_metadata.update(meta)
    fp32.short_description = 'SongFormer worship MuQ-only label head (MuQ30+MuQ120 layer-10 features).'
    fp32.save(str(out / HEAD_FP32))
    print('Palettizing head (palette6)...', flush=True)
    palette6, lut_ops = palettize_head(fp32)
    palette6.save(str(out / HEAD_PALETTE6))
    print('Converting MuQ section context to Core ML FP32...', flush=True)
    native_context, context_wrapper = export_context(ct, context, width)
    native_context.user_defined_metadata.update(dict(protocol=PROTOCOL, precision='fp32', kind='section_context_residual',
        variant='muq', input_width=str(width), context_checkpoint_sha256=t['sha_file'](context_dir / 'best.pt'),
        label_order=','.join(LABELS), max_sections=str(MAX_SECTIONS)))
    native_context.save(str(out / CONTEXT))
    checks = {'head_fp32': [], 'head_palette6': [], 'context': []}
    loaded = {name: ct.models.MLModel(str(out / package), compute_units=ct.ComputeUnit.CPU_ONLY)
              for name, package in (('fp32', HEAD_FP32), ('palette6', HEAD_PALETTE6), ('context', CONTEXT))}
    generator = torch.Generator().manual_seed(7)
    with torch.inference_mode():
        for frames in (3, 93, 750, 1500):
            x = torch.randn(1, frames, width, generator=generator) * .2
            reference = wrapper(x).numpy()
            for name in ('fp32', 'palette6'):
                actual = loaded[name].predict({'input_embeddings': x.numpy()})['function_logits']
                checks['head_' + name].append({'frames': frames, **parity(reference, actual)})
        for n in (1, 2, 7, 16, 65):
            a, b = torch.randn(n, width, generator=generator), torch.rand(n, 6, generator=generator)
            actual = loaded['context'].predict(native_inputs(a.numpy(), b.numpy(), MAX_SECTIONS, width))['context_residual']
            checks['context'].append({'sections': n, **numerical(context_wrapper(a, b).numpy(), actual)})
    sizes = {name: package_bytes(out / package) for name, package in
             (('head_fp32', HEAD_FP32), ('head_palette6', HEAD_PALETTE6), ('context_fp32', CONTEXT))}
    report = {'protocol': PROTOCOL, 'variant': 'muq', 'input_width': width, 'head_checkpoint': str(head_path),
              'head_checkpoint_sha256': t['sha_file'](head_path), 'context_checkpoint_sha256': t['sha_file'](context_dir / 'best.pt'),
              'attention_modules_changed': changed, 'trace_checks': trace_checks, 'native_checks': checks,
              'palette6_lut_operations_added': lut_ops, 'package_bytes': sizes,
              'fp32_synthetic_parity_passed': all(c['passed'] for c in checks['head_fp32']),
              'context_synthetic_parity_passed': all(c['passed'] for c in checks['context']),
              'palette6_synthetic_note': 'Palette6 parity on random inputs is informative only; the gate is the real-cache verify step.',
              'plan_id': t['data']['plan_id'], 'provenance_id': t['provenance'], 'coremltools': ct.__version__,
              'torch': str(torch.__version__), 'ready_for_app_integration': False,
              'created_utc': datetime.now(timezone.utc).isoformat()}
    t['write'](out / 'export_report.json', report)
    print(json.dumps(sizes, indent=2))
    if not (report['fp32_synthetic_parity_passed'] and report['context_synthetic_parity_passed']):
        raise RuntimeError('FP32 synthetic native parity failed; packages are diagnostic only.')
    print('Export passed. Next: verify on real validation caches.', flush=True)


def run_verify(args):
    if platform.system() != 'Darwin':
        raise RuntimeError('Run on macOS: verification executes Core ML predictions.')
    import coremltools as ct
    head, saved, head_path, context, context_dir, summary, t = load_sources(args)
    export = args.export_dir.expanduser().resolve()
    report = t['read'](export / 'export_report.json')
    if (report['head_checkpoint_sha256'] != t['sha_file'](head_path)
            or report['context_checkpoint_sha256'] != t['sha_file'](context_dir / 'best.pt')
            or not report['fp32_synthetic_parity_passed'] or not report['context_synthetic_parity_passed']):
        raise ValueError('Export report does not belong to these checkpoints or did not pass.')
    cfg, data, provenance = t['cfg'], t['data'], t['provenance']
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    cols, width = saved['columns'], len(saved['columns'])
    reader = ablation.sliced(t['iter_features'], cols)
    cpu = torch.device('cpu'); torch.set_num_threads(2)
    native = {name: ct.models.MLModel(str(export / package), compute_units=ct.ComputeUnit.CPU_ONLY)
              for name, package in (('fp32', HEAD_FP32), ('palette6', HEAD_PALETTE6), ('context', CONTEXT))}
    heads = {'pytorch': head, 'fp32': CoreMLHead(native['fp32']), 'palette6': CoreMLHead(native['palette6'])}
    helpers = t['read'], t['write'], t['fingerprint'], t['sha_file'], t['iter_features']
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=False)
    results, pipelines = {}, {}
    for name, model in heads.items():
        print(f'[{name}] head on {len(rows)} validation songs...', flush=True)
        details = []
        head_scores = t['evaluate'](model, rows, cfg, cpu, provenance, details=details, feature_reader=reader)
        cache = args.cache_dir.expanduser().resolve() / ('muq-coreml-' + name); cache.mkdir(parents=True, exist_ok=True)
        samples = slice_samples(prepare_samples(rows, details, cfg, provenance, cache, helpers), cols)
        if name == 'pytorch':
            final, predictions = evaluate_context(context, samples, rows, cpu)
        else:
            final, predictions, _ = coreml_context(native['context'], samples, rows, width)
        pipelines[name] = predictions
        results[name] = {'head_only': {k: head_scores[k] for k in KEYS}, 'pipeline': {k: final[k] for k in KEYS},
                         'per_class': final['per_class'], 'confusion_matrix': final['confusion_matrix']}
        print(f"[{name}] head acc={head_scores['section_accuracy']:.4f} F1={head_scores['macro_f1_present']:.4f} | "
              f"pipeline acc={final['section_accuracy']:.4f} F1={final['macro_f1_present']:.4f}", flush=True)
    reference = results['pytorch']['pipeline']
    for name in ('fp32', 'palette6'):
        agreement = compare_pipelines(pipelines['pytorch'], pipelines[name])
        results[name]['against_pytorch'] = agreement
        results[name]['gate_passed'] = gate(name, reference, results[name]['pipeline'], agreement)
    recorded = summary['pipeline']
    verification = {'protocol': PROTOCOL, 'split': 'val', 'tracks': len(rows), 'gates': GATES, 'results': results,
                    'pytorch_cpu_reproduces_recorded_validation': all(abs(reference[k] - recorded[k]) <= 1e-6 for k in KEYS),
                    'recorded_validation': recorded, 'package_bytes': report['package_bytes'],
                    'encoder_runtime': 'not run (original PyTorch MuQ feature caches)', 'test_used': False,
                    'ready_for_app_integration': False, 'created_utc': datetime.now(timezone.utc).isoformat()}
    verification['passed'] = results['fp32']['gate_passed'] and results['palette6']['gate_passed']
    t['write'](out / 'verification.json', verification)
    for name in ('fp32', 'palette6'):
        a = results[name]['against_pytorch']
        print(f"{name}: label agreement={a['label_agreement']:.4f} changed={len(a['changed_sections'])} "
              f"max prob error={a['max_final_probability_error']:.2e} -> {'PASS' if results[name]['gate_passed'] else 'FAIL'}")
    print('Verification:', 'PASSED' if verification['passed'] else 'FAILED', '| report:', out / 'verification.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--songstruct-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations')
        p.add_argument('--context-dir', type=Path, required=True, help='.../stream-ablation-w120-*/context/muq')
        p.add_argument('--output-dir', type=Path, required=True)
        if name == 'verify':
            p.add_argument('--export-dir', type=Path, required=True)
            p.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    run_export(args) if args.command == 'export' else run_verify(args)


if __name__ == '__main__':
    main()
