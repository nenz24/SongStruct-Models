"""Export the fine-tuned cached-feature label pipeline to Core ML FP16 (no encoders)."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import numpy as np
import torch
from finetuning.common import CLASS_IDS, LABELS, config, load_head, sha_file, write


class FunctionLogitsWrapper(torch.nn.Module):
    """Dataset 5 is constant; preserve all 128 logits for the existing Swift API."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('dataset_ids', torch.tensor([5], dtype=torch.long))

    def forward(self, input_embeddings):
        return self.model.forward_func({'input_embeddings': input_embeddings,
                                       'dataset_ids': self.dataset_ids,
                                       'masks': None})['function_logits']


def disable_flash(model):
    changed = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Attend' and getattr(module, 'flash', False):
            module.flash = False
            changed.append(name)
    return changed


def parity(reference, candidate, atol=0.03, rtol=0.02):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        return {'passed': False, 'reference_shape': list(a.shape), 'candidate_shape': list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'passed': False, 'reason': 'non-finite logits'}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'passed': bool(np.allclose(a, b, atol=atol, rtol=rtol)),
            'max_abs_error': float(delta.max()), 'mean_abs_error': float(delta.mean()),
            'allowed_frame_argmax_agreement': float((a[..., list(CLASS_IDS)].argmax(-1) ==
                                                     b[..., list(CLASS_IDS)].argmax(-1)).mean()),
            'atol': atol, 'rtol': rtol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs.local/finetuning/runs/worship-v3/best.pt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Run on macOS: this command requires Core ML runtime parity checks.')
    import coremltools as ct
    cfg = config(args.config)
    checkpoint = args.checkpoint.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    if checkpoint.suffix != '.pt':
        raise ValueError('Use trusted training best.pt so experiment metadata can be checked.')
    from finetuning.engine import manifest
    data, provenance = manifest(cfg)
    meta = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (meta.get('plan_id') != data['plan_id'] or meta.get('provenance_id') != provenance
            or meta.get('config', {}).get('features') != cfg['features']):
        raise ValueError('Checkpoint does not match frozen split/cache configuration.')
    del meta
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model, _ = load_head(cfg, torch.device('cpu'), checkpoint)
    model.eval().requires_grad_(False)
    wrapper = FunctionLogitsWrapper(model).eval()
    example = torch.randn(1, 750, 4096) * .2
    with torch.inference_mode():
        original = wrapper(example).numpy()
        changes = disable_flash(model)
        modified = wrapper(example).numpy()
    flash_check = parity(original, modified, atol=1e-4, rtol=1e-4)
    write(out / 'attention_parity.json', flash_check)
    if not flash_check['passed']:
        raise RuntimeError('Export attention path differs from reference; see attention_parity.json.')
    # Explicit cross-length checks below detect traces with frozen shapes/rotary state.
    traced = torch.jit.trace(wrapper, example, check_trace=False, strict=True)
    trace_checks = []
    with torch.inference_mode():
        for frames in (3, 93, 750, 1500):
            x = torch.randn(1, frames, 4096) * .2
            check = parity(wrapper(x).numpy(), traced(x).numpy(), atol=1e-4, rtol=1e-4)
            trace_checks.append({'frames': frames, **check})
    write(out / 'trace_parity.json', trace_checks)
    if not all(c['passed'] for c in trace_checks):
        raise RuntimeError('Trace failed variable-length checks; no Core ML package saved.')
    print('Converting FP16 label pipeline...', flush=True)
    mlmodel = ct.convert(traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name='input_embeddings',
            shape=(1, ct.RangeDim(lower_bound=3, upper_bound=10500, default=750), 4096), dtype=np.float32)],
        outputs=[ct.TensorType(name='function_logits', dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY)
    mlmodel.short_description = 'SongFormer worship-v3 label pipeline; frozen dataset ID 5; encoder features required.'
    mlmodel.user_defined_metadata['checkpoint_sha256'] = sha_file(checkpoint)
    mlmodel.user_defined_metadata['allowed_label_ids'] = ','.join(map(str, CLASS_IDS))
    mlmodel.user_defined_metadata['input_contract'] = 'float32 [1,T,4096], T divisible by 3, 3<=T<=10500'
    package = out / 'SongFormerFunctionHead.mlpackage'
    mlmodel.save(str(package))
    checks = []
    with torch.inference_mode():
        for frames in (3, 93, 750, 1500):
            x = torch.randn(1, frames, 4096) * .2
            candidate = mlmodel.predict({'input_embeddings': x.numpy()})['function_logits']
            checks.append({'frames': frames, **parity(wrapper(x).numpy(), candidate)})
    passed = all(c['passed'] for c in checks)
    write(out / 'export_report.json', {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha_file(checkpoint),
        'package': str(package), 'fp16_synthetic_parity_passed': passed,
        'ready_for_app_integration': False, 'attention_modules_changed': changes,
        'checks': checks, 'torch': torch.__version__, 'coremltools': ct.__version__,
        'x_transformers': importlib.metadata.version('x-transformers'),
        'allowed_label_ids': list(CLASS_IDS), 'label_order': list(LABELS),
        'feature_profile': cfg['features'], 'plan_id': data['plan_id'], 'provenance_id': provenance,
        'note': 'Synthetic checks are a smoke test only. Run verify_coreml on real validation caches before compression/integration.'})
    if not passed:
        raise RuntimeError('FP16 parity failed; package is diagnostic only, do not integrate.')
    print('Export smoke passed:', package, flush=True)
    print('Next: deployment.verify_coreml on real validation caches, including full 420s chunks.')


if __name__ == '__main__':
    main()
