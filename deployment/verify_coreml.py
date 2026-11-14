"""Compare Core ML head and PyTorch using identical real validation features and pooling."""
import argparse
from pathlib import Path
import platform
import numpy as np
import torch
from deployment.export_coreml import FunctionLogitsWrapper, parity
from finetuning.common import CLASS_IDS, config, fingerprint, iter_features, load_head, sha_file, write
from finetuning.engine import evaluate, manifest


class CoreMLAdapter:
    def __init__(self, model):
        self.model = model

    def eval(self):
        return self

    def forward_func(self, batch):
        values = batch['input_embeddings'].detach().cpu().numpy().astype(np.float32)
        if values.shape[1] % 3 or not 3 <= values.shape[1] <= 10500:
            raise ValueError('Core ML frame count must be divisible by 3, in [3,10500].')
        output = self.model.predict({'input_embeddings': values})['function_logits']
        if output.shape != (1, values.shape[1] // 3, 128):
            raise ValueError(f'Unexpected Core ML output shape: {output.shape}')
        return {'function_logits': torch.from_numpy(np.asarray(output, dtype=np.float32))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs.local/finetuning/runs/worship-v3/best.pt'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1, help='Validation tracks; 0=all')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Write full diagnostics without failing solely on finite logit tolerance mismatch; does not approve model.')
    parser.add_argument('--candidate-name', default=None, help='Report label, e.g. fp16 or fp32; metadata precision remains separate')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Core ML prediction requires macOS.')
    if args.limit < 0:
        raise ValueError('limit must be >=0')
    import coremltools as ct
    cfg = config(args.config)
    checkpoint, package, output = [p.expanduser().resolve() for p in (args.checkpoint, args.package, args.output)]
    if output.exists():
        raise ValueError('Choose a new output path.')
    data, provenance = manifest(cfg)
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No eligible validation tracks.')
    torch.set_num_threads(2)
    cpu = torch.device('cpu')
    model, _ = load_head(cfg, cpu, checkpoint)
    model.eval()
    reference = FunctionLogitsWrapper(model).eval()
    mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    expected = mlmodel.user_defined_metadata.get('checkpoint_sha256')
    if expected != sha_file(checkpoint):
        raise ValueError('Core ML package checkpoint hash does not match reference.')
    adapter = CoreMLAdapter(mlmodel)
    checks = []
    candidate_name = args.candidate_name or mlmodel.user_defined_metadata.get('precision', 'unknown')
    with torch.inference_mode():
        for row in rows:
            print('Parity:', row['track_id'], flush=True)
            for values, offset, signature in iter_features(cfg, row):
                if signature != provenance:
                    raise ValueError('Feature provenance mismatch.')
                x = torch.from_numpy(np.array(values, copy=True)).unsqueeze(0)
                raw = reference(x).numpy()
                converted = adapter.forward_func({'input_embeddings': x})['function_logits'].numpy()
                checks.append({'track_id': row['track_id'], 'offset': offset,
                               'frames': len(values), **parity(raw, converted)})
    baseline_details, coreml_details = [], []
    baseline = evaluate(model, rows, cfg, cpu, provenance, details=baseline_details)
    converted = evaluate(adapter, rows, cfg, cpu, provenance, details=coreml_details)
    agreement = sum(a['prediction'] == b['prediction'] for a, b in zip(baseline_details, coreml_details)) / len(baseline_details)
    full = len(rows) == sum(r['approved'] and r['split'] == 'val' for r in data['records'])
    passed = all(c['passed'] for c in checks)
    write(output, {'tracks': len(rows), 'full_validation': full, 'plan_id': data['plan_id'],
        'provenance_id': provenance,
        'annotations_id': fingerprint([(r['track_id'], r.get('sections', [])) for r in rows]),
        'checkpoint_sha256': expected, 'package': str(package),
        'logit_parity_passed': passed, 'section_prediction_agreement': agreement,
        'pytorch': baseline, 'coreml': converted,
        **({'coreml_fp16': converted} if candidate_name == 'fp16' else {}),
        'candidate_name': candidate_name, 'precision_metadata': mlmodel.user_defined_metadata.get('precision'),
        'diagnostic_only': args.diagnostic,
        'status': 'numerically_consistent' if passed else 'numerical_mismatch_requires_review',
        'macro_f1_delta': converted['macro_f1_present'] - baseline['macro_f1_present'],
        'chunk_checks': checks,
        'ready_for_app_integration': False,
        'note': 'Head parity only. Encoder parity, section-pooling app implementation and iPhone measurements remain required.'})
    print(f'Section agreement={agreement:.4%}; F1 delta={converted["macro_f1_present"]-baseline["macro_f1_present"]:+.6f}')
    print('Report:', output)
    if not passed and args.diagnostic:
        print('DIAGNOSTIC ONLY: numerical parity FAILED. Completed report is not an integration approval.')
    if not passed and not args.diagnostic:
        raise RuntimeError('Real-feature logit parity failed; inspect report before compression/integration.')


if __name__ == '__main__':
    main()
