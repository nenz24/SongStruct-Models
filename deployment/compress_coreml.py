"""Create independent weight-only INT8 or palette-6 candidates from verified FP32 head."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import numpy as np
from finetuning.common import sha_file, write


def package_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def operation_counts(spec):
    counts = Counter()
    def block_counts(block):
        for op in block.operations:
            counts[op.type] += 1
            for nested in op.blocks:
                block_counts(nested)
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            block_counts(block)
    return dict(counts)


def validate_source_report(report, package, checkpoint_hash):
    if not checkpoint_hash:
        raise ValueError('Source package is missing checkpoint hash metadata.')
    if (not report.get('full_validation') or not report.get('logit_parity_passed')
            or report.get('tracks', 0) < 1):
        raise ValueError('Source must have passed full-validation numerical parity.')
    if Path(report['package']).expanduser().resolve() != package:
        raise ValueError('Validation report is for a different package path.')
    if report.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Validation checkpoint hash differs from package metadata.')
    if report.get('precision_metadata') != 'fp32':
        raise ValueError('Source validation report must identify FP32 metadata.')


def compression_config(api, scheme):
    if scheme == 'int8':
        op = api.OpLinearQuantizerConfig(mode='linear_symmetric', dtype=np.int8,
            granularity='per_channel', weight_threshold=2048)
    elif scheme == 'palette6':
        op = api.OpPalettizerConfig(mode='kmeans', nbits=6,
            granularity='per_grouped_channel', group_size=32,
            num_kmeans_workers=1, weight_threshold=2048)
    else:
        raise ValueError('scheme must be int8 or palette6')
    return api.OptimizationConfig(global_config=op)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--validation-report', type=Path, required=True)
    parser.add_argument('--scheme', choices=['int8', 'palette6'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Run on macOS with coremltools 8.x.')
    import coremltools as ct
    import coremltools.optimize.coreml as optimization
    source, proof, output = [p.expanduser().resolve() for p in (args.package, args.validation_report, args.output_dir)]
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise ValueError('Choose a new output directory; source and prior candidates are preserved.')
    # Do not allow writing inside the source package.
    if source in output.parents:
        raise ValueError('Output cannot be inside source package.')
    source_model = ct.models.MLModel(str(source), compute_units=ct.ComputeUnit.CPU_ONLY)
    metadata = dict(source_model.user_defined_metadata)
    if metadata.get('precision') != 'fp32':
        raise ValueError('Compression source must be the verified FP32 export.')
    if metadata.get('compression_scheme'):
        raise ValueError('Use uncompressed FP32 source, not another compressed candidate.')
    report = json.loads(proof.read_text())
    validate_source_report(report, source, metadata.get('checkpoint_sha256'))
    before_ops = operation_counts(source_model.get_spec())
    print('Compressing:', args.scheme, '| source:', source, flush=True)
    cfg = compression_config(optimization, args.scheme)
    if args.scheme == 'int8':
        candidate = optimization.linear_quantize_weights(source_model, config=cfg)
        expected_ops = ('constexpr_affine_dequantize', 'constexpr_blockwise_shift_scale')
    else:
        candidate = optimization.palettize_weights(source_model, config=cfg)
        expected_ops = ('constexpr_lut_to_dense',)
    after_ops = operation_counts(candidate.get_spec())
    added_ops = sum(after_ops.get(name, 0) - before_ops.get(name, 0) for name in expected_ops)
    if added_ops <= 0:
        raise RuntimeError('No expected compressed-weight operations added; refusing to label this a compressed candidate.')
    candidate.user_defined_metadata.update(metadata)
    candidate.user_defined_metadata['compression_scheme'] = args.scheme
    candidate.user_defined_metadata['compression_scope'] = 'weights_only; float32 compute baseline; activations not quantized'
    candidate.user_defined_metadata['source_validation_sha256'] = sha_file(proof)
    output.mkdir(parents=True, exist_ok=False)
    package = output / 'SongFormerFunctionHead.mlpackage'
    candidate.save(str(package))
    base_size, size = package_bytes(source), package_bytes(package)
    result = {'scheme': args.scheme, 'compression_scope': 'weights_only',
        'activation_quantization': False, 'guaranteed_integer_compute': False,
        'source_package': str(source), 'package': str(package),
        'source_validation_report': str(proof), 'source_validation_sha256': sha_file(proof),
        'checkpoint_sha256': metadata['checkpoint_sha256'],
        'source_bytes': base_size, 'candidate_bytes': size,
        'size_ratio': size / base_size, 'size_reduction_fraction': 1 - size / base_size,
        'source_operations': before_ops, 'candidate_operations': after_ops,
        'compressed_weight_ops_added': added_ops,
        'coremltools': ct.__version__, 'ready_for_app_integration': False,
        'note': 'Head candidate only. Requires fresh validation; no encoder compression or iPhone speedup is claimed.'}
    write(output / 'compression_report.json', result)
    print('Saved:', package)
    print('Size %.2f -> %.2f MiB (%.1f%% reduction)' % (base_size/1024**2, size/1024**2, (1-size/base_size)*100))
    print('Next: verify_coreml --diagnostic --limit 0; compare F1 and per-class metrics.')


if __name__ == '__main__':
    main()
