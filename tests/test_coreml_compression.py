import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from deployment.compress_coreml import compression_config, operation_counts, validate_source_report


class CompressionTests(unittest.TestCase):
    def test_configuration_keeps_weight_only_and_independent_schemes(self):
        api = NS(OpLinearQuantizerConfig=lambda **kw: kw,
                 OpPalettizerConfig=lambda **kw: kw,
                 OptimizationConfig=lambda **kw: kw)
        linear = compression_config(api, 'int8')['global_config']
        palette = compression_config(api, 'palette6')['global_config']
        self.assertEqual(linear['granularity'], 'per_channel')
        self.assertEqual(palette['nbits'], 6)
        self.assertEqual(palette['granularity'], 'per_grouped_channel')
        self.assertEqual(palette['group_size'], 32)
        self.assertEqual(palette['num_kmeans_workers'], 1)
        with self.assertRaises(ValueError):
            compression_config(api, 'invalid')

    def test_source_requires_matching_verified_fp32(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder).resolve()
            report = dict(full_validation=True, logit_parity_passed=True, tracks=32,
                package=str(source), checkpoint_sha256='fixture', precision_metadata='fp32')
            validate_source_report(report, source, 'fixture')
            for key, value in [('full_validation', False), ('logit_parity_passed', False),
                               ('precision_metadata', 'fp16'), ('checkpoint_sha256', 'other'),
                               ('package', str(source / 'other'))]:
                with self.assertRaises(ValueError):
                    validate_source_report({**report, key: value}, source, 'fixture')

    def test_count_compression_operations_inside_nested_blocks(self):
        op = lambda name, blocks=(): NS(type=name, blocks=blocks)
        inner = NS(operations=[op('constexpr_lut_to_dense')])
        outer = NS(operations=[op('const'), op('cond', [inner])])
        spec = NS(mlProgram=NS(functions={'main': NS(block_specializations={'CoreML8': outer})}))
        counts = operation_counts(spec)
        self.assertEqual(counts, {'const': 1, 'cond': 1, 'constexpr_lut_to_dense': 1})


if __name__ == '__main__':
    unittest.main()
