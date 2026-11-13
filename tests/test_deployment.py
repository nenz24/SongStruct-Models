import json
from pathlib import Path
import tempfile
import unittest
from deployment.audit import package_inventory, timed_calls


class AuditTests(unittest.TestCase):
    def test_missing_and_lfs_assets_are_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            roles = ['log_mel', 'muq_conv', 'muq', 'musicfm_conv', 'musicfm', 'head']
            (root / 'songformer_coreml_manifest.json').write_text(json.dumps(
                {'models': {role: role + '.mlpackage' for role in roles}}))
            package = root / 'head.mlpackage'
            package.mkdir()
            (package / 'weight.bin').write_text('version https://git-lfs.github.com/spec/v1\nsize 10000000')
            _, assets = package_inventory(root)
            self.assertFalse(any(a['complete_for_size_measurement'] for a in assets))
            self.assertEqual(len(assets[-1]['git_lfs_pointers']), 1)
            (package / 'weight.bin').write_bytes(b'actual fixture weights')
            _, assets = package_inventory(root)
            self.assertTrue(assets[-1]['complete_for_size_measurement'])

    def test_cpu_benchmark_reports_actual_duration_and_scope(self):
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest('torch/numpy unavailable')
        from unittest.mock import patch
        from deployment.audit import benchmark
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.function_head = torch.nn.Linear(4096, 128)
            def forward_func(self, batch):
                values = batch['input_embeddings'][:, ::3]
                return {'function_logits': self.function_head(values)}
        row = {'approved': True, 'split': 'val', 'track_id': 'fixture'}
        with patch('finetuning.engine.manifest', return_value=({'records': [row], 'plan_id': 'plan'}, 'prov')), \
             patch('finetuning.common.iter_features', return_value=iter([(np.ones((30, 4096), dtype=np.float32), 0, 'prov')])), \
             patch('finetuning.common.load_head', return_value=(Head(), None)):
            result = benchmark({}, Path('unused'), 'cpu', 30, 2)
        self.assertEqual(result['audio_seconds_represented'], 1.2)
        self.assertEqual(result['timing']['repeats'], 2)
        self.assertIn('not Core ML or iPhone', result['scope'])

    def test_warmup_excluded_from_measurement_count(self):
        calls = []
        report = timed_calls(lambda: calls.append(1), lambda: None, 3)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(report['seconds']), 3)
        with self.assertRaises(ValueError):
            timed_calls(lambda: None, lambda: None, 0)
