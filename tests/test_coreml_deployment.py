import unittest
try:
    import numpy as np
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch/numpy unavailable')
class ExportTests(unittest.TestCase):
    def test_wrapper_matches_reference_and_traces_variable_lengths(self):
        from deployment.export_coreml import FunctionLogitsWrapper
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.function_head = torch.nn.Linear(4096, 128)
            def forward_func(self, batch):
                self.last_dataset = batch['dataset_ids']
                assert batch['masks'] is None
                x = batch['input_embeddings'][:, ::3, :]
                return {'function_logits': self.function_head(x)}
        model = Model().eval()
        wrapped = FunctionLogitsWrapper(model).eval()
        x = torch.randn(1, 30, 4096)
        traced = torch.jit.trace(wrapped, x, check_trace=False)
        for frames in (3, 93, 750):
            x = torch.randn(1, frames, 4096)
            with torch.inference_mode():
                expected = model.forward_func({'input_embeddings': x, 'dataset_ids': torch.tensor([5]), 'masks': None})['function_logits']
                self.assertTrue(torch.equal(wrapped(x), expected))
                self.assertTrue(torch.equal(traced(x), expected))
            self.assertEqual(expected.shape, (1, frames//3, 128))
        self.assertEqual(model.last_dataset.item(), 5)

    def test_parity_rejects_wrong_shapes_nan_and_large_drift(self):
        from deployment.export_coreml import parity
        a = np.ones((1, 5, 128), dtype=np.float32)
        self.assertTrue(parity(a, a + .001)['passed'])
        self.assertFalse(parity(a, a + 1)['passed'])
        self.assertFalse(parity(a, a[:, :2])['passed'])
        self.assertFalse(parity(a, a * np.nan)['passed'])

    def test_probability_report_distinguishes_uniform_logit_shift(self):
        from deployment.export_coreml import parity
        reference = np.zeros((1, 4, 128), dtype=np.float32)
        shifted = reference + .15
        result = parity(reference, shifted)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['allowed_probability_max_abs_error'], 0)
        self.assertEqual(result['allowed_frame_argmax_agreement'], 1)
        shifted[..., 3] += 1
        self.assertGreater(parity(reference, shifted)['allowed_probability_max_abs_error'], .01)

    def test_diagnostic_keeps_failed_status_but_finishes_report(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import deployment.verify_coreml as verifier
        from finetuning.common import sha_file
        class Tiny(torch.nn.Module):
            def forward_func(self, batch):
                return {'function_logits': torch.zeros(1, 1, 128)}
        def evaluation(*args, details=None, **kwargs):
            details.append({'prediction': 'verse'})
            return {'macro_f1_present': .7}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'best.pt'
            checkpoint.write_bytes(b'fixture')
            fake = SimpleNamespace(user_defined_metadata={'checkpoint_sha256': sha_file(checkpoint)},
                predict=lambda batch: {'function_logits': np.ones((1, 1, 128), dtype=np.float32)})
            ct = SimpleNamespace(ComputeUnit=SimpleNamespace(CPU_ONLY='cpu'),
                                 models=SimpleNamespace(MLModel=lambda *a, **k: fake))
            row = {'approved': True, 'split': 'val', 'track_id': 'fixture'}
            for diagnostic in (False, True):
                output = Path(folder) / f'{diagnostic}.json'
                argv = ['verify', '--checkpoint', str(checkpoint), '--package', folder, '--output', str(output)]
                if diagnostic:
                    argv.append('--diagnostic')
                with patch.dict(sys.modules, {'coremltools': ct}), \
                     patch.object(verifier.platform, 'system', return_value='Darwin'), \
                     patch.object(verifier, 'config', return_value={}), \
                     patch.object(verifier, 'manifest', return_value=({'records': [row], 'plan_id': 'plan'}, 'prov')), \
                     patch.object(verifier, 'load_head', return_value=(Tiny(), None)), \
                     patch.object(verifier, 'iter_features', return_value=iter([(np.zeros((3,4096), dtype=np.float32), 0, 'prov')])), \
                     patch.object(verifier, 'evaluate', side_effect=evaluation), \
                     patch.object(sys, 'argv', argv):
                    if diagnostic:
                        verifier.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'logit parity failed'):
                            verifier.main()
                report = json.loads(output.read_text())
                self.assertFalse(report['logit_parity_passed'])
                self.assertFalse(report['ready_for_app_integration'])
                self.assertEqual(report['status'], 'numerical_mismatch_requires_review')
                self.assertEqual(report['coreml']['macro_f1_present'], .7)

    def test_coreml_adapter_validates_shape_and_frame_stride(self):
        from deployment.verify_coreml import CoreMLAdapter
        class FakeCoreML:
            def predict(self, batch):
                self.received = batch['input_embeddings']
                return {'function_logits': np.ones((1, self.received.shape[1]//3, 128), dtype=np.float32)}
        fake = FakeCoreML()
        adapter = CoreMLAdapter(fake)
        result = adapter.forward_func({'input_embeddings': torch.zeros(1, 93, 4096)})
        self.assertEqual(result['function_logits'].shape, (1, 31, 128))
        self.assertEqual(fake.received.dtype, np.float32)
        with self.assertRaises(ValueError):
            adapter.forward_func({'input_embeddings': torch.zeros(1, 94, 4096)})


if __name__ == '__main__':
    unittest.main()
