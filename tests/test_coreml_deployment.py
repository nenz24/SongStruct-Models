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
