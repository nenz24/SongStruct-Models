import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('export_muq_pipeline', SCRIPTS / 'export_muq_pipeline.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
import encoder_stream_ablation as ablation
import train_section_context as context

LABELS = module.LABELS
SONGSTRUCT = Path(os.environ.get('SONGSTRUCT_ROOT', '/home/user/nenz24/songstruct-models'))
UPSTREAM = Path(os.environ.get('SONGFORMER_UPSTREAM', '/home/user/ext/SongFormer/src/SongFormer'))
try:
    import coremltools as ct
except ImportError:
    ct = None
# The head converter is pinned to coremltools 8.x (SongStruct deployment/requirements-coreml.txt);
# coremltools 9 with torch 2.15 fails on x-transformers' rotary 'int' op regardless of input width.
REAL = (ct is not None and ct.__version__.startswith('8.')
        and (SONGSTRUCT / 'deployment/export_coreml.py').is_file() and (UPSTREAM / 'models/SongFormer.py').is_file())


def songformer():
    """Real upstream label architecture (random weights); evaluation-only imports stubbed."""
    for name in ('msaf', 'msaf.eval', 'postprocessing', 'postprocessing.functional', 'dataset', 'dataset.custom_types'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['msaf.eval'].compute_results = None
    sys.modules['postprocessing.functional'].postprocess_functional_structure = None
    sys.modules['dataset.custom_types'].MsaInfo = object
    for path in (str(UPSTREAM), str(SONGSTRUCT)):
        if path not in sys.path: sys.path.insert(0, path)
    from omegaconf import OmegaConf
    from models.SongFormer import Model
    torch.manual_seed(0)
    return Model(OmegaConf.load(UPSTREAM / 'configs/SongFormer.yaml')).eval()


class FakeNative:
    def __init__(self, fn): self.fn = fn
    def predict(self, inputs): return self.fn(inputs)


def sample(track, n=3, width=2048):
    p = torch.full((n, 8), .02); p[:, 2] = .86
    details = [dict(track_id=track, start=float(i), end=float(i + 1), label_manual='chorus' if i else 'verse',
                    prediction='chorus') for i in range(n)]
    return {'track_id': track, 'split': 'val', 'audio': torch.randn(n, width), 'timing': torch.rand(n, 6),
            'indices': torch.arange(n), 'probabilities': p, 'labels': torch.tensor([1] + [2] * (n - 1)), 'details': details}


class ExportMuQPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    @unittest.skipUnless(REAL, 'needs coremltools plus SongStruct-Models and upstream SongFormer clones')
    def test_real_head_converts_with_2048_flexible_input_and_palette6_shrinks_weights(self):
        model = songformer()  # also puts SongStruct-Models on sys.path
        from deployment.export_coreml import FunctionLogitsWrapper, disable_flash
        ablation.reduce_projection(model, ablation.columns('muq')); disable_flash(model)
        wrapper = FunctionLogitsWrapper(model).eval()
        fp32, checks = module.convert_head(ct, wrapper, 2048, skip_model_load=True)
        self.assertTrue(all(c['passed'] for c in checks))
        spec = fp32.get_spec()
        (inp,), (out,) = spec.description.input, spec.description.output
        ranges = inp.type.multiArrayType.shapeRange.sizeRanges
        self.assertEqual((inp.name, out.name), ('input_embeddings', 'function_logits'))
        self.assertEqual([(r.lowerBound, r.upperBound) for r in ranges], [(1, 1), (3, module.MAX_FRAMES), (2048, 2048)])
        palette6, added = module.palettize_head(fp32)
        self.assertGreater(added, 0)
        with tempfile.TemporaryDirectory() as folder:
            fp32.save(f'{folder}/a.mlpackage'); palette6.save(f'{folder}/b.mlpackage')
            small, large = module.package_bytes(f'{folder}/b.mlpackage'), module.package_bytes(f'{folder}/a.mlpackage')
        # 22M FP32 parameters -> roughly 6/32 of the bytes plus LUTs.
        self.assertLess(small, large * .3)
        self.assertGreater(large, 80e6)

    @unittest.skipUnless(ct is not None, 'needs coremltools')
    def test_context_exports_with_2048_contract(self):
        torch.manual_seed(1)
        model = context.SectionContextHead(width=2048).eval()
        torch.nn.init.normal_(model.output.weight, std=.01)
        converted, wrapper = module.export_context(ct, model, 2048, skip_model_load=True)
        (audio,) = [f for f in converted.get_spec().description.input if f.name == 'section_audio']
        self.assertEqual(audio.type.multiArrayType.shapeRange.sizeRanges[1].lowerBound, 2048)
        with self.assertRaises(ValueError):
            module.native_inputs(np.zeros((3, 4096)), np.zeros((3, 6)), 512, 2048)

    def test_coreml_head_adapter_validates_output(self):
        good = module.CoreMLHead(FakeNative(lambda i: {'function_logits': np.zeros((1, i['input_embeddings'].shape[1] // 3, 128), np.float32)}))
        out = good.forward_func({'input_embeddings': torch.zeros(1, 9, 2048)})['function_logits']
        self.assertEqual(tuple(out.shape), (1, 3, 128)); self.assertIs(good.eval(), good)
        bad = module.CoreMLHead(FakeNative(lambda i: {'function_logits': np.full((1, 3, 128), np.nan, np.float32)}))
        with self.assertRaises(ValueError): bad.forward_func({'input_embeddings': torch.zeros(1, 9, 2048)})

    def test_coreml_context_matches_pytorch_scoring(self):
        torch.manual_seed(2)
        model = context.SectionContextHead(width=2048).eval(); torch.nn.init.normal_(model.output.weight, std=.5)
        samples = [sample('a'), sample('b', 4)]
        rows = [{'track_id': 'a', 'split': 'val', 'approved': True}, {'track_id': 'b', 'split': 'val', 'approved': True}]
        reference, ref_details = context.evaluate_context(model, samples, rows, torch.device('cpu'))
        with torch.inference_mode():
            native = FakeNative(lambda i: {'context_residual': model(torch.from_numpy(i['section_audio']),
                                                                     torch.from_numpy(i['section_timing'])).numpy()})
            result, details, _ = module.coreml_context(native, samples, rows, 2048)
        self.assertEqual(result['confusion_matrix'], reference['confusion_matrix'])
        agreement = module.compare_pipelines(ref_details, details)
        self.assertEqual(agreement['changed_sections'], []); self.assertLess(agreement['max_final_probability_error'], 1e-6)
        self.assertTrue(module.gate('fp32', reference, result, agreement))
        bad = FakeNative(lambda i: {'context_residual': np.zeros((len(i['section_audio']), 8), np.float64)})
        with self.assertRaises(ValueError): module.coreml_context(bad, samples, rows, 2048)

    def test_export_imports_songstruct_deployment_only_after_load_sources_adds_its_root(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'deployment').mkdir()
            (Path(folder) / 'deployment/__init__.py').write_text('')
            (Path(folder) / 'deployment/export_coreml.py').write_text("raise RuntimeError('songstruct deployment imported')")
            def load_sources(args):
                sys.path.insert(0, folder)
                raise_later = types.SimpleNamespace(columns=list(range(2048)))
                return None, {'columns': raise_later.columns}, None, None, None, None, {}
            hidden = {k: None for k in list(sys.modules) if k == 'deployment' or k.startswith('deployment.')}
            fake_ct = types.ModuleType('coremltools')
            try:
                with patch.object(module.platform, 'system', return_value='Darwin'), \
                        patch.object(module, 'load_sources', load_sources), \
                        patch.dict(sys.modules, {'coremltools': fake_ct}), patch.dict(sys.modules, hidden):
                    for key in hidden: sys.modules.pop(key, None)
                    with self.assertRaisesRegex(RuntimeError, 'songstruct deployment imported'):
                        module.run_export(types.SimpleNamespace(output_dir=Path(folder) / 'out'))
            finally:
                sys.path.remove(folder)

    def test_gates(self):
        ref = {'section_accuracy': .80, 'macro_f1_present': .76}
        changed = {'changed_sections': [{'x': 1}], 'max_final_probability_error': 0.}
        self.assertFalse(module.gate('fp32', ref, ref, changed))
        self.assertFalse(module.gate('fp32', ref, ref, {'changed_sections': [], 'max_final_probability_error': 2e-4}))
        self.assertTrue(module.gate('palette6', ref, {'section_accuracy': .791, 'macro_f1_present': .751}, changed))
        self.assertFalse(module.gate('palette6', ref, {'section_accuracy': .789, 'macro_f1_present': .76}, changed))
        a = [dict(track_id='t', start=0., end=1., label_manual='verse', prediction='verse', probabilities=dict.fromkeys(LABELS, .125))]
        with self.assertRaises(ValueError): module.compare_pipelines(a, [{**a[0], 'start': 2.}])


if __name__ == '__main__':
    unittest.main()
