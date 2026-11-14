import unittest
from pathlib import Path
import numpy as np
from deployment.iphone_project import project_text
from deployment.verify_encoders import rotary, compare_embeddings


class PhoneTests(unittest.TestCase):
    def test_project_contains_distinct_models_and_optimized_target(self):
        text = project_text('com.example.Benchmark', 'ABCDEFGHIJ')
        for name in ['HeadFP32.mlmodelc','HeadINT8.mlmodelc','HeadPalette6.mlmodelc','features.f32','reference_logits.f32']:
            self.assertIn(name, text)
        self.assertIn('SWIFT_OPTIMIZATION_LEVEL = "-O"', text)
        self.assertIn('DEVELOPMENT_TEAM = ABCDEFGHIJ', text)
        self.assertEqual(text.count('{'), text.count('}'))
        with self.assertRaises(ValueError):
            project_text('bad;injection')
        with self.assertRaises(ValueError):
            project_text(team_id='bad')

    def test_preparation_keeps_candidates_separate_and_writes_identical_fixture(self):
        import json, sys, tempfile
        from unittest.mock import patch
        from types import SimpleNamespace as NS
        import deployment.prepare_iphone as stage
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proof = dict(full_validation=True, logit_parity_passed=True, tracks=1,
                checkpoint_sha256='fixture', precision_metadata='fp32',
                plan_id='plan', provenance_id='prov', annotations_id='manual')
            packages = {}
            for name in ('fp32','int8','palette6'):
                package = root / name / 'Head.mlpackage'
                package.mkdir(parents=True)
                (package / 'weights').write_bytes(b'fixture')
                packages[name] = package
            proof['package'] = str(packages['fp32'])
            proof_path = root / 'fp32-validation.json'
            proof_path.write_text(json.dumps(proof))
            for name in ('int8','palette6'):
                (root / f'{name}-validation.json').write_text(json.dumps({**proof,'package':str(packages[name])}))
            comparison = root / 'comparison.json'
            comparison.write_text(json.dumps({'fp32_validation_report': str(proof_path)}))
            def mlmodel(path, **kwargs):
                scheme = Path(path).parent.name
                metadata = {'checkpoint_sha256':'fixture'}
                if scheme != 'fp32': metadata['compression_scheme'] = scheme
                return NS(user_defined_metadata=metadata,
                          predict=lambda inputs: {'function_logits': np.zeros((1,inputs['input_embeddings'].shape[1]//3,128),dtype=np.float32)})
            def compile_model(command, **kwargs):
                (Path(command[4]) / (Path(command[3]).stem + '.mlmodelc')).mkdir()
            row = dict(approved=True,split='val',track_id='fixture')
            output = root / 'benchmark'
            fake = NS(models=NS(MLModel=mlmodel),ComputeUnit=NS(CPU_ONLY='cpu'))
            with patch.dict(sys.modules, {'coremltools':fake}), \
                 patch.object(stage.platform, 'system', return_value='Darwin'), \
                 patch.object(stage, 'config', return_value={}), \
                 patch.object(stage, 'manifest', return_value=({'records':[row],'plan_id':'plan'},'prov')), \
                 patch.object(stage, 'iter_features', return_value=iter([(np.ones((30,4096),dtype=np.float32),0,'prov')])), \
                 patch.object(stage.subprocess, 'run', side_effect=compile_model), \
                 patch.object(sys,'argv',['stage','--comparison',str(comparison),'--output-dir',str(output)]):
                stage.main()
            manifest = json.loads((output/'Assets/benchmark_manifest.json').read_text())
            self.assertEqual(manifest['frames'], 30)
            self.assertEqual([m['resource'] for m in manifest['models']], ['HeadFP32','HeadINT8','HeadPalette6'])
            self.assertEqual((output/'Assets/features.f32').stat().st_size,30*4096*4)
            self.assertTrue((output/'Benchmark.xcodeproj/project.pbxproj').is_file())
            self.assertTrue((output/'BenchmarkRunner.swift').is_file())

    def test_rotary_matches_swift_planes(self):
        frequencies = np.linspace(.001,1,32,dtype=np.float32)
        values = rotary(7, frequencies)
        self.assertEqual(values.shape, (2,7,1,1,64))
        np.testing.assert_allclose(values[0,0], 1)
        np.testing.assert_allclose(values[1,0], 0)
        np.testing.assert_allclose(values[0,3,0,0,:32], np.cos(3*frequencies))
        np.testing.assert_allclose(values[1,3,0,0,32:], np.sin(3*frequencies))
        with self.assertRaises(ValueError):
            rotary(3,[1])

    def test_embedding_metrics_reject_drift(self):
        reference = np.ones((5,1024),dtype=np.float32)
        self.assertTrue(compare_embeddings(reference,reference)['numerical_parity_passed'])
        self.assertFalse(compare_embeddings(reference,reference+1)['numerical_parity_passed'])
        self.assertFalse(compare_embeddings(reference,reference[:2])['numerical_parity_passed'])
        self.assertFalse(compare_embeddings(reference,reference*np.nan)['numerical_parity_passed'])


if __name__ == '__main__':
    unittest.main()
