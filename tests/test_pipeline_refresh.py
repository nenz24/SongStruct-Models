import json
from pathlib import Path
import tempfile
import unittest
from deployment.refresh_pipeline_iphone import refresh


class PipelineRefreshTests(unittest.TestCase):
    def test_refresh_preserves_compiled_assets_and_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'Assets').mkdir()
            project=root/'Benchmark.xcodeproj';project.mkdir()
            (project/'project.pbxproj').write_text('existing signing configuration')
            manifest={'shape_policy':'Original variable lengths, CPU_ONLY; no zero padding or GPU dynamic-shape calls.',
                'sections':[{'start':0,'end':10}], 'class_ids':[0,1,2,3,4,5,6,26]}
            m=root/'Assets/benchmark_manifest.json';m.write_text(json.dumps(manifest))
            weight=root/'Assets/weights.bin';weight.write_bytes(b'preserve assets')
            before=m.read_bytes()
            self.assertEqual(refresh(root),project)
            self.assertEqual(weight.read_bytes(),b'preserve assets')
            self.assertEqual(m.read_bytes(),before)
            self.assertEqual((project/'project.pbxproj').read_text(),'existing signing configuration')
            self.assertIn('PREDICT_START',(root/'BenchmarkRunner.swift').read_text())

    def test_rejects_other_benchmark_before_overwriting_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'Assets').mkdir()
            (root/'Benchmark.xcodeproj').mkdir()
            (root/'Benchmark.xcodeproj/project.pbxproj').write_text('project')
            (root/'Assets/benchmark_manifest.json').write_text('{"shape_policy":"fixed_per_fixture"}')
            source=root/'BenchmarkApp.swift';source.write_text('original')
            with self.assertRaises(ValueError):refresh(root)
            self.assertEqual(source.read_text(),'original')
