import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('install_muq_pipeline', SCRIPTS / 'install_muq_pipeline.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from install_section_context import file_hashes, sha_file
SWIFT = (SCRIPTS.parent / 'SongMap/Data/SongAnalysis/Structure/SongFormerMuQPipeline.swift').read_text()
ANALYZER = (SCRIPTS.parent / 'SongMap/Data/SongAnalysis/Structure/SongFormerAnalyzer.swift').read_text()


def make_probe(root):
    probe = root / 'outputs.local/muq-pipeline-iphone-1'; assets = probe / 'Assets'; assets.mkdir(parents=True)
    for name in module.RESOURCES:
        (assets / (name + '.mlmodelc')).mkdir(); (assets / (name + '.mlmodelc/model.mil')).write_text(name)
    manifest = {'encoder_roles': ['muq'], 'canonical_encoder_functions': {'muq': module.FUNCTIONS},
                'files': {n: file_hashes(assets / (n + '.mlmodelc')) for n in module.RESOURCES},
                'rotary': {'muq': [0.5] * 32}, 'base_checkpoint': 'head-sha', 'context_checkpoint': 'ctx-sha',
                'compiled_encoder_bytes': 184501315}
    (assets / 'benchmark_manifest.json').write_text(json.dumps(manifest))
    (probe / 'preparation.json').write_text(json.dumps({
        'protocol': 'muq-pipeline-phone-preparation-v1', 'native_swift_checks_passed': True, 'native_typecheck_passed': True,
        'mac_reference_passed': True, 'musicfm_bundled': False, 'encoder_roles': ['muq'], 'quality_sha256': 'q',
        'manifest_sha256': sha_file(assets / 'benchmark_manifest.json')}))
    return probe


def batch(probe, **change):
    sha = json.loads((probe / 'preparation.json').read_text())['manifest_sha256']
    runs = [{'report': {'manifest_sha256': sha, 'label_parity_passed': True, 'hardware_identifier': 'iPhone18,1'}}] * 6
    return {'state': 'completed', 'mode': 'cpuOnly', 'requested_runs': 6, 'completed_runs': 6, 'runs': runs, **change}


class InstallMuQPipelineTests(unittest.TestCase):
    def test_install_copies_verified_models_and_writes_the_swift_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); probe = make_probe(root)
            (root / 'phone.json').write_text(json.dumps(batch(probe)))
            destination = module.install(root, probe, root / 'phone.json')
            self.assertEqual(destination, (root / module.DESTINATION).resolve())
            manifest = json.loads((destination / module.MANIFEST).read_text())
            self.assertEqual(manifest['models']['muq'], 'SongFormerMuQ_Encoder.mlmodelc')
            self.assertEqual((manifest['headCheckpointSHA256'], manifest['contextCheckpointSHA256']), ('head-sha', 'ctx-sha'))
            self.assertEqual(manifest['provenance']['phone_runs'], 6)
            for name in module.RESOURCES.values():
                self.assertEqual((destination / (name + '.mlmodelc/model.mil')).is_file(), True)
            self.assertEqual(module.install(root, probe, root / 'phone.json'), destination)  # identical reinstall is a no-op
            (destination / module.MANIFEST).write_text('{}')
            with self.assertRaises(ValueError): module.install(root, probe, root / 'phone.json')

    def test_changed_models_unvalidated_probe_or_failed_phone_batch_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); probe = make_probe(root)
            (probe / 'Assets/head.mlmodelc/model.mil').write_text('tampered')
            with self.assertRaises(ValueError): module.validate_probe(probe)
        for change in ({'musicfm_bundled': True}, {'mac_reference_passed': False}, {'manifest_sha256': 'x'}):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); probe = make_probe(root)
                prep = json.loads((probe / 'preparation.json').read_text()); prep.update(change)
                (probe / 'preparation.json').write_text(json.dumps(prep))
                with self.assertRaises(ValueError): module.validate_probe(probe)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); probe = make_probe(root)
            bad = batch(probe); bad['runs'] = bad['runs'][:5] + [{'report': {**bad['runs'][0]['report'], 'label_parity_passed': False}}]
            for payload in (batch(probe, mode='cpuAndGPU'), batch(probe, state='failed'), bad):
                (root / 'phone.json').write_text(json.dumps(payload))
                with self.assertRaises(ValueError): module.validate_probe(probe, root / 'phone.json')

    def test_phone_batch_is_found_only_when_it_matches_this_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); probe = make_probe(root)
            downloads = root / 'Downloads'; (downloads / 'old').mkdir(parents=True)
            self.assertIsNone(module.find_phone_batch(probe, [downloads, root / 'missing']))
            (downloads / 'old/pipeline-batch-latest.json').write_text(json.dumps(batch(probe, mode='cpuAndGPU')))
            (downloads / 'pipeline-batch-latest 2.json').write_text('not json')
            self.assertIsNone(module.find_phone_batch(probe, [downloads]))
            (downloads / 'pipeline-batch-latest 3.json').write_text(json.dumps(batch(probe)))
            self.assertEqual(module.find_phone_batch(probe, [downloads]), downloads / 'pipeline-batch-latest 3.json')

    def test_manifest_keys_and_constants_match_the_swift_reader(self):
        manifest = module.app_manifest({'manifest_sha256': 'm', 'quality_sha256': 'q'},
                                       {'rotary': {'muq': [0.] * 32}, 'base_checkpoint': 'h', 'context_checkpoint': 'c',
                                        'compiled_encoder_bytes': 1}, None, {})
        block = re.search(r'struct Specification: Decodable \{(.*?)\n    \}\n', SWIFT, re.S).group(1)
        fields = set(re.findall(r'let (\w+):', block)) - {'log_mel', 'muq_conv', 'muq', 'head', 'context'}
        self.assertTrue(fields <= set(manifest), fields - set(manifest))
        self.assertEqual(set(manifest['models']), {'log_mel', 'muq_conv', 'muq', 'head', 'context'})
        self.assertIn(f'static let manifestName = "{module.MANIFEST[:-5]}"', SWIFT)
        for literal in ('decoded.pipeline == "muq-only-v1"', 'decoded.featureWidth == 2048',
                        '["750": "short_30", "3000": "long_120", "partial": "partial"]', '[0, 1, 2, 3, 4, 5, 6, 26]'):
            self.assertIn(literal, SWIFT)
        self.assertEqual(manifest['labelOrder'], module.LABELS)
        self.assertIn('launchArgument("songformer-muq", in: args)', ANALYZER)
        self.assertIn('SongFormerMuQPipeline(diagnostics: diagnostics)', ANALYZER)
        self.assertIn('muqOnly = true; worship = false', ANALYZER)  # Release runs the MuQ-only pipeline

    def test_swift_pipeline_mirrors_the_validated_probe_contract(self):
        for literal in ('"rotary_cos"', '"rotary_sin"', '[2, count, 1, 1, 64]', 'configuration.computeUnits = .cpuOnly',
                        'width: width)', '"input_embeddings"', '"section_audio"', 'metadata["variant"] == "muq"'):
            self.assertIn(literal, SWIFT)
        code = '\n'.join(line for line in SWIFT.splitlines() if not line.strip().startswith('//'))
        self.assertNotIn('cpuAndGPU', code); self.assertNotIn('musicfm', code.lower())


if __name__ == '__main__':
    unittest.main()
