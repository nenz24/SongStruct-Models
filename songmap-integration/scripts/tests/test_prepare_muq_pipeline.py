import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('prepare_muq_pipeline_iphone', SCRIPTS / 'prepare_muq_pipeline_iphone.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
LABELS = module.LABELS
RUNNER = (SCRIPTS / 'encoder_pipeline_iphone/BenchmarkRunner.swift').read_text()
fingerprint = lambda value: json.dumps(value, sort_keys=True)


def detail(start, prediction, p=.9):
    probabilities = dict.fromkeys(LABELS, (1 - p) / 7); probabilities[prediction] = p
    return dict(start=start, end=start + 10, label_manual='chorus', prediction=prediction, probabilities=probabilities)


class PrepareMuQPipelineTests(unittest.TestCase):
    def test_quality_must_be_the_passed_full_muq_validation_with_its_selection(self):
        selection = {'protocol': module.QUALITY_PROTOCOL, 'x': 1}
        selection['identity'] = fingerprint({k: v for k, v in selection.items()})
        q = dict(protocol=module.QUALITY_PROTOCOL, split='val', full_validation=True, passed=True,
                 reference_reproduces_verification=True, musicfm_loaded=False, test_used=False,
                 selection_identity=selection['identity'])
        module.validate_quality(q, selection, fingerprint)
        for change in ({'passed': False}, {'full_validation': False}, {'musicfm_loaded': True}, {'test_used': True},
                       {'split': 'test'}, {'selection_identity': 'other'}, {'reference_reproduces_verification': False}):
            with self.assertRaises(ValueError): module.validate_quality({**q, **change}, selection, fingerprint)
        with self.assertRaises(ValueError): module.validate_quality(q, {**selection, 'x': 2}, fingerprint)

    def test_mac_reference_must_reproduce_validated_labels_and_probabilities(self):
        validated = [detail(0., 'verse'), detail(10., 'chorus')]
        self.assertTrue(module.reproduce(validated, [dict(d) for d in validated])['passed'])
        nudged = [dict(d, probabilities={**d['probabilities'], 'verse': d['probabilities']['verse'] + 5e-5}) for d in validated]
        self.assertTrue(module.reproduce(validated, nudged)['passed'])
        drifted = [dict(d, probabilities={**d['probabilities'], 'verse': d['probabilities']['verse'] + 2e-4}) for d in validated]
        self.assertFalse(module.reproduce(validated, drifted)['passed'])
        changed = module.reproduce(validated, [validated[0], detail(10., 'bridge')])
        self.assertEqual((changed['passed'], changed['changed_sections']), (False, [(10.0, 20.0)]))
        with self.assertRaises(ValueError): module.reproduce(validated, validated[:1])
        with self.assertRaises(ValueError): module.reproduce(validated, [validated[0], detail(30., 'chorus')])

    def test_targets_carry_original_section_index_final_and_base_probabilities(self):
        details = [detail(0., 'verse'), detail(20., 'chorus')]
        arrays = {'indices': [0, 2], 'priors': np.full((2, 8), .125, np.float32)}
        targets = module.bundle_targets(details, arrays)
        self.assertEqual([t['index'] for t in targets], [0, 2])
        self.assertEqual(targets[1]['prediction'], 'chorus'); self.assertEqual(len(targets[0]['probabilities']), 8)
        self.assertEqual(targets[0]['base_probabilities'], [.125] * 8)
        with self.assertRaises(ValueError): module.bundle_targets(details, {**arrays, 'indices': [0]})

    def test_manifest_matches_the_swift_runner_muq_contract(self):
        bank = {'evidence': {'repair_sha256': 'r', 'source_state_sha256': 's'}, 'compiled_bytes': 184501315}
        files = {name: {'model.mil': 'h'} for name in module.RESOURCES}
        m = module.manifest_data(quality_sha='q', bank=bank, bank_report_sha='b', rotary={'muq': [0.] * 32},
                                 export={'head_checkpoint_sha256': 'h', 'context_checkpoint_sha256': 'c'},
                                 files=files, tracks=[], probe_source_sha='p')
        # Mirrors the guards in BenchmarkRunner.manifest() for encoder_roles == ["muq"].
        self.assertEqual(m['encoder_roles'], ['muq'])
        self.assertEqual(m['mac_preflight_policy'], 'muq-functional-v1')
        self.assertEqual(m['canonical_encoder_functions'], {'muq': {'750': 'short_30', '3000': 'long_120', 'partial': 'partial'}})
        self.assertEqual(len(m['canonical_encoder_reports']), 1)
        self.assertEqual(set(m['files']), {'log_mel', 'head', 'context', 'muq', 'muq_conv'})
        self.assertEqual((m['static_graph_policy'], m['frontend_route_probe'], m['frontend_gpu_policy']),
                         ('split-before-conv-v2', False, 'disabled'))
        self.assertEqual((m['canonical_encoder_schedule'], m['tensor_transfer_policy']),
                         ('one-role-at-a-time-v1', 'owned-fp32-between-models-v1'))
        self.assertNotIn('musicfm', json.dumps(m))
        with self.assertRaises(ValueError):
            module.manifest_data(quality_sha='q', bank=bank, bank_report_sha='b', rotary={}, export={'head_checkpoint_sha256': 'h',
                                 'context_checkpoint_sha256': 'c'}, files={**files, 'musicfm': {}}, tracks=[], probe_source_sha='p')
        for literal in ('result.encoder_roles == ["muq"]', '"muq-functional-v1"', 'Self.roles(result)'):
            self.assertIn(literal, RUNNER)

    def test_runner_has_no_hardcoded_four_stream_width_left(self):
        code = '\n'.join(line for line in RUNNER.splitlines() if not line.strip().startswith('//') and '///' not in line)
        self.assertEqual(re.findall(r'\b4096\b', code), [])
        self.assertIn('width: featureWidth', code)
        self.assertIn('ordered[streamIndex*roles.count+roleIndex]', code)


if __name__ == '__main__':
    unittest.main()
