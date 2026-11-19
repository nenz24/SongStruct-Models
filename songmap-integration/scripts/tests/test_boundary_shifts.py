import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'review_boundary_shifts.py'
sys.path.insert(0,str(SCRIPT.parent))
from generate_boundary_shift_fixture import expected_result, cases
spec = importlib.util.spec_from_file_location('boundary_shifts', SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture(old=10, actual=12):
    sections = [dict(start=0., end=float(old), label='bridge'), dict(start=float(old), end=24., label='chorus')]
    frames = []
    for i in range(96):
        p = [.01]*8; p[3 if i*.25 < actual else 2] = .93
        frames.append(dict(time=i*.25, duration=.25, probabilities=p))
    return dict(title='synthetic', id='fixture', state='completed',
        contextEvidence=dict(selectedEpoch=5, precision='fp32', labelOrder=module.LABELS,
            checkpointSHA256='fixture-checkpoint', sections=copy.deepcopy(sections)),
        structureEvidence=dict(labelOrder=module.LABELS, frames=frames, bpm=120., beatsPerBar=4,
            downbeatOffset=0., assembledSections=sections,
            appleBoundaries=[dict(start=s['start'], end=s['end']) for s in sections]))


class BoundaryShiftTests(unittest.TestCase):
    def test_evidence_can_shift_either_way_without_changing_labels_or_source(self):
        for old, actual in ((10, 12), (12, 10)):
            with self.subTest(old=old):
                report = fixture(old, actual); snapshot = copy.deepcopy(report)
                result = module.propose(report)
                self.assertEqual(result['candidate_count'], 1)
                self.assertEqual(result['boundaries'][0]['candidate_seconds'], actual)
                self.assertEqual(result['simulated_sections'][0]['end'], actual)
                self.assertEqual(result['simulated_sections'][1]['start'], actual)
                self.assertEqual([s['label'] for s in result['simulated_sections']], ['bridge','chorus'])
                self.assertEqual(report, snapshot)

    def test_correct_boundary_retains_identity_and_large_moves_are_not_invented(self):
        for old, actual in ((10, 10), (10, 16)):
            r = module.propose(fixture(old, actual))
            self.assertEqual(r['candidate_count'], 0)
            self.assertEqual(r['boundaries'][0]['candidate_seconds'], old)

    def test_missing_frames_and_weak_anchors_abstain(self):
        r = fixture(); r['structureEvidence']['frames'] = [f for f in r['structureEvidence']['frames'] if not 10 <= f['time'] < 12]
        self.assertEqual(module.propose(r)['candidate_count'], 0)
        r = fixture()
        for f in r['structureEvidence']['frames']:
            if 6 <= f['time'] < 8:
                f['probabilities'] = [.125]*8
        self.assertEqual(module.propose(r)['candidate_count'], 0)

    def test_same_labels_silence_and_gaps_preserve_existing_ranges(self):
        for mode in ('same', 'silence', 'gap'):
            r = fixture()
            if mode == 'gap':
                for field in ('assembledSections','appleBoundaries'):
                    r['structureEvidence'][field][1]['start'] = 10.5
                r['contextEvidence']['sections'][1]['start'] = 10.5
            else:
                label = 'bridge' if mode == 'same' else 'silence'
                r['structureEvidence']['assembledSections'][1]['label'] = label
                r['contextEvidence']['sections'][1]['label'] = label
            result = module.propose(r)
            self.assertEqual(result['candidate_count'], 0)
            self.assertEqual(result['simulated_sections'], r['structureEvidence']['assembledSections'])

    def test_rejects_corrupt_or_mismatched_context_and_grid(self):
        mutations = [lambda r:r['structureEvidence']['frames'][0]['probabilities'].__setitem__(0, float('nan')),
            lambda r:r['structureEvidence']['frames'][1].__setitem__('time', 0.),
            lambda r:r['structureEvidence'].__setitem__('bpm', 0.),
            lambda r:r['contextEvidence']['sections'][0].__setitem__('label','chorus'),
            lambda r:r.__setitem__('state','running')]
        for mutate in mutations:
            r = fixture(); mutate(r)
            with self.assertRaises(ValueError): module.propose(r)

    def test_cli_blank_manual_csv_and_bound_reference_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'analysis-fixture.json'; source.write_text(json.dumps(fixture()))
            out = root/'review'
            command = [sys.executable, str(SCRIPT), '--report', str(source), '--output-dir', str(out)]
            subprocess.run(command, check=True, capture_output=True)
            payload = json.loads((out/'review.json').read_text()); tracks = payload['tracks']
            self.assertFalse(payload['maps_changed']); self.assertFalse(payload['boundary_accuracy_validated'])
            with (out/'boundary_review.csv').open(newline='') as f: rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['reference_seconds'], '')
            reference = root/'manual.csv'
            def write():
                with reference.open('w', newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=module.FIELDS);writer.writeheader();writer.writerows(rows)
            write()
            with self.assertRaises(ValueError): module.reference_metrics(tracks, reference)
            rows[0]['reference_seconds']='12';write()
            metrics=module.reference_metrics(tracks,reference)
            self.assertEqual(metrics['baseline']['mae_seconds'],2)
            self.assertEqual(metrics['candidate']['mae_seconds'],0)
            rows[0]['source_sha256']='stale';write()
            with self.assertRaises(ValueError): module.reference_metrics(tracks,reference)
            # Refuse overwrite and leave source JSON unchanged.
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(json.loads(source.read_text()),fixture())

    def test_native_debug_report_reproduces_original_context_rule_without_cumulative_shift(self):
        report=fixture();native=expected_result(report)
        report['boundaryRefinementEvidence']=native
        report['structureEvidence']['assembledSections']=copy.deepcopy(native['refinedSections'])
        result=module.propose(report)
        self.assertEqual(result['candidate_count'],1)
        self.assertEqual(result['boundaries'][0]['original_seconds'],10)
        self.assertEqual(result['boundaries'][0]['candidate_seconds'],12)
        self.assertTrue(result['native_runtime_comparison']['passed'])
        self.assertEqual(report['contextEvidence']['sections'][0]['end'],10)
        native['decisions'][0]['candidateSeconds']=14
        with self.assertRaises(ValueError):module.propose(report)

    def test_native_debug_evidence_rejects_changed_parameters_or_original_ranges(self):
        for field in ('parameters','originalSections','refinedSections','appliedCount'):
            report=fixture();native=expected_result(report);report['boundaryRefinementEvidence']=native
            report['structureEvidence']['assembledSections']=copy.deepcopy(native['refinedSections'])
            if field=='parameters':native[field]['meanLogGain']=.05
            elif field=='appliedCount':native[field]=99
            else:native[field][0]['end']+=1
            with self.assertRaises(ValueError):module.propose(report)

    def test_native_golden_cases_include_real_report_and_fixed_no_change_scenarios(self):
        fixtures=cases();by_name={c['name']:c for c in fixtures}
        real=by_name['uploaded_who_you_say_i_am']['expected']
        self.assertEqual(real['appliedCount'],1)
        shift=next(d for d in real['decisions'] if d['applied'])
        self.assertAlmostEqual(shift['candidateSeconds'],283.3889655172414)
        self.assertEqual(real['refinedSections'][-2]['start'],323.18)
        for name in ('identity','large_move_abstains','missing_coverage','same_label','silence','gap','short_anchor','weak_anchor'):
            self.assertEqual(by_name[name]['expected']['appliedCount'],0,name)


if __name__ == '__main__': unittest.main()
