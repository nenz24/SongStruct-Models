import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_boundary_audit', SCRIPTS / 'compare_boundary_audit.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
ROOT = SCRIPTS.parent
PROVIDER = (ROOT / 'SongMap/Data/SongAnalysis/OnDeviceSongAnalysisProvider.swift').read_text()
ANALYZER = (ROOT / 'SongMap/Data/SongAnalysis/Structure/MusicUnderstandingAnalyzer.swift').read_text()


def sections(*edges):
    return [{'start': a, 'end': b} for a, b in zip(edges, edges[1:])]


def run(name, *edges, seconds=10.):
    return {'configuration': name, 'seconds': seconds, 'sections': sections(*edges), 'error': None}


class BoundaryAuditTests(unittest.TestCase):
    def test_boundaries_skip_song_edges_and_duplicates(self):
        self.assertEqual(module.boundaries(sections(0., 10., 20., 30.), 30.), [10., 20.])
        self.assertEqual(module.boundaries([{'start': 1.4, 'end': 10.}, {'start': 10.02, 'end': 30.}], 30.), [1.4, 10.])

    def test_matching_is_one_to_one_within_tolerance(self):
        self.assertEqual(module.match([10., 20.], [10.2, 10.3, 23.], .5), [(10., 10.2)])
        result = module.scores([10., 20., 30.], [10.1, 23.])
        self.assertAlmostEqual(result['hit_0.5s']['recall'], 1 / 3); self.assertAlmostEqual(result['hit_0.5s']['precision'], .5)
        self.assertAlmostEqual(result['hit_3s']['f'], 2 * (2 / 3) * 1 / (2 / 3 + 1))
        self.assertAlmostEqual(result['hit_3s']['mean_abs_shift'], (0.1 + 3.) / 2)
        self.assertEqual(module.scores([], [])['hit_0.5s']['f'], 1.)

    def test_verdict_attributes_the_difference(self):
        mac = [10., 20., 30.]
        def item(*edges):
            b = module.boundaries(sections(0., *edges, 40.), 40.)
            return {'boundaries': b, 'vs_mac': module.scores(mac, b)}
        same, shifted, other = item(10., 20., 30.), item(13., 20., 30.), item(12., 25., 33.)
        self.assertIn('app call', module.verdict({module.APP: shifted, module.EXPORTER: same, module.REPEAT: same}))
        self.assertIn('consistent', module.verdict({module.APP: same, module.EXPORTER: same, module.REPEAT: same}))
        self.assertIn('run-to-run', module.verdict({module.APP: same, module.EXPORTER: same, module.REPEAT: shifted}))
        self.assertIn('device', module.verdict({module.APP: shifted, module.EXPORTER: shifted, module.REPEAT: shifted}))
        self.assertIn('partly', module.verdict({module.APP: other, module.EXPORTER: shifted, module.REPEAT: shifted}))
        self.assertIn('incomplete', module.verdict({module.APP: same}))

    def test_compare_matches_reports_to_mac_exports_by_audio_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'apple/Raise').mkdir(parents=True)
            (root / 'apple/Raise/0152 - Raise a Hallelujah.json').write_text(json.dumps({
                'boundary_source': 'apple_musicunderstanding.sections', 'audio_path': '/x/Raise/0152 - Raise a Hallelujah.mp3',
                'duration': 40., 'sections': sections(0., 10., 20., 30., 40.), 'os_version': 'macOS 27'}))
            (root / 'apple/other.json').write_text('[]')
            report = {'title': '0152 - Raise a Hallelujah', 'boundaryAuditEvidence': {'runs': [
                run(module.APP, 1.4, 10., 23., 30., 40.), run(module.EXPORTER, 0., 10., 20., 30., 40.),
                run(module.REPEAT, 0., 10., 20., 30., 40.), {'configuration': module.DEFAULT_TIMING, 'seconds': 1., 'sections': [], 'error': 'x'}]}}
            (root / 'analysis-1.json').write_text(json.dumps(report))
            (root / 'analysis-2.json').write_text(json.dumps({'title': 'No audit'}))
            exports = module.load_mac_exports(root / 'apple')
            self.assertEqual(list(exports), ['0152 - Raise a Hallelujah'])
            results = module.compare(sorted(root.glob('analysis-*.json')), exports)
            self.assertIn('app call', results[0]['verdict'])
            self.assertEqual(results[0]['runs'][module.DEFAULT_TIMING], {'error': 'x'})
            self.assertEqual(results[0]['runs'][module.EXPORTER]['vs_mac']['hit_0.5s']['f'], 1.)
            self.assertIn('no boundary audit', results[1]['verdict'])
            self.assertIn('MuQ path: no', results[1]['hint'])
            started = {'title': 'x', 'events': [{'name': 'music_understanding_boundary_audit', 'kind': 'begin'}]}
            self.assertIn('did not finish', module.missing_audit_hint(started))

    def test_launch_argument_matcher_accepts_smart_dash_variants(self):
        diagnostics = (ROOT / 'SongMap/Data/SongAnalysis/SongAnalysisDiagnostics.swift').read_text()
        body = re.search(r'static func launchArgument\(.*?\n    \}', diagnostics, re.S).group(0)
        # Mirror of the Swift rule: drop leading "-", en dash and em dash, then compare exactly.
        self.assertIn('$0 == "-" || $0 == "\\u{2013}" || $0 == "\\u{2014}"', body)
        strip = lambda value: value.lstrip('-\u2013\u2014')
        for typed in ('--songformer-muq', '-songformer-muq', '\u2014songformer-muq', '\u2013songformer-muq'):
            self.assertEqual(strip(typed), 'songformer-muq')
        self.assertNotEqual(strip('--songformer-muq-boundary-refinement'), 'songformer-muq')

    def test_swift_audit_is_debug_only_and_records_without_changing_the_analysis(self):
        block = PROVIDER[PROVIDER.index('if SongAnalysisDiagnostics.launchArgument("musicunderstanding-boundary-audit")') - 200:]
        self.assertIn('#if DEBUG', block[:220])
        self.assertIn('appSections: music.sections', PROVIDER)
        self.assertNotRegex(PROVIDER, r'music\s*=\s*.*structureSections')
        for name in (module.APP, module.EXPORTER, module.REPEAT, module.DEFAULT_TIMING):
            self.assertIn(f'"{name}"', PROVIDER)
        self.assertIn('AVURLAssetPreferPreciseDurationAndTimingKey: true', ANALYZER)
        self.assertIn('session.analyze(for: [.structure])', ANALYZER)


if __name__ == '__main__':
    unittest.main()
