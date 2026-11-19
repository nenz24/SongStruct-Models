import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('timing',Path(__file__).resolve().parents[1]/'summarize_analysis_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TimingReportTests(unittest.TestCase):
    def report(self,events,state='completed'):
        return {'schemaVersion':1,'title':'song','state':state,'totalSeconds':80,'events':events}

    def test_repeated_nested_metrics_do_not_inflate_parent_stage(self):
        r=module.summarize(self.report([
            {'name':'songformer','kind':'stage','seconds':70},
            {'name':'songformer.predict.muq','kind':'metric','seconds':10},
            {'name':'songformer.predict.muq','kind':'metric','seconds':5}]))
        self.assertEqual(r['stages_seconds'],{'songformer':70})
        self.assertEqual(r['nested_metrics_seconds'],{'songformer.predict.muq':15})
        self.assertEqual(r['counts']['songformer.predict.muq'],2)

    def test_partial_stage_and_optional_failures_stay_visible(self):
        r=self.report([{'name':'lyrics_failure','kind':'event','detail':'transcription error'}],state='failed')
        r['activeStage']='lyrics';r['error']='failed'
        result=module.summarize(r)
        self.assertEqual(result['active_stage'],'lyrics')
        self.assertEqual(result['events'][0]['name'],'lyrics_failure')
        self.assertEqual(result['error'],'failed')

    def test_invalid_duration_is_rejected(self):
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):module.summarize(self.report([{'name':'x','kind':'stage','seconds':value}]))

    def test_context_changes_are_distinct_from_base_frame_evidence(self):
        r=self.report([])
        r['contextEvidence']={'selectedEpoch':5,'precision':'fp32','computeUnits':'CPU_ONLY',
            'checkpointSHA256':'checkpoint','lowAudioCoverageSections':[3],
            'sections':[{'baselineLabel':'chorus','label':'bridge','start':10,'end':20},
                        {'baselineLabel':'verse','label':'verse','start':20,'end':30}]}
        result=module.summarize(r)['section_context']
        self.assertEqual(result['sections'],2)
        self.assertEqual(len(result['changed_sections']),1)
        self.assertEqual(result['changed_sections'][0]['label'],'bridge')

    def test_debug_boundary_summary_distinguishes_applied_and_abstained_decisions(self):
        r=self.report([])
        r['boundaryRefinementEvidence']={'protocolVersion':'context-boundary-shift-review-v1',
            'appliedCount':1,'barSeconds':2,'decisions':[
                {'applied':True,'originalSeconds':10,'candidateSeconds':12},
                {'applied':False,'originalSeconds':20,'candidateSeconds':20}]}
        result=module.summarize(r)['boundary_refinement']
        self.assertEqual(result['applied_count'],1)
        self.assertEqual(len(result['applied_decisions']),1)
        self.assertEqual(result['applied_decisions'][0]['candidateSeconds'],12)
