import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'audit_section_range.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SectionAuditTests(unittest.TestCase):
    def report(self,offset=0):
        return {'title':'song','structureEvidence':{'labelOrder':['chorus','bridge'],
            'frames':[{'time':offset,'duration':.12,'probabilities':[.9,.1]},
                {'time':offset+.12,'duration':.12,'probabilities':[.1,.9]}],
            'appleBoundaries':[{'start':offset,'end':offset+.24}],
            'assembledSections':[{'start':offset,'end':offset+.24,'label':'chorus'}]}}
    def test_overlap_probability_pooling_not_frame_majority(self):
        result=module.audit(self.report(),.10,.24)
        self.assertEqual(result['pooled_prediction'],'bridge')
        self.assertAlmostEqual(result['probabilities']['bridge'],(.02*.1+.12*.9)/.14)
        self.assertAlmostEqual(result['coverage_fraction'],1)
    def test_chunk_offset_and_no_coverage_are_explicit(self):
        result=module.audit(self.report(419.88),419.98,420.12)
        self.assertEqual(result['pooled_prediction'],'bridge')
        self.assertAlmostEqual(result['coverage_fraction'],1)
        empty=module.audit(self.report(),1,2)
        self.assertIsNone(empty['pooled_prediction'])
        self.assertEqual(empty['coverage_fraction'],0)
    def test_old_report_and_invalid_intervals_rejected(self):
        with self.assertRaises(ValueError):module.audit({'title':'old'},0,1)
        for start,end in [(2,1),(-1,1),(0,float('nan'))]:
            with self.assertRaises(ValueError):module.audit(self.report(),start,end)

    def test_context_labels_are_exposed_without_rewriting_frame_predictions(self):
        r=self.report()
        r['contextEvidence']={'sections':[{'start':0,'end':.24,'baselineLabel':'chorus','label':'bridge'}]}
        result=module.audit(r,0,.1)
        self.assertEqual(result['pooled_prediction'],'chorus')
        self.assertEqual(result['overlapping_context_sections'][0]['label'],'bridge')
        self.assertIn('before section-context',result['pooled_prediction_scope'])

    def test_boundary_audit_keeps_original_context_and_final_ranges_distinct(self):
        r=self.report()
        r['boundaryRefinementEvidence']={'decisions':[{'applied':True,'originalSeconds':.1,'candidateSeconds':.2}]}
        result=module.audit(r,.12,.18)
        self.assertEqual(len(result['overlapping_boundary_shifts']),1)
        self.assertEqual(module.audit(r,.21,.23)['overlapping_boundary_shifts'],[])
