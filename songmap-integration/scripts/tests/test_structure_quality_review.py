import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('review',Path(__file__).resolve().parents[1]/'review_structure_quality.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def report(length=20,transition=12):
    return {'title':'synthetic','structureEvidence':{'labelOrder':['chorus','outro'],
        'frames':[{'time':i,'duration':1,'probabilities':[.95,.05] if i<transition else [.05,.95]} for i in range(length)],
        'appleBoundaries':[{'start':0,'end':length}],
        'assembledSections':[{'start':0,'end':length,'label':'chorus'}]}}


class StructureReviewTests(unittest.TestCase):
    def test_sustained_split_is_proposed_without_changing_input(self):
        r=report();result=module.propose(r)
        self.assertEqual(len(result['candidates']),1)
        self.assertEqual(result['candidates'][0]['suggested_split_seconds'],12)
        self.assertFalse(result['maps_changed'])
        self.assertFalse(result['candidates'][0]['applied'])
        self.assertEqual(r['structureEvidence']['assembledSections'][0]['end'],20)

    def test_brief_change_and_tiny_pieces_are_rejected(self):
        r=report();r['structureEvidence']['frames'][12]['probabilities']=[.05,.95]
        for f in r['structureEvidence']['frames'][13:]:f['probabilities']=[.95,.05]
        self.assertFalse(module.propose(r)['candidates'])
        self.assertFalse(module.propose(report(20,18))['candidates'])

    def test_missing_coverage_or_frame_overlap_is_rejected(self):
        r=report();r['structureEvidence']['frames']=r['structureEvidence']['frames'][3:]
        self.assertFalse(module.propose(r)['candidates'])
        r=report();r['structureEvidence']['frames'][1]['time']=0
        with self.assertRaises(ValueError):module.propose(r)
        with self.assertRaises(ValueError):module.propose(report(),confidence=2)

    def test_bridge_audit_uses_manual_labels_and_counts_both_directions(self):
        r={'full_validation':True,'sections':[
            {'track_id':'a','label_manual':'bridge','prediction':'chorus'},
            {'track_id':'b','label_manual':'bridge','prediction':'bridge'},
            {'track_id':'c','label_manual':'chorus','prediction':'bridge'}]}
        result=module.bridge_audit(r)
        self.assertEqual(result['manual_bridge_support'],2)
        self.assertEqual(result['manual_bridge_predicted_as']['chorus'],1)
        self.assertEqual(len(result['bridge_chorus_cases']),2)
        self.assertFalse(result['manual_labels_changed'])
        self.assertFalse(result['training_started'])

    def test_test_set_and_unlabelled_predictions_are_not_used_as_manual_truth(self):
        with self.assertRaises(ValueError):module.bridge_audit({'split':'test','sections':[]})
        with self.assertRaises(ValueError):module.bridge_audit({'split':'val','sections':[{'prediction':'bridge'}]})
