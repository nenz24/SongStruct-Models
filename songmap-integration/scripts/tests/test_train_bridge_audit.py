import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'audit_train_bridge.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    records=[{'track_id':'song','group_id':'group','split':'train','approved':True,'duration':120,
        'sections':[{'start':20,'end':40,'label':'bridge'},{'start':60,'end':80,'label':'chorus'}]}]
    details=[]
    for start,end,truth,pred,confidence in [(20,40,'bridge','chorus',.9),(60,80,'chorus','bridge',.7)]:
        probs={name:0. for name in module.LABELS};probs[pred]=confidence;probs[truth]=1-confidence
        details.append({'track_id':'song','start':start,'end':end,'label_manual':truth,
            'prediction':pred,'probabilities':probs,'coverage':1})
    return records,details


class TrainBridgeAuditTests(unittest.TestCase):
    def test_both_directions_are_kept_and_manual_gaps_are_visible(self):
        records,details=fixture();before=copy.deepcopy(records)
        result=module.summarize(records,details)
        self.assertEqual(result['bridge_to_chorus_count'],1)
        self.assertEqual(result['chorus_to_bridge_count'],1)
        self.assertEqual(result['high_confidence_wrong_cases_ge_0_8'],1)
        self.assertEqual(result['cases'][0]['gap_to_next_eligible_seconds'],20)
        self.assertEqual(records,before)
        self.assertFalse(result['training_started']);self.assertFalse(result['val_test_used'])

    def test_holdout_records_are_rejected(self):
        for split in ('val','test'):
            records,details=fixture();records[0]['split']=split
            with self.assertRaises(ValueError):module.summarize(records,details)

    def test_missing_duplicate_or_changed_manual_target_is_rejected(self):
        records,details=fixture()
        with self.assertRaises(ValueError):module.summarize(records,details[:1])
        with self.assertRaises(ValueError):module.summarize(records,details+[details[0]])
        changed=copy.deepcopy(details);changed[0]['label_manual']='chorus'
        with self.assertRaises(ValueError):module.summarize(records,changed)

    def test_nonfinite_or_unnormalized_probabilities_are_rejected(self):
        for value in (float('nan'),-.1,1.1):
            records,details=fixture();details[0]['probabilities']['chorus']=value
            with self.assertRaises(ValueError):module.summarize(records,details)
