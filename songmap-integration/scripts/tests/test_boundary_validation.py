import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('validation',Path(__file__).resolve().parents[1]/'evaluate_boundary_proposals.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BoundaryValidationTests(unittest.TestCase):
    def report(self):
        return {'title':'fixture','structureEvidence':{'labelOrder':['chorus','outro'],
            'frames':[{'time':i,'duration':1,'probabilities':[.9,.1] if i<12 else [.1,.9]} for i in range(20)],
            'appleBoundaries':[{'start':0,'end':20}],
            'assembledSections':[{'start':0,'end':20,'label':'chorus'}]}}

    def test_simulation_caps_one_split_and_preserves_original_map(self):
        r=self.report();original=r['structureEvidence']['assembledSections'][0]
        proposals=[{'parent_section':dict(original),'suggested_split_seconds':12},
            {'parent_section':dict(original),'suggested_split_seconds':14}]
        result=module.refinement(r,proposals)
        self.assertEqual([(s['start'],s['end'],s['label']) for s in result],[(0,12,'chorus'),(12,20,'outro')])
        self.assertEqual(original,{'start':0,'end':20,'label':'chorus'})

    def test_duration_metric_detects_change_hidden_by_majority_section_score(self):
        truth=[{'start':0,'end':20,'label':'chorus'}]
        before=module.score(truth,[{'start':0,'end':20,'label':'chorus'}],['chorus','outro'])
        after=module.score(truth,[{'start':0,'end':12,'label':'chorus'},{'start':12,'end':20,'label':'outro'}],['chorus','outro'])
        self.assertEqual(before['manual_section_duration_vote_accuracy'],after['manual_section_duration_vote_accuracy'])
        self.assertAlmostEqual(after['eligible_duration_label_agreement'],.6)

    def test_held_time_is_not_scored_and_unknown_predictions_are_not_truth(self):
        truth=[{'start':0,'end':5,'label':'chorus'}]
        sections=[{'start':0,'end':5,'label':'chorus'},{'start':5,'end':20,'label':'outro'}]
        self.assertEqual(module.score(truth,sections,['chorus','outro'])['eligible_duration_label_agreement'],1)
        with self.assertRaises(ValueError):module.score(truth,[{'start':0,'end':5,'label':None}],['chorus','outro'])

    def test_overall_weights_tracks_by_eligible_duration(self):
        a=module.score([{'start':0,'end':10,'label':'chorus'}],[{'start':0,'end':10,'label':'chorus'}],['chorus','outro'])
        b=module.score([{'start':0,'end':30,'label':'chorus'}],[{'start':0,'end':30,'label':'outro'}],['chorus','outro'])
        a['eligible_duration_seconds']=10;b['eligible_duration_seconds']=30
        result=module.overall([a,b],['chorus','outro'])
        self.assertEqual(result['manual_section_duration_vote_accuracy'],.5)
        self.assertEqual(result['eligible_duration_label_agreement'],.25)
