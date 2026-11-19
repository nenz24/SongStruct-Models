import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import experiment_lyric_labels as m

PHRASES={'x':'Like a rushing wind Jesus breathe within',
         'y':'Like a mighty storm Stir within my soul','z':'Lord have Your way in me'}


def row(index,start,end,label,role,specs,status=None):
    return dict(section_index=index,start=start,end=end,label=label,dominant_matched_role=role,
        status=status or ('compatible_with_matched_lyrics' if label==role else 'label_disagreement_requires_review'),
        matches=[dict(label=role,reference_id=role,reference_phrase=PHRASES[key],start=a,end=b,
                      text_similarity=.95,mean_word_probability=.9) for key,a,b in specs])


def model(label,p=.8):
    values=[(1-p)/7]*8;values[m.LABELS.index(label)]=p
    return dict(probabilities=values)


def simple():
    rows=[row(1,0,12,'bridge','bridge',[('x',0,5),('y',6,11)]),
          row(2,12,24,'chorus','bridge',[('x',12,17),('y',18,23)])]
    return dict(sections=rows),[model('bridge'),model('chorus')]


class LyricLabelExperimentTests(unittest.TestCase):
    def test_two_independent_phrases_can_correct_only_label_in_either_direction(self):
        for reverse in (False,True):
            review,models=simple()
            if reverse:
                for r in review['sections']:
                    r['label']='chorus' if r['label']=='bridge' else 'bridge';r['dominant_matched_role']='chorus'
                    for h in r['matches']:h['label']='chorus';h['reference_id']='chorus'
                models=[model('chorus'),model('bridge')]
            original=copy.deepcopy(review);r=m.simulate(review,models)
            self.assertEqual(r['candidate_change_count'],1)
            self.assertEqual(r['simulated_sections'][1]['label'],'chorus' if reverse else 'bridge')
            self.assertEqual([(s['start'],s['end']) for s in r['simulated_sections']],[(0,12),(12,24)])
            self.assertEqual(review,original)

    def test_partial_mixed_and_instrumental_sections_stay_unchanged(self):
        for mode in ('partial','mixed','instrumental'):
            review,models=simple();target=review['sections'][1]
            if mode=='partial':target['end']=40;target['status']='partial_lyric_conflict_requires_review'
            elif mode=='mixed':target['status']='mixed_lyric_roles_requires_review'
            else:target['label']='inst';models[1]=model('inst')
            self.assertEqual(m.simulate(review,models)['candidate_change_count'],0)

    def test_one_shared_occurrence_across_boundary_is_not_repetition(self):
        review,models=simple()
        for r in review['sections']:
            for h in r['matches']:h['start']=0;h['end']=24
        result=m.simulate(review,models)
        self.assertEqual(result['candidate_change_count'],0)
        self.assertEqual(result['decisions'][1]['unique_repeated_phrases'],0)

    def test_repeating_one_phrase_does_not_create_two_distinct_phrases(self):
        review,models=simple()
        for r in review['sections']:
            for h in r['matches']:h['reference_phrase']=PHRASES['x']
        result=m.simulate(review,models)
        self.assertEqual(result['candidate_change_count'],0)
        self.assertEqual(result['decisions'][1]['unique_repeated_phrases'],1)

    def test_no_anchor_low_anchor_and_high_target_probability_abstain(self):
        for mode in ('no_anchor','low_anchor','high_target'):
            review,models=simple()
            if mode=='no_anchor':
                review['sections'][0]['label']='chorus';review['sections'][0]['status']='label_disagreement_requires_review';models[0]=model('chorus')
            elif mode=='low_anchor':models[0]=model('bridge',.4)
            else:models[1]=model('chorus',.98)
            self.assertEqual(m.simulate(review,models)['candidate_change_count'],0)

    def test_candidates_cannot_become_anchors_for_a_cascade(self):
        review,models=simple()
        review['sections'][1]['matches'].append(dict(label='bridge',reference_id='bridge',reference_phrase=PHRASES['z'],
            start=20,end=24,text_similarity=.95,mean_word_probability=.9))
        review['sections'].append(row(3,24,36,'chorus','bridge',[('y',24,29),('z',30,35)]));models.append(model('chorus'))
        result=m.simulate(review,models)
        self.assertEqual(result['decisions'][1]['candidate_label'],'bridge')
        self.assertEqual(result['decisions'][2]['candidate_label'],'chorus')
        self.assertEqual([a['section_index'] for a in result['original_anchor_sections']],[1])

    def test_model_identity_probabilities_and_ranges_must_match(self):
        folder=ROOT/'experiments/structure/lyric_review_examples'
        packet=json.loads((folder/'i-surrender.json').read_text());evidence=json.loads((folder/'i-surrender-model.json').read_text())
        for mode in ('source','time','nan','argmax'):
            changed=copy.deepcopy(evidence)
            if mode=='source':changed['audio_sha256']='0'*64
            elif mode=='time':changed['sections'][0]['end']+=1
            elif mode=='nan':changed['sections'][0]['probabilities'][0]=float('nan')
            else:changed['sections'][0]['probabilities']=[.01]*8;changed['sections'][0]['probabilities'][2]=.93
            with self.assertRaises(ValueError):m.validate_model(packet,changed)

    def test_real_example_preserves_every_boundary_and_keeps_uncertain_cases(self):
        folder=ROOT/'experiments/structure/lyric_review_examples'
        packet=json.loads((folder/'i-surrender.json').read_text());evidence=json.loads((folder/'i-surrender-model.json').read_text())
        r=m.experiment(packet,evidence)
        self.assertEqual(r['candidate_change_count'],1)
        self.assertEqual(r['simulated_sections'][13]['label'],'bridge')
        self.assertEqual(r['simulated_sections'][10]['label'],'chorus')
        self.assertEqual(r['simulated_sections'][11]['label'],'chorus')
        self.assertEqual(r['simulated_sections'][14]['label'],'chorus')
        self.assertTrue(r['boundaries_exactly_preserved']);self.assertFalse(r['app_maps_changed'])
        self.assertFalse(r['ready_for_app_integration'])

    def test_cli_creates_separate_comparison_and_refuses_overwrite(self):
        folder=ROOT/'experiments/structure/lyric_review_examples'
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'comparison'
            cmd=[sys.executable,str(ROOT/'scripts/experiment_lyric_labels.py'),'--input',str(folder/'i-surrender.json'),
                 '--model-evidence',str(folder/'i-surrender-model.json'),'--output-dir',str(out)]
            subprocess.run(cmd,check=True,capture_output=True)
            self.assertTrue((out/'comparison.csv').is_file())
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
