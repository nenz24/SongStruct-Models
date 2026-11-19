import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'review_lyric_structure.py'
spec=importlib.util.spec_from_file_location('lyric_review',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def words(text,start):
    return [dict(start=start+i,end=start+i+.8,text=w,probability=.95) for i,w in enumerate(text.split())]


def packet():
    phrase='Like a rushing wind Jesus breathe within'
    return dict(schema_version=1,title='synthetic',label_order=list(m.LABELS),
        sections=[dict(start=0.,end=12.,label='bridge'),dict(start=12.,end=24.,label='chorus')],
        reference_sections=[dict(id='bridge',label='bridge',phrases=[phrase])],
        asr=dict(engine='synthetic',segments=[dict(words=words(phrase,2)+words(phrase,14))]))


class LyricStructureTests(unittest.TestCase):
    def test_repeated_bridge_can_disagree_with_chorus_without_editing_input(self):
        p=packet();before=copy.deepcopy(p);r=m.audit(p)
        self.assertEqual(len(r['phrase_matches']),2)
        self.assertEqual(r['sections'][0]['status'],'compatible_with_matched_lyrics')
        self.assertEqual(r['sections'][1]['status'],'label_disagreement_requires_review')
        self.assertEqual(r['sections'][1]['dominant_matched_role'],'bridge')
        self.assertEqual(p,before)
        self.assertFalse(r['labels_changed']);self.assertFalse(r['boundaries_changed'])

    def test_partial_vocal_phrase_does_not_relabel_entire_instrumental(self):
        p=packet();p['sections']=[dict(start=0.,end=24.,label='inst')]
        p['asr']['segments']=[dict(words=words('I want to know You more',7))]
        p['reference_sections']=[dict(id='chorus',label='chorus',phrases=['I want to know You more'])]
        r=m.audit(p)['sections'][0]
        self.assertEqual(r['status'],'partial_lyric_conflict_requires_review')
        self.assertEqual(r['label'],'inst')

    def test_shared_or_short_phrases_are_not_unique_section_evidence(self):
        p=packet();phrase=p['reference_sections'][0]['phrases'][0]
        p['reference_sections'].append(dict(id='chorus',label='chorus',phrases=[phrase,'I surrender']))
        r=m.audit(p)
        self.assertEqual(r['phrase_matches'],[])
        self.assertTrue(all(s['status']=='no_usable_lyric_evidence' for s in r['sections']))

    def test_missing_transcript_match_does_not_confirm_instrumental(self):
        p=packet();p['asr']['segments']=[dict(words=words('Unrelated words without a lyric match',2))]
        r=m.audit(p)
        self.assertEqual(r['disagreements'],0)
        self.assertTrue(all(s['status']=='no_usable_lyric_evidence' for s in r['sections']))

    def test_same_role_multiple_reference_sections_is_a_granularity_review(self):
        p=packet();p['sections']=[dict(start=0.,end=24.,label='verse')]
        a='first verse phrase sung here';b='second verse phrase sung later'
        p['reference_sections']=[dict(id='v1',label='verse',phrases=[a]),dict(id='v2',label='verse',phrases=[b])]
        p['asr']['segments']=[dict(words=words(a,1)+words(b,13))]
        r=m.audit(p)['sections'][0]
        self.assertEqual(r['status'],'same_role_multiple_references_review_granularity')

    def test_long_silence_cannot_be_bridged_to_force_a_phrase_match(self):
        p=packet();w=words(p['reference_sections'][0]['phrases'][0],1)
        for item in w[4:]:item['start']+=8;item['end']+=8
        p['asr']['segments']=[dict(words=w)]
        # Check that no accepted phrase spans the artificial eight-second gap.
        for hit in m.audit(p)['phrase_matches']:
            self.assertFalse(hit['start']<5 and hit['end']>13)

    def test_invalid_or_unusable_word_data_fails(self):
        for mode in ('nan','reversed','overlap_sections','no_words'):
            p=packet()
            if mode=='nan':p['asr']['segments'][0]['words'][0]['probability']=float('nan')
            elif mode=='reversed':p['asr']['segments'][0]['words'][1]['start']=0
            elif mode=='overlap_sections':p['sections'][1]['start']=11
            else:p['asr']['segments']=[]
            with self.assertRaises(ValueError):m.audit(p)

    def test_changed_recording_or_duration_is_rejected(self):
        p=packet();p['source']={'audio_sha256':'0'*64};p['asr']['audio_sha256']='1'*64
        with self.assertRaises(ValueError):m.audit(p)
        p['asr']['audio_sha256']='0'*64;p['source']['audio_duration_seconds']=24
        p['asr']['duration_seconds']=60
        with self.assertRaises(ValueError):m.audit(p)

    def test_cli_preserves_input_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input.json';source.write_text(json.dumps(packet()))
            original=source.read_bytes();out=root/'review'
            cmd=[sys.executable,str(SCRIPT),'--input',str(source),'--output-dir',str(out)]
            subprocess.run(cmd,check=True,capture_output=True)
            r=json.loads((out/'review.json').read_text())
            self.assertTrue((out/'section_review.csv').is_file());self.assertFalse(r['ready_for_app_integration'])
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual(source.read_bytes(),original)


if __name__=='__main__':unittest.main()
