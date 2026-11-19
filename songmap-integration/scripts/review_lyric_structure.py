#!/usr/bin/env python3
"""Audit app labels against supplied lyric references and existing timestamped ASR; never relabel maps."""
import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re

LABELS = ['intro','verse','chorus','bridge','inst','outro','silence','pre-chorus']
PARAMETERS = dict(minimum_phrase_words=5, text_similarity=.72, minimum_word_probability=.2,
                  maximum_word_gap_seconds=4., minimum_overlap_seconds=.5, substantial_coverage=.5)


def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def normalize(text): return ' '.join(re.findall(r'[^\W_]+',text.casefold(),flags=re.UNICODE))


def validate(packet):
    if packet.get('schema_version')!=1 or packet.get('label_order')!=LABELS:
        raise ValueError('Unknown input schema or label order.')
    source=packet.get('source',{});transcript=packet['asr']
    audio_id=source.get('audio_sha256');asr_id=transcript.get('audio_sha256')
    if audio_id is not None or asr_id is not None:
        if not isinstance(audio_id,str) or not re.fullmatch(r'[0-9a-f]{64}',audio_id) or audio_id!=asr_id:
            raise ValueError('Audio/transcript source identity differs.')
    if 'audio_duration_seconds' in source and 'duration_seconds' in transcript:
        a,b=source['audio_duration_seconds'],transcript['duration_seconds']
        if not finite(a) or not finite(b) or abs(a-b)>.1:
            raise ValueError('Audio/transcript durations differ.')
    sections=packet['sections']; previous=0.
    if not sections: raise ValueError('No app sections.')
    for s in sections:
        if (not all(finite(s[k]) for k in ('start','end')) or not previous-1e-6<=s['start']<s['end']
                or s['label'] not in LABELS): raise ValueError('Invalid or overlapping app sections.')
        previous=s['end']
    references=packet['reference_sections']; ids=[]
    for ref in references:
        if (not isinstance(ref['id'],str) or not ref['id'] or ref['label'] not in LABELS
                or not isinstance(ref['phrases'],list) or not ref['phrases']
                or any(not isinstance(p,str) or not normalize(p) for p in ref['phrases'])):
            raise ValueError('Invalid supplied lyric reference.')
        ids.append(ref['id'])
    if not ids or len(set(ids))!=len(ids):raise ValueError('Reference IDs must be unique.')
    words=[];previous=0.; duration=sections[-1]['end']
    for segment in packet['asr']['segments']:
        for w in segment['words']:
            start,end,p=w['start'],w['end'],w['probability']
            if (not all(finite(x) for x in (start,end,p)) or not 0<=p<=1 or not 0<=start<=end<=duration+1
                    or start<previous-1e-6 or not isinstance(w['text'],str)):
                raise ValueError('Invalid or unordered ASR word timestamps/probabilities.')
            previous=start
            text=normalize(w['text'])
            if text and end>start and p>=PARAMETERS['minimum_word_probability']:
                words.append(dict(start=start,end=end,text=text,probability=p))
    if not words: raise ValueError('No usable timestamped ASR words. Missing words do not prove instrumental audio.')
    return sections,references,words


def union_seconds(intervals):
    total=0.;end=-math.inf
    for left,right in sorted(intervals):
        if right>max(left,end):total+=right-max(left,end)
        end=max(end,right)
    return total


def match_phrases(references,words):
    roles={}
    for ref in references:
        for phrase in ref['phrases']:roles.setdefault(normalize(phrase),set()).add(ref['label'])
    matches=[];ignored=[]
    for ref in references:
        for phrase_index,phrase in enumerate(ref['phrases']):
            text=normalize(phrase);n=len(text.split())
            if n<PARAMETERS['minimum_phrase_words'] or len(roles[text])!=1:
                ignored.append(dict(reference_id=ref['id'],phrase=phrase,
                    reason='short_or_ambiguous_phrase'));continue
            candidates=[]
            for start in range(len(words)):
                for length in range(max(3,n-2),n+4):
                    window=words[start:start+length]
                    if len(window)!=length:continue
                    if any(b['start']-a['end']>PARAMETERS['maximum_word_gap_seconds'] for a,b in zip(window,window[1:])):continue
                    hypothesis=' '.join(w['text'] for w in window)
                    score=SequenceMatcher(None,text,hypothesis,autojunk=False).ratio()
                    if score<PARAMETERS['text_similarity']:continue
                    candidates.append(dict(reference_id=ref['id'],label=ref['label'],phrase_index=phrase_index,
                        reference_phrase=phrase,asr_text=hypothesis,start=window[0]['start'],end=window[-1]['end'],
                        text_similarity=score,mean_word_probability=sum(w['probability'] for w in window)/length))
            # Retain separate repetitions; suppress overlapping windows for the same reference phrase.
            accepted=[]
            for c in sorted(candidates,key=lambda x:(-x['text_similarity'],x['end']-x['start'],x['start'])):
                if any(max(c['start'],a['start'])<min(c['end'],a['end']) for a in accepted):continue
                accepted.append(c)
            matches.extend(accepted)
    return sorted(matches,key=lambda x:(x['start'],x['end'],x['reference_id'])),ignored


def audit(packet):
    sections,references,words=validate(packet);matches,ignored=match_phrases(references,words)
    rows=[]
    for i,section in enumerate(sections,1):
        length=section['end']-section['start'];overlaps=[];evidence={}
        for m in matches:
            start=max(section['start'],m['start']);end=min(section['end'],m['end'])
            if end-start<PARAMETERS['minimum_overlap_seconds']:continue
            overlaps.append(m);evidence.setdefault(m['label'],[]).append((start,end))
        coverage={label:union_seconds(intervals)/length for label,intervals in evidence.items()}
        ids=sorted({m['reference_id'] for m in overlaps})
        role=max(coverage,key=coverage.get) if coverage else None
        status='no_usable_lyric_evidence'
        if role is not None:
            if len(coverage)>1:status='mixed_lyric_roles_requires_review'
            elif role!=section['label']:
                status='label_disagreement_requires_review' if coverage[role]>=PARAMETERS['substantial_coverage'] else 'partial_lyric_conflict_requires_review'
            elif len(ids)>1:status='same_role_multiple_references_review_granularity'
            else:status='compatible_with_matched_lyrics'
        rows.append(dict(section_index=i,**section,status=status,dominant_matched_role=role,
            matched_role_coverage=coverage,reference_ids=ids,matches=overlaps))
    return dict(protocol='lyric-structure-review-v1',title=packet['title'],parameters=dict(PARAMETERS),
        source=packet.get('source',{}),asr_engine=packet['asr'].get('engine'),
        reference_source=packet.get('reference_source','Supplied reference; performance order not assumed.'),
        sections=rows,phrase_matches=matches,ignored_reference_phrases=ignored,
        disagreements=sum('conflict' in r['status'] or 'disagreement' in r['status'] or r['status']=='mixed_lyric_roles_requires_review' for r in rows),
        maps_changed=False,labels_changed=False,boundaries_changed=False,training_started=False,
        independent_accuracy_measured=False,ready_for_app_integration=False,
        limitations=['Text similarity is a matching score, not calibrated confidence or human listening.',
            'ASR contains word/timing errors and omissions. Missing matches do not establish instrumental or outro sections.',
            'Matched vocal spans may precede musical boundaries as pickups; no time is automatically used as a new boundary.',
            'Partial matches/ad-libs do not relabel an entire section. All disagreements require review.',
            'Different supplied reference IDs with the same role can indicate granularity choices rather than an Apple error.',
            'The supplied reference can differ from the performance; repetitions are searched without forcing reference order.',
            'This is an illustrative diagnostic rule with manually chosen defaults, not validated model improvement.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    raw=a.input.expanduser().read_bytes();result=audit(json.loads(raw));result['input_sha256']=hashlib.sha256(raw).hexdigest()
    out=a.output_dir.expanduser();out.mkdir(parents=True,exist_ok=False)
    (out/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['section_index','start','end','label','status','dominant_matched_role','matched_role_coverage','reference_ids']
    with (out/'section_review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in result['sections']:
            row={k:r[k] for k in fields};row['matched_role_coverage']=json.dumps(row['matched_role_coverage']);row['reference_ids']='; '.join(row['reference_ids']);w.writerow(row)
    for row in result['sections']:
        if row['status'] not in ('compatible_with_matched_lyrics','no_usable_lyric_evidence'):
            print(f'{row["start"]:.2f}--{row["end"]:.2f}s: app={row["label"]}; lyric cues={row["dominant_matched_role"]}; {row["status"]}')
    print('Saved:',out/'review.json')


if __name__=='__main__':main()
