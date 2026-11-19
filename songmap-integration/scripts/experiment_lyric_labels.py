#!/usr/bin/env python3
"""Simulate conservative bridge/chorus label changes from repeated lyric evidence; preserve every boundary."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from review_lyric_structure import audit, normalize, union_seconds, LABELS

PARAMETERS=dict(target_coverage=.75, anchor_coverage=.30, anchor_probability=.60,
    strong_text_similarity=.85, strong_word_probability=.50, minimum_unique_repeated_phrases=2,
    maximum_original_probability=.95, allowed_labels=['bridge','chorus'])


def validate_model(packet,evidence):
    source=packet.get('source',{})
    if evidence.get('schema_version')!=1 or evidence.get('label_order')!=LABELS:
        raise ValueError('Unknown original model-evidence schema.')
    for key in ('audio_sha256','analysis_report_sha256'):
        value=evidence.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) or value!=source.get(key):
            raise ValueError('Model/lyric source identity differs: '+key)
    sections=evidence['sections']
    if len(sections)!=len(packet['sections']):raise ValueError('Original section counts differ.')
    for original,row in zip(packet['sections'],sections):
        if any(original[k]!=row[k] for k in ('start','end','label')):
            raise ValueError('Original model labels or boundaries differ from lyric input.')
        p=row['probabilities']
        if len(p)!=8 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in p) or abs(sum(p)-1)>1e-4:
            raise ValueError('Invalid original eight-class probabilities.')
        if LABELS[max(range(8),key=lambda i:p[i])]!=row['label']:
            raise ValueError('Original model label does not match its eight-class argmax.')
    return sections


def strong_matches(row,role):
    return [h for h in row['matches'] if h['label']==role and
        h['text_similarity']>=PARAMETERS['strong_text_similarity'] and
        h['mean_word_probability']>=PARAMETERS['strong_word_probability']]


def coverage(row,matches):
    spans=[(max(row['start'],h['start']),min(row['end'],h['end'])) for h in matches
        if min(row['end'],h['end'])>max(row['start'],h['start'])]
    return union_seconds(spans)/(row['end']-row['start'])


def phrase_key(hit):return (hit['reference_id'],normalize(hit['reference_phrase']))


def simulate(review,model_sections):
    rows=review['sections'];anchors=[]
    # Freeze anchor eligibility before proposing anything; proposed labels never become anchors.
    for row,model in zip(rows,model_sections):
        role=row['label'];hits=strong_matches(row,role)
        probability=model['probabilities'][LABELS.index(role)]
        if (role in PARAMETERS['allowed_labels'] and row['status']=='compatible_with_matched_lyrics'
                and probability>=PARAMETERS['anchor_probability'] and coverage(row,hits)>=PARAMETERS['anchor_coverage']):
            anchors.append(dict(section_index=row['section_index'],label=role,model_probability=probability,matches=hits))
    decisions=[];simulated=[]
    for row,model in zip(rows,model_sections):
        label=row['label'];role=row['dominant_matched_role'];hits=strong_matches(row,role) if role else []
        supported_coverage=coverage(row,hits)
        probability=model['probabilities'][LABELS.index(label)]
        result=dict(section_index=row['section_index'],start=row['start'],end=row['end'],original_label=label,
            candidate_label=label,matched_role=role,strong_match_coverage=supported_coverage,
            original_model_probability=probability,status='keep_original',support=[],unique_repeated_phrases=0)
        if label==role:
            result['status']='keep_already_consistent'
        elif label not in PARAMETERS['allowed_labels'] or role not in PARAMETERS['allowed_labels']:
            result['status']='keep_outside_bridge_chorus_pair'
        elif row['status']!='label_disagreement_requires_review':
            result['status']='keep_mixed_or_partial_evidence'
        elif supported_coverage<PARAMETERS['target_coverage']:
            result['status']='keep_insufficient_strong_coverage'
        elif probability>PARAMETERS['maximum_original_probability']:
            result['status']='keep_high_original_probability_for_review'
        else:
            for hit in hits:
                for anchor in anchors:
                    if anchor['label']!=role or anchor['section_index']==row['section_index']:continue
                    for source in anchor['matches']:
                        # A single phrase crossing two sections is not a second occurrence.
                        if phrase_key(hit)!=phrase_key(source) or not (source['end']<=hit['start'] or hit['end']<=source['start']):continue
                        support=dict(anchor_section=anchor['section_index'],anchor_model_probability=anchor['model_probability'],
                            reference_id=hit['reference_id'],reference_phrase=hit['reference_phrase'],
                            anchor_occurrence=[source['start'],source['end']],target_occurrence=[hit['start'],hit['end']])
                        if support not in result['support']:result['support'].append(support)
            keys={(s['reference_id'],normalize(s['reference_phrase'])) for s in result['support']}
            result['unique_repeated_phrases']=len(keys)
            if len(keys)>=PARAMETERS['minimum_unique_repeated_phrases']:
                result['candidate_label']=role;result['status']='candidate_label_change'
            else:result['status']='keep_insufficient_independent_repetitions'
        decisions.append(result)
        simulated.append(dict(start=row['start'],end=row['end'],label=result['candidate_label']))
    return dict(decisions=decisions,simulated_sections=simulated,
        candidate_change_count=sum(d['candidate_label']!=d['original_label'] for d in decisions),
        original_anchor_sections=[dict(section_index=a['section_index'],label=a['label'],model_probability=a['model_probability']) for a in anchors])


def experiment(packet,evidence):
    review=audit(packet);models=validate_model(packet,evidence);result=simulate(review,models)
    for original,candidate in zip(packet['sections'],result['simulated_sections']):
        if any(original[k]!=candidate[k] for k in ('start','end')):raise RuntimeError('Boundary invariant violated.')
    result.update(protocol='lyric-label-experiment-v1',title=packet['title'],parameters=copy.deepcopy(PARAMETERS),
        audit_parameters=review['parameters'],source=packet['source'],model_checkpoint_sha256=evidence.get('context_checkpoint_sha256'),
        reference_source=review['reference_source'],boundaries_exactly_preserved=True,
        app_maps_changed=False,model_weights_changed=False,training_started=False,
        independent_accuracy_measured=False,ready_for_app_integration=False,
        limitations=['The supplied lyric reference provides roles; this is not a new audio-only SongFormer result.',
            'ASR is automatic and may be wrong. Text/coverage/repetition thresholds are uncalibrated experimental defaults.',
            'Only original model predictions can be anchors; candidates do not propagate or change probabilities.',
            'Several partial/mixed sections deliberately remain unchanged, including instrumental overlaps and uncertain tails.',
            'One illustrative recording informed development; there is no held-out accuracy or iPhone runtime claim.',
            'Candidate changes are a separate local simulation, not applied app labels or new training annotations.'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--model-evidence',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    raw=a.input.expanduser().read_bytes();model_raw=a.model_evidence.expanduser().read_bytes()
    result=experiment(json.loads(raw),json.loads(model_raw))
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(),model_evidence_sha256=hashlib.sha256(model_raw).hexdigest())
    out=a.output_dir.expanduser();out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['section_index','start','end','original_label','candidate_label','status','strong_match_coverage',
            'original_model_probability','unique_repeated_phrases']
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in result['decisions']:w.writerow({k:row[k] for k in fields})
    for row in result['decisions']:
        if row['status']=='candidate_label_change':
            print(f'{row["start"]:.2f}--{row["end"]:.2f}s: {row["original_label"]} -> {row["candidate_label"]}; '
                f'{row["unique_repeated_phrases"]} distinct repeated phrases')
    print('Candidate changes:',result['candidate_change_count'],'| boundaries preserved:',result['boundaries_exactly_preserved'])
    print('Saved:',out/'comparison.json')


if __name__=='__main__':main()
