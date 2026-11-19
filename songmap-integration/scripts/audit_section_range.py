#!/usr/bin/env python3
"""Inspect a manually identified time range using saved Apple boundaries and frame probabilities."""
import argparse
import json
import math
from pathlib import Path


def audit(report,start,end):
    if not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end:raise ValueError('Invalid range.')
    evidence=report.get('structureEvidence')
    if not evidence:raise ValueError('Report has no structure evidence; rerun analysis with the updated app.')
    labels=evidence['labelOrder'];sums=[0.]*len(labels);coverage=0.
    for frame in evidence['frames']:
        probabilities=frame['probabilities']
        if len(probabilities)!=len(labels) or any(not math.isfinite(p) or p<0 for p in probabilities):raise ValueError('Invalid frame probabilities.')
        weight=max(0,min(frame['time']+frame['duration'],end)-max(frame['time'],start))
        if weight:
            sums=[a+p*weight for a,p in zip(sums,probabilities)];coverage+=weight
    probabilities={label:value/coverage for label,value in zip(labels,sums)} if coverage else {}
    result={'title':report['title'],'range_seconds':[start,end],'coverage_fraction':coverage/(end-start),
        'pooled_prediction':max(probabilities,key=probabilities.get) if probabilities else None,
        'probabilities':probabilities,
        'overlapping_apple_boundaries':[r for r in evidence['appleBoundaries'] if r['start']<end and r['end']>start],
        'overlapping_assembled_sections':[s for s in evidence['assembledSections'] if s['start']<end and s['end']>start],
        'note':'The requested interval is a diagnostic reference, not a changed training target or automatic label correction.'}
    if report.get('contextEvidence'):
        result['pooled_prediction_scope']='Base SongFormer frame probabilities before section-context correction.'
        result['overlapping_context_sections']=[s for s in report['contextEvidence']['sections'] if s['start']<end and s['end']>start]
    if report.get('boundaryRefinementEvidence'):
        result['overlapping_boundary_shifts']=[r for r in report['boundaryRefinementEvidence']['decisions']
            if r['applied'] and min(r['originalSeconds'],r['candidateSeconds'])<end
            and max(r['originalSeconds'],r['candidateSeconds'])>start]
        result['context_range_scope']='Original context input ranges before optional Debug boundary refinement; assembled sections are final.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path)
    p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True)
    a=p.parse_args();print(json.dumps(audit(json.loads(a.report.read_text()),a.start,a.end),indent=2))
