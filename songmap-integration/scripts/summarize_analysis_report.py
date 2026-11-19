#!/usr/bin/env python3
"""Summarize exported app timing reports, keeping nested metrics separate from parent stages."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def summarize(report):
    if report['schemaVersion']!=1:raise ValueError('Unknown report schema.')
    groups={'stage':defaultdict(float),'metric':defaultdict(float)}
    counts=defaultdict(int)
    events=[]
    for event in report['events']:
        kind=event['kind'];seconds=event.get('seconds')
        if kind in groups:
            if seconds is None or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid duration.')
            groups[kind][event['name']]+=seconds
            counts[event['name']]+=1
        elif kind=='event':events.append({'name':event['name'],'detail':event.get('detail')})
    result={'title':report['title'],'state':report['state'],'total_seconds':report['totalSeconds'],
        'active_stage':report.get('activeStage'),'error':report.get('error'),
        'stages_seconds':dict(groups['stage']),'nested_metrics_seconds':dict(groups['metric']),
        'counts':dict(counts),'events':events,
        'note':'Nested metrics overlap their parent stage and must not be added to stage totals. Optional-stage failures may still yield a completed song analysis.'}
    context=report.get('contextEvidence')
    if context:
        result['section_context']={'epoch':context['selectedEpoch'],'precision':context['precision'],
            'compute_units':context['computeUnits'],'sections':len(context['sections']),
            'changed_sections':[s for s in context['sections'] if s['baselineLabel']!=s['label']],
            'low_audio_coverage_sections':context['lowAudioCoverageSections'],
            'checkpoint_sha256':context['checkpointSHA256']}
    refinement=report.get('boundaryRefinementEvidence')
    if refinement:
        result['boundary_refinement']={'protocol':refinement['protocolVersion'],
            'applied_count':refinement['appliedCount'],'bar_seconds':refinement['barSeconds'],
            'applied_decisions':[r for r in refinement['decisions'] if r['applied']],
            'scope':'Experimental Debug post-context boundary shifts. Context evidence retains original input ranges.'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path)
    a=p.parse_args();print(json.dumps(summarize(json.loads(a.report.read_text())),indent=2))
