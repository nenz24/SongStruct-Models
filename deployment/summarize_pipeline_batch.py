"""Summarize repeated iPhone pipeline diagnostics without equating successful runs with production approval."""
import argparse
from pathlib import Path
from finetuning.common import read, write


def summarize(batch):
    runs=batch['runs'];requested=batch['requested_runs']
    if not isinstance(requested,int) or requested<1:raise ValueError('Invalid requested run count.')
    indices=[r['run_index'] for r in runs]
    if indices!=list(range(1,len(runs)+1)) or len(runs)>requested:raise ValueError('Run sequence invalid.')
    rows=[];identities=set()
    for run in runs:
        row={'run_index':run['run_index'],'state':run['state']}
        for key in ('footprint_before_run_bytes','footprint_after_run_bytes','footprint_after_idle_bytes'):
            if key in run:row[key]=run[key]
        if run['state']=='completed':
            r=run['report']
            identities.add((r['track_id'],r['source_validation_sha256'],r['requested_compute_units']))
            row.update(pipeline_seconds=r['pcm_pipeline_seconds'],label_agreement=r['section_agreement_with_mac'],
                max_probability_error=r['max_section_probability_error'],finite=r['all_predictions_finite'],
                low_coverage_sections=r['low_coverage_sections'],
                largest_footprint_sample_bytes=max(r['process_footprint_snapshots_bytes'],default=0))
        elif 'error' in run:row['error']=run['error']
        rows.append(row)
    if len(identities)>1:raise ValueError('Batch mixes tracks, model references or routing modes.')
    completed=sum(r['state']=='completed' for r in runs)
    if batch['completed_runs']!=completed:raise ValueError('Completed-run count inconsistent.')
    return {'batch_state':batch['state'],'requested_runs':requested,'completed_runs':completed,
        'all_requested_runs_completed':batch['state']=='completed' and completed==requested,
        'runs':rows,'ready_for_app_integration':False,
        'note':'Footprint samples are not peak memory. Finite successful runs and matching labels do not prove numeric repeatability or resolve earlier crashes.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(read(a.batch));write(a.output,result)
    for r in result['runs']:print(r)
    print('All requested runs completed:',result['all_requested_runs_completed'])


if __name__=='__main__':main()
