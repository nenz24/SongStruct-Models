"""Evaluate a compressed head on existing MusicFM candidate features; no encoder execution."""
import argparse
from pathlib import Path
import platform
import torch
from deployment.evaluate_pipeline import cached_features
from deployment.verify_coreml import CoreMLAdapter
from finetuning.common import config, fingerprint, read, sha_file, write
from finetuning.engine import evaluate, manifest


def check_same_inventory(left,right):
    for key in ('plan_id','provenance_id','annotations_id','checkpoint_sha256','tracks'):
        if left[key]!=right[key]:
            raise ValueError(f'Validation inventories differ: {key}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-pipeline-report',type=Path,required=True)
    parser.add_argument('--head-validation-report',type=Path,required=True)
    parser.add_argument('--head-compression-report',type=Path,required=True)
    parser.add_argument('--config')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Run Core ML prediction on macOS.')
    import coremltools as ct
    pipeline_path,head_path,compression_path,output=[p.expanduser().resolve() for p in (
        args.encoder_pipeline_report,args.head_validation_report,args.head_compression_report,args.output)]
    if output.exists():
        raise ValueError('Choose a new output path.')
    pipeline,head,compression=read(pipeline_path),read(head_path),read(compression_path)
    if not pipeline['full_validation'] or not head['full_validation']:
        raise ValueError('Both reports must cover full validation.')
    check_same_inventory(pipeline,head)
    scheme=compression['scheme']
    if scheme not in ('palette6','int8') or head.get('compression_scheme')!=scheme:
        raise ValueError('Expected an independently validated compressed head.')
    package=Path(head['package']).expanduser().resolve()
    if package!=Path(compression['package']).expanduser().resolve() or compression['checkpoint_sha256']!=head['checkpoint_sha256']:
        raise ValueError('Head compression and validation package identities differ.')
    cfg=config(args.config)
    data,provenance=manifest(cfg)
    rows=[r for r in data['records'] if r['approved'] and r['split']=='val']
    annotations=fingerprint([(r['track_id'],r['sections']) for r in rows])
    if data['plan_id']!=pipeline['plan_id'] or provenance!=pipeline['provenance_id'] or annotations!=pipeline['annotations_id']:
        raise ValueError('Local split/cache/annotations changed since candidate validation.')
    cache=Path(pipeline['diagnostic_cache']).expanduser().resolve()
    identity=pipeline['candidate_encoder_id']
    meta=read(cache/'encoder_provenance.json')
    if meta['id']!=identity or fingerprint({k:v for k,v in meta.items() if k!='id'})!=identity:
        raise ValueError('Candidate encoder cache identity mismatch.')
    # Read only existing features. No encoder model or audio file is opened.
    torch.set_num_threads(2)
    model=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    if model.user_defined_metadata.get('checkpoint_sha256')!=pipeline['checkpoint_sha256'] or model.user_defined_metadata.get('compression_scheme')!=scheme:
        raise ValueError('Head package metadata does not match reports.')
    details=[]
    result=evaluate(CoreMLAdapter(model),rows,cfg,torch.device('cpu'),identity,details=details,
        feature_reader=lambda _cfg,row:cached_features(cache,row,identity))
    reference=pipeline['coreml_encoders_with_fp32_head']
    write(output,{'tracks':len(rows),'full_validation':True,'plan_id':data['plan_id'],
        'provenance_id':provenance,'annotations_id':annotations,'checkpoint_sha256':pipeline['checkpoint_sha256'],
        'candidate_encoder_id':identity,'encoder_pipeline_report':str(pipeline_path),
        'encoder_pipeline_report_sha256':sha_file(pipeline_path),'head_package':str(package),
        'head_scheme':scheme,'musicfm_override':pipeline['musicfm_override'],
        'head_compression_report_sha256':sha_file(compression_path),'metrics':result,
        'fp32_head_same_encoder_metrics':reference,
        'macro_f1_delta_from_fp32_head_same_encoder':result['macro_f1_present']-reference['macro_f1_present'],
        'sections':details,'encoders_executed':False,'ready_for_app_integration':False,
        'scope':'Existing candidate features -> compressed head -> fixed Apple section pooling. CPU Core ML on Mac only.'})
    print(f'MusicFM={pipeline["musicfm_override"].get("compression_scheme","fp32")}; head={scheme}; F1={result["macro_f1_present"]:.6f}; accuracy={result["section_accuracy"]:.4%}')
    print('Report:',output)


if __name__=='__main__':
    main()
