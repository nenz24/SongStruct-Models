"""Evaluate original-snapshot MuQ FP32 while reusing validated MusicFM palette6 streams."""
import argparse
from itertools import zip_longest
from pathlib import Path
import platform
import tempfile
import shutil
import numpy as np
from deployment.coreml_features import CoreMLEncoders, EncoderWindowError
from deployment.musicfm_candidates import musicfm_chunks
from deployment.evaluate_pipeline import cached_features
from deployment.verify_coreml import CoreMLAdapter
from deployment.verify_encoders import rotary
from finetuning.common import config, fingerprint, read, sha_file, write
from finetuning.engine import manifest, evaluate


def replace_muq(produced, baseline):
    for new, old in zip_longest(produced, baseline):
        if new is None or old is None:
            raise ValueError('MuQ and baseline chunk inventories differ.')
        offset, short, long = new
        reference, old_offset, _ = old
        count = len(reference)
        if offset != old_offset or reference.ndim != 2 or reference.shape[1] != 4096:
            raise ValueError('MusicFM baseline chunk offset/shape mismatch.')
        if any(a.ndim != 2 or a.shape[1] != 1024 or not count <= len(a) <= count+4 for a in (short,long)):
            raise ValueError('MuQ frames cannot align to cached MusicFM.')
        values = np.array(reference, dtype=np.float32, copy=True)
        values[:,1024:2048] = short[:count]
        values[:,3072:4096] = long[:count]
        if not np.isfinite(values).all():
            raise ValueError('Non-finite MuQ/MusicFM features.')
        yield offset, values


def validate_source(proof, original, baseline):
    if (proof.get('source_kind') != 'original_training_snapshot' or
            proof.get('compute_precision') != 'float32' or not proof.get('source_window_passed')):
        raise ValueError('Verified original-snapshot MuQ FP32 export required.')
    for key in ('muq_repo','muq_revision'):
        if proof[key] != original[key]:
            raise ValueError('MuQ snapshot differs from training: '+key)
    if not proof.get('muq_snapshot_digest'):
        raise ValueError('MuQ snapshot digest missing.')
    if proof['app_manifest_sha256'] != baseline['app_manifest_sha256']:
        raise ValueError('App manifest changed since MuQ source export.')
    for role in ('log_mel','muq_conv'):
        if proof['frontend_packages'][role] != baseline['packages'][role]:
            raise ValueError('MuQ frontend changed: '+role)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songmap-root',type=Path,required=True)
    p.add_argument('--baseline-combination',type=Path,required=True)
    p.add_argument('--muq-export-report',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1,help='0=all validation tracks')
    p.add_argument('--config')
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Core ML runtime requires macOS.')
    if a.limit<0:raise ValueError('limit must be >=0')
    import coremltools as ct
    import librosa
    import torch
    torch.set_num_threads(2)
    cfg=config(a.config)
    data,original_id=manifest(cfg)
    all_rows=[r for r in data['records'] if r['approved'] and r['split']=='val']
    rows=all_rows[:a.limit] if a.limit else all_rows
    if not rows:raise ValueError('No validation tracks.')
    baseline_path=a.baseline_combination.expanduser().resolve()
    base=read(baseline_path)
    if (not base['full_validation'] or base['head_scheme']!='palette6' or
            base['musicfm_override'].get('compression_scheme')!='palette6'):
        raise ValueError('Use full-validation MusicFM palette6 + head palette6 combination report.')
    annotations=fingerprint([(r['track_id'],r['sections']) for r in all_rows])
    if (base['plan_id']!=data['plan_id'] or base['provenance_id']!=original_id or
            base['annotations_id']!=annotations or base['tracks']!=len(all_rows)):
        raise ValueError('Validation inventory/labels changed.')
    pipeline_path=Path(base['encoder_pipeline_report'])
    if sha_file(pipeline_path)!=base['encoder_pipeline_report_sha256']:
        raise ValueError('Baseline encoder report changed.')
    pipeline=read(pipeline_path)
    reuse=Path(pipeline['diagnostic_cache']).resolve()
    old=read(reuse/'encoder_provenance.json')
    if (old['id']!=base['candidate_encoder_id'] or
            fingerprint({k:v for k,v in old.items() if k!='id'})!=old['id']):
        raise ValueError('Baseline cache provenance invalid.')
    proof_path=a.muq_export_report.expanduser().resolve()
    proof=read(proof_path)
    original=read(Path(cfg['work_dir'])/'encoder_provenance.json')
    validate_source(proof,original,old)
    if proof['plan_id']!=data['plan_id'] or proof['provenance_id']!=original_id:
        raise ValueError('MuQ source export used a different dataset/cache.')
    enc=CoreMLEncoders(a.songmap_root,cfg['features'])
    for key in ('coremltools','librosa','platform','compute_units','feature_settings','app_manifest_sha256'):
        if old[key]!=enc.provenance[key]:raise ValueError('Baseline environment changed: '+key)
    for role in ('log_mel','muq_conv'):
        if old['packages'][role]!=enc.provenance['packages'][role]:raise ValueError('Frontend changed: '+role)
    package=Path(proof['package']).expanduser().resolve()
    if not package.is_dir():raise ValueError('MuQ package missing.')
    enc.package_paths['muq']=package
    identity={'baseline_encoder_id':old['id'],'muq_export_report_sha256':sha_file(proof_path),
        'muq_package_digest':fingerprint([(str(f.relative_to(package)),sha_file(f)) for f in sorted(package.rglob('*')) if f.is_file()]),
        'environment':enc.provenance,'reused_musicfm':True}
    candidate_id=fingerprint(identity)
    cache=a.cache_dir.expanduser().resolve()
    output=a.output.expanduser().resolve()
    if cache==reuse or output.exists():raise ValueError('Choose separate candidate cache and new output report.')
    cache.mkdir(parents=True,exist_ok=True)
    meta=cache/'encoder_provenance.json'
    if meta.exists() and read(meta)['id']!=candidate_id:raise ValueError('Candidate cache stale; choose a new directory.')
    write(meta,{'id':candidate_id,**identity})
    pending=[]
    for row in rows:
        list(cached_features(reuse,row,old['id']))
        if (cache/row['id']/'metadata.json').exists():list(cached_features(cache,row,candidate_id))
        else:pending.append(row)
    needed=sum(r['duration'] for r in pending)*25*4096*4
    if shutil.disk_usage(cache).free<needed+1024**3:raise ValueError(f'Need about {needed/1024**3:.1f} GiB cache plus 1 GiB free.')
    if pending:
        enc.load(('log_mel','muq_conv','muq'))
        md=enc.models['muq'].user_defined_metadata
        for key in ('muq_repo','muq_revision','muq_snapshot_digest'):
            if md.get(key)!=proof[key]:raise ValueError('MuQ package metadata mismatch: '+key)
        if md.get('precision')!='fp32' or md.get('hidden_state_index')!='10' or md.get('compression_scheme'):
            raise ValueError('Use the uncompressed FP32 MuQ source.')
    def encode(audio):
        mel=enc.predict('log_mel',{enc.first_input('log_mel'):np.pad(audio,(1024,1024),mode='reflect')[None]})
        conv=enc.predict('muq_conv',{enc.first_input('muq_conv'):mel})
        if conv.ndim!=3 or conv.shape[0]!=1 or conv.shape[2]!=1024:raise ValueError('Invalid MuQ conv shape.')
        values=enc.predict('muq',{'conv_in':conv,'rotary':rotary(conv.shape[1],enc.app['rotary_inv_freq']['muq'])})
        if values.shape!=conv.shape:raise ValueError('MuQ output shape changed.')
        return values[0]
    for i,row in enumerate(pending,1):
        print(f'MuQ [{i}/{len(pending)}]: {row["track_id"]}',flush=True)
        if sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Audio changed.')
        wav,_=librosa.load(row['audio_path'],sr=24000,mono=True)
        folder=cache/row['id'];folder.mkdir(parents=True,exist_ok=True)
        chunks=[]
        try:
            for offset,values in replace_muq(musicfm_chunks(wav,encode,cfg['features']),cached_features(reuse,row,old['id'])):
                name=f'chunk_{int(offset):06d}.npy'
                with tempfile.NamedTemporaryFile(dir=folder,delete=False) as h:
                    tmp=Path(h.name);np.save(h,values,allow_pickle=False)
                tmp.replace(folder/name)
                chunks.append({'file':name,'offset':offset,'frames':len(values)})
                print(f'  {offset:.1f}s: {len(values)} frames',flush=True)
        except EncoderWindowError as exc:
            write(cache/'last_encoder_failure.json',{**exc.details,'track_id':row['track_id'],'candidate_encoder_id':candidate_id})
            raise
        write(folder/'metadata.json',{'audio_sha256':row['audio_sha256'],'candidate_encoder_id':candidate_id,'chunks':chunks})
    enc.models.clear()
    import gc;gc.collect()
    head=ct.models.MLModel(base['head_package'],compute_units=ct.ComputeUnit.CPU_ONLY)
    md=head.user_defined_metadata
    if md.get('checkpoint_sha256')!=base['checkpoint_sha256'] or md.get('compression_scheme')!='palette6':
        raise ValueError('Head package changed.')
    adapter=CoreMLAdapter(head)
    details=[];baseline_details=[]
    reference=evaluate(adapter,rows,cfg,torch.device('cpu'),old['id'],details=baseline_details,
        feature_reader=lambda _cfg,row:cached_features(reuse,row,old['id']))
    if len(rows)==len(all_rows):
        for key in ('macro_f1_present','section_accuracy'):
            if abs(reference[key]-base['metrics'][key])>1e-10:
                raise ValueError('Recomputed baseline differs from approved combination report: '+key)
    metrics=evaluate(adapter,rows,cfg,torch.device('cpu'),candidate_id,details=details,
        feature_reader=lambda _cfg,row:cached_features(cache,row,candidate_id))
    changed=[]
    for x,y in zip(baseline_details,details):
        if (x['track_id'],x['start'],x['end'])!=(y['track_id'],y['start'],y['end']):raise ValueError('Section inventory differs.')
        if x['prediction']!=y['prediction']:changed.append({'baseline':x,'candidate':y})
    report={'tracks':len(rows),'full_validation':len(rows)==len(all_rows),'plan_id':data['plan_id'],
        'provenance_id':original_id,'annotations_id':annotations,'checkpoint_sha256':base['checkpoint_sha256'],
        'muq_export_report':str(proof_path),'muq_export_report_sha256':sha_file(proof_path),
        'candidate_encoder_id':candidate_id,'diagnostic_cache':str(cache),'baseline_combination':str(baseline_path),
        'baseline_combination_sha256':sha_file(baseline_path),'baseline_metrics':reference,'metrics':metrics,
        'macro_f1_delta':metrics['macro_f1_present']-reference['macro_f1_present'],
        'changed_sections':changed,'sections':details,'reused_musicfm':True,'head_scheme':'palette6',
        'muq_scheme':'fp32','encoder_prediction_timings_this_run':enc.timings,'ready_for_app_integration':False,
        'scope':'Mac CPU Core ML with existing app frontend/rotary, cached MusicFM palette6, head palette6 and fixed Apple boundaries. Not iPhone or end-to-end timing.'}
    write(output,report)
    print(f'MuQ FP32 F1={metrics["macro_f1_present"]:.6f}; accuracy={metrics["section_accuracy"]:.4%}; delta={report["macro_f1_delta"]:+.6f}')
    print('Report:',output)


if __name__=='__main__':main()
