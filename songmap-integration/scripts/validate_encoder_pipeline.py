#!/usr/bin/env python3
"""Validate static encoder candidates on fixed manual sections, keeping legacy partial windows unchanged."""
import argparse
import gc
from itertools import zip_longest
import math
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from audit_manual_context import LABELS, predict_sample, score, validate_details
from context_coreml import load_inputs, package_digest
from encoder_acceleration import PROTOCOL as GRAPH_PROTOCOL, candidate_inputs
from evaluate_frozen_context import verify_scores, changes
from train_section_context import source_sections, pool_audio, geometry, coverage_review

PROTOCOL='encoder-pipeline-quality-v1'


def validate_export(proof,role,plan,provenance):
    if (proof.get('protocol')!=GRAPH_PROTOCOL or proof.get('encoder')!=role or proof.get('compression_scheme')!='palette6'
            or not proof.get('same_weight_rewrite_passed') or proof.get('plan_id')!=plan or proof.get('provenance_id')!=provenance):
        raise ValueError('Wrong/incomplete graph export: '+role)
    checks=proof.get('checks',[]);fixtures=proof.get('fixtures',[])
    if ({c.get('frames') for c in checks}!={750,3000} or len(checks)!=2
            or {f.get('frames') for f in fixtures}!={750,3000} or len(fixtures)!=2):
        raise ValueError('Both static profiles are required: '+role)
    for check in checks:
        if not all(check.get(k,{}).get('passed') for k in ('native_wrapper','rewrite_vs_original','trace_vs_original','coreml_fp32_vs_original')):
            raise ValueError('Original-weight graph parity failed: '+role)
    for f in fixtures:
        if f.get('resource')!=f'{role}_{f["frames"]}':raise ValueError('Unexpected candidate resource name.')


def overlay_chunk(audio,base,offset,encode,settings):
    """Re-encode FULL windows only. Partial legacy stream columns remain bitwise unchanged; no padding."""
    rate=settings['sample_rate'];n=len(base)
    if (base.ndim!=2 or base.shape[1]!=4096 or not 3<=n<=10500 or n%3 or not np.isfinite(base).all()
            or abs(n-len(audio)/rate*25)>5):raise ValueError('Baseline chunk length/layout differs from decoded audio.')
    output=np.array(base,dtype=np.float32,copy=True);events=[]
    for seconds,columns in ((30,(0,1024)),(120,(2048,3072))):
        size=seconds*rate
        for begin in range(0,len(audio),size):
            stop=min(begin+size,len(audio));frame=begin//rate*25
            take=min(seconds*25,max(0,n-frame))
            for role,column in zip(('musicfm','muq'),columns):
                full=stop-begin==size
                event=dict(role=role,stream_seconds=seconds,start=offset+begin/rate,end=offset+stop/rate,
                    retained_fused_frames=take,route='new_static' if full else 'legacy_partial_cache_unchanged')
                if full:
                    values=np.asarray(encode(role,audio[begin:stop],seconds*25))
                    if values.shape!=(seconds*25,1024) or not np.isfinite(values).all():raise ValueError('Static encoder shape/nonfinite failure.')
                    if take<seconds*25-4:raise ValueError('Unexpected truncation of a full encoder window.')
                    output[frame:frame+take,column:column+1024]=values[:take]
                events.append(event)
    return output,events


def overlaid_chunks(waveform,baseline,candidate_id,encode,settings):
    rate=settings['sample_rate'];size=settings['head_window_seconds']*rate
    expected=(start for start in range(0,len(waveform),size) if len(waveform[start:start+size])>2048)
    for start,old in zip_longest(expected,baseline):
        if start is None or old is None:raise ValueError('Baseline/audio head-chunk inventories differ.')
        base,offset,_=old
        if abs(offset-start/rate)>1e-6:raise ValueError('Baseline chunk offset differs from the audio clock.')
        values,events=overlay_chunk(waveform[start:start+size],base,offset,encode,settings)
        yield values,offset,candidate_id,events


def quality_comparison(candidate,baseline):
    delta={label:candidate['per_class'][label]['f1']-baseline['per_class'][label]['f1'] for label in LABELS}
    acc=candidate['section_accuracy']-baseline['section_accuracy'];f1=candidate['macro_f1_present']-baseline['macro_f1_present']
    return dict(accuracy_delta=acc,macro_f1_delta=f1,per_class_f1_delta=delta,
        no_observed_drop=acc>=-1e-12 and f1>=-1e-12 and all(v>=-1e-12 for v in delta.values()))


def cached_result(info,array_path,identity,read,fingerprint,sha_file):
    if not info.exists():return None
    item=read(info)
    if (item.get('identity')!=identity or item.get('payload_sha256')!=fingerprint({k:v for k,v in item.items() if k!='payload_sha256'})
            or not array_path.is_file() or item.get('array_sha256')!=sha_file(array_path)):
        raise ValueError('Track result cache changed or incomplete.')
    if not math.isfinite(item['nll_sum']) or item['nll_sum']<0:raise ValueError('Invalid saved context loss.')
    return item


class StaticEncoders:
    def __init__(self,ct,app_root,app,proofs,proof_paths,digest):
        self.ct=ct;self.app_root=app_root;self.app=app;self.proofs=proofs;self.paths=proof_paths
        self.digest=digest;self.frontends={};self.active={};self.calls=[]
    def encode(self,role,audio,frames):
        from deployment.verify_encoders import rotary
        import time
        def frontend(name):
            if name not in self.frontends:self.frontends[name]=self.ct.models.MLModel(str(self.app_root/self.app['models'][name]),compute_units=self.ct.ComputeUnit.CPU_ONLY)
            return self.frontends[name]
        def predict_first(model,values):
            name=sorted(f.name for f in model.get_spec().description.input)[0]
            result=model.predict({name:np.array(values,dtype=np.float32,order='C',copy=True)})
            arrays=[np.asarray(v) for v in result.values() if isinstance(v,np.ndarray)]
            if len(arrays)!=1 or not np.isfinite(arrays[0]).all():raise ValueError('Invalid frontend output.')
            return arrays[0]
        mel=predict_first(frontend('log_mel'),np.pad(audio,(1024,1024),mode='reflect')[None])
        conv=predict_first(frontend(role+'_conv'),mel)
        if conv.shape!=(1,frames,1024):raise ValueError('Full-window frontend frame shape differs.')
        inputs=candidate_inputs(conv,rotary(frames,self.app['rotary_inv_freq'][role]),frames)
        if role not in self.active or self.active[role][0]!=frames:
            self.active.pop(role,None);gc.collect()
            fixture=next(f for f in self.proofs[role]['fixtures'] if f['frames']==frames)
            path=self.paths[role].parent/'Assets'/(fixture['resource']+'.mlmodelc')
            self.active[role]=(frames,self.ct.models.CompiledMLModel(str(path),compute_units=self.ct.ComputeUnit.CPU_ONLY))
        started=time.perf_counter();values=np.asarray(self.active[role][1].predict(inputs)['embedding'])
        self.calls.append(dict(role=role,frames=frames,prediction_seconds=time.perf_counter()-started))
        if values.shape!=(1,frames,1024) or not np.isfinite(values).all():raise ValueError('Invalid static embedding.')
        return values[0]
    def release(self):
        self.active.clear();self.frontends.clear();gc.collect()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,required=True);p.add_argument('--songmap-root',type=Path,required=True)
    p.add_argument('--palette-validation',type=Path,required=True);p.add_argument('--frozen-evaluation',type=Path,required=True)
    p.add_argument('--palette6-report',type=Path);p.add_argument('--baseline-pool-cache',type=Path,required=True)
    p.add_argument('--musicfm-export',type=Path,required=True);p.add_argument('--muq-export',type=Path,required=True)
    p.add_argument('--graph-repair',type=Path,help='Validate the bound split-convolution repair, keeping the original export gates')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=1)
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run real Core ML validation on the Mac.')
    if a.limit<0:raise ValueError('limit must be >=0')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    sys.path.insert(0,str(a.songstruct_root))
    from finetuning.common import config,read,write,fingerprint,sha_file
    from finetuning.engine import manifest,logits,pooled_sections
    from deployment.verify_coreml import CoreMLAdapter
    from evaluate_palette6_context import palette_features
    import coremltools as ct
    import librosa
    cfg=config(a.songstruct_root/'finetuning/config.json');data,provenance=manifest(cfg)
    torch.set_num_threads(2)
    # Shared original source gates and exact reproduction of the selected palette6 context validation.
    model,baseline_samples,all_rows,context_identity,_=load_inputs(SimpleNamespace(songstruct_root=a.songstruct_root,
        palette_validation=a.palette_validation,frozen_evaluation=a.frozen_evaluation,palette6_report=a.palette6_report,cache_dir=a.baseline_pool_cache))
    selected=read(a.palette_validation.parent/'selection.json')
    source_path=a.palette6_report
    if source_path is None:source_path=next(q for q in (a.songstruct_root/'outputs.local').glob('muq-compression-*/palette6-all-validation.json') if sha_file(q)==selected['palette6_report_sha256'])
    source=read(source_path)
    def artifact(value):
        path=Path(value).expanduser();return path.resolve() if path.is_absolute() else (a.songstruct_root/path).resolve()
    baseline_cache=artifact(source['diagnostic_cache']);header_path=baseline_cache/'encoder_provenance.json';header=read(header_path)
    if header['id']!=source['candidate_encoder_id'] or fingerprint({k:v for k,v in header.items() if k!='id'})!=header['id']:
        raise ValueError('Baseline fused-cache identity differs.')
    if header['environment']['librosa']!=librosa.__version__:raise ValueError('Use the same audio decoder version as the baseline cache.')
    base_path=artifact(source['baseline_combination'])
    if sha_file(base_path)!=source['baseline_combination_sha256']:raise ValueError('Source head combination changed.')
    base=read(base_path);head_package=artifact(base['head_package'])
    app_root=a.songmap_root/'SongMap/Resources/MLModels/SongFormer';app_path=app_root/'songformer_coreml_manifest.json';app=read(app_path)
    paths=dict(musicfm=a.musicfm_export,muq=a.muq_export);proofs={k:read(v) for k,v in paths.items()}
    def digest(path):return package_digest(path,fingerprint,sha_file)
    for role,proof in proofs.items():
        validate_export(proof,role,data['plan_id'],provenance)
        if proof['app_manifest_sha256']!=sha_file(app_path):raise ValueError('App source profile changed.')
        for key in ('log_mel',role+'_conv'):
            actual=digest(app_root/app['models'][key])
            if proof['frontend_hashes'][key]!=actual or actual!=header['environment']['packages'][key]:raise ValueError('Frontend differs from frozen baseline: '+key)
        for fixture in proof['fixtures']:
            path=paths[role].parent/'Assets'/(fixture['resource']+'.mlmodelc')
            if not path.is_dir() or digest(path)!=fixture['resource_digest']:raise ValueError('Compiled graph asset missing/changed: '+str(path))
    parent_paths=dict(paths)
    if a.graph_repair:
        from repair_encoder_glu import load_overrides
        paths,proofs=load_overrides(a.graph_repair,parent_paths,proofs)
    if cfg['features']['sample_rate']!=24000 or cfg['features']['head_window_seconds']!=420 or cfg['features']['long_window_seconds']!=120:
        raise ValueError('Expected unchanged 24-kHz, 30/120/420 profile.')
    out=a.output_dir
    if any(out==path or path in out.parents for path in (baseline_cache,a.baseline_pool_cache,*[q.parent for q in (*paths.values(),*parent_paths.values())])):
        raise ValueError('Output must be separate from source caches/graph exports.')
    input_identity=dict(protocol=PROTOCOL,context_identity=context_identity,palette6_source_sha256=sha_file(source_path),
        head_package_digest=digest(head_package),encoder_exports={k:sha_file(v) for k,v in parent_paths.items()},
        graph_resource_digests={k:{str(f['frames']):f['resource_digest'] for f in p['fixtures']} for k,p in proofs.items()},
        baseline_header_sha256=sha_file(header_path),app_manifest_sha256=sha_file(app_path),
        source_code_sha256=sha_file(Path(__file__)),coremltools=ct.__version__,librosa=librosa.__version__,
        partial_window_policy='reuse verified legacy CPU palette6 partial streams bitwise; no padding',compute_units='CPU_ONLY')
    if a.graph_repair:
        input_identity.update(static_graph_policy='split-before-conv-v2',graph_repair_sha256=sha_file(a.graph_repair),
            repair_exports={k:sha_file(v) for k,v in paths.items()})
    identity=fingerprint(input_identity)
    if (out/'selection.json').exists() and read(out/'selection.json')!={'identity':identity,**input_identity}:raise ValueError('Validation source/runtime changed; choose a new output directory.')
    out.mkdir(parents=True,exist_ok=True);write(out/'selection.json',{'identity':identity,**input_identity})
    stored=out/'per-track';stored.mkdir(exist_ok=True)
    enc=StaticEncoders(ct,app_root,app,proofs,paths,digest);head=None
    rows=all_rows[:a.limit] if a.limit else all_rows;samples_by={s['track_id']:s for s in baseline_samples}
    candidate_details=[];reference_details=[];loss_sum=0.;reference_loss=0.;window_events=[];coverage_events=[]
    for number,row in enumerate(rows,1):
        print(f'Encoder quality [{number}/{len(rows)}]: {row["track_id"]}',flush=True)
        sample=samples_by[row['track_id']];baseline_sections,baseline_loss=predict_sample(model,sample,torch.device('cpu'))
        reference_details.extend(baseline_sections);reference_loss+=baseline_loss
        metadata_path=baseline_cache/row['id']/'metadata.json';meta=read(metadata_path)
        raw_hashes=[sha_file(baseline_cache/row['id']/chunk['file']) for chunk in meta['chunks']]
        sample_identity=fingerprint(dict(run=identity,sample=sample['identity'],metadata=sha_file(metadata_path),raw_feature_hashes=raw_hashes))
        info=stored/(row['id']+'.json');array_path=stored/(row['id']+'.npz')
        item=cached_result(info,array_path,sample_identity,read,fingerprint,sha_file)
        if item is not None:
            print('  verified compact result reused',flush=True)
        else:
            if sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Validation audio changed.')
            intervals,indices=source_sections(read(row['source_json']),row)
            def baseline_chunks():return palette_features(baseline_cache,row,header['id'],read)
            means,original_coverage=pool_audio(baseline_chunks(),intervals,header['id'])
            if not np.allclose(means,sample['audio'].numpy(),atol=1e-6,rtol=1e-6):raise ValueError('Original fused cache does not reproduce pooled baseline audio.')
            del means
            waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True)
            if head is None:
                native_head=ct.models.MLModel(str(head_package),compute_units=ct.ComputeUnit.CPU_ONLY)
                if native_head.user_defined_metadata.get('checkpoint_sha256')!=source['checkpoint_sha256'] or native_head.user_defined_metadata.get('compression_scheme')!='palette6':
                    raise ValueError('Frozen label head metadata differs.')
                head=CoreMLAdapter(native_head)
            sums=np.zeros((len(indices),8),dtype=np.float64);head_coverage=np.zeros(len(indices))
            old_sums=np.zeros_like(sums);old_coverage=np.zeros_like(head_coverage);events=[];chunks_meta=[]
            def accumulate(values,offset,target_sums,target_coverage):
                with torch.inference_mode():
                    scores=logits(head,values,torch.device('cpu'))
                    for i,p,covered in pooled_sections(scores,row['sections'],offset):
                        target_sums[i]+=np.asarray(p.tolist())*covered;target_coverage[i]+=covered
            def inputs():
                for values,offset,signature in baseline_chunks():
                    accumulate(values,offset,old_sums,old_coverage)
                    yield values,offset,signature
            def observed_chunks():
                for values,offset,signature,observations in overlaid_chunks(waveform,inputs(),identity,enc.encode,cfg['features']):
                    print(f'  head window {offset:.1f}s: {len(values)} frames',flush=True)
                    events.extend(observations);chunks_meta.append(dict(offset=offset,frames=len(values)))
                    accumulate(values,offset,sums,head_coverage)
                    yield values,offset,signature
            audio,coverage=pool_audio(observed_chunks(),intervals,identity)
            if np.any(head_coverage<=0) or np.any(old_coverage<=0):raise ValueError('Manual targets lost head coverage.')
            old_prob=old_sums/old_coverage[:,None]
            error=float(np.max(np.abs(old_prob-sample['probabilities'].numpy())))
            if error>1e-5:raise ValueError('Same native head failed baseline prior reproduction: '+str(error))
            diagnostics=coverage_review(row,intervals,indices,coverage,dict(chunks=chunks_meta))
            probabilities=(sums/head_coverage[:,None]).astype(np.float32)
            candidate={**sample,'audio':torch.from_numpy(audio),'timing':torch.from_numpy(geometry(intervals,row['duration'])),
                'probabilities':torch.from_numpy(probabilities)}
            produced,loss=predict_sample(model,candidate,torch.device('cpu'));validate_details([row],produced)
            with tempfile.NamedTemporaryFile(dir=stored,delete=False) as handle:
                temporary=Path(handle.name);np.savez(handle,audio=audio,coverage=coverage,probabilities=probabilities)
            temporary.replace(array_path)
            item=dict(identity=sample_identity,array_sha256=sha_file(array_path),sections=produced,nll_sum=loss,
                baseline_head_max_probability_error=error,windows=events,coverage_diagnostics=diagnostics)
            item['payload_sha256']=fingerprint(item);write(info,item)
            del waveform,audio,candidate;enc.release()
        validate_details([row],item['sections']);candidate_details.extend(item['sections']);loss_sum+=item['nll_sum']
        window_events.extend(item['windows']);coverage_events.extend(item['coverage_diagnostics'])
    enc.release();reference=score(reference_details);reference['section_nll']=reference_loss/len(reference_details)
    candidate=score(candidate_details);candidate['section_nll']=loss_sum/len(candidate_details)
    old=read(a.palette_validation)
    if len(rows)==len(all_rows):verify_scores(reference,old['palette6_features_with_context'])
    original=old['original_features_with_context'] if len(rows)==len(all_rows) else None
    if (any(sha_file(path)!=input_identity['encoder_exports'][role] for role,path in parent_paths.items())
            or digest(head_package)!=input_identity['head_package_digest']
            or sha_file(header_path)!=input_identity['baseline_header_sha256']
            or sha_file(app_path)!=input_identity['app_manifest_sha256']):raise ValueError('Frozen source changed during evaluation.')
    if a.graph_repair:
        if (sha_file(a.graph_repair)!=input_identity['graph_repair_sha256']
                or any(sha_file(path)!=input_identity['repair_exports'][role] for role,path in paths.items())):
            raise ValueError('Graph repair changed during evaluation.')
        for role,proof in proofs.items():
            for fixture in proof['fixtures']:
                compiled=paths[role].parent/'Assets'/(fixture['resource']+'.mlmodelc')
                if digest(compiled)!=fixture['resource_digest']:raise ValueError('Repaired compiled asset changed during evaluation.')
    routes={route:dict(windows=sum(e['route']==route for e in window_events),
        stream_frames=sum(e['retained_fused_frames'] for e in window_events if e['route']==route))
        for route in ('new_static','legacy_partial_cache_unchanged')}
    report=dict(protocol=PROTOCOL,split='val',tracks=len(rows),full_validation=len(rows)==len(all_rows),
        selection_identity=identity,plan_id=data['plan_id'],provenance_id=provenance,
        baseline_palette6_context=reference,candidate=candidate,comparison_to_current_pipeline=quality_comparison(candidate,reference),
        original_training_features_context=original,comparison_to_original_context=quality_comparison(candidate,original) if original else None,
        section_agreement=sum(x['prediction']==y['prediction'] for x,y in zip(reference_details,candidate_details))/len(candidate_details),
        changed_sections=changes(reference_details,candidate_details),routes=routes,coverage_diagnostics=coverage_events,
        encoder_prediction_timings_this_run=enc.calls,encoder_compute_units='CPU_ONLY',context_runtime='frozen epoch5 PyTorch',
        weights_updated=False,test_used=False,manual_labels_changed=False,boundaries_changed=False,ready_for_app_integration=False,
        limitations=['Full original manual validation inventory when full_validation=true; legacy partial-window embeddings are inherited unchanged from the verified CPU cache.',
            'Static full-window encoders and native palette6 base head execute on Mac CPU; context stays frozen PyTorch. Not a GPU or end-to-end iPhone benchmark.',
            'No padding, changed audio window sizes, retraining, lyric rules or boundary edits. Validation has guided prior experiments; no population-level noninferiority claim.'])
    if a.graph_repair:report.update(static_graph_policy=input_identity['static_graph_policy'],graph_repair_sha256=input_identity['graph_repair_sha256'])
    destination=out/('validation-all.json' if report['full_validation'] else f'validation-first-{len(rows)}.json');write(destination,report)
    print('Baseline F1:',reference['macro_f1_present'],'Candidate F1:',candidate['macro_f1_present'],flush=True)
    print('Quality vs current:',report['comparison_to_current_pipeline'],flush=True);print('Report:',destination,flush=True)


if __name__=='__main__':main()
