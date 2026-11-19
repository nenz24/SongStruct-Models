#!/usr/bin/env python3
"""Evaluate unified 6-bit encoder weights on all manual validation targets; re-encode partials too."""
import argparse
import gc
from itertools import zip_longest
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from audit_manual_context import LABELS,predict_sample,score,validate_details
from canonical_encoder_weights import PROTOCOL as EXPORT_PROTOCOL,CANONICAL,PARAMETERS
from context_coreml import load_inputs
from encoder_acceleration import candidate_inputs
from evaluate_frozen_context import verify_scores,changes
from install_section_context import read,sha_file,fingerprint,package_digest
from prepare_encoder_pipeline_iphone import pcm_chunks,validate_quality,validate_repair_selection
from repair_encoder_glu import write
from train_section_context import source_sections,pool_audio,geometry,coverage_review
from validate_encoder_pipeline import cached_result,quality_comparison

PROTOCOL='canonical-encoder-quality-v1'


def validate_candidate(report,role,quality_hash,repair_hash):
    if (report.get('protocol')!=EXPORT_PROTOCOL or report.get('encoder')!=role or report.get('canonical_profile')!=CANONICAL
            or not report.get('candidate_for_quality_validation') or not report.get('native_cpu_passed') or report.get('error')
            or report.get('weights_unified') is not True or report.get('weights_updated') is not True or report.get('weights_requantized') is not False or report.get('retrained') is not False
            or report.get('evidence',{}).get('quality_sha256')!=quality_hash or report.get('evidence',{}).get('repair_sha256')!=repair_hash):
        raise ValueError('Wrong/incomplete canonical encoder export: '+role)
    profiles=report.get('profiles',{})
    if set(profiles)!={'short_30','long_120','partial'} or any(not p.get('non_parameter_graph_unchanged') or p.get('parameters')!=PARAMETERS for p in profiles.values()):
        raise ValueError('Canonical parameter mapping is incomplete.')
    long=profiles[CANONICAL]
    if (not isinstance(long.get('original_graph_sha256'),str) or len(long['original_graph_sha256'])!=64
            or long['original_graph_sha256']!=long.get('unified_graph_sha256')):raise ValueError('Long control graph changed.')
    checks=report.get('native_checks',[])
    if (len(checks)!=6 or {(c.get('function'),c.get('frames')) for c in checks}!={('short_30',750),('long_120',3000),('partial',594),('partial',739),('partial',1500),('partial',2989)}
            or not all(c.get('passed') and len(c.get('packed_vs_unified',[]))==2 and all(v.get('passed') for v in c['packed_vs_unified']) for c in checks)
            or not (next(c for c in checks if c['function']=='long_120').get('vs_original_profile') or {}).get('passed')):
        raise ValueError('Required native checks missing or failed.')


class Encoders:
    def __init__(self,ct,app_root,app,reports,report_paths):
        self.ct=ct;self.app_root=app_root;self.app=app;self.reports=reports;self.paths=report_paths;self.front={};self.active={};self.calls=[]
    def frontend(self,role,tensor):
        key=tuple(tensor.shape)
        if role not in self.front or self.front[role][0]!=key:
            self.front.pop(role,None);gc.collect()
            self.front[role]=(key,self.ct.models.MLModel(str(self.app_root/self.app['models'][role]),compute_units=self.ct.ComputeUnit.CPU_ONLY))
        model=self.front[role][1];name=sorted(f.name for f in model.get_spec().description.input)[0]
        values=model.predict({name:np.array(tensor,dtype=np.float32,order='C',copy=True)})
        if len(values)!=1:raise ValueError('Unexpected frontend outputs.')
        output=np.array(next(iter(values.values())),dtype=np.float32,order='C',copy=True)
        if not np.isfinite(output).all():raise ValueError('Non-finite frontend output.')
        return output
    def encode_pair(self,audio,seconds):
        from deployment.verify_encoders import rotary
        mel=self.frontend('log_mel',np.pad(audio,(1024,1024),mode='reflect')[None]);outputs=[]
        full=len(audio)==seconds*24000
        function=('short_30' if seconds==30 else 'long_120') if full else 'partial'
        for role in ('musicfm','muq'):
            conv=self.frontend(role+'_conv',mel);frames=conv.shape[1]
            if conv.shape!=(1,frames,1024):raise ValueError('Frontend shape changed.')
            positions=rotary(frames,self.app['rotary_inv_freq'][role])
            inputs=candidate_inputs(conv,positions,frames) if full else dict(conv_in=conv,rotary=positions)
            identity=(function,frames)
            if role not in self.active or self.active[role][0]!=identity:
                self.active.pop(role,None);gc.collect()
                path=self.paths[role].parent/'Compiled'/(role+'_Canonical.mlmodelc')
                self.active[role]=(identity,self.ct.models.CompiledMLModel(str(path),function_name=function,compute_units=self.ct.ComputeUnit.CPU_ONLY))
            started=time.perf_counter();result=self.active[role][1].predict(inputs)
            if len(result)!=1:raise ValueError('Unexpected encoder output.')
            output=np.array(next(iter(result.values())),dtype=np.float32,order='C',copy=True)
            self.calls.append(dict(role=role,function=function,frames=frames,seconds=time.perf_counter()-started))
            if output.shape!=conv.shape or not np.isfinite(output).all():raise ValueError('Canonical encoder shape/nonfinite failure.')
            outputs.append(output[0])
        return outputs
    def release(self):self.front.clear();self.active.clear();gc.collect()


def reencoded_chunks(waveform,baseline,encode_pair,settings,signature):
    rate=settings['sample_rate'];step=settings['head_window_seconds']*rate
    starts=(s for s in range(0,len(waveform),step) if len(waveform[s:s+step])>2048)
    for start,previous in zip_longest(starts,baseline):
        if start is None or previous is None:raise ValueError('Audio/reference chunk inventories differ.')
        old,offset,_=previous
        if abs(offset-start/rate)>1e-6:raise ValueError('Head-chunk timing differs.')
        observations=[];counters={30:0,120:0}
        def observed(audio,seconds):
            begin=counters[seconds]*seconds*rate;counters[seconds]+=1
            output=encode_pair(audio,seconds)
            for role,values in zip(('musicfm','muq'),output):
                observations.append(dict(role=role,stream_seconds=seconds,start=offset+begin/rate,end=offset+(begin+len(audio))/rate,
                    frames=len(values),route='canonical_full' if len(audio)==seconds*rate else 'canonical_partial_cpu'))
            return output
        chunks=list(pcm_chunks(waveform[start:start+step],observed,settings))
        if len(chunks)!=1 or chunks[0][0]!=0.:raise ValueError('Unexpected local head chunk.')
        values=chunks[0][1]
        if values.shape!=old.shape:raise ValueError('Fused timing/shape changed; do not adjust boundaries or pad features.')
        yield values,offset,signature,observations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('songstruct-root','songmap-root','reference-quality','graph-repair','palette-validation','frozen-evaluation','baseline-pool-cache','musicfm-report','muq-report','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=1);a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run real Core ML evaluation on Mac.')
    if a.limit<0:raise ValueError('limit must be >=0')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    sys.path.insert(0,str(a.songstruct_root))
    from finetuning.common import config
    from finetuning.engine import manifest,logits,pooled_sections
    import finetuning.engine as engine_module
    import deployment.verify_coreml as head_adapter_module
    from deployment.verify_coreml import CoreMLAdapter
    from evaluate_palette6_context import palette_features
    import coremltools as ct
    import librosa
    cfg=config(a.songstruct_root/'finetuning/config.json');data,provenance=manifest(cfg);torch.set_num_threads(2)
    if (cfg['features']['sample_rate'],cfg['features']['long_window_seconds'],cfg['features']['head_window_seconds'])!=(24000,120,420):raise ValueError('Window profile changed.')
    reference=read(a.reference_quality);selection=read(a.reference_quality.parent/'selection.json')
    validate_quality(reference,selection);validate_repair_selection(selection,a.graph_repair,sha_file)
    if selection['identity']!=fingerprint({k:v for k,v in selection.items() if k!='identity'}):raise ValueError('Reference identity changed.')
    model,samples,all_rows,context_identity,_=load_inputs(SimpleNamespace(songstruct_root=a.songstruct_root,palette_validation=a.palette_validation,
        frozen_evaluation=a.frozen_evaluation,palette6_report=None,cache_dir=a.baseline_pool_cache))
    if context_identity!=selection['context_identity'] or reference['plan_id']!=data['plan_id'] or reference['provenance_id']!=provenance:raise ValueError('Reference dataset/context changed.')
    paths=dict(musicfm=a.musicfm_report,muq=a.muq_report);reports={k:read(v) for k,v in paths.items()}
    for role,r in reports.items():
        validate_candidate(r,role,sha_file(a.reference_quality),sha_file(a.graph_repair))
        if r['evidence']['plan_id']!=data['plan_id'] or r['evidence']['provenance_id']!=provenance:raise ValueError('Canonical dataset identity differs.')
        if package_digest(Path(r['package']))!=r['package_digest'] or package_digest(paths[role].parent/'Compiled'/(role+'_Canonical.mlmodelc'))!=r['compiled_digest']:raise ValueError('Canonical artifact changed.')
    def artifact(value):
        value=Path(value).expanduser();return value.resolve() if value.is_absolute() else (a.songstruct_root/value).resolve()
    source_path=next((q for q in (a.songstruct_root/'outputs.local').glob('muq-compression-*/palette6-all-validation.json') if sha_file(q)==selection['palette6_source_sha256']),None)
    if source_path is None:raise ValueError('Frozen source pipeline missing.')
    source=read(source_path);cache=artifact(source['diagnostic_cache']);header=read(cache/'encoder_provenance.json')
    if sha_file(cache/'encoder_provenance.json')!=selection['baseline_header_sha256'] or header['environment']['librosa']!=librosa.__version__:raise ValueError('Baseline cache/decoder changed.')
    combo_path=artifact(source['baseline_combination'])
    if sha_file(combo_path)!=source['baseline_combination_sha256']:raise ValueError('Head combination changed.')
    head_path=artifact(read(combo_path)['head_package'])
    if package_digest(head_path)!=selection['head_package_digest']:raise ValueError('Head changed.')
    app_root=a.songmap_root/'SongMap/Resources/MLModels/SongFormer';app_path=app_root/'songformer_coreml_manifest.json';app=read(app_path)
    if sha_file(app_path)!=selection['app_manifest_sha256']:raise ValueError('App frontend profile changed.')
    for role in ('log_mel','musicfm_conv','muq_conv'):
        if package_digest(app_root/app['models'][role])!=header['environment']['packages'][role]:raise ValueError('Frontend changed: '+role)
    out=a.output_dir
    protected=(cache,a.baseline_pool_cache,a.frozen_evaluation,a.reference_quality.parent,*(path.parent for path in paths.values()))
    if (a.songmap_root/'outputs.local') not in out.parents or any(out==path or path in out.parents for path in protected):raise ValueError('Use separate staged validation output.')
    identity_data=dict(protocol=PROTOCOL,reference_quality_sha256=sha_file(a.reference_quality),repair_sha256=sha_file(a.graph_repair),
        reports={k:sha_file(v) for k,v in paths.items()},packages={k:r['package_digest'] for k,r in reports.items()},
        compiled={k:r['compiled_digest'] for k,r in reports.items()},context_identity=context_identity,
        head_digest=selection['head_package_digest'],app_manifest_sha256=sha_file(app_path),
        script_sha256=sha_file(Path(__file__)),extractor_sha256=sha_file(Path(__file__).parent/'prepare_encoder_pipeline_iphone.py'),
        engine_sha256=sha_file(Path(engine_module.__file__)),head_adapter_sha256=sha_file(Path(head_adapter_module.__file__)),
        coremltools=ct.__version__,librosa=librosa.__version__,all_windows_reencoded=True,compute_units='CPU_ONLY')
    identity=fingerprint(identity_data);locked=dict(identity=identity,**identity_data)
    if (out/'selection.json').exists() and read(out/'selection.json')!=locked:raise ValueError('Canonical evaluation identity changed; choose new output.')
    out.mkdir(parents=True,exist_ok=True);write(out/'selection.json',locked);stored=out/'per-track';stored.mkdir(exist_ok=True)
    backend=Encoders(ct,app_root,app,reports,paths);native_head=ct.models.MLModel(str(head_path),compute_units=ct.ComputeUnit.CPU_ONLY);head=CoreMLAdapter(native_head)
    samples={s['track_id']:s for s in samples};rows=all_rows[:a.limit] if a.limit else all_rows
    candidate_details=[];baseline_details=[];loss_sum=0.;baseline_loss=0.;windows=[];coverage_diagnostics=[]
    for number,row in enumerate(rows,1):
        print(f'Canonical quality [{number}/{len(rows)}]: {row["track_id"]}',flush=True)
        sample=samples[row['track_id']];meta_path=cache/row['id']/'metadata.json';meta=read(meta_path)
        reference_identity=fingerprint(dict(run=selection['identity'],sample=sample['identity'],metadata=sha_file(meta_path),raw_feature_hashes=[sha_file(cache/row['id']/c['file']) for c in meta['chunks']]))
        baseline_info=a.reference_quality.parent/'per-track'/(row['id']+'.json');baseline_array=baseline_info.with_suffix('.npz')
        baseline=cached_result(baseline_info,baseline_array,reference_identity,read,fingerprint,sha_file)
        if baseline is None:raise ValueError('Verified reference per-track result missing.')
        validate_details([row],baseline['sections']);baseline_details.extend(baseline['sections']);baseline_loss+=baseline['nll_sum']
        sample_identity=fingerprint(dict(run=identity,sample=sample['identity'],reference=sha_file(baseline_info)))
        info=stored/(row['id']+'.json');array_path=info.with_suffix('.npz');item=cached_result(info,array_path,sample_identity,read,fingerprint,sha_file)
        if item is None:
            if sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Audio changed.')
            intervals,indices=source_sections(read(row['source_json']),row)
            waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True);sums=np.zeros((len(indices),8),dtype=np.float64);covered=np.zeros(len(indices));events=[];chunks=[]
            old_sums=np.zeros_like(sums);old_covered=np.zeros_like(covered)
            def reference_chunks():
                for values,offset,signature in palette_features(cache,row,header['id'],read):
                    with torch.inference_mode():
                        scores=logits(head,values,torch.device('cpu'))
                        for i,p,seconds in pooled_sections(scores,row['sections'],offset):old_sums[i]+=np.array(p.tolist())*seconds;old_covered[i]+=seconds
                    yield values,offset,signature
            def observed():
                base=reference_chunks()
                for values,offset,signature,observations in reencoded_chunks(waveform,base,backend.encode_pair,cfg['features'],identity):
                    print(f'  all encoder streams, including partials: {offset:.1f}s, {len(values)} frames',flush=True)
                    events.extend(observations);chunks.append(dict(offset=offset,frames=len(values)))
                    with torch.inference_mode():
                        scores=logits(head,values,torch.device('cpu'))
                        for i,p,seconds in pooled_sections(scores,row['sections'],offset):sums[i]+=np.array(p.tolist())*seconds;covered[i]+=seconds
                    yield values,offset,signature
            audio,coverage=pool_audio(observed(),intervals,identity)
            if np.any(covered<=0) or np.any(old_covered<=0):raise ValueError('Targets lost probability coverage.')
            old_head_error=float(np.max(np.abs(old_sums/old_covered[:,None]-sample['probabilities'].numpy())))
            if old_head_error>1e-5:raise ValueError('Frozen head failed baseline prior reproduction: '+str(old_head_error))
            probabilities=(sums/covered[:,None]).astype(np.float32)
            new_sample={**sample,'audio':torch.from_numpy(audio),'timing':torch.from_numpy(geometry(intervals,row['duration'])),'probabilities':torch.from_numpy(probabilities)}
            sections,loss=predict_sample(model,new_sample,torch.device('cpu'));validate_details([row],sections)
            diagnostics=coverage_review(row,intervals,indices,coverage,dict(chunks=chunks))
            with tempfile.NamedTemporaryFile(dir=stored,delete=False) as handle:
                temporary=Path(handle.name);np.savez(handle,audio=audio,coverage=coverage,probabilities=probabilities)
            temporary.replace(array_path);item=dict(identity=sample_identity,array_sha256=sha_file(array_path),sections=sections,nll_sum=loss,windows=events,coverage_diagnostics=diagnostics,baseline_head_max_probability_error=old_head_error)
            item['payload_sha256']=fingerprint(item);write(info,item);del waveform,audio,new_sample;backend.release()
        else:print('  verified compact canonical result reused',flush=True)
        validate_details([row],item['sections']);candidate_details.extend(item['sections']);loss_sum+=item['nll_sum'];windows.extend(item['windows']);coverage_diagnostics.extend(item['coverage_diagnostics'])
    backend.release();baseline=score(baseline_details);candidate=score(candidate_details)
    baseline['section_nll']=baseline_loss/len(baseline_details);candidate['section_nll']=loss_sum/len(candidate_details)
    if len(rows)==len(all_rows):verify_scores(baseline,reference['candidate'])
    for role,r in reports.items():
        if sha_file(paths[role])!=identity_data['reports'][role] or package_digest(Path(r['package']))!=r['package_digest'] or package_digest(paths[role].parent/'Compiled'/(role+'_Canonical.mlmodelc'))!=r['compiled_digest']:raise ValueError('Candidate changed during quality evaluation.')
    if package_digest(head_path)!=identity_data['head_digest'] or sha_file(app_path)!=identity_data['app_manifest_sha256']:raise ValueError('Head/frontend manifest changed during evaluation.')
    for role in ('log_mel','musicfm_conv','muq_conv'):
        if package_digest(app_root/app['models'][role])!=header['environment']['packages'][role]:raise ValueError('Frontend changed during evaluation.')
    result=dict(protocol=PROTOCOL,split='val',tracks=len(rows),full_validation=len(rows)==len(all_rows),selection_identity=identity,
        plan_id=data['plan_id'],provenance_id=provenance,baseline_current_pipeline=baseline,candidate=candidate,
        comparison_to_current_pipeline=quality_comparison(candidate,baseline),section_agreement=sum(a['prediction']==b['prediction'] for a,b in zip(baseline_details,candidate_details))/len(candidate_details),
        changed_sections=changes(baseline_details,candidate_details),coverage_diagnostics=coverage_diagnostics,
        routes={name:sum(v['route']==name for v in windows) for name in ('canonical_full','canonical_partial_cpu')},
        weights_updated=True,weights_unified=True,weights_requantized=False,retrained=False,test_used=False,manual_labels_changed=False,boundaries_changed=False,
        artifact_sizes=dict(previous_compiled_encoder_bytes=sum(r['prior_compiled_bytes'] for r in reports.values()),canonical_compiled_encoder_bytes=sum(r['compiled_bytes'] for r in reports.values())),
        all_windows_reencoded=True,partial_embeddings_reused=False,canonical_profile=CANONICAL,encoder_compute_units='CPU_ONLY',ready_for_app_integration=False,
        limitations=['Both full and partial encoders run on Mac CPU; no old partial embeddings become candidate input.',
            'One fixed long-profile bank was chosen before evaluating; validation has guided previous experiments, test is not used.',
            'GPU/device speed, memory and compiled multi-function stability are unverified for this candidate.'])
    destination=out/('validation-all.json' if result['full_validation'] else 'validation-one.json');write(destination,result)
    print('Current F1:',baseline['macro_f1_present'],'Canonical candidate F1:',candidate['macro_f1_present'],flush=True)
    print('Comparison:',result['comparison_to_current_pipeline'],'Report:',destination,flush=True)


if __name__=='__main__':main()
