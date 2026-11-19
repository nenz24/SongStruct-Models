#!/usr/bin/env python3
"""Prepare an isolated real PCM -> static/legacy encoders -> head -> native context iPhone probe."""
import argparse
import gc
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

from audit_manual_context import LABELS, validate_details
from context_coreml import package_digest
from encoder_acceleration import candidate_inputs, numerical, ATOL, RTOL
from install_section_context import file_hashes
from train_section_context import source_sections, pool_audio, geometry
from validate_encoder_pipeline import PROTOCOL as QUALITY_PROTOCOL, validate_export, cached_result

PROTOCOL='encoder-pipeline-phone-v2'
PREFLIGHT_POLICIES=('strict-feature','functional-benchmark')
FRONTEND_GPU_POLICY='full-30s-only-v1'


class PreflightMismatch(ValueError):
    def __init__(self,check):
        self.check=check
        super().__init__('Mac PCM preflight failed: '+json.dumps(check,allow_nan=False))


def feature_parity(reference,actual):
    """Use the SAME elementwise FP32 atol/rtol as encoder export, not max-absolute alone."""
    reference=np.asarray(reference);actual=np.asarray(actual)
    check=numerical(actual.astype(np.float64),reference.astype(np.float64),atol=ATOL,rtol=RTOL)
    check['policy']='encoder-export-elementwise-fp32-v1'
    check['shape_and_finite_passed']=not bool(check.get('reason'))
    if check.get('reason'):return check
    error=np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    allowed=ATOL+RTOL*np.abs(reference.astype(np.float64))
    ratio=error/allowed;index=np.unravel_index(np.argmax(ratio),ratio.shape)
    check.update(max_tolerance_ratio=float(ratio[index]),failed_values=int(np.count_nonzero(error>allowed)),
        absolute_only_failed_values=int(np.count_nonzero(error>ATOL)),worst_index=[int(i) for i in index],
        worst_reference=float(reference[index]),worst_actual=float(actual[index]),worst_allowed_error=float(allowed[index]))
    return check


def pcm_parity_check(audio,probabilities,reference_audio,reference_priors):
    features=feature_parity(reference_audio,audio)
    # Probabilities remain on an absolute scale; do not relax their gate with feature magnitudes.
    priors=numerical(reference_priors,probabilities,atol=1e-4,rtol=0.)
    return dict(passed=features['passed'] and priors['passed'],audio_features=features,base_probabilities=priors)


def validate_quality(report,selection):
    if (report.get('protocol')!=QUALITY_PROTOCOL or report.get('split')!='val' or not report.get('full_validation')
            or report.get('section_agreement')!=1 or report.get('changed_sections')!=[]
            or not report.get('comparison_to_current_pipeline',{}).get('no_observed_drop')
            or report.get('weights_updated') is not False or report.get('test_used') is not False
            or report.get('manual_labels_changed') is not False or report.get('boundaries_changed') is not False
            or report.get('selection_identity')!=selection.get('identity')):
        raise ValueError('Completed unchanged-label candidate validation is required.')
    for key in ('section_accuracy','macro_f1_present','confusion_matrix'):
        if report['candidate'][key]!=report['baseline_palette6_context'][key]:raise ValueError('Current pipeline metrics differ.')


def validate_repair_selection(selection,repair_path,sha):
    expected=selection.get('graph_repair_sha256')
    if expected is not None and (repair_path is None or sha(repair_path)!=expected):
        raise ValueError('Use the exact graph repair validated by the selected quality report.')


def validate_frontend_probe(report,selection,repair_path,sha):
    validate_quality(report,selection)
    validate_repair_selection(selection,repair_path,sha)
    expected=selection.get('graph_repair_sha256')
    if (not expected or repair_path is None or sha(repair_path)!=expected
            or selection.get('static_graph_policy')!='split-before-conv-v2'
            or report.get('static_graph_policy')!='split-before-conv-v2'
            or report.get('graph_repair_sha256')!=expected):
        raise ValueError('Frontend route probe requires full unchanged-label validation of this exact repaired graph.')


def validate_preflight_policy(policy,frontend_probe):
    if policy not in PREFLIGHT_POLICIES or (policy=='functional-benchmark' and not frontend_probe):
        raise ValueError('Functional-output admission is restricted to the isolated, quality-gated frontend benchmark.')


def choose_tracks(rows,requested=None):
    if any(r['split']!='val' or not r['approved'] for r in rows):raise ValueError('Only approved validation songs can be bundled.')
    if len({r['track_id'] for r in rows})!=len(rows):raise ValueError('Duplicate track.')
    if requested:
        lookup={r['track_id']:r for r in rows}
        if len(set(requested))!=len(requested) or any(k not in lookup for k in requested):raise ValueError('Requested tracks must be distinct approved validation tracks.')
        chosen=[lookup[k] for k in requested]
    else:
        if len(rows)<2:raise ValueError('Two validation songs required.')
        chosen=[rows[0],max((r for r in rows if r['track_id']!=rows[0]['track_id']),key=lambda r:r['duration'])]
    if len(chosen)!=2 or not any(r['duration']>420 for r in chosen):raise ValueError('Bundle exactly two songs, including one >420s, to exercise head/shape transitions.')
    return chosen


def native_context_check(native,audio,timing,probabilities,indices,details,strict=True):
    residual=np.asarray(native.predict({'section_audio':np.array(audio,dtype=np.float32,order='C',copy=True),
        'section_timing':np.array(timing,dtype=np.float32,order='C',copy=True)})['context_residual'])
    if residual.shape!=(len(audio),8) or not np.isfinite(residual).all():raise ValueError('Invalid native context output.')
    scores=np.log(np.maximum(probabilities.astype(np.float32),np.float32(1e-8)))+residual[indices]
    p=np.exp(scores-scores.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    reference=np.array([[item['probabilities'][label] for label in LABELS] for item in details])
    agreed=[LABELS[int(np.argmax(row))]==item['prediction'] for row,item in zip(p,details)]
    error=float(np.max(np.abs(p-reference)))
    check=dict(passed=all(agreed) and error<=1e-4,max_probability_error=error,
        section_agreement=sum(agreed)/len(agreed),changed_eligible_indices=[int(indices[i]) for i,ok in enumerate(agreed) if not ok])
    if strict and not check['passed']:raise ValueError('Native context differs from completed quality result: '+json.dumps(check))
    return check


def pcm_chunks(waveform,encode_pair,settings,width=1024):
    """Keep the requested stream size explicit: a 30s tail of a 120s stream stays a legacy partial."""
    rate=settings['sample_rate'];step=settings['head_window_seconds']*rate
    for offset in range(0,len(waveform),step):
        audio=waveform[offset:offset+step]
        if len(audio)<=2048:break
        arrays=[]
        for seconds in (30,120):
            parts=[]
            for start in range(0,len(audio),seconds*rate):
                window=audio[start:start+seconds*rate]
                if len(window)<=2048:continue
                pair=encode_pair(window,seconds)
                if len(pair)!=2:raise ValueError('Two encoder streams required.')
                if any(a.ndim!=2 or a.shape[1]!=width or not np.isfinite(a).all() for a in pair):raise ValueError('Invalid encoder arrays.')
                count=min(len(a) for a in pair)
                parts.append([a[:count] for a in pair])
            if not parts:raise ValueError('No valid encoder windows.')
            arrays.extend(np.concatenate([pair[k] for pair in parts]) for k in (0,1))
        lengths=[len(a) for a in arrays]
        if max(lengths)-min(lengths)>4:raise ValueError('Four encoder streams do not align.')
        frames=min(lengths)//3*3
        if not 3<=frames<=10500:raise ValueError('Invalid fused frame count.')
        yield offset/rate,np.concatenate([a[:frames] for a in arrays],axis=-1).astype(np.float32)


class MacPipeline:
    def __init__(self,ct,assets,input_names,rotary):
        self.ct=ct;self.assets=assets;self.input_names=input_names;self.rotary=rotary;self.models={}
    def load(self,key,frames=None):
        slot=key.split('_')[0] if key in ('musicfm_750','musicfm_3000','muq_750','muq_3000') else key
        identity=key if frames is None else key+':'+str(frames)
        if slot not in self.models or self.models[slot][0]!=identity:
            self.models.pop(slot,None);gc.collect()
            self.models[slot]=(identity,self.ct.models.CompiledMLModel(str(self.assets/(key+'.mlmodelc')),compute_units=self.ct.ComputeUnit.CPU_ONLY))
        return self.models[slot][1]
    def array(self,outputs):
        arrays=[np.asarray(v) for v in outputs.values() if isinstance(v,np.ndarray)]
        if len(arrays)!=1 or not np.isfinite(arrays[0]).all():raise ValueError('Invalid Mac pipeline output.')
        return np.array(arrays[0],dtype=np.float32,order='C',copy=True)
    def pair(self,audio,stream_seconds):
        from deployment.verify_encoders import rotary
        full=len(audio)==stream_seconds*24000
        mel=self.array(self.load('log_mel').predict({self.input_names['log_mel']:np.pad(audio,(1024,1024),mode='reflect')[None].astype(np.float32)}))
        result=[]
        for role in ('musicfm','muq'):
            conv=self.array(self.load(role+'_conv',frames=int(mel.shape[-1])).predict({self.input_names[role+'_conv']:mel}));frames=conv.shape[1]
            if conv.shape!=(1,frames,1024):raise ValueError('Invalid convolution output.')
            positions=rotary(frames,self.rotary[role]);key=f'{role}_{frames}' if full else role
            values=candidate_inputs(conv,positions,frames) if full else dict(conv_in=conv,rotary=positions)
            output=self.array(self.load(key,frames=frames).predict(values))
            if output.shape!=conv.shape:raise ValueError('Encoder shape mismatch.')
            result.append(output[0])
        return result
    def release(self):self.models.clear();gc.collect()


def mac_preflight(backend,waveform,row,intervals,indices,reference_audio,reference_priors,details,settings,policy='strict-feature',chunk_source=None):
    if policy not in PREFLIGHT_POLICIES:raise ValueError('Unknown preflight policy.')
    from deployment.verify_coreml import CoreMLAdapter
    from finetuning.engine import logits,pooled_sections
    adapter=CoreMLAdapter(backend.load('head'));sums=np.zeros_like(reference_priors,dtype=np.float64);covered=np.zeros(len(indices))
    def observed():
        chunks=chunk_source(waveform,settings) if chunk_source is not None else pcm_chunks(waveform,backend.pair,settings)
        for offset,values in chunks:
            with torch.inference_mode():
                scores=logits(adapter,values,torch.device('cpu'))
                for i,p,seconds in pooled_sections(scores,row['sections'],offset):sums[i]+=np.array(p.tolist())*seconds;covered[i]+=seconds
            yield values,offset,'prepared'
    audio,coverage=pool_audio(observed(),intervals,'prepared')
    if np.any(covered<=0):raise ValueError('Mac pipeline lost head coverage.')
    probabilities=(sums/covered[:,None]).astype(np.float32)
    parity=pcm_parity_check(audio,probabilities,reference_audio,reference_priors)
    check=dict(track_id=row['track_id'],passed=False,preflight_policy=policy,numerical_passed=parity['passed'],functional_passed=False,
        **{k:v for k,v in parity.items() if k!='passed'})
    check.update(max_audio_mean_error=parity['audio_features'].get('max_abs_error'),
        max_base_probability_error=parity['base_probabilities'].get('max_abs_error'))
    if parity['audio_features']['shape_and_finite_passed'] and audio.ndim==2 and audio.shape[1]==4096:
        check['stream_features']={name:feature_parity(reference_audio[:,begin:begin+1024],audio[:,begin:begin+1024])
            for name,begin in (('musicfm_30',0),('muq_30',1024),('musicfm_120',2048),('muq_120',3072))}
        worst=parity['audio_features']['worst_index'][0]
        if worst<len(intervals):check['worst_source_interval']=dict(index=worst,**intervals[worst])
    try:
        # Complete the endpoint diagnostic when finite inputs exist, even if feature tolerance failed.
        if not parity['audio_features']['shape_and_finite_passed'] or parity['base_probabilities'].get('reason'):
            check['native_context']=dict(passed=False,not_run='invalid_shape_or_nonfinite_input');raise PreflightMismatch(check)
        try:
            check['native_context']=native_context_check(backend.load('context'),audio,geometry(intervals,row['duration']),probabilities,indices,details,strict=False)
        except ValueError as error:
            check['native_context']=dict(passed=False,error=str(error));raise PreflightMismatch(check) from error
        check['functional_passed']=parity['base_probabilities']['passed'] and check['native_context']['passed']
        check['passed']=check['functional_passed'] and (check['numerical_passed'] or policy=='functional-benchmark')
        if not check['passed']:raise PreflightMismatch(check)
        return check
    finally:backend.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,required=True);p.add_argument('--songmap-root',type=Path,required=True)
    p.add_argument('--quality-report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--graph-repair',type=Path,help='Same-weight split-convolution repair-report.json; required for GPU candidate mode')
    p.add_argument('--benchmark-frontend-routes',action='store_true',help='Enable isolated CPU vs CPU+GPU convolution frontend comparison after full repaired-graph quality validation')
    p.add_argument('--preflight-policy',choices=PREFLIGHT_POLICIES,default='strict-feature',help='Default requires feature and final-output parity; functional-benchmark keeps feature mismatch diagnostic and requires unchanged final labels/probabilities')
    p.add_argument('--track-id',action='append');a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Prepare on the Mac with Xcode and the existing SongFormer environment.')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    root=a.songstruct_root;songmap=a.songmap_root;out=a.output_dir;source=Path(__file__).resolve().parent
    if out.exists():raise ValueError('Use a fresh output directory; source/app assets are never overwritten.')
    validate_preflight_policy(a.preflight_policy,a.benchmark_frontend_routes)
    sys.path.insert(0,str(root))
    from finetuning.common import config,read,write,fingerprint,sha_file
    from finetuning.engine import manifest
    from deployment.iphone_project import project_text
    import coremltools as ct
    import librosa
    cfg=config(root/'finetuning/config.json');data,provenance=manifest(cfg);torch.set_num_threads(2)
    if cfg['features']['long_window_seconds']!=120:raise ValueError('Expected the fixed 30/120/420-second profile.')
    q=read(a.quality_report);sel=read(a.quality_report.parent/'selection.json');validate_quality(q,sel)
    validate_repair_selection(sel,a.graph_repair,sha_file)
    if a.benchmark_frontend_routes:validate_frontend_probe(q,sel,a.graph_repair,sha_file)
    if sel['identity']!=fingerprint({k:v for k,v in sel.items() if k!='identity'}):raise ValueError('Quality selection digest differs.')
    if q['plan_id']!=data['plan_id'] or q['provenance_id']!=provenance:raise ValueError('Dataset/provenance changed.')
    rows=[r for r in data['records'] if r['approved'] and r['split']=='val'];chosen=choose_tracks(rows,a.track_id)
    if q['tracks']!=len(rows) or q['candidate']['sections']!=sum(len(r['sections']) for r in rows):raise ValueError('Validation inventory changed.')
    def digest(folder):return package_digest(folder,fingerprint,sha_file)
    def artifact(value):
        value=Path(value).expanduser();return value.resolve() if value.is_absolute() else (root/value).resolve()
    def find(paths,expected):
        result=next((path.resolve() for path in paths if sha_file(path)==expected),None)
        if result is None:raise ValueError('Source report missing: '+expected)
        return result
    palette_path=find((root/'outputs.local').glob('muq-compression-*/palette6-all-validation.json'),sel['palette6_source_sha256'])
    palette=read(palette_path);baseline_cache=artifact(palette['diagnostic_cache'])
    combo_path=artifact(palette['baseline_combination'])
    if sha_file(combo_path)!=palette['baseline_combination_sha256']:raise ValueError('Head/source combination changed.')
    combo=read(combo_path);pipeline_path=artifact(combo['encoder_pipeline_report'])
    if sha_file(pipeline_path)!=combo['encoder_pipeline_report_sha256']:raise ValueError('Legacy MusicFM source changed.')
    pipeline=read(pipeline_path);fm_path=find((root/'outputs.local').glob('musicfm-compression-*/palette6/compression_report.json'),pipeline['musicfm_override']['report_sha256'])
    fm=read(fm_path);muq_path=artifact(palette['muq_export_report']);muq=read(muq_path)
    if sha_file(muq_path)!=palette['muq_export_report_sha256']:raise ValueError('Legacy MuQ source changed.')
    old_fm_header=read(artifact(pipeline['diagnostic_cache'])/'encoder_provenance.json')
    if old_fm_header['id']!=pipeline['candidate_encoder_id'] or fingerprint({k:v for k,v in old_fm_header.items() if k!='id'})!=old_fm_header['id']:raise ValueError('Legacy MusicFM provenance differs.')
    legacy=dict(musicfm=artifact(fm['package']),muq=artifact(muq['package']),head=artifact(combo['head_package']))
    if digest(legacy['musicfm'])!=old_fm_header['packages']['musicfm'] or digest(legacy['muq'])!=muq['package_digest'] or digest(legacy['head'])!=sel['head_package_digest']:
        raise ValueError('Legacy dynamic/head source package differs from validation.')
    app_root=songmap/'SongMap/Resources/MLModels/SongFormer';app_path=app_root/'songformer_coreml_manifest.json';app=read(app_path)
    if sha_file(app_path)!=sel['app_manifest_sha256']:raise ValueError('Original frontend manifest changed.')
    header_path=baseline_cache/'encoder_provenance.json';header=read(header_path)
    if sha_file(header_path)!=sel['baseline_header_sha256'] or header['environment']['librosa']!=librosa.__version__:raise ValueError('Baseline decoder/cache changed.')
    frontends={key:app_root/app['models'][key] for key in ('log_mel','musicfm_conv','muq_conv')}
    for key,path in frontends.items():
        if digest(path)!=header['environment']['packages'][key]:raise ValueError('Frontend source changed: '+key)
    graph_paths={role:find((songmap/'outputs.local').glob('encoder-acceleration-*/'+role+'/export-report.json'),sel['encoder_exports'][role]) for role in ('musicfm','muq')}
    graphs={role:read(path) for role,path in graph_paths.items()}
    for role,proof in graphs.items():validate_export(proof,role,data['plan_id'],provenance)
    static_policy='v1-cpu-only'
    if a.graph_repair:
        from repair_encoder_glu import load_overrides,POLICY
        graph_paths,graphs=load_overrides(a.graph_repair,graph_paths,graphs);static_policy=POLICY
    context_sha=sel['context_identity']['context_checkpoint_sha256'];parities=[]
    for path in (songmap/'outputs.local').glob('context-coreml-fp32-*/parity/validation-all.json'):
        v=read(path)
        if (v.get('full_validation') and v.get('passed') and v.get('status')=='numerically_consistent'
                and v.get('context_checkpoint_sha256')==context_sha and v.get('selected_epoch')==5
                and v.get('palette_validation_sha256')==sel['context_identity']['palette_validation_sha256']):parities.append(path)
    if not parities:raise ValueError('Matching full native context parity report required.')
    parity_path=max(parities,key=lambda p:p.stat().st_mtime);parity=read(parity_path);context_report_path=parity_path.parent.parent/'export/export_report.json';ctx=read(context_report_path)
    if (ctx['context_checkpoint_sha256']!=context_sha or ctx['precision']!='fp32' or not ctx['native_synthetic_parity_passed']
            or ctx['package_digest']!=parity['package_digest'] or ctx['plan_id']!=data['plan_id'] or ctx['provenance_id']!=provenance):raise ValueError('Native context identity differs.')
    legacy['context']=artifact(ctx['package'])
    if digest(legacy['context'])!=ctx['package_digest']:raise ValueError('Native context package changed.')
    # Compile the ACTUAL sources used in the app math/bridge. Native typecheck happens before large assets are copied.
    math_source=songmap/'SongMap/Data/SongAnalysis/Structure/SongFormerContextMath.swift'
    bridge_source=songmap/'SongMap/Data/SongAnalysis/Structure/SongFormerTensorBridge.swift'
    if not math_source.is_file() or not bridge_source.is_file():raise ValueError('Actual app context math/tensor bridge sources are missing.')
    out.mkdir(parents=True);assets=out/'Assets';assets.mkdir()
    shutil.copyfile(source/'encoder_pipeline_iphone/BenchmarkApp.swift',out/'BenchmarkApp.swift')
    runner=(source/'encoder_pipeline_iphone/BenchmarkRunner.swift').read_text()+'\n'+(source/'encoder_pipeline_iphone/ProbeOwnedTensorCopy.swift').read_text()+'\n'+math_source.read_text()+'\n'+bridge_source.read_text()
    (out/'BenchmarkRunner.swift').write_text(runner)
    sdk=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-path'],text=True).strip()
    subprocess.run(['xcrun','swiftc','-typecheck','-parse-as-library','-target','arm64-apple-ios18.0','-sdk',sdk,
        str(out/'BenchmarkApp.swift'),str(out/'BenchmarkRunner.swift')],check=True)
    for role,path in {**frontends,**legacy}.items():
        print('Compiling probe resource:',role,flush=True)
        work=out/'CompileWork'/role;work.mkdir(parents=True)
        subprocess.run(['xcrun','coremlcompiler','compile',str(path),str(work)],check=True)
        compiled=work/(path.stem+'.mlmodelc')
        if not compiled.is_dir():raise ValueError('Compiled model missing: '+str(compiled))
        shutil.move(str(compiled),assets/(role+'.mlmodelc'))
    for role,proof in graphs.items():
        for f in proof['fixtures']:
            original=graph_paths[role].parent/'Assets'/(f['resource']+'.mlmodelc')
            if digest(original)!=f['resource_digest']:raise ValueError('Static compiled source changed.')
            shutil.copytree(original,assets/(f['resource']+'.mlmodelc'))
    input_names={role:sorted(i.name for i in ct.models.MLModel(str(path),skip_model_load=True).get_spec().description.input)[0] for role,path in frontends.items()}
    backend=MacPipeline(ct,assets,input_names,app['rotary_inv_freq']);tracks=[];checks=[]
    baseline_samples=dict(sel['context_identity']['samples'])
    for number,row in enumerate(chosen):
        print('Checking actual PCM pipeline on Mac:',row['track_id'],flush=True)
        if row['annotation_sha256']!=fingerprint(row['sections']) or sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Audio/manual annotation changed.')
        meta_path=baseline_cache/row['id']/'metadata.json';meta=read(meta_path)
        sample_id=fingerprint(dict(run=sel['identity'],sample=baseline_samples[row['track_id']],metadata=sha_file(meta_path),
            raw_feature_hashes=[sha_file(baseline_cache/row['id']/c['file']) for c in meta['chunks']]))
        cache=a.quality_report.parent/'per-track';array_path=cache/(row['id']+'.npz')
        result=cached_result(cache/(row['id']+'.json'),array_path,sample_id,read,fingerprint,sha_file)
        if result is None:raise ValueError('Completed quality track result missing.')
        validate_details([row],result['sections']);intervals,indices=source_sections(read(row['source_json']),row)
        with np.load(array_path,allow_pickle=False) as arrays:
            audio=np.array(arrays['audio'],copy=True);priors=np.array(arrays['probabilities'],copy=True)
        waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True)
        try:
            checks.append(mac_preflight(backend,waveform,row,intervals,indices,audio,priors,result['sections'],cfg['features'],policy=a.preflight_policy))
        except PreflightMismatch as error:
            report=out/'mac-preflight.json'
            failed=checks+[error.check]
            write(report,dict(checks=failed,passed=False,preflight_policy=a.preflight_policy,
                feature_parity_passed=all(c['audio_features']['passed'] for c in failed),
                functional_passed=all(c['functional_passed'] for c in failed),ready_for_app_integration=False))
            raise ValueError(f'Mac PCM parity failed. Diagnostic report: {report}') from error
        print('  PCM parity:',json.dumps(checks[-1],allow_nan=False),flush=True)
        pcm=f'track{number}_pcm';means=f'track{number}_means'
        np.asarray(waveform,dtype='<f4').tofile(assets/(pcm+'.f32'));audio.astype('<f4').tofile(assets/(means+'.f32'))
        targets=[dict(index=index,label=section['label'],prediction=det['prediction'],
            probabilities=[det['probabilities'][label] for label in LABELS],base_probabilities=prior.tolist())
            for index,section,det,prior in zip(indices,row['sections'],result['sections'],priors)]
        tracks.append(dict(id=row['track_id'],title=Path(row['track_id']).stem,duration=row['duration'],samples=len(waveform),
            pcm=pcm,pcm_sha256=sha_file(assets/(pcm+'.f32')),means=means,means_sha256=sha_file(assets/(means+'.f32')),intervals=intervals,targets=targets))
        write(out/'mac-preflight.json',dict(checks=checks,passed=all(c['passed'] for c in checks),preflight_policy=a.preflight_policy,
            feature_parity_passed=all(c['audio_features']['passed'] for c in checks),functional_passed=all(c['functional_passed'] for c in checks),
            ready_for_app_integration=False))
    models={path.stem:file_hashes(path) for path in assets.glob('*.mlmodelc')}
    manifest_data=dict(version=PROTOCOL,quality_sha256=sha_file(a.quality_report),static_graph_policy=static_policy,
        frontend_route_probe=a.benchmark_frontend_routes,graph_repair_sha256=sha_file(a.graph_repair) if a.graph_repair else None,
        frontend_gpu_policy=FRONTEND_GPU_POLICY if a.benchmark_frontend_routes else 'disabled',
        mac_preflight_policy=a.preflight_policy,mac_feature_parity_passed=all(c['audio_features']['passed'] for c in checks),
        probe_source_sha256=fingerprint(dict(runner=sha_file(out/'BenchmarkRunner.swift'),view=sha_file(out/'BenchmarkApp.swift'))),graph_states={k:v['source_state_sha256'] for k,v in graphs.items()},
        base_checkpoint=palette['checkpoint_sha256'],context_checkpoint=context_sha,rotary=app['rotary_inv_freq'],files=models,tracks=tracks)
    write(assets/'benchmark_manifest.json',manifest_data)
    resources=[p.name for p in assets.iterdir()]
    project=out/'Benchmark.xcodeproj';project.mkdir()
    bundle='com.danendra.EncoderGLURepairProbe' if a.graph_repair else 'com.danendra.EncoderPipelineProbe'
    title='GLU Repair' if a.graph_repair else 'Pipeline CPU'
    if a.benchmark_frontend_routes:bundle='com.danendra.FrontendShortRouteProbe';title='Frontend 30s'
    (project/'project.pbxproj').write_text(project_text(bundle,resource_names=resources).replace(
        'GENERATE_INFOPLIST_FILE = YES;',f'GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_CFBundleDisplayName = "{title}";'))
    write(out/'preparation.json',dict(protocol=PROTOCOL,quality_sha256=sha_file(a.quality_report),manifest_sha256=sha_file(assets/'benchmark_manifest.json'),
        preparation_script_sha256=sha_file(Path(__file__)),runner_sha256=sha_file(out/'BenchmarkRunner.swift'),view_sha256=sha_file(out/'BenchmarkApp.swift'),
        template_runner_sha256=sha_file(source/'encoder_pipeline_iphone/BenchmarkRunner.swift'),template_view_sha256=sha_file(source/'encoder_pipeline_iphone/BenchmarkApp.swift'),
        app_math_sha256=sha_file(math_source),app_bridge_sha256=sha_file(bridge_source),
        native_typecheck_passed=True,mac_pcm_pipeline_passed=True,checks=checks,static_graph_policy=static_policy,
        frontend_route_probe=a.benchmark_frontend_routes,
        frontend_gpu_policy=FRONTEND_GPU_POLICY if a.benchmark_frontend_routes else 'disabled',
        mac_preflight_policy=a.preflight_policy,mac_pcm_feature_parity_passed=all(c['audio_features']['passed'] for c in checks),
        mac_pcm_functional_passed=all(c['functional_passed'] for c in checks),
        graph_repair_sha256=sha_file(a.graph_repair) if a.graph_repair else None,temporary_assets_bytes=sum(p.stat().st_size for p in assets.rglob('*') if p.is_file()),
        temporary_model_bytes=sum(p.stat().st_size for folder in assets.glob('*.mlmodelc') for p in folder.rglob('*') if p.is_file()),
        ready_for_app_integration=False,notes=['Static/dynamic models duplicate weights temporarily; not production packaging.',
            'Phone starts from bundled resampled PCM and fixed boundaries; decoder/Apple/chords/lyrics are excluded.',
            'Mode comparison uses new statics in both modes and actual legacy CPU tails. Memory snapshots are not peak memory.']))
    print('Prepared:',project,flush=True)


if __name__=='__main__':main()
