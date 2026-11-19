#!/usr/bin/env python3
"""Stage the quality-validated shared encoder banks in an isolated physical-iPhone probe."""
import argparse
import gc
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

from audit_manual_context import LABELS,validate_details
from canonical_encoder_quality import PROTOCOL as QUALITY_PROTOCOL,validate_candidate
from install_section_context import read,sha_file,fingerprint,package_digest,file_hashes
from prepare_encoder_pipeline_iphone import MacPipeline,mac_preflight,PreflightMismatch,choose_tracks,validate_quality
from repair_encoder_glu import write
from train_section_context import source_sections
from validate_encoder_pipeline import cached_result,quality_comparison

POLICY='long-profile-shared-palette-v1'
FUNCTIONS={'750':'short_30','3000':'long_120','partial':'partial'}
SCHEDULE='one-role-at-a-time-v1'
TRANSFER='owned-fp32-between-models-v1'
CPU_RECEIPT_SHA='066006ed14b4850c4bce05a5445b5a53ff5bc962ce30ec1ad85082a6b4a00770'
PREFLIGHT_POLICIES=('strict-feature','owned-buffer-diagnostic')


def validate_owned_diagnostic_control(receipt,quality_sha,selection,q):
    """Admit only this reviewed CPU-functional candidate to a non-production buffer experiment."""
    if (receipt.get('protocol')!='reviewed-canonical-cpu-control-v1' or receipt.get('mode')!='cpuOnly'
            or receipt.get('completed_runs')!=6 or receipt.get('requested_runs')!=6
            or receipt.get('all_functional_checks_passed') is not True or receipt.get('numerical_identity_claimed') is not False
            or receipt.get('quality_sha256')!=quality_sha or receipt.get('encoder_reports')!=selection['reports']
            or receipt.get('compiled_encoder_bytes')!=q['artifact_sizes']['canonical_compiled_encoder_bytes']
            or receipt.get('encoder_schedule')!=SCHEDULE or len(receipt.get('runs',[]))!=6):
        raise ValueError('Owned-buffer diagnostic requires the exact reviewed CPU control and canonical artifacts.')
    for index,run in enumerate(receipt['runs']):
        if (run.get('pass_index')!=index//2+1 or run.get('track_index')!=index%2 or run.get('label_agreement')!=1
                or not isinstance(run.get('targets'),int) or run['targets']<=0
                or any(not isinstance(run.get(k),(int,float)) or not math.isfinite(run[k]) or not 0<=run[k]<=1e-4
                    for k in ('base_probability_max_abs_error','final_probability_max_abs_error'))):
            raise ValueError('Reviewed CPU control is incomplete or fails functional gates.')
    if sum(v['targets'] for v in receipt['runs'])!=receipt.get('repeated_targets'):raise ValueError('Reviewed CPU decision count differs.')


def finalize_preflight_admission(check,policy):
    if policy not in PREFLIGHT_POLICIES:raise ValueError('Unknown canonical preflight policy.')
    def bounded(value):return isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1e-4
    functional=bool(check.get('functional_passed') and check.get('base_probabilities',{}).get('passed')
        and check.get('native_context',{}).get('passed') and check.get('native_context',{}).get('section_agreement')==1
        and check.get('native_context',{}).get('changed_eligible_indices')==[]
        and bounded(check.get('base_probabilities',{}).get('max_abs_error'))
        and bounded(check.get('native_context',{}).get('max_probability_error'))
        and check.get('audio_features',{}).get('shape_and_finite_passed'))
    check['engine_preflight_policy']=check.get('preflight_policy')
    check['preflight_policy']=policy
    check['admission_policy']=policy
    check['passed']=functional and (check.get('numerical_passed') is True or policy=='owned-buffer-diagnostic')
    # Preserve numerical_passed/audio_features.passed: diagnostic admission is not numerical equivalence.
    if not check['passed']:raise PreflightMismatch(check)
    return check


def serial_chunks(waveform,encode,release,settings,width=1024):
    """Same windows/timestamps/fusion order, with one encoder's model lifetime at a time."""
    rate=settings['sample_rate'];step=settings['head_window_seconds']*rate
    for offset in range(0,len(waveform),step):
        audio=waveform[offset:offset+step]
        if len(audio)<=2048:break
        streams={};window_frames={}
        for role in ('musicfm','muq'):
            try:
                for seconds in (30,120):
                    parts=[]
                    for start in range(0,len(audio),seconds*rate):
                        window=audio[start:start+seconds*rate]
                        if len(window)<=2048:continue
                        values=encode(role,window,seconds)
                        if values.ndim!=2 or values.shape[1]!=width or not np.isfinite(values).all():raise ValueError('Invalid serial encoder output.')
                        key=(seconds,start)
                        if role=='musicfm':window_frames[key]=len(values)
                        elif len(values)!=window_frames[key]:raise ValueError('Serial per-window frame counts differ; refusing shifted alignment.')
                        parts.append(values)
                    if not parts:raise ValueError('No valid serial encoder windows.')
                    streams[(role,seconds)]=np.concatenate(parts)
            finally:release()
        ordered=[streams[(role,seconds)] for seconds in (30,120) for role in ('musicfm','muq')]
        lengths=[len(v) for v in ordered]
        if max(lengths)-min(lengths)>4:raise ValueError('Serial encoder streams do not align.')
        frames=min(lengths)//3*3
        if not 3<=frames<=10500:raise ValueError('Invalid serial fused frame count.')
        yield offset/rate,np.concatenate([v[:frames] for v in ordered],axis=-1).astype(np.float32)


def validate_canonical_quality(q,selection):
    if (q.get('protocol')!=QUALITY_PROTOCOL or q.get('split')!='val' or not q.get('full_validation')
            or q.get('section_agreement')!=1 or q.get('changed_sections')!=[] or q.get('coverage_diagnostics')!=[]
            or q.get('weights_updated') is not True or q.get('weights_unified') is not True
            or q.get('weights_requantized') is not False or q.get('retrained') is not False
            or q.get('test_used') is not False or q.get('manual_labels_changed') is not False or q.get('boundaries_changed') is not False
            or q.get('all_windows_reencoded') is not True or q.get('partial_embeddings_reused') is not False
            or q.get('encoder_compute_units')!='CPU_ONLY' or q.get('canonical_profile')!='long_120'
            or q.get('selection_identity')!=selection.get('identity')
            or selection.get('protocol')!=QUALITY_PROTOCOL
            or selection.get('identity')!=fingerprint({k:v for k,v in selection.items() if k!='identity'})):
        raise ValueError('Use completed unchanged-label full canonical validation and its exact selection.')
    baseline=q['baseline_current_pipeline'];candidate=q['candidate']
    if quality_comparison(candidate,baseline)!=q.get('comparison_to_current_pipeline') or not q['comparison_to_current_pipeline']['no_observed_drop']:
        raise ValueError('Canonical metrics/comparison differ.')
    for key in ('sections','section_accuracy','macro_f1_present','duration_weighted_accuracy','per_class','confusion_matrix','label_order'):
        if candidate[key]!=baseline[key]:raise ValueError('Canonical labels/per-class metrics differ from reviewed baseline.')
    sizes=q['artifact_sizes']
    if not 0<sizes['canonical_compiled_encoder_bytes']<sizes['previous_compiled_encoder_bytes']:
        raise ValueError('Canonical compiled encoders must actually be smaller.')


def locate(paths,expected):
    result=next((p.resolve() for p in paths if p.is_file() and sha_file(p)==expected),None)
    if result is None:raise ValueError('Required source report not found: '+expected)
    return result


class CanonicalMacPipeline(MacPipeline):
    def load(self,key,frames=None):
        role=next((r for r in ('musicfm','muq') if key==r or key in (r+'_750',r+'_3000')),None)
        if role is None:return super().load(key,frames)
        function=FUNCTIONS[key.rsplit('_',1)[1]] if key!=role else 'partial'
        identity=(role,function,frames)
        if role not in self.models or self.models[role][0]!=identity:
            self.models.pop(role,None);gc.collect()
            self.models[role]=(identity,self.ct.models.CompiledMLModel(str(self.assets/(role+'.mlmodelc')),
                function_name=function,compute_units=self.ct.ComputeUnit.CPU_ONLY))
        return self.models[role][1]

    def encode_role(self,role,audio,seconds):
        from deployment.verify_encoders import rotary
        mel=self.array(self.load('log_mel',frames=len(audio)).predict({self.input_names['log_mel']:np.pad(audio,(1024,1024),mode='reflect')[None].astype(np.float32)}))
        conv=self.array(self.load(role+'_conv',frames=int(mel.shape[-1])).predict({self.input_names[role+'_conv']:mel}));frames=conv.shape[1]
        if conv.shape!=(1,frames,1024):raise ValueError('Invalid serial convolution output.')
        full=len(audio)==seconds*24000
        if full and frames!=seconds*25:raise ValueError('Canonical full-window frames changed.')
        positions=rotary(frames,self.rotary[role]);key=f'{role}_{frames}' if full else role
        from encoder_acceleration import candidate_inputs
        inputs=candidate_inputs(conv,positions,frames) if full else dict(conv_in=conv,rotary=positions)
        result=self.array(self.load(key,frames=frames).predict(inputs))
        if result.shape!=conv.shape:raise ValueError('Serial encoder output shape differs.')
        return result[0]

    def chunks(self,waveform,settings):return serial_chunks(waveform,self.encode_role,self.release,settings)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('songstruct-root','songmap-root','quality-report','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-policy',choices=PREFLIGHT_POLICIES,default='strict-feature')
    p.add_argument('--track-id',action='append');a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Prepare on the Mac with Xcode and the SongFormer environment.')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    root=a.songstruct_root;songmap=a.songmap_root;out=a.output_dir;source=Path(__file__).resolve().parent
    if out.exists() or (songmap/'outputs.local') not in out.parents:raise ValueError('Use a fresh output directory under outputs.local.')
    sys.path.insert(0,str(root))
    from finetuning.common import config
    from finetuning.engine import manifest
    from deployment.iphone_project import project_text
    import coremltools as ct
    import librosa
    cfg=config(root/'finetuning/config.json');data,provenance=manifest(cfg);torch.set_num_threads(2)
    if (cfg['features']['sample_rate'],cfg['features']['long_window_seconds'],cfg['features']['head_window_seconds'])!=(24000,120,420):raise ValueError('Window profile changed.')
    q=read(a.quality_report);selection_path=a.quality_report.parent/'selection.json';selection=read(selection_path)
    validate_canonical_quality(q,selection)
    if q['plan_id']!=data['plan_id'] or q['provenance_id']!=provenance or selection['librosa']!=librosa.__version__ or selection['coremltools']!=ct.__version__:raise ValueError('Dataset or native runtime/decoder changed.')
    rows=[r for r in data['records'] if r['approved'] and r['split']=='val'];chosen=choose_tracks(rows,a.track_id)
    if q['tracks']!=len(rows) or q['candidate']['sections']!=sum(len(r['sections']) for r in rows):raise ValueError('Validation inventory changed.')
    locked={}
    def lock(path):
        path=Path(path);locked[path]=sha_file(path);return read(path)
    lock(a.quality_report);lock(selection_path)
    reference_path=locate((songmap/'outputs.local').glob('encoder-glu-quality-*/validation-all.json'),selection['reference_quality_sha256'])
    reference=lock(reference_path);ref_sel=lock(reference_path.parent/'selection.json');validate_quality(reference,ref_sel)
    if ref_sel['identity']!=fingerprint({k:v for k,v in ref_sel.items() if k!='identity'}) or ref_sel['context_identity']!=selection['context_identity']:raise ValueError('Reference/context selection changed.')
    reports={};report_paths={};encoders={};model_hashes={}
    for role in ('musicfm','muq'):
        report_paths[role]=locate((songmap/'outputs.local').glob('encoder-canonical-bank-*/*/canonical-report.json'),selection['reports'][role])
        report=lock(report_paths[role]);reports[role]=report
        validate_candidate(report,role,selection['reference_quality_sha256'],selection['repair_sha256'])
        if report['evidence']['plan_id']!=data['plan_id'] or report['evidence']['provenance_id']!=provenance:raise ValueError('Encoder dataset identity differs.')
        package=Path(report['package']);compiled=report_paths[role].parent/'Compiled'/(role+'_Canonical.mlmodelc')
        if package_digest(package)!=report['package_digest'] or package_digest(compiled)!=report['compiled_digest']:raise ValueError('Canonical artifact changed: '+role)
        if selection['packages'][role]!=report['package_digest'] or selection['compiled'][role]!=report['compiled_digest']:raise ValueError('Canonical selection/artifact differs.')
        model_hashes[package]=report['package_digest'];model_hashes[compiled]=report['compiled_digest'];encoders[role]=compiled
    if sum(r['compiled_bytes'] for r in reports.values())!=q['artifact_sizes']['canonical_compiled_encoder_bytes']:raise ValueError('Compiled size inventory differs.')
    control_receipt_sha=None
    if a.preflight_policy=='owned-buffer-diagnostic':
        receipt_path=source/'evidence/canonical_cpu_control_v1.json'
        if sha_file(receipt_path)!=CPU_RECEIPT_SHA:raise ValueError('Reviewed CPU control receipt changed.')
        validate_owned_diagnostic_control(lock(receipt_path),sha_file(a.quality_report),selection,q)
        control_receipt_sha=CPU_RECEIPT_SHA
    def artifact(value):
        value=Path(value).expanduser();return value.resolve() if value.is_absolute() else (root/value).resolve()
    palette_path=locate((root/'outputs.local').glob('muq-compression-*/palette6-all-validation.json'),ref_sel['palette6_source_sha256'])
    palette=lock(palette_path);header_path=artifact(palette['diagnostic_cache'])/'encoder_provenance.json'
    if sha_file(header_path)!=ref_sel['baseline_header_sha256']:raise ValueError('Original frontend provenance changed.')
    header=lock(header_path);combo_path=artifact(palette['baseline_combination'])
    if sha_file(combo_path)!=palette['baseline_combination_sha256']:raise ValueError('Head combination changed.')
    combo=lock(combo_path);head=artifact(combo['head_package'])
    if package_digest(head)!=selection['head_digest'] or selection['head_digest']!=ref_sel['head_package_digest']:raise ValueError('Head changed.')
    app_root=songmap/'SongMap/Resources/MLModels/SongFormer';app_path=app_root/'songformer_coreml_manifest.json'
    if sha_file(app_path)!=selection['app_manifest_sha256'] or sha_file(app_path)!=ref_sel['app_manifest_sha256']:raise ValueError('Frontend manifest changed.')
    app=lock(app_path);models={role:app_root/app['models'][role] for role in ('log_mel','musicfm_conv','muq_conv')};models['head']=head
    for role in ('log_mel','musicfm_conv','muq_conv'):
        if package_digest(models[role])!=header['environment']['packages'][role]:raise ValueError('Frontend changed: '+role)
    context_sha=selection['context_identity']['context_checkpoint_sha256'];parities=[]
    for path in (songmap/'outputs.local').glob('context-coreml-fp32-*/parity/validation-all.json'):
        v=read(path)
        if (v.get('full_validation') and v.get('passed') and v.get('status')=='numerically_consistent'
                and v.get('context_checkpoint_sha256')==context_sha and v.get('selected_epoch')==5
                and v.get('palette_validation_sha256')==selection['context_identity']['palette_validation_sha256']):parities.append(path)
    if not parities:raise ValueError('Matching full native context parity report is required.')
    parity_path=max(parities,key=lambda p:p.stat().st_mtime);parity=lock(parity_path);context_path=parity_path.parent.parent/'export/export_report.json';ctx=lock(context_path)
    if (ctx['context_checkpoint_sha256']!=context_sha or ctx['precision']!='fp32' or not ctx['native_synthetic_parity_passed']
            or ctx['package_digest']!=parity['package_digest'] or ctx['plan_id']!=data['plan_id'] or ctx['provenance_id']!=provenance):raise ValueError('Native context identity differs.')
    models['context']=artifact(ctx['package'])
    if package_digest(models['context'])!=ctx['package_digest']:raise ValueError('Native context changed.')
    model_hashes.update({path:package_digest(path) for path in models.values()})
    if any(path==out or path in out.parents for path in [a.quality_report.parent,reference_path.parent,*model_hashes]):raise ValueError('Keep probe output outside validation/source model folders.')
    math_source=songmap/'SongMap/Data/SongAnalysis/Structure/SongFormerContextMath.swift';bridge_source=songmap/'SongMap/Data/SongAnalysis/Structure/SongFormerTensorBridge.swift'
    out.mkdir(parents=True);assets=out/'Assets';assets.mkdir()
    shutil.copyfile(source/'encoder_pipeline_iphone/BenchmarkApp.swift',out/'BenchmarkApp.swift')
    copy_source=source/'encoder_pipeline_iphone/ProbeOwnedTensorCopy.swift'
    runner=(source/'encoder_pipeline_iphone/BenchmarkRunner.swift').read_text()+'\n'+copy_source.read_text()+'\n'+math_source.read_text()+'\n'+bridge_source.read_text();(out/'BenchmarkRunner.swift').write_text(runner)
    # Execute the actual Swift buffer helper on the Mac before large model assets are staged.
    test_source=out/'OwnedTensorNativeTests.swift'
    test_source.write_text(copy_source.read_text()+'\n'+bridge_source.read_text()+
        '\n@main struct OwnedTensorNativeTests { static func main() throws { try ProbeOwnedTensorCopy.selfTest(); print("Native tensor ownership tests passed") } }\n')
    mac_sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
    executable=out/'OwnedTensorNativeTests'
    subprocess.run(['xcrun','swiftc','-parse-as-library','-sdk',mac_sdk,str(test_source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)
    sdk=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-path'],text=True).strip()
    subprocess.run(['xcrun','swiftc','-typecheck','-parse-as-library','-target','arm64-apple-ios18.0','-sdk',sdk,str(out/'BenchmarkApp.swift'),str(out/'BenchmarkRunner.swift')],check=True)
    for role,path in models.items():
        print('Compiling unchanged frontend/head/context:',role,flush=True);work=out/'CompileWork'/role;work.mkdir(parents=True)
        subprocess.run(['xcrun','coremlcompiler','compile',str(path),str(work)],check=True)
        shutil.move(str(work/(path.stem+'.mlmodelc')),assets/(role+'.mlmodelc'))
    for role,path in encoders.items():
        shutil.copytree(path,assets/(role+'.mlmodelc'))
        if package_digest(assets/(role+'.mlmodelc'))!=reports[role]['compiled_digest']:raise ValueError('Copied canonical model changed.')
    input_names={role:sorted(i.name for i in ct.models.MLModel(str(models[role]),skip_model_load=True).get_spec().description.input)[0] for role in ('log_mel','musicfm_conv','muq_conv')}
    backend=CanonicalMacPipeline(ct,assets,input_names,app['rotary_inv_freq']);tracks=[];checks=[]
    baseline_samples=dict(selection['context_identity']['samples'])
    for number,row in enumerate(chosen):
        print('Canonical actual PCM preflight:',row['track_id'],flush=True)
        if row['annotation_sha256']!=fingerprint(row['sections']) or sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Audio/annotation changed.')
        original_info=reference_path.parent/'per-track'/(row['id']+'.json')
        sample_id=fingerprint(dict(run=selection['identity'],sample=baseline_samples[row['track_id']],reference=sha_file(original_info)))
        cache=a.quality_report.parent/'per-track';array_path=cache/(row['id']+'.npz');result=cached_result(cache/(row['id']+'.json'),array_path,sample_id,read,fingerprint,sha_file)
        if result is None:raise ValueError('Verified canonical per-track result missing.')
        validate_details([row],result['sections']);intervals,indices=source_sections(read(row['source_json']),row)
        with np.load(array_path,allow_pickle=False) as arrays:
            audio=np.array(arrays['audio'],copy=True);priors=np.array(arrays['probabilities'],copy=True)
        waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True)
        try:
            engine_policy='functional-benchmark' if a.preflight_policy=='owned-buffer-diagnostic' else 'strict-feature'
            check=mac_preflight(backend,waveform,row,intervals,indices,audio,priors,result['sections'],cfg['features'],policy=engine_policy,chunk_source=backend.chunks)
            checks.append(finalize_preflight_admission(check,a.preflight_policy))
        except PreflightMismatch as error:
            write(out/'mac-preflight.json',dict(checks=checks+[error.check],passed=False,admission_policy=a.preflight_policy,ready_for_app_integration=False));raise
        write(out/'mac-preflight.json',dict(checks=checks,passed=all(c['passed'] for c in checks),admission_policy=a.preflight_policy,
            numerical_passed=all(c['numerical_passed'] for c in checks),functional_passed=all(c['functional_passed'] for c in checks),ready_for_app_integration=False))
        pcm=f'track{number}_pcm';means=f'track{number}_means';np.asarray(waveform,dtype='<f4').tofile(assets/(pcm+'.f32'));audio.astype('<f4').tofile(assets/(means+'.f32'))
        targets=[dict(index=index,label=section['label'],prediction=det['prediction'],probabilities=[det['probabilities'][label] for label in LABELS],base_probabilities=prior.tolist())
            for index,section,det,prior in zip(indices,row['sections'],result['sections'],priors)]
        tracks.append(dict(id=row['track_id'],title=Path(row['track_id']).stem,duration=row['duration'],samples=len(waveform),pcm=pcm,pcm_sha256=sha_file(assets/(pcm+'.f32')),means=means,means_sha256=sha_file(assets/(means+'.f32')),intervals=intervals,targets=targets))
    if any(sha_file(path)!=sha for path,sha in locked.items()) or any(package_digest(path)!=sha for path,sha in model_hashes.items()):raise ValueError('Evidence/source model changed during preparation.')
    inventory={path.stem:file_hashes(path) for path in assets.glob('*.mlmodelc')}
    if set(inventory)!={'log_mel','musicfm_conv','muq_conv','head','context','musicfm','muq'}:raise ValueError('Unexpected duplicated model resources.')
    manifest_data=dict(version='encoder-pipeline-phone-v2',quality_sha256=sha_file(a.quality_report),static_graph_policy='split-before-conv-v2',frontend_route_probe=False,frontend_gpu_policy='disabled',
        graph_repair_sha256=selection['repair_sha256'],mac_preflight_policy=a.preflight_policy,mac_feature_parity_passed=all(c['audio_features']['passed'] for c in checks),
        cpu_control_receipt_sha256=control_receipt_sha,
        canonical_encoder_policy=POLICY,canonical_encoder_functions={r:FUNCTIONS for r in reports},canonical_encoder_reports=selection['reports'],compiled_encoder_bytes=sum(r['compiled_bytes'] for r in reports.values()),
        canonical_encoder_schedule=SCHEDULE,
        tensor_transfer_policy=TRANSFER,
        probe_source_sha256=fingerprint(dict(runner=sha_file(out/'BenchmarkRunner.swift'),view=sha_file(out/'BenchmarkApp.swift'))),graph_states={r:v['evidence']['source_state_sha256'] for r,v in reports.items()},
        base_checkpoint=palette['checkpoint_sha256'],context_checkpoint=context_sha,rotary=app['rotary_inv_freq'],files=inventory,tracks=tracks)
    write(assets/'benchmark_manifest.json',manifest_data);project=out/'Benchmark.xcodeproj';project.mkdir()
    (project/'project.pbxproj').write_text(project_text('com.danendra.CanonicalOwnedPipelineProbe',resource_names=[p.name for p in assets.iterdir()]).replace(
        'GENERATE_INFOPLIST_FILE = YES;','GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_CFBundleDisplayName = "Canonical Owned";'))
    write(out/'preparation.json',dict(protocol='canonical-encoder-phone-preparation-v1',quality_sha256=sha_file(a.quality_report),manifest_sha256=sha_file(assets/'benchmark_manifest.json'),
        preparation_script_sha256=sha_file(Path(__file__)),runner_sha256=sha_file(out/'BenchmarkRunner.swift'),view_sha256=sha_file(out/'BenchmarkApp.swift'),app_math_sha256=sha_file(math_source),app_bridge_sha256=sha_file(bridge_source),
        native_typecheck_passed=True,mac_pcm_pipeline_passed=True,checks=checks,canonical_encoder_policy=POLICY,compiled_encoder_bytes=manifest_data['compiled_encoder_bytes'],
        mac_preflight_policy=a.preflight_policy,mac_pcm_functional_passed=all(c['functional_passed'] for c in checks),mac_pcm_feature_parity_passed=all(c['audio_features']['passed'] for c in checks),
        cpu_control_receipt_sha256=control_receipt_sha,
        canonical_encoder_schedule=SCHEDULE,
        tensor_transfer_policy=TRANSFER,native_owned_tensor_self_test_passed=True,owned_tensor_source_sha256=sha_file(copy_source),
        bundle_identifier='com.danendra.CanonicalOwnedPipelineProbe',
        temporary_model_bytes=sum(p.stat().st_size for folder in assets.glob('*.mlmodelc') for p in folder.rglob('*') if p.is_file()),ready_for_app_integration=False,
        notes=['Exactly one compiled multi-function package per encoder; no separate static or old partial model assets.',
            'Serial role phases preserve each window and final stream order; lower native peak memory and GPU parity are not established until the phone runs.',
            'First run CPU on one track, then CPU+GPU on that same track. Partial functions and all frontends remain CPU.',
            'New native multifunction GPU stability, memory and latency remain unverified until the phone runs; sampled footprint is not peak memory.']))
    print('Prepared:',project,flush=True)


if __name__=='__main__':main()
