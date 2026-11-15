"""Export MuQ hidden-state 10 from the exact cached training snapshot, with explicit FP32 compute."""
import argparse
from pathlib import Path
import platform
import numpy as np
import torch
from deployment.coreml_features import native_input
from deployment.export_musicfm_fp32 import MusicFMEncoderL10, numerical
from deployment.verify_encoders import rotary
from finetuning.common import config, fingerprint, read, sha_file, write
from finetuning.engine import manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songmap-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config')
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Run on macOS with the SongFormer Python environment.')
    import coremltools as ct
    import librosa
    from huggingface_hub import snapshot_download
    from muq import MuQ
    cfg=config(args.config)
    out=args.output_dir.expanduser().resolve()
    root=args.songmap_root.expanduser().resolve()/'SongMap/Resources/MLModels/SongFormer'
    if out.exists():
        raise ValueError('Choose a new output directory.')
    app=read(root/'songformer_coreml_manifest.json')
    if (app['sample_rate'] != 24000 or app['encoder_layer'] != 10
            or cfg['features']['long_window_seconds'] != 120):
        raise ValueError('Expected the validated 24 kHz, layer-10, 30/120-second profile.')
    data,source_id=manifest(cfg)
    provenance=read(Path(cfg['work_dir'])/'encoder_provenance.json')
    if provenance['muq_repo']!=cfg['features']['muq_repo']:
        raise ValueError('MuQ source repository differs from frozen cache provenance.')
    # Do not silently fetch a newer model or use a different snapshot.
    snapshot=Path(snapshot_download(repo_id=provenance['muq_repo'],revision=provenance['muq_revision'],local_files_only=True))
    if snapshot.name!=provenance['muq_revision']:
        raise ValueError('Local MuQ snapshot commit differs from training.')
    row=next(r for r in data['records'] if r['approved'] and r['split']=='val')
    if sha_file(row['audio_path'])!=row['audio_sha256']:
        raise ValueError('Audio changed since prepare.')
    waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True,duration=120)
    audio=np.array(waveform[:120*24000],dtype=np.float32,copy=True)
    if len(audio)<=2048:
        raise ValueError('Audio too short.')
    torch.set_num_threads(2)
    print('Loading original MuQ snapshot:',snapshot.name,flush=True)
    muq=MuQ.from_pretrained(str(snapshot)).cpu().float().eval().requires_grad_(False)
    encoder=muq.model.conformer
    if encoder.config.position_embeddings_type!='rotary':
        raise ValueError('Expected rotary MuQ conformer.')
    wrapper=MusicFMEncoderL10(encoder).eval().requires_grad_(False)
    out.mkdir(parents=True,exist_ok=False)
    report={'muq_repo':provenance['muq_repo'],'muq_revision':provenance['muq_revision'],
        'muq_snapshot_digest':fingerprint([(str(p.relative_to(snapshot)),sha_file(p)) for p in sorted(snapshot.rglob('*')) if p.is_file()]),
        'plan_id':data['plan_id'],'provenance_id':source_id,'track_id':row['track_id'],
        'compute_precision':'float32','ready_for_app_integration':False,
        'source_kind':'original_training_snapshot','coremltools':ct.__version__,'torch':torch.__version__,
        'app_manifest_sha256':sha_file(root/'songformer_coreml_manifest.json'),
        'frontend_packages':{role:fingerprint([(str(p.relative_to(root/app['models'][role])),sha_file(p))
            for p in sorted((root/app['models'][role]).rglob('*')) if p.is_file()])
            for role in ('log_mel','muq_conv')}}
    def persist():write(out/'export_report.json',report)
    def first_input(model):return sorted(f.name for f in model.get_spec().description.input)[0]
    def one_array(outputs):
        arrays=[np.asarray(outputs[k]) for k in sorted(outputs) if isinstance(outputs[k],np.ndarray)]
        if len(arrays)!=1 or not np.isfinite(arrays[0]).all():
            raise ValueError('App frontend output invalid.')
        return arrays[0]
    logmel=ct.models.MLModel(str(root/app['models']['log_mel']),compute_units=ct.ComputeUnit.CPU_ONLY)
    mel=one_array(logmel.predict({first_input(logmel):native_input(np.pad(audio,(1024,1024),mode='reflect')[None])}))
    conv_model=ct.models.MLModel(str(root/app['models']['muq_conv']),compute_units=ct.ComputeUnit.CPU_ONLY)
    app_conv=native_input(one_array(conv_model.predict({first_input(conv_model):native_input(mel)})))
    if app_conv.ndim!=3 or app_conv.shape[0]!=1 or app_conv.shape[-1]!=1024:
        raise ValueError('Unexpected app MuQ convolution shape.')
    app_rotary=rotary(app_conv.shape[1],app['rotary_inv_freq']['muq'])
    del logmel,conv_model,mel
    with torch.inference_mode():
        normalized=muq.model.normalize(muq.model.preprocessing(torch.from_numpy(audio.copy())[None],features=['melspec_2048']))
        native_conv=muq.model.conv(normalized['melspec_2048'])
        source_rotary=encoder.embed_positions(torch.from_numpy(app_conv)).clone()
        native_reference=encoder(torch.from_numpy(app_conv),output_hidden_states=True)['hidden_states'][10]
        wrapped=wrapper(torch.from_numpy(app_conv),source_rotary)
        report['wrapper_vs_native_encoder']=numerical(native_reference.numpy(),wrapped.numpy())
        report['frontend_conv_app_vs_original']=numerical(native_conv.numpy(),app_conv)
        native_audio_rotary=encoder.embed_positions(native_conv).clone()
        native_audio_reference=wrapper(native_conv,native_audio_rotary).numpy()
        app_reference=wrapper(torch.from_numpy(app_conv),torch.from_numpy(app_rotary)).numpy()
        report['combined_frontend_rotary_effect']=numerical(native_audio_reference,app_reference)
    np.save(out/'conv_in.npy',app_conv,allow_pickle=False)
    np.save(out/'rotary.npy',app_rotary,allow_pickle=False)
    np.save(out/'pytorch_reference.npy',app_reference,allow_pickle=False)
    report['fixture_hashes']={k:sha_file(out/f'{k}.npy') for k in ('conv_in','rotary')}
    persist()
    if not report['wrapper_vs_native_encoder']['numerical_parity_passed']:
        raise RuntimeError('MuQ wrapper does not match native hidden-state 10.')
    trace_frames=min(93,app_conv.shape[1])
    traced=torch.jit.trace(wrapper,(torch.from_numpy(app_conv[:,:trace_frames].copy()),
        torch.from_numpy(app_rotary[:,:trace_frames].copy())),strict=True,check_trace=False)
    with torch.inference_mode():
        report['trace_long_window']=numerical(app_reference,traced(torch.from_numpy(app_conv),torch.from_numpy(app_rotary)).numpy())
    persist()
    if not report['trace_long_window']['numerical_parity_passed']:
        raise RuntimeError('MuQ variable-length trace failed.')
    if app_conv.shape[1] > 3000:
        raise ValueError('MuQ fixture exceeds the supported 120-second input range.')
    frames=ct.RangeDim(lower_bound=1,upper_bound=3000,default=750)
    print('Converting original MuQ L10 to FP32...',flush=True)
    model=ct.convert(traced,convert_to='mlprogram',
        inputs=[ct.TensorType(name='conv_in',shape=(1,frames,1024),dtype=np.float32),
                ct.TensorType(name='rotary',shape=(2,frames,1,1,64),dtype=np.float32)],
        outputs=[ct.TensorType(name='embedding',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,compute_units=ct.ComputeUnit.CPU_ONLY)
    for name in ('muq_repo','muq_revision','muq_snapshot_digest'):
        model.user_defined_metadata[name]=report[name]
    model.user_defined_metadata['precision']='fp32'
    model.user_defined_metadata['hidden_state_index']='10'
    package=out/'MuQ_L10_FP32.mlpackage'
    model.save(str(package))
    checks=[]
    for count in sorted(set((min(750,app_conv.shape[1]),app_conv.shape[1]))):
        inputs={'conv_in':native_input(app_conv[:,:count]),'rotary':native_input(app_rotary[:,:count])}
        with torch.inference_mode():
            expected=wrapper(torch.from_numpy(inputs['conv_in']),torch.from_numpy(inputs['rotary'])).numpy()
        actual=model.predict(inputs)['embedding']
        checks.append({'frames':count,**numerical(expected,actual)})
    report.update(package=str(package),checks=checks,
                  source_window_passed=all(c['numerical_parity_passed'] for c in checks),
                  source_rotary_inv_freq=encoder.embed_positions.inv_freq.detach().cpu().tolist(),
                  note='App frontend/rotary retained for controlled evaluation. Short/long real-input checks only; full validation required.')
    persist()
    if not report['source_window_passed']:
        raise RuntimeError('MuQ FP32 real-input parity failed; inspect export_report.json.')
    print('MuQ FP32 export passed:',out/'export_report.json')


if __name__=='__main__':
    main()
