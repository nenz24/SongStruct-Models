"""Export original MusicFM hidden-state 10 with FP32 compute and explicit rotary input."""
import argparse
import inspect
from pathlib import Path
import platform
import numpy as np
import torch
from deployment.coreml_features import native_input
from deployment.diagnose_musicfm import invalid_frames
from deployment.verify_encoders import compare_embeddings
from finetuning.common import config, fingerprint, read, sha_file, upstream, write


class MusicFMEncoderL10(torch.nn.Module):
    """Hidden_states[10] is after layers 0..9, before final encoder layer norm."""
    def __init__(self, encoder):
        super().__init__()
        if len(encoder.layers) < 10:
            raise ValueError('MusicFM encoder needs at least ten layers.')
        self.dropout=encoder.dropout
        self.layers=torch.nn.ModuleList(list(encoder.layers[:10]))
        for layer in self.layers:
            if 'relative_position_embeddings' not in inspect.signature(layer.forward).parameters:
                raise ValueError('Unsupported transformers layer API; relative_position_embeddings required.')

    def forward(self, conv_in, rotary):
        values=self.dropout(conv_in)
        for layer in self.layers:
            values=layer(values,attention_mask=None,relative_position_embeddings=rotary,
                         output_attentions=False)[0]
        return values


def numerical(reference,candidate):
    reference,candidate=np.asarray(reference),np.asarray(candidate)
    result={'reference':invalid_frames(reference),'candidate':invalid_frames(candidate)}
    if reference.shape==candidate.shape:
        result.update(compare_embeddings(reference,candidate))
    else:
        result.update(numerical_parity_passed=False,reason='shape_mismatch')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir',type=Path,required=True)
    parser.add_argument('--songmap-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config')
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Run on macOS with coremltools 8.x.')
    import coremltools as ct
    cfg=config(args.config)
    diagnostic,out=[p.expanduser().resolve() for p in (args.diagnostic_dir,args.output_dir)]
    if out.exists():
        raise ValueError('Choose a new output directory.')
    recorded=read(diagnostic/'diagnostic.json')
    audio=np.load(diagnostic/'audio_window.npy',allow_pickle=False)
    inputs={name:native_input(np.load(diagnostic/f'{name}.npy',allow_pickle=False)) for name in ('conv_in','rotary')}
    for name in inputs:
        if sha_file(diagnostic/f'{name}.npy')!=recorded['input_file_hashes'][name]:
            raise ValueError('Saved diagnostic inputs changed.')
    if (audio.ndim!=1 or not np.isfinite(audio).all() or
            len(audio)!=recorded['end_sample']-recorded['start_sample']):
        raise ValueError('Unexpected diagnostic audio window.')
    root=args.songmap_root.expanduser().resolve()/'SongMap/Resources/MLModels/SongFormer'
    app=read(root/'songformer_coreml_manifest.json')
    for role in ('log_mel','musicfm_conv','musicfm'):
        old_package=root/app['models'][role]
        digest=fingerprint([(str(p.relative_to(old_package)),sha_file(p))
            for p in sorted(old_package.rglob('*')) if p.is_file()])
        if digest!=recorded['source_package_hashes'][role]:
            raise ValueError(f'Original {role} package changed since exact-window diagnosis.')
    provenance=read(Path(cfg['work_dir'])/'encoder_provenance.json')
    folder=upstream(cfg)
    weights=folder/'ckpts/MusicFM/pretrained_msd.pt'
    stats=folder/'ckpts/MusicFM/msd_stats.json'
    if sha_file(weights)!=provenance['musicfm_sha256'] or sha_file(stats)!=provenance['stats_sha256']:
        raise ValueError('Original MusicFM weights/stats do not match training provenance.')
    import transformers
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    torch.set_num_threads(2)
    fm=MusicFM25Hz(is_flash=False,stat_path=str(stats),model_path=str(weights)).cpu().float().eval().requires_grad_(False)
    if fm.conformer.config.position_embeddings_type!='rotary':
        raise ValueError('Expected original rotary MusicFM architecture.')
    wrapper=MusicFMEncoderL10(fm.conformer).eval().requires_grad_(False)
    out.mkdir(parents=True,exist_ok=False)
    report={'diagnostic_dir':str(diagnostic),'track_id':recorded['track_id'],
        'start_sample':recorded['start_sample'],'end_sample':recorded['end_sample'],
        'musicfm_weights_sha256':provenance['musicfm_sha256'],'musicfm_stats_sha256':provenance['stats_sha256'],
        'torch':torch.__version__,'transformers':transformers.__version__,'coremltools':ct.__version__,
        'ready_for_app_integration':False,'compute_precision':'float32'}
    def persist(): write(out/'repair_report.json',report)
    conv=torch.from_numpy(inputs['conv_in'])
    app_rotary=torch.from_numpy(inputs['rotary'])
    print('Comparing native MusicFM paths and exact saved inputs...',flush=True)
    with torch.inference_mode():
        normalized=fm.normalize(fm.preprocessing(torch.from_numpy(audio.copy())[None],features=['melspec_2048']))
        native_conv=fm.conv(normalized['melspec_2048'])
        source_rotary=fm.conformer.embed_positions(conv).clone()
        native_reference=fm.conformer(conv,output_hidden_states=True)['hidden_states'][10]
        wrapped_source=wrapper(conv,source_rotary)
        report['wrapper_vs_native_encoder']=numerical(native_reference.numpy(),wrapped_source.numpy())
        reference_app_inputs=wrapper(conv,app_rotary).numpy()
        source_inputs=wrapper(conv,source_rotary).numpy()
        native_audio_rotary=fm.conformer.embed_positions(native_conv).clone()
        native_audio_reference=wrapper(native_conv,native_audio_rotary).numpy()
        report['frontend_conv_app_vs_original']=numerical(native_conv.numpy(),conv.numpy())
        report['rotary_effect_same_conv']=numerical(source_inputs,reference_app_inputs)
        report['combined_frontend_and_rotary_effect']=numerical(native_audio_reference,reference_app_inputs)
    np.save(out/'pytorch_same_app_inputs.npy',reference_app_inputs,allow_pickle=False)
    persist()
    if not report['wrapper_vs_native_encoder']['numerical_parity_passed']:
        raise RuntimeError('L10 wrapper differs from native transformers; see repair_report.json.')
    # Diagnose original Core ML encoder on EXACTLY the same conv/rotary inputs.
    old=ct.models.MLModel(str(root/app['models']['musicfm']),compute_units=ct.ComputeUnit.CPU_ONLY)
    old_result=old.predict({name:native_input(value) for name,value in inputs.items()})
    old_arrays=[np.asarray(old_result[k]) for k in sorted(old_result) if isinstance(old_result[k],np.ndarray)]
    if len(old_arrays)!=1:
        raise ValueError('Unexpected original MusicFM output contract.')
    report['old_coreml_vs_torch_same_inputs']=numerical(reference_app_inputs,old_arrays[0])
    del old,old_result,old_arrays
    persist()
    # Supplied rotary makes this trace independent of HF's cached positional state.
    trace_frames=93
    trace_conv=torch.from_numpy(inputs['conv_in'][:,:trace_frames].copy())
    trace_rotary=torch.from_numpy(inputs['rotary'][:,:trace_frames].copy())
    traced=torch.jit.trace(wrapper,(trace_conv,trace_rotary),strict=True,check_trace=False)
    with torch.inference_mode():
        trace_result=traced(conv,app_rotary).numpy()
    report['trace_exact_window_parity']=numerical(reference_app_inputs,trace_result)
    persist()
    if not report['trace_exact_window_parity']['numerical_parity_passed']:
        raise RuntimeError('Trace failed long-window parity; see repair_report.json.')
    print('Converting MusicFM layer-10 encoder to Core ML FP32...',flush=True)
    frames=ct.RangeDim(lower_bound=1,upper_bound=3000,default=750)
    candidate=ct.convert(traced,convert_to='mlprogram',
        inputs=[ct.TensorType(name='conv_in',shape=(1,frames,1024),dtype=np.float32),
                ct.TensorType(name='rotary',shape=(2,frames,1,1,64),dtype=np.float32)],
        outputs=[ct.TensorType(name='embedding',dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY)
    candidate.user_defined_metadata['precision']='fp32'
    candidate.user_defined_metadata['musicfm_weights_sha256']=provenance['musicfm_sha256']
    candidate.user_defined_metadata['musicfm_stats_sha256']=provenance['stats_sha256']
    candidate.user_defined_metadata['hidden_state_index']='10'
    package=out/'MusicFM_L10_FP32.mlpackage'
    candidate.save(str(package))
    repaired=candidate.predict({name:native_input(value) for name,value in inputs.items()})['embedding']
    report['fp32_coreml_vs_torch_same_inputs']=numerical(reference_app_inputs,repaired)
    source_fixed=candidate.predict({'conv_in':native_input(conv.numpy()),'rotary':native_input(source_rotary.numpy())})['embedding']
    report['fp32_source_rotary_vs_torch']=numerical(source_inputs,source_fixed)
    audio_fixed=candidate.predict({'conv_in':native_input(native_conv.numpy()),'rotary':native_input(native_audio_rotary.numpy())})['embedding']
    report['fp32_native_frontend_vs_torch']=numerical(native_audio_reference,audio_fixed)
    report['source_rotary_inv_freq']=fm.conformer.embed_positions.inv_freq.detach().cpu().tolist()
    report['package']=str(package)
    report['exact_window_passed']=all(report[key]['numerical_parity_passed'] for key in (
        'fp32_coreml_vs_torch_same_inputs','fp32_source_rotary_vs_torch','fp32_native_frontend_vs_torch'))
    report['note']='Exact failed-window checks only. Frontend/rotary effects are reported separately; full validation still required.'
    persist()
    if not report['exact_window_passed']:
        raise RuntimeError('FP32 candidate still fails exact-window parity; inspect repair_report.json.')
    print('Exact-window FP32 checks passed:',out/'repair_report.json')


if __name__=='__main__':
    main()
