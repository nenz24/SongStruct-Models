"""Replay an exact failed MusicFM window, preserving inputs and NaN locations for diagnosis."""
import argparse
from collections import Counter
import gc
from pathlib import Path
import platform
import time
import numpy as np
from deployment.coreml_features import native_input, array_summary
from deployment.verify_encoders import rotary, compare_embeddings
from finetuning.common import config, fingerprint, read, sha_file, upstream, write
from finetuning.engine import manifest


def invalid_frames(values):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != 1024:
        raise ValueError(f'Unexpected MusicFM output shape: {values.shape}')
    counts = (~np.isfinite(values[0])).sum(axis=-1)
    bad = np.flatnonzero(counts)
    return {**array_summary(values), 'bad_frame_count': len(bad),
            'bad_frame_indices': bad.tolist(),
            'invalid_channels_per_bad_frame': counts[bad].tolist(),
            'fully_invalid_frame_indices': np.flatnonzero(counts == 1024).tolist()}


def repeat_predictions(model, inputs, repeats, output_dir, prefix='same_instance'):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    results=[]
    for index in range(repeats):
        # Copy the exact saved inputs for every call, avoiding mmap/stride bridge issues.
        start=time.perf_counter()
        outputs=model.predict({k:native_input(v) for k,v in inputs.items()})
        arrays=[np.asarray(outputs[k]) for k in sorted(outputs) if isinstance(outputs[k],np.ndarray)]
        if len(arrays)!=1:
            raise ValueError('Expected one MusicFM output array.')
        elapsed=time.perf_counter()-start
        values=np.array(arrays[0],copy=True)
        name=f'{prefix}_{index+1}.npy'
        np.save(output_dir/name,values,allow_pickle=False)
        stats=invalid_frames(values)
        result=dict(call=index+1,seconds=elapsed,output_file=name,**stats)
        results.append(result)
        print(f'{prefix} call {index+1}: NaN={stats["nan_count"]}, Inf={stats["inf_count"]}, bad frames={stats["bad_frame_count"]}',flush=True)
    return results


def precision_inventory(spec):
    """Count declared output tensor dtypes, not parameter counts or runtime placement."""
    counts=Counter()
    def visit(block):
        for op in block.operations:
            for output in op.outputs:
                if output.type.HasField('tensorType'):
                    tensor=output.type.tensorType
                    enum=tensor.DESCRIPTOR.fields_by_name['dataType'].enum_type
                    counts[enum.values_by_number[tensor.dataType].name]+=1
            for inner in op.blocks:
                visit(inner)
    try:
        for function in spec.mlProgram.functions.values():
            for block in function.block_specializations.values():
                visit(block)
        return {'declared_tensor_dtype_counts':dict(counts),
                'note':'Declared MIL variable output types; not proof of kernel execution precision.'}
    except (AttributeError,KeyError,ValueError) as exc:
        return {'inspection_error':str(exc)}


def exact_window(waveform, failure):
    start,end=failure['start_sample'],failure['end_sample']
    if (not isinstance(start,int) or not isinstance(end,int) or start < 0 or end <= start
            or end > len(waveform) or failure['sample_rate'] != 24000):
        raise ValueError('Failure sample coordinates do not match decoded audio.')
    result=np.array(waveform[start:end],dtype=np.float32,copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Decoded audio contains NaN/Inf.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure',type=Path,default=Path('outputs.local/coreml-encoder-cache/last_encoder_failure.json'))
    parser.add_argument('--songmap-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--skip-pytorch',action='store_true',help='Core ML replay only; default also runs original PyTorch MusicFM on CPU')
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Run on macOS with coremltools and the SongFormer Python environment.')
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    import coremltools as ct
    import librosa
    cfg=config(args.config)
    out,failure_path=[p.expanduser().resolve() for p in (args.output_dir,args.failure)]
    if out.exists():
        raise ValueError('Choose a new output directory.')
    failure=read(failure_path)
    if failure['component']!='musicfm':
        raise ValueError('This reproducer targets a MusicFM encoder failure.')
    data,provenance=manifest(cfg)
    row=next((r for r in data['records'] if r['track_id']==failure['track_id']),None)
    if not row or row['split']!='val':
        raise ValueError('Failed track must belong to the frozen validation split.')
    if sha_file(row['audio_path'])!=row['audio_sha256'] or row['audio_sha256']!=failure['audio_sha256']:
        raise ValueError('Audio changed since failure recording.')
    root=args.songmap_root.expanduser().resolve()/'SongMap/Resources/MLModels/SongFormer'
    app=read(root/'songformer_coreml_manifest.json')
    recorded=read(failure_path.parent/'encoder_provenance.json')
    if recorded['id']!=failure['candidate_encoder_id']:
        raise ValueError('Failure and recorded encoder provenance do not match.')
    source_hashes={}
    for role in ('log_mel','musicfm_conv','musicfm'):
        package=root/app['models'][role]
        source_hashes[role]=fingerprint([(str(p.relative_to(package)),sha_file(p))
            for p in sorted(package.rglob('*')) if p.is_file()])
        if source_hashes[role]!=recorded['packages'][role]:
            raise ValueError(f'{role} package changed since failure.')
    if sha_file(root/'songformer_coreml_manifest.json')!=recorded['app_manifest_sha256']:
        raise ValueError('App manifest changed since failure.')
    waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True)
    audio=exact_window(waveform,failure)
    preceding_start=max(failure.get('head_offset_samples',0),
                        failure['start_sample']-failure['stream_window_seconds']*24000)
    preceding_audio=(np.array(waveform[preceding_start:failure['start_sample']],dtype=np.float32,copy=True)
                     if preceding_start<failure['start_sample'] else None)
    del waveform
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'audio_window.npy',audio,allow_pickle=False)
    def load(role):
        return ct.models.MLModel(str(root/app['models'][role]),compute_units=ct.ComputeUnit.CPU_ONLY)
    def first_input(model):
        return sorted(v.name for v in model.get_spec().description.input)[0]
    def one_array(outputs):
        arrays=[np.asarray(outputs[k]) for k in sorted(outputs) if isinstance(outputs[k],np.ndarray)]
        if len(arrays)!=1 or not np.isfinite(arrays[0]).all():
            raise ValueError('Pre-encoder component failed or produced NaN/Inf.')
        return arrays[0]
    mel_model=load('log_mel')
    mel=one_array(mel_model.predict({first_input(mel_model):native_input(np.pad(audio,(1024,1024),mode='reflect')[None])}))
    del mel_model
    conv_model=load('musicfm_conv')
    convolved=one_array(conv_model.predict({first_input(conv_model):native_input(mel)}))
    del conv_model,mel
    if convolved.ndim!=3 or convolved.shape[0]!=1 or convolved.shape[-1]!=1024:
        raise ValueError('Unexpected MusicFM conv shape.')
    inputs={'conv_in':native_input(convolved),
            'rotary':rotary(convolved.shape[1],app['rotary_inv_freq']['musicfm'])}
    del convolved
    for name,value in inputs.items():
        np.save(out/f'{name}.npy',value,allow_pickle=False)
    report={'track_id':row['track_id'],'failure_report':str(failure_path),
        'start_sample':failure['start_sample'],'end_sample':failure['end_sample'],'sample_rate':24000,
        'audio_summary':array_summary(audio),'inputs':{k:array_summary(v) for k,v in inputs.items()},
        'input_file_hashes':{k:sha_file(out/f'{k}.npy') for k in inputs},
        'provenance_id':provenance,'compute_units':'CPU_ONLY','ready_for_app_integration':False,
        'coremltools':ct.__version__,'librosa':librosa.__version__,'platform':platform.platform(),
        'source_package_hashes':source_hashes,
        'note':'Exact-window diagnosis; no clipping, normalization, NaN replacement or pipeline fallback.'}
    model=load('musicfm')
    report['coreml_precision']=precision_inventory(model.get_spec())
    write(out/'diagnostic.json',report)
    report['same_instance']=repeat_predictions(model,inputs,args.repeats,out)
    del model
    gc.collect()
    model=load('musicfm')
    report['fresh_instance']=repeat_predictions(model,inputs,1,out,prefix='fresh_instance')
    del model
    gc.collect()
    report['same_instance_bad_masks_identical']=all(
        np.array_equal(~np.isfinite(np.load(out/report['same_instance'][0]['output_file'])),
                       ~np.isfinite(np.load(out/item['output_file']))) for item in report['same_instance'][1:])
    first_output=np.load(out/report['same_instance'][0]['output_file'],allow_pickle=False)
    report['same_instance_outputs_identical']=all(np.array_equal(first_output,
        np.load(out/item['output_file'],allow_pickle=False),equal_nan=True) for item in report['same_instance'][1:])
    report['fresh_instance_bad_mask_matches']=bool(np.array_equal(~np.isfinite(first_output),
        ~np.isfinite(np.load(out/report['fresh_instance'][0]['output_file'],allow_pickle=False))))
    del first_output
    write(out/'diagnostic.json',report)
    if preceding_audio is not None and len(preceding_audio)>2048:
        print('Checking preceding-window -> failing-window transition...',flush=True)
        mel_model=load('log_mel')
        preceding_mel=one_array(mel_model.predict({first_input(mel_model):native_input(
            np.pad(preceding_audio,(1024,1024),mode='reflect')[None])}))
        del mel_model,preceding_audio
        conv_model=load('musicfm_conv')
        preceding_conv=one_array(conv_model.predict({first_input(conv_model):native_input(preceding_mel)}))
        del conv_model,preceding_mel
        preceding_inputs={'conv_in':native_input(preceding_conv),
            'rotary':rotary(preceding_conv.shape[1],app['rotary_inv_freq']['musicfm'])}
        del preceding_conv
        model=load('musicfm')
        report['preceding_window']={'start_sample':preceding_start,'end_sample':failure['start_sample'],
            'inputs':{k:array_summary(v) for k,v in preceding_inputs.items()},
            'calls':repeat_predictions(model,preceding_inputs,1,out,prefix='preceding_window')}
        report['after_preceding_window']=repeat_predictions(model,inputs,1,out,prefix='after_preceding_window')
        del model,preceding_inputs
        gc.collect()
        write(out/'diagnostic.json',report)
    if not args.skip_pytorch:
        import torch
        torch.set_num_threads(2)
        provenance_record=read(Path(cfg['work_dir'])/'encoder_provenance.json')
        folder=upstream(cfg)
        weights=folder/'ckpts/MusicFM/pretrained_msd.pt'
        stats=folder/'ckpts/MusicFM/msd_stats.json'
        if (sha_file(weights)!=provenance_record['musicfm_sha256'] or
                sha_file(stats)!=provenance_record['stats_sha256']):
            raise ValueError('Original MusicFM weights/stats differ from training cache provenance.')
        from musicfm.model.musicfm_25hz import MusicFM25Hz
        print('Running original PyTorch MusicFM on CPU for exact audio window...',flush=True)
        fm=MusicFM25Hz(is_flash=False,stat_path=str(stats),model_path=str(weights)).cpu().float().eval()
        started=time.perf_counter()
        with torch.inference_mode():
            reference=fm.get_predictions(torch.from_numpy(audio.copy()).unsqueeze(0))[1][10].float().cpu().numpy()
        np.save(out/'pytorch_musicfm.npy',reference,allow_pickle=False)
        report['pytorch']={'seconds':time.perf_counter()-started,**invalid_frames(reference)}
        comparisons=[]
        for item in report['same_instance']+report['fresh_instance']+report.get('after_preceding_window',[]):
            candidate=np.load(out/item['output_file'],allow_pickle=False)
            if candidate.shape!=reference.shape:
                comparisons.append({'output_file':item['output_file'],'shapes_match':False})
                continue
            finite_frames=np.isfinite(candidate[0]).all(-1)&np.isfinite(reference[0]).all(-1)
            result={'output_file':item['output_file'],'finite_frames_compared':int(finite_frames.sum()),
                    'invalid_frames_excluded_from_similarity_only':True}
            if finite_frames.any():
                result.update(compare_embeddings(reference[0,finite_frames],candidate[0,finite_frames]))
            comparisons.append(result)
        report['finite_subset_comparisons']=comparisons
        report['pytorch_device']='cpu'
        write(out/'diagnostic.json',report)
    print('Diagnostic:',out/'diagnostic.json')
    print('Original cache, models and annotations were not modified.')


if __name__=='__main__':
    main()
