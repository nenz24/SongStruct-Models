#!/usr/bin/env python3
"""Export an isolated, same-weight encoder graph candidate; no training or app routing changes."""
import argparse
import copy
import gc
import hashlib
import inspect
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn

PROTOCOL = 'encoder-rank3-explicit-glu-v1'
ATOL = RTOL = 1e-4


class Rank3RotaryAttention(nn.Module):
    """Batch-one HF Conformer attention, with rotation BEFORE q/k projections as in the source."""
    def __init__(self, source):
        super().__init__()
        if (source.position_embeddings_type != 'rotary' or source.head_size % 2
                or source.linear_q.in_features != source.num_heads * source.head_size):
            raise ValueError('Only the original rotary attention architecture is supported.')
        self.num_heads, self.head_size = source.num_heads, source.head_size
        self.width = self.num_heads * self.head_size
        for name in ('linear_q','linear_k','linear_v','linear_out','dropout'):
            setattr(self, name, getattr(source, name))

    def forward(self, hidden_states, attention_mask=None, relative_position_embeddings=None, output_attentions=False):
        if attention_mask is not None: raise ValueError('This batch-one encoder is unmasked.')
        # Public candidate inputs are rank two cos/sin; stack is rank three, not the old rank-five input.
        n = hidden_states.shape[1]
        values = hidden_states[0].reshape(n, self.num_heads, self.head_size)
        left, right = values[..., :self.head_size//2], values[..., self.head_size//2:]
        rotated = torch.cat((-right, left), dim=-1)
        cos, sin = relative_position_embeddings[0].unsqueeze(1), relative_position_embeddings[1].unsqueeze(1)
        rotated = (values*cos + rotated*sin).reshape(1, n, self.width)
        q = self.linear_q(rotated)[0].reshape(n, self.num_heads, self.head_size).permute(1,0,2)
        k = self.linear_k(rotated)[0].reshape(n, self.num_heads, self.head_size).permute(1,0,2)
        v = self.linear_v(hidden_states)[0].reshape(n, self.num_heads, self.head_size).permute(1,0,2)
        probabilities = self.dropout(torch.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(self.head_size),dim=-1))
        attended = torch.bmm(probabilities,v).permute(1,0,2).reshape(1,n,self.width)
        return self.linear_out(attended), probabilities.unsqueeze(0)


class ExplicitGLUConvolution(nn.Module):
    """Keep all original convolution/norm/activation weights; expose the two GLU branches explicitly."""
    def __init__(self, source):
        super().__init__()
        if source.glu.dim != 1 or source.pointwise_conv1.out_channels != 2*source.pointwise_conv1.in_channels:
            raise ValueError('Unexpected conformer GLU contract.')
        self.width=source.pointwise_conv1.in_channels
        for name in ('layer_norm','pointwise_conv1','depthwise_conv','batch_norm','activation','pointwise_conv2','dropout'):
            setattr(self,name,getattr(source,name))

    def forward(self, hidden_states):
        values=self.pointwise_conv1(self.layer_norm(hidden_states).transpose(1,2))
        values=values[:,:self.width,:]*torch.sigmoid(values[:,self.width:,:])
        values=self.activation(self.batch_norm(self.depthwise_conv(values)))
        return self.dropout(self.pointwise_conv2(values)).transpose(1,2)


class ReferenceL10(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        if len(encoder.layers)<10: raise ValueError('At least ten conformer layers are required.')
        self.dropout=encoder.dropout
        self.layers=nn.ModuleList(list(encoder.layers[:10]))

    def forward(self, conv_in, rotary):
        values=self.dropout(conv_in)
        for layer in self.layers:
            values=layer(values,attention_mask=None,relative_position_embeddings=rotary,output_attentions=False)[0]
        return values


class LoweredL10(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        if len(encoder.layers)<10 or encoder.training or any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('A frozen eval-mode encoder with at least ten layers is required.')
        self.dropout=encoder.dropout
        layers=[]
        for original in encoder.layers[:10]:
            if 'relative_position_embeddings' not in inspect.signature(original.forward).parameters:
                raise ValueError('Unsupported native conformer forward API.')
            # Share immutable parameters without replacing any submodule in the original encoder.
            layer=copy.copy(original);layer._modules=original._modules.copy()
            layer.self_attn=Rank3RotaryAttention(original.self_attn)
            layer.conv_module=ExplicitGLUConvolution(original.conv_module)
            layers.append(layer)
        self.layers=nn.ModuleList(layers)

    def forward(self, conv_in, rotary_cos, rotary_sin):
        positions=torch.stack((rotary_cos,rotary_sin),dim=0)
        values=self.dropout(conv_in)
        for layer in self.layers:
            values=layer(values,attention_mask=None,relative_position_embeddings=positions,output_attentions=False)[0]
        return values


def candidate_inputs(conv,rotary,frames,width=1024,head_size=64):
    conv,rotary=np.asarray(conv),np.asarray(rotary)
    if (conv.ndim!=3 or conv.shape[0]!=1 or conv.shape[2]!=width or not 1<=frames<=conv.shape[1]
            or rotary.shape!=(2,conv.shape[1],1,1,head_size) or not np.isfinite(conv).all() or not np.isfinite(rotary).all()):
        raise ValueError('Expected finite batch-one source convolution/rotary tensors.')
    return dict(conv_in=np.array(conv[:,:frames],dtype=np.float32,order='C',copy=True),
        rotary_cos=np.array(rotary[0,:frames,0,0,:],dtype=np.float32,order='C',copy=True),
        rotary_sin=np.array(rotary[1,:frames,0,0,:],dtype=np.float32,order='C',copy=True))


def numerical(a,b,atol=ATOL,rtol=RTOL):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False,reason='shape_or_nonfinite')
    difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(passed=bool(np.allclose(a,b,atol=atol,rtol=rtol)),max_abs_error=float(difference.max()),
        mean_abs_error=float(difference.mean()),atol=atol,rtol=rtol)


def state_digest(module):
    h=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def convert_fixed(ct,trace,frames,skip_model_load=False,width=1024,head_size=64):
    # Fully concrete shapes are traced independently; there is no post-conversion shape materialization.
    # Prevent a later default precision pass from adding FP16 casts to this FP32 graph.
    pipeline=ct.PassPipeline.DEFAULT
    pipeline=copy.deepcopy(pipeline)
    pipeline.remove_passes(['common::add_fp16_cast'])
    model=ct.convert(trace,convert_to='mlprogram',
        inputs=[ct.TensorType(name='conv_in',shape=(1,frames,width),dtype=np.float32),
            ct.TensorType(name='rotary_cos',shape=(frames,head_size),dtype=np.float32),
            ct.TensorType(name='rotary_sin',shape=(frames,head_size),dtype=np.float32)],
        outputs=[ct.TensorType(name='embedding',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=skip_model_load,pass_pipeline=pipeline)
    spec=model.get_spec()
    expected={'conv_in':[1,frames,width],'rotary_cos':[frames,head_size],'rotary_sin':[frames,head_size]}
    if {i.name:list(i.type.multiArrayType.shape) for i in spec.description.input}!=expected:
        raise ValueError('Static Core ML input contract differs.')
    def inspect_block(block):
        for op in block.operations:
            for value in op.outputs:
                if value.type.HasField('tensorType'):
                    kind=value.type.tensorType
                    if kind.dataType==10: raise ValueError('Unexpected FLOAT16 declaration.')
            for child in op.blocks:inspect_block(child)
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():inspect_block(block)
    return model


def load_encoder(role,cfg,provenance,source_report):
    if role=='musicfm':
        from finetuning.common import upstream,sha_file
        folder=upstream(cfg);weights=folder/'ckpts/MusicFM/pretrained_msd.pt';stats=folder/'ckpts/MusicFM/msd_stats.json'
        if sha_file(weights)!=provenance['musicfm_sha256'] or sha_file(stats)!=provenance['stats_sha256']:
            raise ValueError('MusicFM checkpoint/statistics differ from frozen training provenance.')
        if (not source_report.get('exact_window_passed') or source_report.get('compute_precision')!='float32'
                or source_report.get('musicfm_weights_sha256')!=provenance['musicfm_sha256']
                or source_report.get('musicfm_stats_sha256')!=provenance['stats_sha256']):
            raise ValueError('Original MusicFM export proof differs.')
        from musicfm.model.musicfm_25hz import MusicFM25Hz
        holder=MusicFM25Hz(is_flash=False,stat_path=str(stats),model_path=str(weights)).cpu().float().eval().requires_grad_(False)
        encoder=holder.conformer
    else:
        from huggingface_hub import snapshot_download
        from muq import MuQ
        from finetuning.common import fingerprint,sha_file
        if cfg['features']['muq_repo']!=provenance['muq_repo']:raise ValueError('MuQ repository differs.')
        snapshot=Path(snapshot_download(repo_id=provenance['muq_repo'],revision=provenance['muq_revision'],local_files_only=True))
        if snapshot.name!=provenance['muq_revision']:raise ValueError('MuQ snapshot differs.')
        digest=fingerprint([(str(q.relative_to(snapshot)),sha_file(q)) for q in sorted(snapshot.rglob('*')) if q.is_file()])
        if (not source_report.get('source_window_passed') or source_report.get('source_kind')!='original_training_snapshot'
                or source_report.get('compute_precision')!='float32'
                or source_report.get('muq_repo')!=provenance['muq_repo']
                or source_report.get('muq_revision')!=provenance['muq_revision']
                or source_report.get('muq_snapshot_digest')!=digest):
            raise ValueError('Original MuQ snapshot/export proof differs.')
        holder=MuQ.from_pretrained(str(snapshot)).cpu().float().eval().requires_grad_(False)
        encoder=holder.model.conformer
    if encoder.config.position_embeddings_type!='rotary':raise ValueError('Original rotary encoder required.')
    return holder,encoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,required=True);p.add_argument('--songmap-root',type=Path,required=True)
    p.add_argument('--encoder',choices=['musicfm','muq'],required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-report',type=Path,required=True,help='Original MusicFM repair_report.json or MuQ export_report.json')
    p.add_argument('--scheme',choices=['palette6','int8'],default='palette6')
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Export/compile on the Mac with Xcode and the existing SongFormer environment.')
    # Resolve before upstream() changes the process working directory.
    root,songmap,out,source_report_path=[path.expanduser().resolve() for path in (a.songstruct_root,a.songmap_root,a.output_dir,a.source_report)]
    script=Path(__file__).resolve();template=script.parent/'encoder_acceleration_iphone'
    if out.exists():raise ValueError('Choose a fresh output directory; original assets are never overwritten.')
    sys.path.insert(0,str(root))
    from finetuning.common import config,read,write,sha_file,fingerprint
    from finetuning.engine import manifest
    from deployment.verify_encoders import rotary
    from deployment.compress_coreml import compression_config,operation_counts,package_bytes
    from deployment.iphone_project import project_text
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    import librosa
    import shutil
    import subprocess
    cfg=config(root/'finetuning/config.json');data,source_id=manifest(cfg)
    provenance_path=Path(cfg['work_dir'])/'encoder_provenance.json';provenance=read(provenance_path)
    if provenance['id']!=source_id or fingerprint({k:v for k,v in provenance.items() if k!='id'})!=source_id:
        raise ValueError('Frozen encoder provenance changed.')
    if cfg['features']['long_window_seconds']!=120 or cfg['features']['encoder_layer']!=10:
        raise ValueError('Expected original 30/120-second L10 profile.')
    app_root=songmap/'SongMap/Resources/MLModels/SongFormer';app_manifest=app_root/'songformer_coreml_manifest.json';app=read(app_manifest)
    if app['sample_rate']!=24000 or app['encoder_layer']!=10 or app['encoder_window_seconds']!=120:
        raise ValueError('App encoder profile differs.')
    row=next((r for r in data['records'] if r['approved'] and r['split']=='val' and r['duration']>=120),None)
    if row is None or sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Unchanged validation audio >=120s required.')
    source_paths={key:app_root/app['models'][key] for key in ('log_mel',a.encoder+'_conv')}
    def digest(path):return fingerprint([(str(q.relative_to(path)),sha_file(q)) for q in sorted(path.rglob('*')) if q.is_file()])
    source_hashes={key:digest(path) for key,path in source_paths.items()}
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    source_report_hash=sha_file(source_report_path)
    holder,encoder=load_encoder(a.encoder,cfg,provenance,read(source_report_path))
    if encoder.layers[0].self_attn.head_size!=64 or encoder.layers[0].self_attn.num_heads!=16:
        raise ValueError('Expected original 1024-wide, 16-head, 64-wide head architecture.')
    original=ReferenceL10(encoder).eval().requires_grad_(False)
    original_hash=state_digest(original)
    candidate=LoweredL10(encoder).eval().requires_grad_(False)
    if state_digest(candidate)!=original_hash:raise ValueError('Graph rewrite changed parameters/buffers.')
    out.mkdir(parents=True,exist_ok=False);assets=out/'Assets';assets.mkdir()
    proof=dict(protocol=PROTOCOL,encoder=a.encoder,compression_scheme=a.scheme,plan_id=data['plan_id'],
        provenance_id=source_id,source_provenance_sha256=sha_file(provenance_path),source_state_sha256=original_hash,
        source_report_sha256=source_report_hash,source_report=str(source_report_path),
        app_manifest_sha256=sha_file(app_manifest),frontend_hashes=source_hashes,track_id=row['track_id'],
        audio_sha256=row['audio_sha256'],source_code_sha256=sha_file(script),torch=str(torch.__version__),
        attention_source_sha256=hashlib.sha256(inspect.getsource(type(encoder.layers[0].self_attn)).encode()).hexdigest(),
        coremltools=ct.__version__,checks=[],ready_for_app_integration=False,full_validation=False,
        scope='Same-weight graph rewrite and lossy weight compression; static encoder only. No app routes, boundaries, labels or training changed.')
    write(out/'export-report.json',proof)
    def array(outputs):
        values=[np.asarray(v) for v in outputs.values() if isinstance(v,np.ndarray)]
        if len(values)!=1 or not np.isfinite(values[0]).all():raise ValueError('Invalid frontend output.')
        return np.array(values[0],dtype=np.float32,order='C',copy=True)
    def native_call(model,values):
        name=sorted(feature.name for feature in model.get_spec().description.input)[0]
        return model.predict({name:np.array(values,dtype=np.float32,order='C',copy=True)})
    waveform,_=librosa.load(row['audio_path'],sr=24000,mono=True,duration=120)
    if len(waveform)!=120*24000:raise ValueError('Expected a full 120-second real fixture.')
    fixtures=[]
    for frames in (750,3000):
        name=f'{a.encoder}_{frames}';print('Preparing real encoder profile:',name,flush=True)
        logmel=ct.models.MLModel(str(source_paths['log_mel']),compute_units=ct.ComputeUnit.CPU_ONLY)
        mel=array(native_call(logmel,np.array(np.pad(waveform[:frames*960],(1024,1024),mode='reflect')[None],dtype=np.float32)))
        del logmel;gc.collect()
        frontend=ct.models.MLModel(str(source_paths[a.encoder+'_conv']),compute_units=ct.ComputeUnit.CPU_ONLY)
        conv=array(native_call(frontend,mel));del frontend,mel;gc.collect()
        if conv.shape!=(1,frames,1024):raise ValueError('Real frontend frame count differs.')
        positions=rotary(frames,app['rotary_inv_freq'][a.encoder])
        inputs=candidate_inputs(conv,positions,frames)
        tensors=tuple(torch.from_numpy(inputs[key]) for key in ('conv_in','rotary_cos','rotary_sin'))
        with torch.inference_mode():
            # Check the local native layer wrapper against the installed encoder using its own rotary.
            native_positions=encoder.embed_positions(torch.from_numpy(conv)).clone()
            native=encoder(torch.from_numpy(conv),output_hidden_states=True)['hidden_states'][10].numpy()
            native_check=numerical(native,original(torch.from_numpy(conv),native_positions).numpy())
            reference=original(torch.from_numpy(conv),torch.from_numpy(positions)).numpy()
            rewrite_check=numerical(reference,candidate(*tensors).numpy())
        check=dict(frames=frames,native_wrapper=native_check,rewrite_vs_original=rewrite_check)
        proof['checks'].append(check);write(out/'export-report.json',proof)
        if not native_check['passed'] or not rewrite_check['passed']:raise RuntimeError('Native/rewrite parity failed; inspect export-report.json.')
        print('Tracing concrete shape:',name,flush=True)
        trace=torch.jit.trace(candidate,tensors,strict=True,check_trace=False)
        with torch.inference_mode():check['trace_vs_original']=numerical(reference,trace(*tensors).numpy())
        if not check['trace_vs_original']['passed']:raise RuntimeError('Trace parity failed.')
        print('Converting static FP32 graph:',name,flush=True)
        model=convert_fixed(ct,trace,frames);del trace;gc.collect()
        fp32_result=np.asarray(model.predict(inputs)['embedding'])
        check['coreml_fp32_vs_original']=numerical(reference,fp32_result)
        write(out/'export-report.json',proof)
        if not check['coreml_fp32_vs_original']['passed']:raise RuntimeError('FP32 Core ML parity failed; compression not started.')
        before=operation_counts(model.get_spec());cfg_compression=compression_config(optimize,a.scheme)
        print('Compressing same-weight graph:',name,a.scheme,flush=True)
        model=(optimize.palettize_weights(model,config=cfg_compression) if a.scheme=='palette6'
               else optimize.linear_quantize_weights(model,config=cfg_compression))
        counts=operation_counts(model.get_spec());types=('constexpr_lut_to_dense',) if a.scheme=='palette6' else ('constexpr_affine_dequantize','constexpr_blockwise_shift_scale')
        if sum(counts.get(k,0)-before.get(k,0) for k in types)<=0:raise RuntimeError('No compressed-weight operations were introduced.')
        model.user_defined_metadata.update(graph_protocol=PROTOCOL,encoder=a.encoder,source_state_sha256=original_hash,
            compression_scheme=a.scheme,precision='fp32',hidden_state_index='10')
        package=out/(name+'.mlpackage');model.save(str(package))
        baseline=np.array(model.predict(inputs)['embedding'],dtype=np.float32,copy=True)
        check['compression_vs_fp32']=numerical(fp32_result,baseline)
        if baseline.shape!=reference.shape or not np.isfinite(baseline).all():raise RuntimeError('Compressed candidate is non-finite/wrong shape.')
        # iPhone compares compute routes to this SAME compressed candidate on Mac CPU.
        for key,values in {**inputs,'reference':baseline}.items():values.astype('<f4',copy=False).tofile(assets/(name+'_'+key+'.f32'))
        del model;gc.collect()
        subprocess.run(['xcrun','coremlcompiler','compile',str(package),str(assets)],check=True)
        compiled=assets/(name+'.mlmodelc')
        if not compiled.is_dir():raise RuntimeError('Compiled encoder missing.')
        fixtures.append(dict(name=name,frames=frames,resource=name,package_bytes=package_bytes(package),compiled_bytes=package_bytes(compiled),
            resource_digest=digest(compiled),files={key:name+'_'+key for key in (*inputs,'reference')},
            hashes={key:sha_file(assets/(name+'_'+key+'.f32')) for key in (*inputs,'reference')}))
        write(out/'export-report.json',proof)
    if (state_digest(original)!=original_hash or state_digest(candidate)!=original_hash
            or sha_file(source_report_path)!=source_report_hash or sha_file(row['audio_path'])!=row['audio_sha256'] or {k:digest(v) for k,v in source_paths.items()}!=source_hashes):
        raise RuntimeError('Source weights/audio/frontends changed.')
    proof.update(same_weight_rewrite_passed=True,fixtures=fixtures,
        limitations=['Real validation-window checks only; no full-song class metrics or iPhone speed measured.',
            'Numerical compression drift is reported, not accepted as label parity.',
            'Accelerator compatibility remains unproven until repeated physical-device runs, including cold runs.'])
    write(out/'export-report.json',proof)
    write(assets/'benchmark_manifest.json',dict(protocol=PROTOCOL,encoder=a.encoder,compression_scheme=a.scheme,
        source_state_sha256=original_hash,export_report_sha256=sha_file(out/'export-report.json'),fixtures=fixtures))
    for name in ('BenchmarkApp.swift','BenchmarkRunner.swift'):shutil.copyfile(template/name,out/name)
    resources=['benchmark_manifest.json']+[f['resource']+'.mlmodelc' for f in fixtures]+[value+'.f32' for f in fixtures for value in f['files'].values()]
    project=out/'Benchmark.xcodeproj';project.mkdir()
    (project/'project.pbxproj').write_text(project_text('com.danendra.EncoderGraphProbe.'+a.encoder,resource_names=resources).replace(
        'GENERATE_INFOPLIST_FILE = YES;', 'GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_CFBundleDisplayName = "'+a.encoder+' Graph";'))
    print('Prepared:',project,flush=True)
    print('Start with cpuOnly. Then test cpuAndNeuralEngine and cpuAndGPU separately after relaunch. Share JSON reports.',flush=True)


if __name__=='__main__':main()
