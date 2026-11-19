#!/usr/bin/env python3
"""Split the existing quantized GLU projection BEFORE convolution; preserve compressed weight payloads."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

from encoder_acceleration import numerical
from install_section_context import read,sha_file,fingerprint,package_digest

POLICY='split-before-conv-v2'
PROTOCOL='encoder-glu-repair-v2'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as handle:
        temporary=Path(handle.name);json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
    temporary.replace(path)


def compressed_payloads(program):
    from coremltools.converters.mil.mil import types
    result=[]
    def visit(block):
        for op in block.operations:
            if op.op_type.startswith('constexpr_'):
                inputs={}
                for key,value in op.inputs.items():
                    if value is None:continue
                    if not hasattr(value,'val') or value.val is None:raise ValueError('Compressed-weight input is not a fixed tensor.')
                    array=np.asarray(value.val)
                    inputs[key]=dict(dtype=str(array.dtype),mil_dtype=types.builtin_to_string(value.dtype),shape=list(array.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
                result.append(fingerprint(dict(type=op.op_type,inputs=inputs)))
            for child in op.blocks:visit(child)
    for function in program.functions.values():visit(function)
    return sorted(result)


def slice_half(op,width):
    if op.op_type!='slice_by_index' or len(op.outputs)!=1:raise ValueError('Unsupported GLU consumer; expected a channel slice.')
    shape=op.x.shape;rank=len(shape)
    def field(name,default):
        value=op.inputs.get(name)
        return np.asarray(value.val) if value is not None else np.asarray(default)
    begin=field('begin',np.zeros(rank));end=field('end',shape)
    begin_mask=field('begin_mask',np.zeros(rank,dtype=bool));end_mask=field('end_mask',np.zeros(rank,dtype=bool))
    stride=field('stride',np.ones(rank));squeeze=field('squeeze_mask',np.zeros(rank,dtype=bool))
    if np.any(squeeze) or np.any(stride!=1) or len(shape)<3 or shape[1]!=2*width:raise ValueError('Unsupported GLU slice layout.')
    ranges=[]
    for i,size in enumerate(shape):
        ranges.append(slice(None if begin_mask[i] else int(begin[i]),None if end_mask[i] else int(end[i]),1).indices(int(size))[:2])
    if any(ranges[i]!=(0,int(shape[i])) for i in range(rank) if i!=1):raise ValueError('GLU consumer also slices a non-channel axis.')
    if ranges[1]==(0,width):return 0
    if ranges[1]==(width,2*width):return 1
    raise ValueError('GLU slice is not an exact channel half.')


def split_glu(program,width=1024,expected=10):
    from coremltools.converters.mil import Builder as mb
    changed=[]
    for function in program.functions.values():
        block=function
        for op in list(block.operations):
            if op.op_type!='conv' or tuple(op.weight.shape[:2])!=(2*width,width):continue
            if int(op.groups.val)!=1 or any(int(d)!=1 for d in op.weight.shape[2:]):raise ValueError('Expected a pointwise non-grouped GLU convolution.')
            root=op.outputs[0];view=None
            consumers=list(root.child_ops)
            if len(consumers)==1 and consumers[0].op_type=='squeeze':
                view=consumers[0];root=view.outputs[0];consumers=list(root.child_ops)
                if 1 in np.asarray(view.axes.val).tolist():raise ValueError('GLU view squeezes the channel axis.')
            if len(consumers)!=2:raise ValueError('Expected exactly two consumers of the doubled-channel activation.')
            halves={slice_half(child,width):child for child in consumers}
            if set(halves)!={0,1}:raise ValueError('Both GLU halves are required.')
            with block:
                outputs=[]
                for half in (0,1):
                    begin=np.zeros(len(op.weight.shape),dtype=np.int32);end=np.array(op.weight.shape,dtype=np.int32)
                    begin[0]=half*width;end[0]=(half+1)*width
                    weight=mb.slice_by_index(x=op.weight,begin=begin,end=end,before_op=op,name=op.name+f'_weight_half{half}')
                    args=dict(op.inputs);args['weight']=weight
                    if args.get('bias') is not None:
                        args['bias']=mb.slice_by_index(x=args['bias'],begin=[half*width],end=[(half+1)*width],before_op=op)
                    output=mb.conv(**args,before_op=op,name=op.name+f'_half{half}')
                    if view is not None:output=mb.squeeze(x=output,axes=view.axes,before_op=op)
                    outputs.append(output)
                for half,old in halves.items():
                    block.replace_uses_of_var_after_op(anchor_op=op,old_var=old.outputs[0],new_var=outputs[half])
                block.remove_ops(consumers+([view] if view is not None else [])+[op])
            changed.append(op.name)
    if len(changed)!=expected:raise ValueError(f'Expected {expected} GLU projections; found {len(changed)}.')
    return changed


def wide_convolutions(program,width=1024):
    return [op.name for function in program.functions.values() for op in function.operations
            if op.op_type=='conv' and int(op.outputs[0].shape[1])>width]


def repair_model(ct,source,width=1024,expected=10):
    from coremltools.converters.mil.frontend.milproto.load import load
    spec=source.get_spec();program=load(spec,spec.specificationVersion,source.weights_dir)
    before=compressed_payloads(program)
    if not before:raise ValueError('Use the existing compressed candidate, not a fresh uncompressed model.')
    changed=split_glu(program,width,expected)
    if before!=compressed_payloads(program):raise RuntimeError('Compressed parameters changed during graph surgery.')
    pipeline=copy.deepcopy(ct.PassPipeline.DEFAULT);pipeline.remove_passes({'common::add_fp16_cast','common::add_int16_cast'})
    candidate=ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target(spec.specificationVersion),compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,pass_pipeline=pipeline)
    after=compressed_payloads(candidate._mil_program)
    if before!=after:raise RuntimeError('Conversion changed/removed compressed weight payloads.')
    wide=wide_convolutions(candidate._mil_program,width)
    if wide:raise RuntimeError('A doubled-channel convolution was reintroduced: '+str(wide))
    candidate.user_defined_metadata.update(dict(source.user_defined_metadata))
    candidate.user_defined_metadata['graph_repair_policy']=POLICY
    return candidate,dict(changed_convolutions=changed,compressed_payload_count=len(before),compressed_payload_sha256=fingerprint(before),
        compressed_payloads_unchanged=True,wide_convolution_count_after=0,weights_requantized=False)


def load_overrides(summary_path,parent_paths,parent_graphs,read_fn=read,sha_fn=sha_file,digest_fn=package_digest):
    summary_path=Path(summary_path).resolve();summary=read_fn(summary_path)
    expected={k:sha_fn(v) for k,v in parent_paths.items()}
    if (summary.get('protocol')!=PROTOCOL or summary.get('policy')!=POLICY or summary.get('weights_requantized') is not False
            or summary.get('parent_exports')!=expected or set(summary.get('roles',{}))!=set(expected)):
        raise ValueError('Graph repair does not belong to the validated original candidates.')
    paths={};graphs={}
    for role,entry in summary['roles'].items():
        path=(summary_path.parent/entry['report']).resolve();path.relative_to(summary_path.parent)
        if sha_fn(path)!=entry['sha256']:raise ValueError('Graph repair report changed.')
        proof=read_fn(path);parent=parent_graphs[role]
        if (proof.get('protocol')!=PROTOCOL or proof.get('policy')!=POLICY or proof.get('encoder')!=role
                or proof.get('parent_export_sha256')!=expected[role] or proof.get('weights_requantized') is not False
                or any(proof.get(k)!=parent[k] for k in ('source_state_sha256','plan_id','provenance_id'))):
            raise ValueError('Graph repair source weights/data changed.')
        checks=proof.get('checks',[]);fixtures=proof.get('fixtures',[])
        if len(checks)!=2 or {c.get('frames') for c in checks}!={750,3000} or len(fixtures)!=2 or {f.get('frames') for f in fixtures}!={750,3000}:
            raise ValueError('Both repaired static profiles are required.')
        for check in checks:
            if (not check.get('compressed_payloads_unchanged') or check.get('weights_requantized') is not False
                    or check.get('wide_convolution_count_after')!=0 or check.get('compressed_payload_count',0)<=0
                    or len(set(check.get('changed_convolutions',[])))!=10 or not check.get('source_fixture',{}).get('passed')
                    or len(check.get('cpu_checks',[]))!=3 or not all(c.get('passed') for c in check['cpu_checks'])):
                raise ValueError('Graph repair did not preserve compressed weights/CPU outputs.')
        for f in fixtures:
            if f.get('resource')!=f'{role}_{f["frames"]}':raise ValueError('Unexpected repaired resource name.')
            compiled=path.parent/'Assets'/(f['resource']+'.mlmodelc')
            if digest_fn(compiled)!=f['resource_digest']:raise ValueError('Repaired compiled model changed.')
        paths[role]=path;graphs[role]=proof
    return paths,graphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--musicfm-export',type=Path,required=True);p.add_argument('--muq-export',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run real-fixture CPU checks and compilation on the Mac.')
    import coremltools as ct
    paths={role:path.expanduser().resolve() for role,path in dict(musicfm=a.musicfm_export,muq=a.muq_export).items()}
    out=a.output_dir.expanduser().resolve()
    if out.exists():raise ValueError('Use a fresh output directory; source candidates stay unchanged.')
    out.mkdir(parents=True)
    summary=dict(protocol=PROTOCOL,policy=POLICY,parent_exports={k:sha_file(v) for k,v in paths.items()},roles={},
        weights_requantized=False,ready_for_app_integration=False,iphone_stability_verified=False)
    for role,path in paths.items():
        parent=read(path)
        if parent.get('protocol')!='encoder-rank3-explicit-glu-v1' or parent.get('compression_scheme')!='palette6' or not parent.get('same_weight_rewrite_passed'):
            raise ValueError('Expected the measured palette6 source export.')
        if len(parent.get('fixtures',[]))!=2 or {f['frames'] for f in parent['fixtures']}!={750,3000}:
            raise ValueError('Both measured static profiles are required.')
        target=out/role;assets=target/'Assets';assets.mkdir(parents=True)
        fixtures=[];checks=[]
        for f in parent['fixtures']:
            frames=f['frames'];name=f'{role}_{frames}';source_package=path.parent/(name+'.mlpackage')
            source=ct.models.MLModel(str(source_package),compute_units=ct.ComputeUnit.CPU_ONLY)
            metadata=dict(source.user_defined_metadata)
            if (metadata.get('graph_protocol')!=parent['protocol'] or metadata.get('encoder')!=role
                    or metadata.get('source_state_sha256')!=parent['source_state_sha256']
                    or metadata.get('compression_scheme')!='palette6' or metadata.get('precision')!='fp32'):
                raise ValueError('Source package metadata differs from the measured candidate.')
            inputs={};reference=None
            for key,resource in f['files'].items():
                original=path.parent/'Assets'/(resource+'.f32')
                if sha_file(original)!=f['hashes'][key]:raise ValueError('Original real fixture changed.')
                data=np.fromfile(original,dtype='<f4')
                shapes={'conv_in':(1,frames,1024),'rotary_cos':(frames,64),'rotary_sin':(frames,64),'reference':(1,frames,1024)}
                data=data.reshape(shapes[key]).astype(np.float32)
                if key=='reference':reference=data
                else:inputs[key]=data
                shutil.copyfile(original,assets/(resource+'.f32'))
            original=np.asarray(source.predict(inputs)['embedding'])
            source_check=numerical(reference,original)
            if not source_check['passed']:raise ValueError('Parent package no longer matches the measured fixture.')
            print('Splitting quantized GLU projections:',name,flush=True)
            candidate,proof=repair_model(ct,source);package=target/(name+'.mlpackage');candidate.save(str(package))
            native=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
            repeated=[]
            for call in range(3):repeated.append(numerical(original,np.asarray(native.predict(inputs)['embedding'])))
            if not all(c['passed'] for c in repeated):raise ValueError('Repaired graph failed original-compressed CPU parity.')
            newf=dict(f);newf['package_bytes']=sum(q.stat().st_size for q in package.rglob('*') if q.is_file())
            subprocess.run(['xcrun','coremlcompiler','compile',str(package),str(assets)],check=True)
            newf['compiled_bytes']=sum(q.stat().st_size for q in (assets/(name+'.mlmodelc')).rglob('*') if q.is_file())
            newf['resource_digest']=package_digest(assets/(name+'.mlmodelc'));fixtures.append(newf)
            checks.append(dict(frames=frames,source_fixture=source_check,cpu_checks=repeated,**proof))
            del native,candidate,source
            import gc;gc.collect()
        report=dict(protocol=PROTOCOL,policy=POLICY,encoder=role,parent_export_sha256=sha_file(path),
            source_state_sha256=parent['source_state_sha256'],plan_id=parent['plan_id'],provenance_id=parent['provenance_id'],
            fixtures=fixtures,checks=checks,weights_requantized=False,ready_for_app_integration=False,iphone_stability_verified=False)
        write(target/'repair-export.json',report)
        summary['roles'][role]=dict(report=role+'/repair-export.json',sha256=sha_file(target/'repair-export.json'))
        write(out/'repair-report.json',summary)
    print('Repair ready for isolated pipeline stress test:',out/'repair-report.json',flush=True)
    print('No GPU fix or whole-dataset label parity is claimed until subsequent checks pass.',flush=True)


if __name__=='__main__':main()
