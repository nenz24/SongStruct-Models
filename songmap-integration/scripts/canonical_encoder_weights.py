#!/usr/bin/env python3
"""Experimental reuse of one existing 6-bit parameter bank; quality must be reevaluated."""
import argparse
import gc
import hashlib
from pathlib import Path
import platform
import re
import subprocess

import numpy as np

from install_section_context import read,sha_file,fingerprint,package_digest
from package_encoder_weights import FUNCTIONS,model_program,function_digest,contract,size_bytes,pack,sources_for,real_inputs
from repair_encoder_glu import write
from encoder_acceleration import numerical

PROTOCOL='encoder-canonical-palette-v1'
CANONICAL='long_120'
PARAMETERS=110
MAX_RELATIVE_PARAMETER_RMS=.1  # Sanity guard for name/layout binding, not a quality acceptance criterion.


def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def trimmed_shape(shape):
    shape=list(shape)
    while len(shape)>2 and shape[-1]==1:shape.pop()
    return tuple(shape)


def encoded_hash(indices,lut):
    return fingerprint(dict(indices=array_hash(indices.reshape(-1)),lut=array_hash(lut.reshape(-1)),
        shape=trimmed_shape(indices.shape),groups=trimmed_shape(lut.shape[:-2]),bits=6,dtype='fp32'))


def validate_palette(op,key):
    from coremltools.converters.mil.mil import types
    info=dict(indices_dtype=types.builtin_to_string(op.indices.dtype),lut_dtype=types.builtin_to_string(op.lut.dtype),
        indices_shape=list(op.indices.shape),lut_shape=list(op.lut.shape),vector_axis=None if op.vector_axis is None else str(op.vector_axis.val))
    shape=tuple(op.indices.shape);lut_shape=tuple(op.lut.shape)
    if (info['indices_dtype']!='uint6' or info['lut_dtype']!='fp32'
            or op.indices.val is None or op.lut.val is None
            or len(lut_shape)!=len(shape)+2 or lut_shape[-2:]!=(64,1)
            or op.vector_axis is not None
            or any(g<=0 or n%g for n,g in zip(shape,lut_shape[:-2]))
            or tuple(op.outputs[0].shape)!=shape):
        raise ValueError('Expected fixed scalar FP32 palettes with 64 entries per group: '+key+' '+str(info))


def folded_depthwise_binding(op):
    """Identify anonymous folded weights by their unique, unchanged convolution bias/site."""
    from coremltools.converters.mil.mil import types
    children=list(op.outputs[0].child_ops);shape=tuple(op.outputs[0].shape)
    if len(children)!=1 or children[0].op_type!='conv' or children[0].weight is not op.outputs[0]:
        raise ValueError('Anonymous palette requires exactly one direct convolution weight consumer: '+op.name)
    conv=children[0];bias=conv.inputs.get('bias')
    if (len(shape)!=3 or shape[1]!=1 or shape[2]<=1 or conv.groups.val!=shape[0]
            or bias is None or bias.op.op_type!='const' or bias.val is None
            or types.builtin_to_string(bias.dtype)!='fp32' or tuple(bias.shape)!=(shape[0],)
            or not np.isfinite(bias.val).all()):
        raise ValueError('Anonymous palette is not a fixed-bias depthwise convolution: '+op.name)
    settings={}
    for name in ('groups','strides','dilations','pad_type','pad'):
        value=conv.inputs.get(name)
        if value is None or value.val is None:raise ValueError('Convolution binding requires fixed '+name)
        array=np.asarray(value.val)
        settings[name]=dict(dtype=types.builtin_to_string(value.dtype),shape=list(array.shape),value=array.tolist())
    signature=dict(policy='unique-folded-depthwise-bias-v1',weight_shape=list(shape),bias_dtype='fp32',
        bias_shape=list(bias.shape),bias_sha256=array_hash(bias.val),convolution=settings)
    return 'folded_depthwise_'+fingerprint(signature),signature


def parameter_binding(op):
    match=re.fullmatch(r'(.+)_palettized_indices(?:_\d+)?',op.indices.op.name)
    if not match:raise ValueError('Unrecognized parameter identity: '+op.indices.op.name)
    original=match[1]
    if re.fullmatch(r'const_\d+',original):
        key,signature=folded_depthwise_binding(op)
        return key,dict(method='unique-folded-depthwise-bias-v1',original_parameter=original,
            consumer=op.outputs[0].child_ops[0].name,signature=signature)
    return original,dict(method='exact-parameter-name',original_parameter=original)


def parameters(function):
    result={}
    for op in function.operations:
        if op.blocks:raise ValueError('Control-flow parameter mapping requires a separate verifier.')
        if op.op_type!='constexpr_lut_to_dense':continue
        if op.indices.op.op_type!='const':raise ValueError('Expected fixed 6-bit index constants.')
        key,binding=parameter_binding(op)
        if key in result:raise ValueError('Ambiguous duplicate parameter identity/bias binding: '+key)
        validate_palette(op,key)
        result[key]=op
    return result


def build_bank(function,expected=PARAMETERS):
    ops=parameters(function)
    if len(ops)!=expected:raise ValueError(f'Expected {expected} named parameters, found {len(ops)}.')
    bank={}
    for key,op in ops.items():
        indices=np.array(op.indices.val,copy=True);lut=np.array(op.lut.val,copy=True)
        dense=np.asarray(op.materialized_val_inference())
        if not np.isfinite(dense).all():raise ValueError('Non-finite canonical parameter.')
        bank[key]=dict(indices=indices.reshape(-1),lut=lut,shape=tuple(dense.shape),dense_sha256=array_hash(dense.reshape(-1)),
            encoded_sha256=encoded_hash(indices,lut),binding=parameter_binding(op)[1])
    return bank


def shape_compatible(left,right):
    return trimmed_shape(left)==trimmed_shape(right)


def bank_inputs(source,shape):
    """Keep each channel group's LUT; only align trailing singleton weight axes."""
    if not shape_compatible(shape,source['shape']):raise ValueError('Unsupported parameter layout change.')
    groups=tuple(source['lut'].shape[:-2])
    if len(shape)<len(groups):
        if any(g!=1 for g in groups[len(shape):]):raise ValueError('Cannot remove a palette grouping axis.')
        groups=groups[:len(shape)]
    else:groups=groups+(1,)*(len(shape)-len(groups))
    if any(g<=0 or n%g for n,g in zip(shape,groups)):raise ValueError('Palette groups do not divide the target shape.')
    return dict(indices=source['indices'].reshape(shape),lut=source['lut'].reshape(groups+(64,1)))


def validate_bindings(ops,bank):
    if set(ops)!=set(bank):raise ValueError('Parameter IDs differ; no fuzzy matching allowed: '+str(dict(missing=sorted(set(bank)-set(ops)),extra=sorted(set(ops)-set(bank)))))
    for key,op in ops.items():
        if not shape_compatible(op.outputs[0].shape,bank[key]['shape']):raise ValueError('Unsupported parameter layout change: '+key)


def unify(program,bank):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil.ops.defs.iOS18.compression import constexpr_lut_to_dense
    from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
    function=program.functions['main'];ops=parameters(function)
    validate_bindings(ops,bank)
    before=function_digest(function,{op.outputs[0]:key for key,op in ops.items()});leaves={};differences=[]
    with function:
        for key,op in ops.items():
            source=bank[key];shape=tuple(op.outputs[0].shape)
            if not shape_compatible(shape,source['shape']):raise ValueError('Unsupported parameter layout change: '+key)
            old=np.asarray(op.materialized_val_inference()).reshape(-1)
            encoded=bank_inputs(source,shape)
            canonical=np.asarray(constexpr_lut_to_dense.decompress(encoded['indices'],encoded['lut'],None)).reshape(-1)
            delta=old.astype(np.float64)-canonical.astype(np.float64)
            rms=float(np.sqrt(np.mean(delta*delta)));base=float(np.sqrt(np.mean(canonical.astype(np.float64)**2)))
            if rms/max(base,1e-12)>MAX_RELATIVE_PARAMETER_RMS:
                raise ValueError('Large parameter mismatch; refusing presumed quantization-only replacement: '+key+' relative RMS='+str(rms/max(base,1e-12)))
            differences.append(dict(parameter=key,old_shape=list(shape),canonical_shape=list(source['shape']),
                original_binding=parameter_binding(op)[1],canonical_binding=source['binding'],
                original_lut_shape=list(op.lut.shape),canonical_lut_shape=list(source['lut'].shape),unified_lut_shape=list(encoded['lut'].shape),
                old_decoded_sha256=array_hash(old),canonical_decoded_sha256=source['dense_sha256'],
                changed=array_hash(old)!=source['dense_sha256'],max_abs_error=float(np.max(np.abs(delta))),relative_rms_error=rms/max(base,1e-12)))
            restored=mb.constexpr_lut_to_dense(**encoded,before_op=op,name=key+'_canonical_weight')
            if array_hash(np.asarray(restored.op.materialized_val_inference()).reshape(-1))!=source['dense_sha256']:raise ValueError('Canonical encoding changed decoded values.')
            function.replace_uses_of_var_after_op(anchor_op=op,old_var=op.outputs[0],new_var=restored,force_replace=True)
            function.remove_ops([op]);leaves[restored]=key
    PASS_REGISTRY['common::dead_code_elimination'](program)
    after=function_digest(function,leaves)
    if before!=after:raise ValueError('Non-parameter computation changed.')
    return dict(structure_sha256=before,non_parameter_graph_unchanged=True,parameters=len(ops),differences=differences)


def verify_unified(function,bank):
    found={}
    for op in function.operations:
        if op.op_type!='constexpr_lut_to_dense':continue
        match=re.fullmatch(r'(.+)_canonical_weight(?:_\d+)?',op.name)
        if not match:raise ValueError('Unexpected LUT outside canonical bank: '+op.name)
        key=match[1]
        if key in found or key not in bank:raise ValueError('Duplicate/unknown canonical parameter: '+key)
        validate_palette(op,key)
        if key.startswith('folded_depthwise_') and folded_depthwise_binding(op)[0]!=key:raise ValueError('Canonical folded-weight consumer/bias binding changed: '+key)
        if encoded_hash(op.indices.val,op.lut.val)!=bank[key]['encoded_sha256']:raise ValueError('Canonical payload changed: '+key)
        if array_hash(np.asarray(op.materialized_val_inference()).reshape(-1))!=bank[key]['dense_sha256']:raise ValueError('Canonical decoded values changed: '+key)
        if not shape_compatible(op.outputs[0].shape,bank[key]['shape']):raise ValueError('Restored parameter layout changed.')
        found[key]=op.outputs[0]
    if set(found)!=set(bank):raise ValueError('Serialized canonical parameter inventory changed.')
    return function_digest(function,{v:k for k,v in found.items()})


def export_profile(ct,source,destination,bank):
    from coremltools.converters.mil.converter import mil_convert
    original,program=model_program(ct,source);spec=original.get_spec();expected_contract=contract(spec)
    original_graph=function_digest(program.functions['main'])
    proof=unify(program,bank);program.skip_all_passes=True
    target=mil_convert(program,convert_to='mlprogram',convert_from='milinternal',specification_version=spec.specificationVersion,
        model_description=spec.description,compute_units=ct.ComputeUnit.CPU_ONLY,skip_model_load=True)
    target.user_defined_metadata.update(weights_unified='true',canonical_profile=CANONICAL)
    target.save(str(destination));del original,program,target;gc.collect()
    target,reloaded=model_program(ct,destination)
    if contract(target.get_spec())!=expected_contract or verify_unified(reloaded.functions['main'],bank)!=proof['structure_sha256']:
        raise ValueError('Serialized profile graph/contract changed.')
    proof['original_graph_sha256']=original_graph;proof['unified_graph_sha256']=function_digest(reloaded.functions['main'])
    del target,reloaded;gc.collect()
    return proof


def native_checks(ct,old,new,package,proof,directory,on_check=None):
    cases=[]
    for fixture in proof['fixtures']:
        data=real_inputs(directory,fixture);n=fixture['frames']
        cases.append(('short_30' if n==750 else 'long_120',n,{k:v for k,v in data.items() if k!='reference'}))
    long=real_inputs(directory,next(f for f in proof['fixtures'] if f['frames']==3000))
    for n in (594,739,1500,2989):
        cases.append(('partial',n,dict(conv_in=long['conv_in'][:,:n].copy(),rotary=np.stack([long['rotary_cos'][:n],long['rotary_sin'][:n]])[:,:,None,None,:].copy())))
    checks=[]
    for name,n,inputs in cases:
        print('Canonical CPU check:',name,n,flush=True)
        source=ct.models.MLModel(str(new[name]),compute_units=ct.ComputeUnit.CPU_ONLY)
        expected=np.asarray(next(iter(source.predict(inputs).values()))).copy();del source;gc.collect()
        if expected.shape!=(1,n,1024) or not np.isfinite(expected).all():raise ValueError('Unified output shape/nonfinite failure.')
        previous=None
        if name!='partial' or n==594:
            source=ct.models.MLModel(str(old[name]),compute_units=ct.ComputeUnit.CPU_ONLY)
            previous=numerical(np.asarray(next(iter(source.predict(inputs).values()))),expected);del source;gc.collect()
        combined=ct.models.MLModel(str(package),function_name=name,compute_units=ct.ComputeUnit.CPU_ONLY)
        repeated=[numerical(expected,np.asarray(next(iter(combined.predict(inputs).values())))) for _ in range(2)]
        del combined;gc.collect()
        item=dict(function=name,frames=n,packed_vs_unified=repeated,vs_original_profile=previous,
            passed=all(v['passed'] for v in repeated) and (name!='long_120' or previous['passed']))
        checks.append(item)
        if on_check:on_check(checks)
        if not item['passed']:raise ValueError('Packed/unified parity or canonical long control failed: '+str(item))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('songstruct-root','songmap-root','quality-report','graph-repair','packaging-report','audit-report','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--encoder',choices=('musicfm','muq'),required=True);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    if platform.system()!='Darwin':raise RuntimeError('Run real model candidate export and native controls on the Mac.')
    import coremltools as ct
    if a.output_dir.exists() or (a.songmap_root/'outputs.local') not in a.output_dir.parents:raise ValueError('Use a fresh staging directory under outputs.local.')
    old,proof,directory,evidence=sources_for(a.songstruct_root,a.songmap_root,a.quality_report,a.graph_repair,a.encoder)
    prior=read(a.packaging_report);audit=read(a.audit_report)
    if (prior.get('protocol')!='encoder-shared-weights-v1' or not prior.get('native_cpu_passed') or prior.get('error')
            or audit.get('protocol')!='encoder-sharing-audit-v1' or audit.get('source_models_changed') is not False or audit.get('weights_requantized') is not False
            or prior.get('encoder')!=a.encoder or audit.get('encoder')!=a.encoder or audit.get('packaging_report_sha256')!=sha_file(a.packaging_report)
            or prior.get('evidence')!=evidence or audit.get('evidence')!=evidence or audit.get('source_digests')!=prior.get('source_digests')
            or any(package_digest(path)!=prior['source_digests'][name] for name,path in old.items())):
        raise ValueError('Reviewed audit/source evidence changed.')
    a.output_dir.mkdir(parents=True)
    report=dict(protocol=PROTOCOL,encoder=a.encoder,evidence=evidence,canonical_profile=CANONICAL,
        audit_sha256=sha_file(a.audit_report),prior_packaging_sha256=sha_file(a.packaging_report),source_digests=prior['source_digests'],
        script_sha256=sha_file(Path(__file__)),weights_updated=True,weights_unified=True,weights_requantized=False,retrained=False,
        graphs_rewritten=True,full_quality_passed=False,ready_for_app_integration=False,prior_compiled_bytes=prior['compiled_bytes'],
        parameter_mapping_policy='exact-name-or-unique-folded-depthwise-bias-v1')
    try:
        model,program=model_program(ct,old[CANONICAL]);bank=build_bank(program.functions['main']);del model,program;gc.collect()
        report['bank_sha256']=fingerprint({k:v['encoded_sha256'] for k,v in sorted(bank.items())});report['profiles']={};new={}
        report['parameter_bindings']={}
        for name in FUNCTIONS:
            print('Checking all parameter bindings before export:',name,flush=True)
            model,program=model_program(ct,old[name]);ops=parameters(program.functions['main'])
            report['parameter_bindings'][name]={key:parameter_binding(op)[1] for key,op in ops.items()}
            write(a.output_dir/'canonical-report.json',report);validate_bindings(ops,bank)
            del ops,model,program;gc.collect()
        for name in FUNCTIONS:
            print('Reusing long-profile 6-bit parameters:',name,flush=True)
            new[name]=a.output_dir/(name+'.mlpackage');report['profiles'][name]=export_profile(ct,old[name],new[name],bank)
            if name==CANONICAL and report['profiles'][name]['original_graph_sha256']!=report['profiles'][name]['unified_graph_sha256']:raise ValueError('Canonical long-profile computation changed.')
            write(a.output_dir/'canonical-report.json',report)
        package=a.output_dir/(a.encoder+'_Canonical.mlpackage');packed=pack(ct,new,package)
        model,program=model_program(ct,package)
        for name in FUNCTIONS:
            if verify_unified(program.functions[name],bank)!=report['profiles'][name]['structure_sha256']:raise ValueError('Packed function computation changed.')
        del model,program,bank;gc.collect()
        report['packing']=packed;report['package']=str(package);report['package_digest']=package_digest(package)
        def record(checks):
            report['native_checks']=checks;write(a.output_dir/'canonical-report.json',report)
        report['native_checks']=native_checks(ct,old,new,package,proof,directory,on_check=record)
        report['native_cpu_passed']=True
        compiled=a.output_dir/'Compiled';compiled.mkdir();subprocess.run(['xcrun','coremlcompiler','compile',str(package),str(compiled)],check=True)
        resource=compiled/(package.stem+'.mlmodelc');report['compiled_bytes']=size_bytes(resource);report['compiled_digest']=package_digest(resource)
        if not resource.is_dir() or not report['compiled_bytes']:raise ValueError('Missing compiled candidate.')
        if (any(package_digest(path)!=prior['source_digests'][name] for name,path in old.items())
                or any(package_digest(path)!=packed['source_digests'][name] for name,path in new.items())):raise ValueError('Source models changed.')
        if package_digest(package)!=report['package_digest']:raise ValueError('Candidate changed during native checks.')
        if (sha_file(a.quality_report)!=evidence['quality_sha256'] or sha_file(a.graph_repair)!=evidence['repair_sha256']
                or sha_file(a.packaging_report)!=report['prior_packaging_sha256'] or sha_file(a.audit_report)!=report['audit_sha256']):raise ValueError('Evidence changed during export.')
        report['smaller_than_previous_compiled']=report['compiled_bytes']<prior['compiled_bytes']
        report['candidate_for_quality_validation']=True;write(a.output_dir/'canonical-report.json',report)
    except Exception as error:
        report.update(error=str(error),candidate_for_quality_validation=False);write(a.output_dir/'canonical-report.json',report);raise
    print('Compiled MiB:',report['compiled_bytes']/2**20,'Previous:',prior['compiled_bytes']/2**20,flush=True)
    print('Weights in short/partial changed; full validation is required:',a.output_dir/'canonical-report.json',flush=True)


if __name__=='__main__':main()
