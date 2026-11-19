#!/usr/bin/env python3
"""Share existing encoder constants across Core ML functions; no training or quantization."""
import argparse
import gc
import hashlib
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import numpy as np

from encoder_acceleration import numerical
from install_section_context import read, sha_file, fingerprint, package_digest, find_report
from prepare_encoder_pipeline_iphone import validate_quality, validate_repair_selection
from repair_encoder_glu import load_overrides, compressed_payloads, write
from validate_encoder_pipeline import validate_export

PROTOCOL='encoder-shared-weights-v1'
FUNCTIONS=('short_30','long_120','partial')


def model_program(ct,path):
    from coremltools.converters.mil.frontend.milproto.load import load
    model=ct.models.MLModel(str(path),skip_model_load=True)
    spec=model.get_spec()
    return model,load(spec,spec.specificationVersion,model.weights_dir)


def function_digest(function,parameter_leaves=None):
    """Canonical computational DAG: ignore variable names/blob offsets, retain values and wiring."""
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.mil.types.symbolic import is_symbolic
    variables={};symbols={}
    def tensor_type(value):
        shape=[]
        for dim in getattr(value,'shape',()):
            if is_symbolic(dim):
                key=str(dim)
                if key not in symbols:symbols[key]=len(symbols)
                shape.append(dict(symbol=symbols[key]))
            else:shape.append(int(dim))
        return dict(dtype=types.builtin_to_string(value.dtype),shape=shape)
    for name,value in function.inputs.items():variables[value]=fingerprint(dict(input=name,type=tensor_type(value)))
    leaves=parameter_leaves or {}
    for value,name in leaves.items():variables[value]=fingerprint(dict(parameter=name,type=tensor_type(value)))
    def binding(value):
        if value is None:return None
        if isinstance(value,(tuple,list)):return [binding(v) for v in value]
        if value not in variables:raise ValueError('Unresolved graph binding; refusing unverifiable sharing.')
        return variables[value]
    for op in function.operations:
        if op.blocks:raise ValueError('Control-flow graph requires an additional packaging verifier.')
        if op.op_type=='const':
            for value in op.outputs:
                if value in leaves:continue
                array=np.asarray(value.val)
                variables[value]=fingerprint(dict(constant=tensor_type(value),numpy_dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest()))
        else:
            inputs={key:binding(value) for key,value in sorted(op.inputs.items())}
            for index,value in enumerate(op.outputs):
                if value in leaves:continue
                variables[value]=fingerprint(dict(operation=op.op_type,inputs=inputs,output=index,type=tensor_type(value)))
    return fingerprint(dict(inputs={name:variables[value] for name,value in function.inputs.items()},outputs=[variables[v] for v in function.outputs]))


def payload_digest(function):
    return fingerprint(compressed_payloads(SimpleNamespace(functions={'selected':function})))


def contract(spec,function=None):
    description=spec.description if function is None else next(f for f in spec.description.functions if f.name==function)
    return {name:[value.SerializeToString(deterministic=True).hex() for value in getattr(description,name)] for name in ('input','output','state')}


def size_bytes(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def pack(ct,sources,destination):
    if set(sources)!=set(FUNCTIONS):raise ValueError('Exactly short_30, long_120 and partial sources are required.')
    destination=Path(destination)
    if destination.exists():raise ValueError('Use a fresh output package.')
    descriptor=ct.utils.MultiFunctionDescriptor();before={}
    for name in FUNCTIONS:
        source,program=model_program(ct,sources[name])
        if set(program.functions)!={'main'}:raise ValueError('Source encoder must contain only main.')
        before[name]=dict(graph=function_digest(program.functions['main']),payloads=payload_digest(program.functions['main']),
            contract=contract(source.get_spec()),source_package_digest=package_digest(sources[name]))
        descriptor.add_function(str(sources[name]),'main',name)
        del source,program;gc.collect()
    descriptor.default_function_name='short_30'
    # Core ML's utility performs cross-function constant deduplication and skips graph optimization passes.
    ct.utils.save_multifunction(descriptor,str(destination))
    model,program=model_program(ct,destination)
    if set(program.functions)!=set(FUNCTIONS) or model.get_spec().description.defaultFunctionName!='short_30':raise ValueError('Packed function inventory changed.')
    checks={}
    for name in FUNCTIONS:
        current=dict(graph=function_digest(program.functions[name]),payloads=payload_digest(program.functions[name]),contract=contract(model.get_spec(),name))
        checks[name]={key:current[key]==before[name][key] for key in current}
        if not all(checks[name].values()):raise ValueError('Sharing changed graph, compressed values or input/output contract: '+name+' '+str(checks[name]))
        if package_digest(sources[name])!=before[name]['source_package_digest']:raise ValueError('Source encoder changed during packaging.')
    del model,program;gc.collect()
    original=sum(size_bytes(path) for path in sources.values());packed=size_bytes(destination)
    return dict(checks=checks,source_digests={k:v['source_package_digest'] for k,v in before.items()},
        source_graph_digests={k:v['graph'] for k,v in before.items()},source_payload_digests={k:v['payloads'] for k,v in before.items()},
        source_bytes=original,packed_bytes=packed,size_reduction_fraction=1-packed/original,
        smaller=packed<original,package_digest=package_digest(destination),weights_requantized=False,graphs_rewritten=False)


def sources_for(root,songmap,quality_path,repair_path,role):
    quality=read(quality_path);selection=read(quality_path.parent/'selection.json')
    validate_quality(quality,selection);validate_repair_selection(selection,repair_path,sha_file)
    if (selection['identity']!=fingerprint({k:v for k,v in selection.items() if k!='identity'})
            or selection.get('static_graph_policy')!='split-before-conv-v2' or quality.get('graph_repair_sha256')!=sha_file(repair_path)):
        raise ValueError('Completed repaired-graph quality with unchanged labels is required.')
    paths={k:find_report((songmap/'outputs.local').glob('encoder-acceleration-*/'+k+'/export-report.json'),v) for k,v in selection['encoder_exports'].items()}
    if set(paths)!={'musicfm','muq'} or any(v is None for v in paths.values()):raise ValueError('Validated original encoder exports missing.')
    parents={k:read(v) for k,v in paths.items()}
    for k,v in parents.items():validate_export(v,k,quality['plan_id'],quality['provenance_id'])
    repaired,reports=load_overrides(repair_path,paths,parents)
    def artifact(value):
        value=Path(value).expanduser();return value.resolve() if value.is_absolute() else (root/value).resolve()
    palette_path=find_report((root/'outputs.local').glob('muq-compression-*/palette6-all-validation.json'),selection['palette6_source_sha256'])
    if palette_path is None:raise ValueError('Original palette6 source report missing.')
    palette=read(palette_path)
    if role=='muq':
        report_path=artifact(palette['muq_export_report']);old=read(report_path)
        if sha_file(report_path)!=palette['muq_export_report_sha256']:raise ValueError('Legacy MuQ report changed.')
        expected=old['package_digest']
    else:
        combo_path=artifact(palette['baseline_combination'])
        if sha_file(combo_path)!=palette['baseline_combination_sha256']:raise ValueError('Head/source combination changed.')
        combo=read(combo_path);pipeline_path=artifact(combo['encoder_pipeline_report'])
        if sha_file(pipeline_path)!=combo['encoder_pipeline_report_sha256']:raise ValueError('Legacy MusicFM pipeline changed.')
        pipeline=read(pipeline_path)
        report_path=find_report((root/'outputs.local').glob('musicfm-compression-*/palette6/compression_report.json'),pipeline['musicfm_override']['report_sha256'])
        if report_path is None:raise ValueError('Legacy MusicFM report missing.')
        old=read(report_path);header=read(artifact(pipeline['diagnostic_cache'])/'encoder_provenance.json')
        if header['id']!=pipeline['candidate_encoder_id'] or fingerprint({k:v for k,v in header.items() if k!='id'})!=header['id']:raise ValueError('Legacy MusicFM provenance changed.')
        expected=header['packages']['musicfm']
    partial=artifact(old['package'])
    if package_digest(partial)!=expected:raise ValueError('Legacy partial encoder changed.')
    directory=repaired[role].parent
    sources=dict(short_30=directory/(role+'_750.mlpackage'),long_120=directory/(role+'_3000.mlpackage'),partial=partial)
    evidence=dict(quality_sha256=sha_file(quality_path),quality_selection=selection['identity'],repair_sha256=sha_file(repair_path),
        repair_export_sha256=sha_file(repaired[role]),legacy_report_sha256=sha_file(report_path),
        plan_id=quality['plan_id'],provenance_id=quality['provenance_id'],source_state_sha256=reports[role]['source_state_sha256'])
    return sources,reports[role],directory,evidence


def real_inputs(directory,fixture):
    arrays={};frames=fixture['frames']
    for key,name in fixture['files'].items():
        path=directory/'Assets'/(name+'.f32')
        if sha_file(path)!=fixture['hashes'][key]:raise ValueError('Real fixture changed.')
        shapes={'conv_in':(1,frames,1024),'rotary_cos':(frames,64),'rotary_sin':(frames,64),'reference':(1,frames,1024)}
        arrays[key]=np.fromfile(path,dtype='<f4').reshape(shapes[key]).copy()
    return arrays


def native_checks(ct,sources,package,proof,directory,on_check=None):
    checks=[]
    cases=[]
    for fixture in proof['fixtures']:
        data=real_inputs(directory,fixture);frames=fixture['frames']
        cases.append((('short_30' if frames==750 else 'long_120'),frames,{k:v for k,v in data.items() if k!='reference'},data['reference']))
    long=real_inputs(directory,next(f for f in proof['fixtures'] if f['frames']==3000))
    # Real feature prefixes test dynamic signatures; these are not full-song quality measurements.
    for frames in (594,739,1500,2989):
        values=dict(conv_in=long['conv_in'][:,:frames].copy(),rotary=np.stack([long['rotary_cos'][:frames],long['rotary_sin'][:frames]])[:,:,None,None,:].copy())
        cases.append(('partial',frames,values,None))
    for name,frames,inputs,reference in cases:
        print('CPU function parity:',name,frames,flush=True)
        original=ct.models.MLModel(str(sources[name]),compute_units=ct.ComputeUnit.CPU_ONLY)
        result=original.predict(inputs)
        if len(result)!=1:raise ValueError('Expected single encoder output.')
        expected=np.asarray(next(iter(result.values()))).copy();del original;gc.collect()
        if expected.shape!=(1,frames,1024) or not np.isfinite(expected).all():raise ValueError('Source output shape/nonfinite failure.')
        source_check=numerical(reference,expected) if reference is not None else None
        if source_check is not None and not source_check['passed']:
            checks.append(dict(function=name,frames=frames,source_fixture=source_check,passed=False,repeats=[]))
            if on_check:on_check(checks)
            raise ValueError('Raw source no longer reproduces the validated real fixture: '+str(checks[-1]))
        combined=ct.models.MLModel(str(package),function_name=name,compute_units=ct.ComputeUnit.CPU_ONLY)
        repeats=[]
        for _ in range(2):
            output=combined.predict(inputs)
            if len(output)!=1:raise ValueError('Unexpected packed output contract.')
            repeats.append(numerical(expected,np.asarray(next(iter(output.values())))))
        del combined;gc.collect()
        check=dict(function=name,frames=frames,source_fixture=source_check,repeats=repeats,passed=all(v['passed'] for v in repeats))
        checks.append(check)
        if on_check:on_check(checks)
        if not check['passed']:raise ValueError('Packed CPU function differs from source: '+str(check))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,required=True);p.add_argument('--songmap-root',type=Path,required=True)
    p.add_argument('--quality-report',type=Path,required=True);p.add_argument('--graph-repair',type=Path,required=True)
    p.add_argument('--encoder',choices=('musicfm','muq'),required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run real native parity and compilation on the Mac.')
    import coremltools as ct
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    if a.output_dir.exists():raise ValueError('Use a fresh directory; original models are never overwritten.')
    sources,proof,directory,evidence=sources_for(a.songstruct_root,a.songmap_root,a.quality_report,a.graph_repair,a.encoder)
    if (a.songmap_root/'outputs.local') not in a.output_dir.parents or any(path==a.output_dir or path in a.output_dir.parents for path in sources.values()):
        raise ValueError('Stage outside source model packages, under SongMap outputs.local only.')
    a.output_dir.mkdir(parents=True)
    package=a.output_dir/(a.encoder+'_Shared.mlpackage')
    report=dict(protocol=PROTOCOL,encoder=a.encoder,evidence=evidence,sources={k:str(v) for k,v in sources.items()},
        coremltools=ct.__version__,script_sha256=sha_file(Path(__file__)),native_cpu_passed=False,ready_for_app_integration=False,
        selected_route=dict(frontends='CPU_ONLY',short_30='CPU_AND_GPU',long_120='CPU_AND_GPU',partial='CPU_ONLY',head='CPU_ONLY',context='CPU_ONLY'),
        limitations=['Raw and compiled sizes are measured separately; neither is an installed-app size.',
            'Native checks use real feature fixtures/prefixes, not all validation songs or iPhone GPU execution.',
            'Multi-function loading may change native GPU compilation behavior; real pipeline/device verification is still required.'])
    try:
        report.update(pack(ct,sources,package));write(a.output_dir/'packaging-report.json',report)
        for fixture in proof['fixtures']:
            name='short_30' if fixture['frames']==750 else 'long_120'
            check=next(c for c in proof['checks'] if c['frames']==fixture['frames'])
            if report['source_payload_digests'][name]!=check['compressed_payload_sha256']:
                raise ValueError('Static source compressed values differ from the validated GLU repair.')
        def record(checks):
            report['native_checks']=checks;write(a.output_dir/'packaging-report.json',report)
        report['native_checks']=native_checks(ct,sources,package,proof,directory,on_check=record)
        report['native_cpu_passed']=True
        compiled=a.output_dir/'Compiled';compiled.mkdir()
        subprocess.run(['xcrun','coremlcompiler','compile',str(package),str(compiled)],check=True)
        resource=compiled/(package.stem+'.mlmodelc')
        if not resource.is_dir():raise ValueError('Packed compiled model missing.')
        report.update(compiled_bytes=size_bytes(resource),compiled_digest=package_digest(resource))
        if any(package_digest(path)!=report['source_digests'][name] for name,path in sources.items()):raise ValueError('Source changed during native checks.')
        if package_digest(package)!=report['package_digest']:raise ValueError('Packed artifact changed during native checks.')
        if sha_file(a.quality_report)!=evidence['quality_sha256'] or sha_file(a.graph_repair)!=evidence['repair_sha256']:
            raise ValueError('Quality/repair evidence changed during packaging.')
        report['candidate_for_device_test']=report['smaller'] and report['native_cpu_passed']
        write(a.output_dir/'packaging-report.json',report)
    except Exception as error:
        report['error']=str(error);report['candidate_for_device_test']=False;write(a.output_dir/'packaging-report.json',report);raise
    print('Source packages MiB:',report['source_bytes']/2**20,'Shared package MiB:',report['packed_bytes']/2**20,flush=True)
    print('Compiled shared MiB:',report['compiled_bytes']/2**20,'Report:',a.output_dir/'packaging-report.json',flush=True)


if __name__=='__main__':main()
