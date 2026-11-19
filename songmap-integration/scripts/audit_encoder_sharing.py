#!/usr/bin/env python3
"""Read-only audit of exact constants and decoded palette weights across encoder profiles."""
import argparse
from collections import defaultdict
import gc
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from install_section_context import read,sha_file,fingerprint,package_digest
from package_encoder_weights import FUNCTIONS,model_program
from repair_encoder_glu import write

PROTOCOL='encoder-sharing-audit-v1'


def signature(array,dtype):
    value=np.asarray(array)
    return dict(dtype=dtype,shape=list(value.shape),value_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def storage_bytes(array,dtype):
    # Tensor payload estimate only; excludes alignment/serialization and shared-blob/fusion decisions.
    match=re.search(r'(\d+)$',dtype)
    bits=int(match.group(1)) if match else (1 if dtype=='bool' else 8*np.asarray(array).dtype.itemsize)
    return math.ceil(np.asarray(array).size*bits/8)


def collect(function,profile,progress=None):
    from coremltools.converters.mil.mil import types
    constants=[];palettes=[]
    def visit(block):
        for op in block.operations:
            if op.op_type=='const':
                value=op.outputs[0];array=np.asarray(value.val)
                if array.size>=64 and array.dtype.kind in 'biuf':
                    dtype=types.builtin_to_string(value.dtype)
                    constants.append(dict(profile=profile,operation=op.name,**signature(array,dtype),
                        payload_bytes=storage_bytes(array,dtype),tag=('present:'+str(op.weight_key)) if hasattr(op,'weight_key') else 'absent'))
            if op.op_type=='constexpr_lut_to_dense':
                encoded={};size=0
                for key,value in sorted(op.inputs.items()):
                    if value.val is None:raise ValueError('Chained LUT inputs need an explicit additional audit.')
                    dtype=types.builtin_to_string(value.dtype);array=np.asarray(value.val)
                    encoded[key]=signature(array,dtype);size+=storage_bytes(array,dtype)
                dense=np.asarray(op.materialized_val_inference())
                if not np.isfinite(dense).all():raise ValueError('Non-finite decoded constant.')
                dtype=types.builtin_to_string(op.outputs[0].dtype)
                palettes.append(dict(profile=profile,operation=op.name,**signature(dense,dtype),
                    elements=int(dense.size),encoded_sha256=fingerprint(encoded),encoded_payload_bytes=size))
                del dense
                if progress and len(palettes)%20==0:progress(len(palettes))
            for child in op.blocks:visit(child)
    visit(function)
    return constants,palettes


def grouped(records,flatten=False):
    groups=defaultdict(list)
    for item in records:
        shape=('elements',item['elements']) if flatten else tuple(item['shape'])
        groups[(item['dtype'],shape,item['value_sha256'])].append(item)
    return groups.values()


def summarize(constants,palettes):
    exact=[];equivalent=[];layout=[]
    for group in grouped(constants):
        profiles=sorted({v['profile'] for v in group})
        if len(profiles)<2:continue
        per_profile={p:min(v['payload_bytes'] for v in group if v['profile']==p) for p in profiles}
        tags=sorted({v['tag'] for v in group})
        exact.append(dict(profiles=profiles,dtype=group[0]['dtype'],shape=group[0]['shape'],
            estimated_cross_profile_redundancy_bytes=sum(per_profile.values())-min(per_profile.values()),
            different_tags=len(tags)>1,tags=tags,occurrences=[dict(profile=v['profile'],operation=v['operation']) for v in group]))
    for group in grouped(palettes):
        profiles=sorted({v['profile'] for v in group})
        encodings={v['encoded_sha256'] for v in group}
        if len(profiles)<2 or len(encodings)<2:continue
        costs={p:min(v['encoded_payload_bytes'] for v in group if v['profile']==p) for p in profiles}
        equivalent.append(dict(profiles=profiles,dtype=group[0]['dtype'],shape=group[0]['shape'],encoding_variants=len(encodings),
            estimated_equivalent_encoding_redundancy_bytes=sum(costs.values())-min(costs.values()),
            occurrences=[dict(profile=v['profile'],operation=v['operation'],encoding=v['encoded_sha256']) for v in group]))
    for group in grouped(palettes,flatten=True):
        profiles=sorted({v['profile'] for v in group});shapes=sorted({tuple(v['shape']) for v in group})
        if len(profiles)<2 or len(shapes)<2:continue
        layout.append(dict(profiles=profiles,dtype=group[0]['dtype'],elements=group[0]['elements'],shapes=[list(s) for s in shapes],
            occurrences=[dict(profile=v['profile'],operation=v['operation']) for v in group]))
    exact.sort(key=lambda x:x['estimated_cross_profile_redundancy_bytes'],reverse=True)
    equivalent.sort(key=lambda x:x['estimated_equivalent_encoding_redundancy_bytes'],reverse=True)
    return dict(profile_counts={p:dict(constant_tensors=sum(v['profile']==p for v in constants),palette_weights=sum(v['profile']==p for v in palettes)) for p in FUNCTIONS},
        exact_constant_groups=exact,decoded_identical_different_encoding_groups=equivalent,decoded_flat_identical_different_shape_groups=layout,
        estimated_exact_cross_profile_payload_redundancy_bytes=sum(v['estimated_cross_profile_redundancy_bytes'] for v in exact),
        estimated_tag_separated_exact_payload_bytes=sum(v['estimated_cross_profile_redundancy_bytes'] for v in exact if v['different_tags']),
        estimated_different_encoding_equivalent_payload_bytes=sum(v['estimated_equivalent_encoding_redundancy_bytes'] for v in equivalent),
        estimates_are_additive=False)


def validate_report(report):
    if (report.get('protocol')!='encoder-shared-weights-v1' or report.get('encoder') not in ('musicfm','muq')
            or not report.get('native_cpu_passed') or report.get('weights_requantized') is not False
            or report.get('graphs_rewritten') is not False or set(report.get('sources',{}))!=set(FUNCTIONS)
            or set(report.get('checks',{}))!=set(FUNCTIONS)
            or any(not all(v.get(k) is True for k in ('graph','payloads','contract')) for v in report['checks'].values())
            or len(report.get('native_checks',[]))!=6 or not all(v.get('passed') for v in report['native_checks']) or report.get('error')):
        raise ValueError('Use the complete source-bound successful packaging report.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packaging-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.packaging_report=a.packaging_report.expanduser().resolve();a.output=a.output.expanduser().resolve()
    report_hash=sha_file(a.packaging_report);report=read(a.packaging_report);validate_report(report)
    sources={name:Path(path).expanduser().resolve() for name,path in report['sources'].items()}
    if (a.output.exists() or 'outputs.local' not in a.output.parts
            or any(a.output==path or path in a.output.parents for path in sources.values())):raise ValueError('Use a fresh audit file under outputs.local, outside model packages.')
    import coremltools as ct
    packed=a.packaging_report.parent/(report['encoder']+'_Shared.mlpackage')
    if package_digest(packed)!=report['package_digest']:raise ValueError('Packed package changed since measured report.')
    files=[dict(path=str(path.relative_to(packed)),bytes=path.stat().st_size) for path in sorted(packed.rglob('*')) if path.is_file()]
    blob_bytes=sum(v['bytes'] for v in files if 'weights' in Path(v['path']).parts)
    constants=[];palettes=[]
    for name in FUNCTIONS:
        path=sources[name]
        if package_digest(path)!=report['source_digests'][name]:raise ValueError('Original source package changed: '+name)
        print('Read-only weight audit:',report['encoder'],name,flush=True)
        model,program=model_program(ct,path)
        if set(program.functions)!={'main'}:raise ValueError('Expected single-function source.')
        c,w=collect(program.functions['main'],name,progress=lambda count:print('  decoded palette weights checked:',count,flush=True));constants.extend(c);palettes.extend(w)
        del c,w,model,program;gc.collect()
        if package_digest(path)!=report['source_digests'][name]:raise ValueError('Source changed during read-only audit.')
    if (sha_file(a.packaging_report)!=report_hash or package_digest(packed)!=report['package_digest']
            or any(package_digest(path)!=report['source_digests'][name] for name,path in sources.items())):
        raise ValueError('Audit evidence/source artifacts changed during reading.')
    result=dict(protocol=PROTOCOL,encoder=report['encoder'],packaging_report_sha256=report_hash,
        audit_script_sha256=sha_file(Path(__file__)),coremltools=ct.__version__,source_digests=report['source_digests'],evidence=report['evidence'],
        measured_source_bytes=report['source_bytes'],measured_packed_bytes=report['packed_bytes'],
        measured_saving_fraction=report['size_reduction_fraction'],packed_files=files,packed_weight_blob_bytes=blob_bytes,
        packed_other_bytes=sum(v['bytes'] for v in files)-blob_bytes,**summarize(constants,palettes),
        source_models_changed=False,weights_requantized=False,ready_for_app_integration=False,
        limitations=['Payload estimates exclude alignment, metadata, fusion and already shared blobs; they are not additional package savings.',
            'Exact constants, equivalent encodings and layout-only groups overlap and must not be added.',
            'Decoded identity is bitwise within the reported dtype/order. Audit neither rewrites encodings nor certifies a proposed rewrite.',
            'Different whole-model payload hashes do not by themselves prove that every weight differs.'])
    write(a.output,result)
    print('Exact shared groups:',len(result['exact_constant_groups']),flush=True)
    print('Equal decoded weights / different encodings:',len(result['decoded_identical_different_encoding_groups']),flush=True)
    print('Equal flattened values / different shapes:',len(result['decoded_flat_identical_different_shape_groups']),flush=True)
    print('Report:',a.output,flush=True)


if __name__=='__main__':main()
