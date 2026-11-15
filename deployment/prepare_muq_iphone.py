"""Prepare an isolated MuQ encoder benchmark with identical real conv/rotary fixtures."""
import argparse
import gc
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import numpy as np
from deployment.compress_coreml import package_bytes
from deployment.compress_muq import validate_baseline, package_digest
from deployment.coreml_features import native_input
from deployment.evaluate_cached_combination import check_same_inventory
from deployment.iphone_project import project_text
from deployment.fixed_musicfm import materialize_package
from finetuning.common import read, sha_file, write


def fixture_inputs(conv,rotary,frames):
    if (conv.ndim!=3 or conv.shape[0]!=1 or conv.shape[-1]!=1024 or rotary.shape!=(2,conv.shape[1],1,1,64)
            or not 1<=frames<=min(3000,conv.shape[1])):
        raise ValueError('Unexpected encoder fixture dimensions.')
    return {'conv_in':native_input(conv[:,:frames]),'rotary':native_input(rotary[:,:frames])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--include-fp32',action='store_true',help='Also bundle the large FP32 source; default bundles only INT8/palette6')
    parser.add_argument('--bundle-id',default='com.danendra.MuQEncoderBenchmark')
    parser.add_argument('--team-id')
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Prepare on macOS with Xcode and coremltools.')
    import coremltools as ct
    comparison,out=[p.expanduser().resolve() for p in (args.comparison,args.output_dir)]
    if out.exists():
        raise ValueError('Choose a new output folder.')
    summary=read(comparison)
    baseline_path=Path(summary['source_validation_report']).expanduser().resolve()
    baseline=read(baseline_path)
    repair_path=Path(baseline['muq_export_report']).expanduser().resolve()
    repair=read(repair_path)
    validate_baseline(repair,baseline,repair_path)
    source=Path(repair['package']).resolve()
    cache_meta=read(Path(baseline['diagnostic_cache'])/'encoder_provenance.json')
    if cache_meta['muq_package_digest']!=package_digest(source):
        raise ValueError('Source MuQ package changed since full validation.')
    diagnostic=repair_path.parent
    arrays={name:np.load(diagnostic/f'{name}.npy',allow_pickle=False) for name in ('conv_in','rotary')}
    for name in arrays:
        if sha_file(diagnostic/f'{name}.npy')!=repair['fixture_hashes'][name]:
            raise ValueError('Saved MuQ fixture changed.')
    packages=[]
    for scheme,resource in [('int8','MuQINT8'),('palette6','MuQPalette6')]:
        candidate_path=comparison.parent/scheme/'compression_report.json'
        candidate=read(candidate_path)
        if (candidate['compression_scheme']!=scheme or not candidate['window_finite_passed']
                or candidate['source_report_sha256']!=sha_file(repair_path)
                or candidate['source_validation_report_sha256']!=sha_file(baseline_path)):
            raise ValueError('MuQ candidate source validation mismatch.')
        validated=read(comparison.parent/f'{scheme}-all-validation.json')
        if (not validated['full_validation'] or validated['muq_scheme']!=scheme
                or validated['muq_export_report_sha256']!=sha_file(candidate_path)
                or validated['metrics']['low_coverage_sections']!=0):
            raise ValueError('MuQ candidate must complete full validation before benchmarking.')
        check_same_inventory(baseline,validated)
        package=Path(candidate['package']).resolve()
        if package_digest(package)!=candidate['package_digest']:
            raise ValueError('Compressed MuQ package changed.')
        packages.append((scheme,resource,package))
    if args.include_fp32:
        packages.append(('fp32','MuQFP32',source))
    resources=[resource+'_'+fixture+'.mlmodelc' for _,resource,_ in packages
               for fixture in ('short','long')]
    for name in ('short','long'):
        resources.extend([f'conv_{name}.f32',f'rotary_{name}.f32',f'reference_{name}.f32'])
    resources.append('benchmark_manifest.json')
    project=project_text(args.bundle_id,args.team_id,resource_names=resources)
    project=project.replace('GENERATE_INFOPLIST_FILE = YES;',
        'GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_CFBundleDisplayName = "MuQ Bench";')
    out.mkdir(parents=True,exist_ok=False)
    assets=out/'Assets'
    assets.mkdir()
    model=ct.models.MLModel(str(source),compute_units=ct.ComputeUnit.CPU_ONLY)
    if (model.user_defined_metadata.get('precision')!='fp32'
            or model.user_defined_metadata.get('muq_snapshot_digest')!=repair['muq_snapshot_digest']):
        raise ValueError('Source reference package metadata differs.')
    fixtures=[]
    for name,frames in [('short',min(750,arrays['conv_in'].shape[1])),('long',arrays['conv_in'].shape[1])]:
        inputs=fixture_inputs(arrays['conv_in'],arrays['rotary'],frames)
        reference=np.array(model.predict(inputs)['embedding'],dtype=np.float32,copy=True)
        if reference.shape!=(1,frames,1024) or not np.isfinite(reference).all():
            raise ValueError('Invalid FP32 reference fixture result.')
        for key,value in [(f'conv_{name}',inputs['conv_in']),(f'rotary_{name}',inputs['rotary']),(f'reference_{name}',reference)]:
            value.astype('<f4',copy=False).tofile(assets/f'{key}.f32')
        fixtures.append({'name':name,'frames':frames,'conv_resource':f'conv_{name}',
            'rotary_resource':f'rotary_{name}','reference_resource':f'reference_{name}',
            'conv_sha256':sha_file(assets/f'conv_{name}.f32'),
            'rotary_sha256':sha_file(assets/f'rotary_{name}.f32')})
    del model
    gc.collect()
    models=[]
    for scheme,resource,package in packages:
        print('Compiling MuQ candidate:',scheme,flush=True)
        check=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
        meta=dict(check.user_defined_metadata)
        del check
        if (meta.get('precision')!='fp32' or meta.get('muq_snapshot_digest')!=repair['muq_snapshot_digest']
                or meta.get('muq_revision')!=repair['muq_revision']
                or meta.get('hidden_state_index')!='10'
                or meta.get('compression_scheme')!=(None if scheme=='fp32' else scheme)):
            raise ValueError('Candidate package metadata mismatch.')
        fixed_resources, fixed_compiled_bytes, fixed_package_bytes = {}, {}, {}
        for fixture in fixtures:
            fixture_name=fixture['name']
            fixed_resource=resource+'_'+fixture_name
            fixed_package=out/'FixedPackages'/f'{fixed_resource}.mlpackage'
            inputs=fixture_inputs(arrays['conv_in'],arrays['rotary'],fixture['frames'])
            print('Materializing fixed shape:',scheme,fixture_name,fixture['frames'],flush=True)
            receipt=materialize_package(ct,package,fixture['frames'],fixed_package,inputs)
            write(out/f'{scheme}-{fixture_name}-static-shape.json',receipt)
            subprocess.run(['xcrun','coremlcompiler','compile',str(fixed_package),str(assets)],check=True)
            compiled=assets/f'{fixed_resource}.mlmodelc'
            if not compiled.is_dir():
                raise RuntimeError('Compiled fixed model folder missing.')
            fixed_resources[fixture_name]=fixed_resource
            fixed_compiled_bytes[fixture_name]=package_bytes(compiled)
            fixed_package_bytes[fixture_name]=package_bytes(fixed_package)
        models.append({'name':scheme,'resource':fixed_resources['short'],
                       'package_bytes':package_bytes(package),'compiled_bytes':sum(fixed_compiled_bytes.values()),
                       'fixed_resources':fixed_resources,'fixed_compiled_bytes':fixed_compiled_bytes,
                       'fixed_package_bytes':fixed_package_bytes})
    write(assets/'benchmark_manifest.json',{'models':models,'fixtures':fixtures,
        'shape_policy':'fixed_per_fixture',
        'bundle_size_note':'Separate static models per fixture duplicate weights for benchmark isolation; not a production packaging decision.',
        'muq_snapshot_digest':repair['muq_snapshot_digest'],
        'muq_revision':repair['muq_revision'],
        'reference':'Mac original-snapshot FP32 MuQ with identical saved conv/rotary input',
        'source_comparison':str(comparison),'track_id':repair['track_id'],
        'source_export_report_sha256':sha_file(repair_path),
        'scope':'Encoder only; short is a prefix of saved long-window conv input, not newly encoded 30s audio.'})
    template=Path(__file__).resolve().parent/'iphone_muq'
    for name in ('BenchmarkApp.swift','BenchmarkRunner.swift'):
        shutil.copyfile(template/name,out/name)
    project_dir=out/'Benchmark.xcodeproj'
    project_dir.mkdir()
    (project_dir/'project.pbxproj').write_text(project)
    print('Open:',project_dir)
    print('Select signing Team and physical iPhone. Test one candidate/fixture per run; share JSON.')


if __name__=='__main__':
    main()
