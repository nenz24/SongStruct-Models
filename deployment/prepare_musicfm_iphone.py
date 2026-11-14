"""Prepare an isolated MusicFM encoder benchmark with identical real conv/rotary fixtures."""
import argparse
import gc
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import numpy as np
from deployment.compress_coreml import package_bytes
from deployment.compress_musicfm import validate_baseline
from deployment.coreml_features import native_input, validate_musicfm_proof
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
    parser.add_argument('--bundle-id',default='com.danendra.MusicFMEncoderBenchmark')
    parser.add_argument('--team-id')
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Prepare on macOS with Xcode and coremltools.')
    import coremltools as ct
    comparison,out=[p.expanduser().resolve() for p in (args.comparison,args.output_dir)]
    if out.exists():
        raise ValueError('Choose a new output folder.')
    summary=read(comparison)
    baseline_path=Path(summary['baseline_pipeline_report']).expanduser().resolve()
    baseline=read(baseline_path)
    repair_path=baseline_path.parent/'repair_report.json'
    repair=read(repair_path)
    validate_baseline(repair,baseline,repair_path)
    source=Path(repair['package']).resolve()
    diagnostic=Path(repair['diagnostic_dir']).resolve()
    original_diagnostic=read(diagnostic/'diagnostic.json')
    arrays={name:np.load(diagnostic/f'{name}.npy',allow_pickle=False) for name in ('conv_in','rotary')}
    for name in arrays:
        if sha_file(diagnostic/f'{name}.npy')!=original_diagnostic['input_file_hashes'][name]:
            raise ValueError('Saved diagnostic inputs changed.')
    packages=[]
    for scheme,resource in [('int8','MusicFMINT8'),('palette6','MusicFMPalette6')]:
        candidate_path=comparison.parent/scheme/'compression_report.json'
        candidate=read(candidate_path)
        validate_musicfm_proof(candidate)
        if candidate['compression_scheme']!=scheme or candidate['source_repair_report_sha256']!=sha_file(repair_path):
            raise ValueError('Candidate source MusicFM differs from baseline.')
        validated=read(comparison.parent/f'{scheme}-all-validation.json')
        if not validated['full_validation'] or validated['musicfm_override']['report_sha256']!=sha_file(candidate_path):
            raise ValueError('Candidate must complete full validation before benchmark preparation.')
        check_same_inventory(baseline,validated)
        packages.append((scheme,resource,Path(candidate['package']).resolve()))
    if args.include_fp32:
        packages.append(('fp32','MusicFMFP32',source))
    resources=[resource+'_'+fixture+'.mlmodelc' for _,resource,_ in packages
               for fixture in ('short','failed_full')]
    for name in ('short','failed_full'):
        resources.extend([f'conv_{name}.f32',f'rotary_{name}.f32',f'reference_{name}.f32'])
    resources.append('benchmark_manifest.json')
    project=project_text(args.bundle_id,args.team_id,resource_names=resources)
    project=project.replace('GENERATE_INFOPLIST_FILE = YES;',
        'GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_CFBundleDisplayName = "MusicFM Bench";')
    out.mkdir(parents=True,exist_ok=False)
    assets=out/'Assets'
    assets.mkdir()
    model=ct.models.MLModel(str(source),compute_units=ct.ComputeUnit.CPU_ONLY)
    if (model.user_defined_metadata.get('precision')!='fp32'
            or model.user_defined_metadata.get('musicfm_weights_sha256')!=repair['musicfm_weights_sha256']):
        raise ValueError('Source reference package metadata differs.')
    fixtures=[]
    for name,frames in [('short',min(750,arrays['conv_in'].shape[1])),('failed_full',arrays['conv_in'].shape[1])]:
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
        print('Compiling MusicFM candidate:',scheme,flush=True)
        check=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
        meta=dict(check.user_defined_metadata)
        del check
        if (meta.get('precision')!='fp32' or meta.get('musicfm_weights_sha256')!=repair['musicfm_weights_sha256']
                or meta.get('musicfm_stats_sha256')!=repair['musicfm_stats_sha256']
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
        'musicfm_weights_sha256':repair['musicfm_weights_sha256'],
        'musicfm_stats_sha256':repair['musicfm_stats_sha256'],
        'reference':'Mac repaired FP32 MusicFM with identical saved conv/rotary input',
        'source_comparison':str(comparison),'track_id':original_diagnostic['track_id'],
        'start_sample':original_diagnostic['start_sample'],'end_sample':original_diagnostic['end_sample'],
        'scope':'Encoder only; short is a prefix of failed-window conv input, not newly encoded 30s audio.'})
    template=Path(__file__).resolve().parent/'iphone_musicfm'
    for name in ('BenchmarkApp.swift','BenchmarkRunner.swift'):
        shutil.copyfile(template/name,out/name)
    project_dir=out/'Benchmark.xcodeproj'
    project_dir.mkdir()
    (project_dir/'project.pbxproj').write_text(project)
    print('Open:',project_dir)
    print('Select signing Team and physical iPhone. Test one candidate/fixture per run; share JSON.')


if __name__=='__main__':
    main()
