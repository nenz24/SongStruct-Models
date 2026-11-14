"""Independent weight compression of verified repaired MusicFM FP32, not the invalid old model."""
import argparse
from pathlib import Path
import platform
import numpy as np
from deployment.compress_coreml import compression_config, operation_counts, package_bytes
from deployment.coreml_features import native_input, validate_musicfm_proof
from deployment.diagnose_musicfm import invalid_frames
from deployment.export_musicfm_fp32 import numerical
from finetuning.common import read, sha_file, write


def validate_baseline(repair, pipeline, repair_path):
    if not repair.get('exact_window_passed') or repair.get('compute_precision')!='float32' or repair.get('compression_scheme'):
        raise ValueError('Use uncompressed MusicFM FP32 with passing exact-window checks.')
    if not pipeline.get('full_validation') or pipeline.get('tracks',0)<1:
        raise ValueError('MusicFM FP32 pipeline must complete full validation.')
    override=pipeline.get('musicfm_override') or {}
    if override.get('report_sha256')!=sha_file(repair_path):
        raise ValueError('Full-validation pipeline does not reference this repair report.')
    for key in ('musicfm_weights_sha256','musicfm_stats_sha256'):
        if not repair.get(key) or repair[key]!=override.get(key):
            raise ValueError('Baseline MusicFM source provenance differs.')
    if override.get('rotary_mode')!='original_app_values_unchanged' or override.get('compression_scheme'):
        raise ValueError('Expected original rotary and uncompressed FP32 MusicFM baseline.')
    metrics=pipeline.get('coreml_encoders_with_fp32_head',{})
    if metrics.get('sections',0)<1 or metrics.get('low_coverage_sections',1)!=0:
        raise ValueError('Full-validation baseline must have supervised sections with full coverage.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-report',type=Path,required=True)
    parser.add_argument('--pipeline-report',type=Path,required=True)
    parser.add_argument('--scheme',choices=['int8','palette6'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Darwin':
        raise RuntimeError('Run on macOS with coremltools 8.x.')
    import coremltools as ct
    import coremltools.optimize.coreml as optimization
    repair_path,pipeline_path,out=[p.expanduser().resolve() for p in (args.repair_report,args.pipeline_report,args.output_dir)]
    if out.exists():
        raise ValueError('Choose a new output directory.')
    repair,pipeline=read(repair_path),read(pipeline_path)
    validate_baseline(repair,pipeline,repair_path)
    source=Path(repair['package']).resolve()
    if source in out.parents:
        raise ValueError('Output cannot be inside source package.')
    original=ct.models.MLModel(str(source),compute_units=ct.ComputeUnit.CPU_ONLY)
    metadata=dict(original.user_defined_metadata)
    if metadata.get('precision')!='fp32' or metadata.get('hidden_state_index')!='10' or metadata.get('compression_scheme'):
        raise ValueError('Source must be uncompressed FP32 MusicFM hidden-state 10.')
    for key in ('musicfm_weights_sha256','musicfm_stats_sha256'):
        if metadata.get(key)!=repair[key]:
            raise ValueError('Source package metadata differs from verified baseline.')
    diagnostic=Path(repair['diagnostic_dir']).resolve()
    diagnosis=read(diagnostic/'diagnostic.json')
    inputs={name:native_input(np.load(diagnostic/f'{name}.npy',allow_pickle=False)) for name in ('conv_in','rotary')}
    for name in inputs:
        if sha_file(diagnostic/f'{name}.npy')!=diagnosis['input_file_hashes'][name]:
            raise ValueError('Exact-window inputs changed.')
    reference=np.array(original.predict(inputs)['embedding'],copy=True)
    if not np.isfinite(reference).all():
        raise ValueError('FP32 source is no longer finite on the exact failed window.')
    print('Compressing repaired MusicFM:',args.scheme,flush=True)
    cfg=compression_config(optimization,args.scheme)
    before=operation_counts(original.get_spec())
    if args.scheme=='int8':
        candidate=optimization.linear_quantize_weights(original,config=cfg)
        types=('constexpr_affine_dequantize','constexpr_blockwise_shift_scale')
    else:
        candidate=optimization.palettize_weights(original,config=cfg)
        types=('constexpr_lut_to_dense',)
    after=operation_counts(candidate.get_spec())
    added=sum(after.get(k,0)-before.get(k,0) for k in types)
    if added<=0:
        raise RuntimeError('No expected compressed-weight operations added.')
    candidate.user_defined_metadata.update(metadata)
    candidate.user_defined_metadata['compression_scheme']=args.scheme
    candidate.user_defined_metadata['compression_scope']='MusicFM encoder weights only; FLOAT32 compute export; no activation quantization'
    candidate.user_defined_metadata['source_repair_report_sha256']=sha_file(repair_path)
    out.mkdir(parents=True,exist_ok=False)
    package=out/'MusicFM_L10.mlpackage'
    candidate.save(str(package))
    # Exercise the known pathological input repeatedly. Lossy numerical drift is reported, not hidden.
    checks=[]
    for index in range(3):
        result=np.asarray(candidate.predict({k:native_input(v) for k,v in inputs.items()})['embedding'])
        check={'call':index+1,**numerical(reference,result)}
        checks.append(check)
        print(f'Exact window {index+1}: NaN={check["candidate"]["nan_count"]}; Inf={check["candidate"]["inf_count"]}',flush=True)
    finite=all(c['candidate']['nan_count']==0 and c['candidate']['inf_count']==0
               and c['candidate']['shape']==list(reference.shape) for c in checks)
    base_size,size=package_bytes(source),package_bytes(package)
    report={'compression_scheme':args.scheme,'compute_precision':'float32','coremltools':ct.__version__,
        'musicfm_weights_sha256':repair['musicfm_weights_sha256'],
        'musicfm_stats_sha256':repair['musicfm_stats_sha256'],
        'source_exact_window_passed':True,'source_full_validation':True,
        'window_finite_passed':finite,'exact_window_passed':all(c['numerical_parity_passed'] for c in checks),
        'package':str(package),'source_package':str(source),
        'source_repair_report':str(repair_path),'source_repair_report_sha256':sha_file(repair_path),
        'source_pipeline_report':str(pipeline_path),'source_pipeline_report_sha256':sha_file(pipeline_path),
        'source_bytes':base_size,'candidate_bytes':size,'size_reduction_fraction':1-size/base_size,
        'source_operations':before,'candidate_operations':after,'compressed_weight_ops_added':added,
        'exact_window_checks':checks,'activation_quantization':False,'guaranteed_integer_compute':False,
        'ready_for_app_integration':False,'candidate_full_validation_pending':True,
        'note':'Finite pathological-window smoke only. Numerical drift expected for lossy weights; complete pipeline validation required.'}
    write(out/'compression_report.json',report)
    if not finite:
        raise RuntimeError('Compressed MusicFM produced non-finite/wrong-shape output; candidate blocked.')
    validate_musicfm_proof(report)
    print(f'Size: {base_size/1024**2:.2f} -> {size/1024**2:.2f} MiB')
    print('Report:',out/'compression_report.json')


if __name__=='__main__':
    main()
