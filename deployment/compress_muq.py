"""Compress the fully validated original-snapshot MuQ FP32 independently to INT8 or palette6."""
import argparse
from pathlib import Path
import platform
import numpy as np
from deployment.compress_coreml import compression_config, operation_counts, package_bytes
from deployment.coreml_features import native_input
from deployment.export_musicfm_fp32 import numerical
from finetuning.common import read, sha_file, fingerprint, write


def package_digest(path):
    path=Path(path)
    return fingerprint([(str(p.relative_to(path)),sha_file(p)) for p in sorted(path.rglob('*')) if p.is_file()])


def validate_baseline(source, validation, source_path):
    if (source.get('source_kind')!='original_training_snapshot' or
            source.get('compute_precision')!='float32' or not source.get('source_window_passed') or
            source.get('compression_scheme')):
        raise ValueError('Use the verified, uncompressed original-snapshot MuQ FP32 export.')
    if (not validation.get('full_validation') or validation.get('muq_scheme')!='fp32' or
            validation.get('head_scheme')!='palette6' or not validation.get('reused_musicfm')):
        raise ValueError('Full MuQ FP32 validation with the fixed palette6 combination required.')
    if validation.get('muq_export_report_sha256')!=sha_file(source_path):
        raise ValueError('Full validation references a different MuQ export.')
    for key in ('plan_id','provenance_id'):
        if source[key]!=validation[key]:raise ValueError('MuQ validation provenance mismatch: '+key)
    metrics=validation['metrics']
    if metrics.get('sections',0)<1 or metrics.get('low_coverage_sections',1)!=0:
        raise ValueError('MuQ FP32 baseline must have full section coverage.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-report',type=Path,required=True)
    p.add_argument('--validation-report',type=Path,required=True)
    p.add_argument('--scheme',choices=['int8','palette6'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run Core ML compression on macOS.')
    import coremltools as ct
    import coremltools.optimize.coreml as opt
    source_path,validation_path,out=[p.expanduser().resolve() for p in (a.source_report,a.validation_report,a.output_dir)]
    if out.exists():raise ValueError('Choose a new output directory.')
    proof,validation=read(source_path),read(validation_path)
    validate_baseline(proof,validation,source_path)
    source=Path(proof['package']).resolve()
    if source in out.parents:raise ValueError('Output cannot be inside source package.')
    cache_meta=read(Path(validation['diagnostic_cache'])/'encoder_provenance.json')
    if (cache_meta['id']!=validation['candidate_encoder_id'] or
            fingerprint({k:v for k,v in cache_meta.items() if k!='id'})!=cache_meta['id'] or
            cache_meta['muq_package_digest']!=package_digest(source)):
        raise ValueError('MuQ package differs from the fully validated source.')
    original=ct.models.MLModel(str(source),compute_units=ct.ComputeUnit.CPU_ONLY)
    metadata=dict(original.user_defined_metadata)
    if metadata.get('precision')!='fp32' or metadata.get('hidden_state_index')!='10' or metadata.get('compression_scheme'):
        raise ValueError('Expected uncompressed FP32 hidden-state 10.')
    for key in ('muq_repo','muq_revision','muq_snapshot_digest'):
        if metadata.get(key)!=proof[key]:raise ValueError('MuQ metadata mismatch: '+key)
    inputs={k:native_input(np.load(source_path.parent/(k+'.npy'),allow_pickle=False)) for k in ('conv_in','rotary')}
    for k in inputs:
        if sha_file(source_path.parent/(k+'.npy'))!=proof['fixture_hashes'][k]:raise ValueError('MuQ fixture changed.')
    print('Compressing MuQ:',a.scheme,flush=True)
    before=operation_counts(original.get_spec())
    cfg=compression_config(opt,a.scheme)
    if a.scheme=='int8':
        candidate=opt.linear_quantize_weights(original,config=cfg)
        kinds=('constexpr_affine_dequantize','constexpr_blockwise_shift_scale')
    else:
        candidate=opt.palettize_weights(original,config=cfg)
        kinds=('constexpr_lut_to_dense',)
    after=operation_counts(candidate.get_spec())
    added=sum(after.get(k,0)-before.get(k,0) for k in kinds)
    if added<=0:raise RuntimeError('No expected compressed-weight operations added.')
    candidate.user_defined_metadata.update(metadata)
    candidate.user_defined_metadata['compression_scheme']=a.scheme
    candidate.user_defined_metadata['source_export_report_sha256']=sha_file(source_path)
    candidate.user_defined_metadata['compression_scope']='MuQ weights only; FLOAT32 compute; no activation quantization'
    out.mkdir(parents=True,exist_ok=False)
    package=out/'MuQ_L10.mlpackage';candidate.save(str(package))
    checks=[]
    for count in sorted(set((min(750,inputs['conv_in'].shape[1]),inputs['conv_in'].shape[1]))):
        values={k:native_input(v[:,:count]) for k,v in inputs.items()}
        reference=np.asarray(original.predict(values)['embedding'])
        if not np.isfinite(reference).all():raise ValueError('FP32 MuQ source produced NaN/Inf.')
        actual=np.asarray(candidate.predict(values)['embedding'])
        checks.append({'frames':count,**numerical(reference,actual)})
    finite=all(c['candidate']['nan_count']==0 and c['candidate']['inf_count']==0 and
        c['candidate']['shape']==c['reference']['shape'] for c in checks)
    size,base=package_bytes(package),package_bytes(source)
    report={k:proof[k] for k in ('source_kind','compute_precision','muq_repo','muq_revision',
        'muq_snapshot_digest','plan_id','provenance_id','app_manifest_sha256','frontend_packages')}
    report.update(compression_scheme=a.scheme,package=str(package),package_digest=package_digest(package),
        source_report=str(source_path),source_report_sha256=sha_file(source_path),
        source_validation_report=str(validation_path),source_validation_report_sha256=sha_file(validation_path),
        source_window_passed=True,source_full_validation=True,window_finite_passed=finite,
        checks=checks,source_bytes=base,candidate_bytes=size,size_reduction_fraction=1-size/base,
        source_operations=before,candidate_operations=after,compressed_weight_ops_added=added,
        activation_quantization=False,guaranteed_integer_compute=False,coremltools=ct.__version__,
        ready_for_app_integration=False,candidate_full_validation_pending=True,
        note='Lossy-weight numerical drift is reported. Full validation and iPhone benchmarking still required.')
    write(out/'compression_report.json',report)
    if not finite:raise RuntimeError('Compressed MuQ returned invalid values or shape; candidate blocked.')
    print(f'Size: {base/1024**2:.2f} -> {size/1024**2:.2f} MiB')
    print('Report:',out/'compression_report.json')


if __name__=='__main__':main()
