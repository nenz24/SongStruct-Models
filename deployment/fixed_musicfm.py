"""Materialize one fixed MusicFM input shape, preserving weight compression and checking CPU parity."""
from pathlib import Path
import tempfile
import numpy as np
from deployment.compress_coreml import operation_counts
from deployment.coreml_features import native_input
from deployment.verify_encoders import compare_embeddings
from deployment.diagnose_musicfm import precision_inventory


def fixed_shapes(frames):
    if not isinstance(frames,int) or not 1<=frames<=3000:
        raise ValueError('MusicFM frame count must be an integer in [1,3000].')
    return {'conv_in':(1,frames,1024),'rotary':(2,frames,1,1,64)}


def assert_fixed_inputs(spec,frames):
    expected=fixed_shapes(frames)
    observed={}
    for feature in spec.description.input:
        array=feature.type.multiArrayType
        observed[feature.name]=tuple(array.shape)
        ranges=list(array.shapeRange.sizeRanges)
        if ranges and any(r.lowerBound!=r.upperBound for r in ranges):
            raise ValueError(f'Input remains flexible after materialization: {feature.name}')
        if len(array.enumeratedShapes.shapes)>1:
            raise ValueError(f'Input has multiple enumerated shapes: {feature.name}')
    if observed!=expected:
        raise ValueError(f'Fixed input contract differs: {observed} vs {expected}')


def materialize_package(ct,source,frames,destination,inputs):
    destination=Path(destination)
    if destination.exists():
        raise ValueError('Static destination exists; choose a new output folder.')
    expected=fixed_shapes(frames)
    if set(inputs)!=set(expected) or any(tuple(inputs[k].shape)!=shape for k,shape in expected.items()):
        raise ValueError('Fixture does not match fixed-shape contract.')
    # Materializer can mutate its MIL program; always reload source for each shape.
    original=ct.models.MLModel(str(source),compute_units=ct.ComputeUnit.CPU_ONLY)
    metadata=dict(original.user_defined_metadata)
    original_types=precision_inventory(original.get_spec())
    reference=np.array(original.predict({k:native_input(v) for k,v in inputs.items()})['embedding'],copy=True)
    if reference.shape!=(1,frames,1024) or not np.isfinite(reference).all():
        raise ValueError('Dynamic source is invalid on CPU for this fixture.')
    with tempfile.TemporaryDirectory() as folder:
        intermediate=Path(folder)/'materialized.mlpackage'
        ct.models.utils.materialize_dynamic_shape_mlmodel(original,
            {'main':expected},str(intermediate),source_function_name='main')
        fixed=ct.models.MLModel(str(intermediate),compute_units=ct.ComputeUnit.CPU_ONLY)
        assert_fixed_inputs(fixed.get_spec(),frames)
        actual=np.asarray(fixed.predict({k:native_input(v) for k,v in inputs.items()})['embedding'])
        check=compare_embeddings(reference,actual)
        check['same_precision_strict_passed']=bool(reference.shape==actual.shape and np.allclose(reference,actual,atol=1e-3,rtol=1e-3))
        fixed_types=precision_inventory(fixed.get_spec())
        source_counts=original_types.get('declared_tensor_dtype_counts',{})
        fixed_counts=fixed_types.get('declared_tensor_dtype_counts',{})
        if source_counts and fixed_counts and source_counts.get('FLOAT16',0)==0 and fixed_counts.get('FLOAT16',0)>0:
            raise RuntimeError('Static materialization unexpectedly introduced FLOAT16 tensor declarations.')
        if not check['numerical_parity_passed'] or not check['same_precision_strict_passed']:
            raise RuntimeError(f'Fixed-shape CPU parity failed: {check}')
        operations=operation_counts(fixed.get_spec())
        scheme=metadata.get('compression_scheme')
        types=('constexpr_lut_to_dense',) if scheme=='palette6' else (
            'constexpr_affine_dequantize','constexpr_blockwise_shift_scale') if scheme=='int8' else ()
        if types and not any(operations.get(name,0)>0 for name in types):
            raise RuntimeError('Materialization removed expected weight-compression operations.')
        fixed.user_defined_metadata.update(metadata)
        fixed.user_defined_metadata['fixed_frames']=str(frames)
        fixed.user_defined_metadata['shape_policy']='fixed_per_fixture'
        destination.parent.mkdir(parents=True,exist_ok=True)
        fixed.save(str(destination))
    return {'frames':frames,'cpu_parity':check,'operations':operations,
            'source_package':str(source),'fixed_package':str(destination),
            'shape_policy':'fixed_per_fixture','iphone_verified':False,
            'source_precision_inventory':original_types,'fixed_precision_inventory':fixed_types}
