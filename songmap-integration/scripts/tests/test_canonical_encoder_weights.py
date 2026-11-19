import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import canonical_encoder_weights as m


def model(ct,frames,profile,offset=0.,extra_axis=False,key='layers_0_conv_weight'):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,16,frames))],opset_version=ct.target.iOS18)
    def program(x):
        shape=(16,16,1,1) if extra_axis else (16,16,1)
        indices=(np.arange(256).reshape(shape)%64).astype(types.np_uint6_dtype)
        weight=mb.constexpr_lut_to_dense(indices=mb.const(val=indices,name=key+'_palettized_indices_'+str(profile)),
            lut=(np.arange(64,dtype=np.float32)/100+offset).reshape((1,)*len(shape)+(64,1)),name=key+'_palettized')
        if extra_axis:
            return mb.squeeze(x=mb.conv(x=mb.expand_dims(x=x,axes=[3]),weight=weight),axes=[3])
        return mb.conv(x=x,weight=weight)
    return ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,skip_model_load=True)


def grouped_model(ct,frames,offset=0.,extra_axis=False,linear=False,axis=0):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,frames,64) if linear else (1,64,frames))],opset_version=ct.target.iOS18)
    def program(x):
        shape=(64,64) if linear else ((64,64,1,1) if extra_axis else (64,64,1))
        indices=(np.arange(4096).reshape(shape)%64).astype(types.np_uint6_dtype)
        # Two groups of 32 output channels, each with its own 64 scalar entries.
        lut=np.stack([np.arange(64,dtype=np.float32)/100+offset,
                      np.arange(64,dtype=np.float32)/100+2+offset])
        groups=[1]*len(shape);groups[axis]=2
        weight=mb.constexpr_lut_to_dense(indices=indices,
            lut=lut.reshape(tuple(groups)+(64,1)),
            name='layers_0_ffn1_intermediate_dense_weight_palettized')
        if linear:return mb.linear(x=x,weight=weight)
        if extra_axis:
            return mb.squeeze(x=mb.conv(x=mb.expand_dims(x=x,axes=[3]),weight=weight),axes=[3])
        return mb.conv(x=x,weight=weight)
    return ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,skip_model_load=True)


def folded_model(ct,frames,start=20,offset=0.,count=10,duplicate_bias=False,bias_offset=0.,shared_weight=False):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,64,frames))],opset_version=ct.target.iOS18)
    def program(x):
        values=x;first=None
        for layer in range(count):
            indices=(np.arange(192).reshape(64,1,3)%64).astype(types.np_uint6_dtype)
            lut=np.stack([np.arange(64,dtype=np.float32)/100+offset+layer*.02,
                          np.arange(64,dtype=np.float32)/100+1+offset+layer*.02]).reshape(2,1,1,64,1)
            weight=first if shared_weight and first is not None else mb.constexpr_lut_to_dense(indices=indices,lut=lut,name=f'const_{start+2*layer}_palettized')
            if first is None:first=weight
            bias=np.linspace(-.1,.1,64,dtype=np.float32)+(0. if duplicate_bias else layer*.05)+bias_offset
            values=mb.conv(x=values,weight=weight,bias=bias,groups=64,pad_type='custom',pad=[1,1],name=f'depthwise_{layer}')
        return values
    return ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,skip_model_load=True)


@unittest.skipUnless(importlib.util.find_spec('coremltools'),'Canonical graph tests require coremltools')
class CanonicalWeightsTests(unittest.TestCase):
    def test_automatic_const_number_overlap_maps_all_ten_folded_weights_by_use(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources={};new={}
            for name,frames,start,offset in [('short_30',7,20,0.),('long_120',11,20,.002),('partial',13,10,.004)]:
                sources[name]=root/(name+'-old.mlpackage');folded_model(ct,frames,start,offset).save(str(sources[name]))
            source_hashes={k:m.package_digest(v) for k,v in sources.items()}
            _,program=m.model_program(ct,sources['long_120']);bank=m.build_bank(program.functions['main'],expected=10)
            canonical_aliases={v['binding']['original_parameter']:k for k,v in bank.items()}
            _,partial=m.model_program(ct,sources['partial'])
            partial_aliases={m.parameter_binding(op)[1]['original_parameter']:k for k,op in m.parameters(partial.functions['main']).items()}
            # The same counter refers to different layers; a constant-name match would be wrong.
            self.assertNotEqual(canonical_aliases['const_20'],partial_aliases['const_20'])
            self.assertEqual(canonical_aliases['const_30'],partial_aliases['const_20'])
            for name in m.FUNCTIONS:
                new[name]=root/(name+'.mlpackage');proof=m.export_profile(ct,sources[name],new[name],bank)
                self.assertTrue(proof['non_parameter_graph_unchanged'])
                self.assertEqual(len(proof['differences']),10)
                if name=='long_120':self.assertEqual(proof['original_graph_sha256'],proof['unified_graph_sha256'])
                _,loaded=m.model_program(ct,new[name]);m.verify_unified(loaded.functions['main'],bank)
            report=m.pack(ct,new,root/'packed.mlpackage');self.assertTrue(all(all(c.values()) for c in report['checks'].values()))
            _,packed=m.model_program(ct,root/'packed.mlpackage')
            for function in packed.functions.values():m.verify_unified(function,bank)
            self.assertEqual(source_hashes,{k:m.package_digest(v) for k,v in sources.items()})

    def test_anonymous_binding_rejects_duplicate_bias_and_multiple_weight_consumers(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            for name,options,reason in [('ambiguous',dict(count=2,duplicate_bias=True),'Ambiguous duplicate'),
                                        ('shared',dict(count=2,shared_weight=True),'exactly one direct')]:
                path=Path(d)/(name+'.mlpackage');folded_model(ct,7,**options).save(str(path))
                _,program=m.model_program(ct,path)
                with self.assertRaisesRegex(ValueError,reason):m.parameters(program.functions['main'])
            path=Path(d)/'ordinary.mlpackage';model(ct,7,0,key='const_10').save(str(path))
            _,program=m.model_program(ct,path)
            with self.assertRaisesRegex(ValueError,'not a fixed-bias depthwise'):m.parameters(program.functions['main'])

    def test_identical_automatic_name_with_changed_bias_cannot_bind_to_canonical_layer(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'long.mlpackage';target=Path(d)/'different.mlpackage'
            folded_model(ct,7,count=1).save(str(source));folded_model(ct,11,count=1,bias_offset=1.).save(str(target))
            _,program=m.model_program(ct,source);bank=m.build_bank(program.functions['main'],expected=1)
            _,program=m.model_program(ct,target)
            with self.assertRaisesRegex(ValueError,'Parameter IDs differ'):m.unify(program,bank)

    def test_grouped_linear_layout_and_group_axis_are_not_flattened_away(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'long.mlpackage';target=root/'linear.mlpackage'
            grouped_model(ct,11,axis=1).save(str(source));grouped_model(ct,7,offset=.01,linear=True,axis=1).save(str(target))
            _,program=m.model_program(ct,source);bank=m.build_bank(program.functions['main'],expected=1)
            original=next(iter(m.parameters(program.functions['main']).values())).materialized_val_inference()
            proof=m.export_profile(ct,target,root/'new.mlpackage',bank)
            self.assertTrue(proof['non_parameter_graph_unchanged'])
            _,program=m.model_program(ct,root/'new.mlpackage')
            op=next(o for o in program.functions['main'].operations if o.op_type=='constexpr_lut_to_dense')
            self.assertEqual(op.lut.shape,(1,2,64,1))
            np.testing.assert_array_equal(op.materialized_val_inference().reshape(-1),original.reshape(-1))
            # Same flat bytes, different grouping axis: must not count as the same encoding.
            self.assertNotEqual(m.encoded_hash(op.indices.val,op.lut.val),
                                m.encoded_hash(op.indices.val,op.lut.val.reshape(2,1,64,1)))
            m.verify_unified(program.functions['main'],bank)
            bad={k:dict(v) for k,v in bank.items()}
            key=next(iter(bad));bad[key]['encoded_sha256']=m.encoded_hash(op.indices.val,op.lut.val.reshape(2,1,64,1))
            with self.assertRaisesRegex(ValueError,'payload changed'):m.verify_unified(program.functions['main'],bad)

    def test_grouped_32_channel_palettes_keep_distinct_groups_across_export_and_packing(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources={};unified={}
            for name,frames,offset,extra in [('short_30',7,0.,False),('long_120',11,.01,False),('partial',13,.02,True)]:
                sources[name]=root/(name+'-old.mlpackage')
                grouped_model(ct,frames,offset,extra).save(str(sources[name]))
            source_hashes={k:m.package_digest(v) for k,v in sources.items()}
            _,program=m.model_program(ct,sources['long_120'])
            original_weight=next(iter(m.parameters(program.functions['main']).values())).materialized_val_inference().copy()
            self.assertEqual(float(original_weight[0,0,0]),float(np.float32(.01)))
            self.assertEqual(float(original_weight[32,0,0]),float(np.float32(2.01)))
            bank=m.build_bank(program.functions['main'],expected=1)
            for name in m.FUNCTIONS:
                unified[name]=root/(name+'.mlpackage');proof=m.export_profile(ct,sources[name],unified[name],bank)
                self.assertTrue(proof['non_parameter_graph_unchanged'])
                self.assertEqual(proof['differences'][0]['changed'],name!='long_120')
                if name=='long_120':self.assertEqual(proof['original_graph_sha256'],proof['unified_graph_sha256'])
                _,exported=m.model_program(ct,unified[name]);op=next(o for o in exported.functions['main'].operations if o.op_type=='constexpr_lut_to_dense')
                self.assertEqual(op.lut.shape[:2],(2,1))
                np.testing.assert_array_equal(op.materialized_val_inference().reshape(-1),original_weight.reshape(-1))
            m.pack(ct,unified,root/'shared.mlpackage')
            _,packed=m.model_program(ct,root/'shared.mlpackage')
            for function in packed.functions.values():m.verify_unified(function,bank)
            self.assertEqual(source_hashes,{k:m.package_digest(v) for k,v in sources.items()})

    def test_linear_parameter_inputs_stay_supported_without_runtime_reshape(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        def dense_model(frames,offset):
            @mb.program(input_specs=[mb.TensorSpec(shape=(1,frames,16))],opset_version=ct.target.iOS18)
            def program(x):
                indices=(np.arange(256).reshape(16,16)%64).astype(types.np_uint6_dtype)
                w=mb.constexpr_lut_to_dense(indices=indices,lut=(np.arange(64,dtype=np.float32)/100+offset).reshape(1,1,64,1),name='layers_0_dense_weight_palettized')
                return mb.linear(x=x,weight=w)
            return ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
                minimum_deployment_target=ct.target.iOS18,skip_model_load=True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'long.mlpackage';other=root/'short.mlpackage'
            dense_model(11,.01).save(str(source));dense_model(7,0.).save(str(other))
            _,program=m.model_program(ct,source);bank=m.build_bank(program.functions['main'],expected=1)
            proof=m.export_profile(ct,other,root/'unified.mlpackage',bank)
            self.assertTrue(proof['non_parameter_graph_unchanged'])

    def test_reuse_long_bank_preserves_nonparameter_graph_and_shares_real_storage(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original={};new={}
            for i,(name,frames) in enumerate(zip(m.FUNCTIONS,(7,11,13))):
                original[name]=root/(name+'-old.mlpackage')
                model(ct,frames,i,offset=.01*i,extra_axis=name=='partial').save(str(original[name]))
            _,long=m.model_program(ct,original['long_120']);bank=m.build_bank(long.functions['main'],expected=1)
            for name in m.FUNCTIONS:
                new[name]=root/(name+'.mlpackage');proof=m.export_profile(ct,original[name],new[name],bank)
                self.assertTrue(proof['non_parameter_graph_unchanged'])
                self.assertEqual(proof['parameters'],1)
                self.assertEqual(proof['differences'][0]['changed'],name!='long_120')
                if name=='long_120':self.assertEqual(proof['original_graph_sha256'],proof['unified_graph_sha256'])
            report=m.pack(ct,new,root/'shared.mlpackage')
            self.assertTrue(report['smaller'])
            _,program=m.model_program(ct,root/'shared.mlpackage')
            for function in program.functions.values():m.verify_unified(function,bank)

    def test_unknown_parameter_ids_and_non_singleton_layout_changes_are_rejected(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'long.mlpackage';model(ct,7,0).save(str(source))
            _,program=m.model_program(ct,source);bank=m.build_bank(program.functions['main'],expected=1)
            wrong=Path(d)/'wrong.mlpackage';model(ct,7,1,key='unrelated').save(str(wrong))
            _,program=m.model_program(ct,wrong)
            with self.assertRaises(ValueError):m.unify(program,bank)
            huge=Path(d)/'huge.mlpackage';model(ct,7,1,offset=10.).save(str(huge))
            _,program=m.model_program(ct,huge)
            with self.assertRaises(ValueError):m.unify(program,bank)
        self.assertTrue(m.shape_compatible((16,16,1),(16,16,1,1)))
        self.assertFalse(m.shape_compatible((16,8,2),(16,16,1)))


if __name__=='__main__':unittest.main()
