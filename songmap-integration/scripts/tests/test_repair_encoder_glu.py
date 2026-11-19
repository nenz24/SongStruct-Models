import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import repair_encoder_glu as m


def toy_program():
    from coremltools.converters.mil import Builder as mb
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,16,7))],opset_version=__import__('coremltools').target.iOS18)
    def program(x):
        weights=np.random.RandomState(3).normal(size=(32,16,1)).astype(np.float32)
        projected=mb.conv(x=x,weight=weights,pad_type='valid',name='wide')
        a=mb.slice_by_index(x=projected,begin=[0,0,0],end=[1,16,7],end_mask=[True,False,True])
        b=mb.slice_by_index(x=projected,begin=[0,16,0],end=[1,32,7],end_mask=[True,True,True])
        return mb.mul(x=a,y=mb.sigmoid(x=b))
    return program


def weight_value(variable):
    if variable.val is not None:return variable.val
    op=variable.op
    if op.op_type.startswith('constexpr_'):return op.materialized_val_inference()
    if op.op_type=='slice_by_index':
        source=weight_value(op.x);rank=len(source.shape)
        def arg(name,default):
            value=op.inputs.get(name);return value.val if value is not None else default
        begin=arg('begin',[0]*rank);end=arg('end',source.shape)
        bm=arg('begin_mask',[False]*rank);em=arg('end_mask',[False]*rank)
        return source[tuple(slice(None if bm[i] else begin[i],None if em[i] else end[i]) for i in range(rank))]
    raise AssertionError('Unexpected weight operation: '+op.op_type)


@unittest.skipUnless(importlib.util.find_spec('coremltools'),'MIL graph tests require coremltools')
class RepairGLUTests(unittest.TestCase):
    def test_graph_splits_convolution_and_removes_both_activation_slices(self):
        program=toy_program();before=next(op.weight.val.copy() for op in program.functions['main'].operations if op.op_type=='conv')
        self.assertEqual(len(m.wide_convolutions(program,width=16)),1)
        changed=m.split_glu(program,width=16,expected=1)
        self.assertEqual(changed,['wide']);self.assertFalse(m.wide_convolutions(program,width=16))
        conv=[op for op in program.functions['main'].operations if op.op_type=='conv']
        self.assertEqual(len(conv),2);self.assertTrue(all(op.outputs[0].shape==(1,16,7) for op in conv))
        np.testing.assert_array_equal(np.concatenate([op.weight.val for op in conv]),before)
        program.validate()

    def test_partial_or_extra_channel_consumers_are_rejected(self):
        program=toy_program();conv=next(op for op in program.functions['main'].operations if op.op_type=='conv')
        slices=list(conv.outputs[0].child_ops)
        self.assertEqual({m.slice_half(op,16) for op in slices},{0,1})
        for width in (8,32):
            with self.assertRaises(ValueError):m.slice_half(slices[0],width)
        with self.assertRaises(ValueError):m.split_glu(toy_program(),width=16,expected=10)

    def test_actual_compressed_mil_roundtrip_keeps_payloads_and_weight_values(self):
        import coremltools as ct
        import coremltools.optimize.coreml as optimize
        source=ct.convert(toy_program(),source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS18,skip_model_load=True)
        source=optimize.palettize_weights(source,config=optimize.OptimizationConfig(global_config=optimize.OpPalettizerConfig(mode='uniform',nbits=6,weight_threshold=0)))
        before=m.compressed_payloads(source._mil_program)
        weights=next(weight_value(op.weight).copy() for op in source._mil_program.functions['main'].operations if op.op_type=='conv')
        self.assertTrue(before)
        repaired,proof=m.repair_model(ct,source,width=16,expected=1)
        self.assertEqual(m.compressed_payloads(repaired._mil_program),before)
        self.assertFalse(proof['weights_requantized']);self.assertTrue(proof['compressed_payloads_unchanged'])
        self.assertFalse(m.wide_convolutions(repaired._mil_program,width=16))
        conv=[op for op in repaired._mil_program.functions['main'].operations if op.op_type=='conv']
        np.testing.assert_array_equal(np.concatenate([weight_value(op.weight) for op in conv]),weights)
        self.assertEqual(len(m.wide_convolutions(source._mil_program,width=16)),1) # Source is unchanged.
        with tempfile.TemporaryDirectory() as d:
            repaired.save(str(Path(d)/'repaired.mlpackage'))
            self.assertTrue((Path(d)/'repaired.mlpackage/Manifest.json').is_file())


class RepairBindingTests(unittest.TestCase):
    def fixture(self, root):
        parents={role:root/(role+'-original.json') for role in ('musicfm','muq')}
        graphs={role:dict(source_state_sha256=role+'-weights',plan_id='plan',provenance_id='data') for role in parents}
        reports={};summary=dict(protocol=m.PROTOCOL,policy=m.POLICY,weights_requantized=False,
            parent_exports={role:'hash:'+str(path) for role,path in parents.items()},roles={})
        for role in parents:
            proof=dict(graphs[role],protocol=m.PROTOCOL,policy=m.POLICY,encoder=role,weights_requantized=False,
                parent_export_sha256=summary['parent_exports'][role],fixtures=[],checks=[])
            for frames in (750,3000):
                proof['fixtures'].append(dict(frames=frames,resource=f'{role}_{frames}',resource_digest='compiled'))
                proof['checks'].append(dict(frames=frames,compressed_payloads_unchanged=True,weights_requantized=False,
                    wide_convolution_count_after=0,compressed_payload_count=20,changed_convolutions=[str(i) for i in range(10)],
                    source_fixture=dict(passed=True),cpu_checks=[dict(passed=True) for _ in range(3)]))
            relative=role+'/repair-export.json';path=root/relative
            reports[path]=proof;summary['roles'][role]=dict(report=relative,sha256='hash:'+str(path))
        summary_path=root/'repair-report.json';reports[summary_path]=summary
        def load():
            return m.load_overrides(summary_path,parents,graphs,read_fn=lambda p:reports[p],
                sha_fn=lambda p:'hash:'+str(p),digest_fn=lambda p:'compiled')
        return reports,summary,load

    def test_repair_is_bound_to_both_validated_parents_and_compiled_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();reports,summary,load=self.fixture(root)
            paths,graphs=load();self.assertEqual(set(paths),{'musicfm','muq'})
            self.assertTrue(all(g['policy']==m.POLICY for g in graphs.values()))
            summary['parent_exports']['muq']='different'
            with self.assertRaises(ValueError):load()

    def test_invalid_repair_cannot_enable_gpu(self):
        for change in ('cpu','weights','wide','missing','compiled','escape','provenance'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();reports,summary,load=self.fixture(root)
                proof=reports[root/'muq/repair-export.json']
                if change=='cpu':proof['checks'][1]['cpu_checks'][2]['passed']=False
                elif change=='weights':proof['checks'][0]['compressed_payloads_unchanged']=False
                elif change=='wide':proof['checks'][1]['wide_convolution_count_after']=1
                elif change=='missing':proof['checks'].pop()
                elif change=='compiled':proof['fixtures'][0]['resource_digest']='modified'
                elif change=='escape':summary['roles']['muq']['report']='../escape.json'
                elif change=='provenance':proof['source_state_sha256']='other-weights'
                with self.assertRaises(ValueError):load()


if __name__=='__main__':unittest.main()
