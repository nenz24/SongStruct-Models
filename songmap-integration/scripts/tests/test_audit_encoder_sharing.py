import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_encoder_sharing as m


def palette_program(permuted=False,changed=False,flat=False):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,16,7))],opset_version=ct.target.iOS18)
    def program(x):
        indices=(np.arange(256).reshape(16,16,1)%64).astype(types.np_uint6_dtype)
        lut=np.arange(64,dtype=np.float32)
        if changed:lut[0]+=.25
        if permuted:lut=lut[::-1].copy();indices=(63-indices).astype(types.np_uint6_dtype)
        if flat:indices=indices.reshape(16,16)
        lut=lut.reshape((1,)*indices.ndim+(64,1))
        weight=mb.constexpr_lut_to_dense(indices=indices,lut=lut)
        if flat:weight=mb.reshape(x=weight,shape=[16,16,1])
        return mb.conv(x=x,weight=weight)
    return program


@unittest.skipUnless(importlib.util.find_spec('coremltools'),'MIL palette audit tests require coremltools')
class SharingAuditTests(unittest.TestCase):
    def test_palette_permutation_is_reported_as_same_decoded_weight_not_same_encoding(self):
        a=palette_program();b=palette_program(permuted=True)
        ca,wa=m.collect(a.functions['main'],'short_30');cb,wb=m.collect(b.functions['main'],'long_120')
        report=m.summarize(ca+cb,wa+wb)
        self.assertEqual(len(report['decoded_identical_different_encoding_groups']),1)
        self.assertEqual(report['decoded_identical_different_encoding_groups'][0]['encoding_variants'],2)
        self.assertFalse(report['estimates_are_additive'])
        c=palette_program(changed=True);cc,wc=m.collect(c.functions['main'],'long_120')
        self.assertFalse(m.summarize(ca+cc,wa+wc)['decoded_identical_different_encoding_groups'])

    def test_flat_equal_values_with_different_shapes_are_kept_as_layout_candidates(self):
        a=palette_program();b=palette_program(flat=True)
        ca,wa=m.collect(a.functions['main'],'short_30');cb,wb=m.collect(b.functions['main'],'partial')
        report=m.summarize(ca+cb,wa+wb)
        self.assertFalse(report['decoded_identical_different_encoding_groups'])
        self.assertEqual(len(report['decoded_flat_identical_different_shape_groups']),1)
        self.assertEqual(report['decoded_flat_identical_different_shape_groups'][0]['elements'],256)

    def test_bit_identical_constants_with_different_tags_are_identified(self):
        a=palette_program();b=palette_program()
        for program,tag in ((a,'source-A'),(b,'source-B')):
            for op in program.functions['main'].operations:
                if op.op_type=='const':op.weight_key=tag
        ca,wa=m.collect(a.functions['main'],'short_30');cb,wb=m.collect(b.functions['main'],'long_120')
        report=m.summarize(ca+cb,wa+wb)
        self.assertGreater(report['estimated_tag_separated_exact_payload_bytes'],0)
        self.assertTrue(any(g['different_tags'] for g in report['exact_constant_groups']))
        self.assertFalse(report['decoded_identical_different_encoding_groups'])


class ReportGateTests(unittest.TestCase):
    def test_failed_or_incomplete_packaging_report_is_rejected(self):
        report=dict(protocol='encoder-shared-weights-v1',encoder='musicfm',native_cpu_passed=True,
            weights_requantized=False,graphs_rewritten=False,sources={p:p for p in m.FUNCTIONS},
            checks={p:dict(graph=True,payloads=True,contract=True) for p in m.FUNCTIONS},native_checks=[dict(passed=True) for _ in range(6)])
        m.validate_report(report)
        for key,value in [('native_cpu_passed',False),('weights_requantized',True),('error','incomplete'),('native_checks',[])]:
            invalid=copy.deepcopy(report);invalid[key]=value
            with self.assertRaises(ValueError):m.validate_report(invalid)
        invalid=copy.deepcopy(report);invalid['checks']['partial']['payloads']=False
        with self.assertRaises(ValueError):m.validate_report(invalid)


if __name__=='__main__':unittest.main()
