import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import encoder_acceleration as m
from conformer_reference import (Wav2Vec2ConformerSelfAttention, Wav2Vec2ConformerConvolutionModule,
                                 Wav2Vec2ConformerEncoderLayer)


def config(width=16,heads=2):
    return SimpleNamespace(hidden_size=width,num_attention_heads=heads,position_embeddings_type='rotary',
        attention_dropout=.1,conv_depthwise_kernel_size=3,hidden_act='swish',conformer_conv_dropout=.1,
        intermediate_size=width*2,activation_dropout=.1,hidden_dropout=.1)


def positions(n,d):
    # Deliberately nontrivial rotary values, to detect rotation after projection or mixed half order.
    values=torch.arange(n)[:,None]*(10000**(-torch.arange(0,d,2).float()/d))[None,:]
    angle=torch.cat((values,values),dim=-1)
    return torch.stack((angle.cos(),angle.sin()))[:,:,None,None,:]


def encoder(cfg=None):
    cfg=cfg or config()
    model=nn.Module();model.dropout=nn.Dropout(.1)
    model.layers=nn.ModuleList([Wav2Vec2ConformerEncoderLayer(cfg) for _ in range(10)])
    return model.eval().requires_grad_(False)


class EncoderAccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_attention_matches_upstream_rotary_before_projection(self):
        for width,heads,n in ((16,2,1),(16,2,17),(32,4,129),(1024,16,17)):
            torch.manual_seed(29)
            original=Wav2Vec2ConformerSelfAttention(config(width,heads)).eval().requires_grad_(False)
            candidate=m.Rank3RotaryAttention(original).eval()
            x=torch.randn(1,n,width);p=positions(n,width//heads)
            with torch.inference_mode():
                a,pa=original(x,relative_position_embeddings=p)
                b,pb=candidate(x,relative_position_embeddings=p[:,:,0,0,:])
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(pa,pb,atol=1e-5,rtol=1e-5)
            self.assertEqual(m.state_digest(original),m.state_digest(candidate))
            with self.assertRaises(ValueError):candidate(x,attention_mask=torch.zeros(n,n),relative_position_embeddings=p[:,:,0,0,:])

    def test_explicit_glu_matches_native_convolution_without_mutation(self):
        torch.manual_seed(7)
        original=Wav2Vec2ConformerConvolutionModule(config(32,4)).eval().requires_grad_(False)
        digest=m.state_digest(original);candidate=m.ExplicitGLUConvolution(original).eval()
        with torch.inference_mode():
            for n in (1,7,129):
                x=torch.randn(1,n,32)
                torch.testing.assert_close(original(x),candidate(x),atol=1e-6,rtol=1e-6)
        self.assertEqual(digest,m.state_digest(original));self.assertEqual(digest,m.state_digest(candidate))
        original.glu.dim=-1
        with self.assertRaises(ValueError):m.ExplicitGLUConvolution(original)

    def test_ten_layers_and_traces_match_without_replacing_native_submodules(self):
        torch.manual_seed(42);native=encoder();before=m.state_digest(native)
        attention_ids=[id(layer.self_attn) for layer in native.layers]
        reference=m.ReferenceL10(native).eval();candidate=m.LoweredL10(native).eval()
        p=positions(7,8)
        trace=torch.jit.trace(candidate,(torch.randn(1,7,16),p[0,:,0,0,:],p[1,:,0,0,:]),strict=True,check_trace=False)
        with torch.inference_mode():
            for n in (1,7,31,129):
                x=torch.randn(1,n,16);p=positions(n,8);inputs=(x,p[0,:,0,0,:],p[1,:,0,0,:])
                expected=reference(x,p)
                torch.testing.assert_close(expected,candidate(*inputs),atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(expected,trace(*inputs),atol=1e-5,rtol=1e-5)
        self.assertEqual(before,m.state_digest(native))
        self.assertEqual(m.state_digest(reference),m.state_digest(candidate))
        self.assertEqual(attention_ids,[id(layer.self_attn) for layer in native.layers])
        self.assertTrue(all(isinstance(layer.self_attn,Wav2Vec2ConformerSelfAttention) for layer in native.layers))

    def test_unsupported_architecture_or_training_encoder_rejected(self):
        native=encoder();native.train()
        with self.assertRaises(ValueError):m.LoweredL10(native)
        native.eval().requires_grad_(True)
        with self.assertRaises(ValueError):m.LoweredL10(native)
        native.requires_grad_(False);native.layers[0].self_attn.position_embeddings_type='relative'
        with self.assertRaises(ValueError):m.LoweredL10(native)

    def test_native_input_conversion_preserves_rotary_order_and_owns_buffers(self):
        conv=np.arange(2*9*1024,dtype=np.float32).reshape(1,9,2048)[:,:,::2]
        rotary=positions(9,64).numpy()
        actual=m.candidate_inputs(conv,rotary,7)
        for value in actual.values():
            self.assertEqual(value.dtype,np.float32);self.assertTrue(value.flags.owndata);self.assertTrue(value.flags.c_contiguous)
        np.testing.assert_array_equal(actual['rotary_cos'],rotary[0,:7,0,0,:])
        np.testing.assert_array_equal(actual['rotary_sin'],rotary[1,:7,0,0,:])
        for a,b,n in ((conv,rotary,0),(conv,rotary,10),(conv.repeat(2,axis=0),rotary,7),
                      (conv,np.zeros((2,8,1,1,64)),7),(conv*float('nan'),rotary,7)):
            with self.assertRaises(ValueError):m.candidate_inputs(a,b,n)

    def test_numerical_gate_catches_shape_nonfinite_and_finite_drift(self):
        a=np.ones((1,7,16),dtype=np.float32)
        self.assertTrue(m.numerical(a,a)['passed'])
        for b in (a+.01,a[:,:6],a*float('nan'),a*float('inf')):
            self.assertFalse(m.numerical(a,b)['passed'])

    @unittest.skipUnless(importlib.util.find_spec('coremltools'),'actual MIL conversion requires coremltools')
    def test_real_mil_conversion_is_static_float32_with_low_rank_inputs(self):
        import coremltools as ct
        native=encoder();candidate=m.LoweredL10(native).eval()
        p=positions(7,8);inputs=(torch.randn(1,7,16),p[0,:,0,0,:],p[1,:,0,0,:])
        trace=torch.jit.trace(candidate,inputs,strict=True,check_trace=False)
        model=m.convert_fixed(ct,trace,7,skip_model_load=True,width=16,head_size=8)
        spec=model.get_spec()
        self.assertEqual([i.name for i in spec.description.input],['conv_in','rotary_cos','rotary_sin'])
        self.assertTrue(all(i.type.multiArrayType.dataType==65568 for i in spec.description.input))
        self.assertEqual(spec.description.output[0].name,'embedding')
        with tempfile.TemporaryDirectory() as d:
            model.save(str(Path(d)/'candidate.mlpackage'))
            self.assertTrue((Path(d)/'candidate.mlpackage/Manifest.json').is_file())


if __name__=='__main__':unittest.main()
