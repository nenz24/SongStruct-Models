import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch unavailable')
class MusicFMExportTests(unittest.TestCase):
    def test_layer_ten_excludes_final_layer_norm_and_later_layers(self):
        from deployment.export_musicfm_fp32 import MusicFMEncoderL10
        class Layer(torch.nn.Module):
            def forward(self,x,attention_mask=None,relative_position_embeddings=None,output_attentions=False):
                return (x+relative_position_embeddings[0,:x.shape[1],0,0,:1]+1,None)
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout=torch.nn.Dropout(.5)
                self.layers=torch.nn.ModuleList([Layer() for _ in range(12)])
                self.layer_norm=torch.nn.LayerNorm(4)
        encoder=Encoder().eval()
        wrapper=MusicFMEncoderL10(encoder).eval()
        x=torch.zeros((1,93,4))
        rotary=torch.ones((2,93,1,1,1))
        traced=torch.jit.trace(wrapper,(x,rotary),check_trace=False)
        for frames in (3,93,250):
            x=torch.zeros((1,frames,4))
            rotary=torch.ones((2,frames,1,1,1))
            expected=torch.full((1,frames,4),20.)
            self.assertTrue(torch.equal(wrapper(x,rotary),expected))
            self.assertTrue(torch.equal(traced(x,rotary),expected))
        self.assertEqual(len(wrapper.layers),10)
        self.assertFalse(any('layer_norm' in n for n,_ in wrapper.named_parameters()))

    def test_override_is_explicit_and_preserves_default_cache_identity(self):
        import json, tempfile, sys
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from deployment.coreml_features import CoreMLEncoders
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            root=base/'SongMap/Resources/MLModels/SongFormer'
            root.mkdir(parents=True)
            roles=['log_mel','musicfm_conv','musicfm','muq_conv','muq','head']
            app={'models':{r:r+'.mlpackage' for r in roles},'encoder_window_seconds':120,
                 'sample_rate':24000,'encoder_layer':10,'head_max_seconds':420,
                 'stream_order':['musicfm_30s','muq_30s','musicfm_long','muq_long']}
            (root/'songformer_coreml_manifest.json').write_text(json.dumps(app))
            for role in roles:
                folder=root/app['models'][role]
                folder.mkdir()
                (folder/'weights').write_bytes(role.encode())
            package=base/'candidate.mlpackage'
            package.mkdir()
            (package/'weights').write_bytes(b'new weights')
            report={'exact_window_passed':True,'compute_precision':'float32','package':str(package),
                    'musicfm_weights_sha256':'weights','musicfm_stats_sha256':'stats'}
            proof=base/'repair_report.json'
            proof.write_text(json.dumps(report))
            settings={'long_window_seconds':120,'sample_rate':24000,'head_window_seconds':420}
            with patch.dict(sys.modules,{'coremltools':NS(__version__='8.3'), 'librosa':NS(__version__='0.11')}):
                original=CoreMLEncoders(base,settings)
                same=CoreMLEncoders(base,settings)
                candidate=CoreMLEncoders(base,settings,musicfm_export_report=proof)
                self.assertEqual(original.id,same.id)
                self.assertNotEqual(original.id,candidate.id)
                self.assertEqual(candidate.package_paths['musicfm'],package)
                self.assertEqual(candidate.app,original.app)
                proof.write_text(json.dumps({**report,'exact_window_passed':False}))
                with self.assertRaisesRegex(ValueError,'exact-window'):
                    CoreMLEncoders(base,settings,musicfm_export_report=proof)

    def test_numerical_comparison_keeps_nan_failure(self):
        from deployment.export_musicfm_fp32 import numerical
        reference=np.zeros((1,3,1024),np.float32)
        candidate=reference.copy()
        candidate[0,0]=np.nan
        result=numerical(reference,candidate)
        self.assertFalse(result['numerical_parity_passed'])
        self.assertEqual(result['candidate']['nan_count'],1024)
        self.assertEqual(result['reference']['nan_count'],0)


if __name__=='__main__':
    unittest.main()
