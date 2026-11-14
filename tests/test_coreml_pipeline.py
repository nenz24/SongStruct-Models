import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from deployment.coreml_features import extract_chunks, native_input


class ExtractTests(unittest.TestCase):
    def test_stream_order_offsets_stride_and_short_tail(self):
        calls=[]
        def encode(audio):
            calls.append(len(audio))
            value=len(calls)
            return np.full((25,1024),value,np.float32),np.full((25,1024),value+10,np.float32)
        settings=dict(sample_rate=24000,head_window_seconds=1,long_window_seconds=120)
        waveform=np.zeros(3*24000+1000,np.float32)
        chunks=list(extract_chunks(waveform,encode,settings))
        self.assertEqual([offset for offset,_ in chunks],[0,1,2])
        self.assertEqual(len(calls),6)
        for index,(_,values) in enumerate(chunks):
            self.assertEqual(values.shape,(24,4096))
            expected=[2*index+1,2*index+11,2*index+2,2*index+12]
            self.assertEqual([values[0,k] for k in (0,1024,2048,3072)],expected)

    def test_diagnostic_cache_checks_identity_and_preserves_source(self):
        import tempfile
        from pathlib import Path
        from finetuning.common import write
        from deployment.evaluate_pipeline import cached_features
        with tempfile.TemporaryDirectory() as temporary:
            cache=Path(temporary)
            folder=cache/'fixture'
            folder.mkdir()
            original=np.ones((6,4096),np.float32)
            np.save(folder/'chunk.npy',original)
            row={'id':'fixture','audio_sha256':'audio'}
            metadata={'audio_sha256':'audio','candidate_encoder_id':'candidate',
                      'chunks':[{'file':'chunk.npy','offset':0}]}
            write(folder/'metadata.json',metadata)
            result=list(cached_features(cache,row,'candidate'))
            np.testing.assert_array_equal(result[0][0],original)
            self.assertEqual(result[0][2],'candidate')
            with self.assertRaisesRegex(ValueError,'stale'):
                list(cached_features(cache,row,'other'))
            write(folder/'metadata.json',{**metadata,'chunks':[]})
            with self.assertRaisesRegex(ValueError,'no chunks'):
                list(cached_features(cache,row,'candidate'))
            np.testing.assert_array_equal(np.load(folder/'chunk.npy'),original)

    def test_native_input_copies_readonly_view(self):
        original=np.ones((30,1024),np.float32)
        original.flags.writeable=False
        result=native_input(original[::-1])
        self.assertTrue(result.flags.owndata)
        self.assertTrue(result.flags.writeable)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(original,result))

    @unittest.skipIf(torch is None,'torch not installed')
    def test_numpy_extraction_matches_frozen_torch_windowing(self):
        from finetuning.features import extract_chunks as reference_chunks
        settings=dict(sample_rate=24000,head_window_seconds=1,long_window_seconds=120)
        wav=np.arange(3*24000+1000,dtype=np.float32)/1000
        def pair(audio):
            n=int(len(audio)*25/24000)
            return (np.full((n,1024),audio[0]+1,np.float32),np.full((n,1024),audio[0]+2,np.float32))
        class MuQ:
            def __call__(self,audio,**kwargs):
                values=pair(audio[0].numpy())[1]
                return {'hidden_states':[None]*10+[torch.from_numpy(values).unsqueeze(0)]}
        class FM:
            def get_predictions(self,audio):
                values=pair(audio[0].numpy())[0]
                return (None,[None]*10+[torch.from_numpy(values).unsqueeze(0)])
        expected=list(reference_chunks(wav,MuQ(),FM(),torch.device('cpu'),settings))
        actual=list(extract_chunks(wav,pair,settings))
        self.assertEqual(len(actual),len(expected))
        for (a,x),(b,y) in zip(actual,expected):
            self.assertEqual(a,b)
            np.testing.assert_array_equal(x,y)

    @unittest.skipIf(torch is None,'torch not installed')
    def test_evaluation_injected_reader_preserves_section_pooling(self):
        from unittest.mock import patch
        from types import SimpleNamespace as NS
        from finetuning.engine import evaluate
        def forward(batch):
            n=batch['input_embeddings'].shape[1]//3
            scores=torch.zeros((1,n,128))
            scores[:,:,1]=2
            return {'function_logits':scores}
        model=NS(eval=lambda:None,forward_func=forward)
        row=dict(track_id='fixture',sections=[dict(start=0,end=.48,label='verse')])
        def reader(cfg,row):
            for offset in (0,.24):
                yield np.ones((6,4096),np.float32),offset,'candidate'
        with patch('finetuning.engine.iter_features',side_effect=AssertionError('Original cache must not be read')):
            result=evaluate(model,[row],{},torch.device('cpu'),'candidate',feature_reader=reader)
        self.assertEqual(result['sections'],1)
        self.assertEqual(result['section_accuracy'],1)
        self.assertEqual(result['low_coverage_sections'],0)
        with self.assertRaisesRegex(ValueError,'Mixed feature provenance'):
            evaluate(model,[row],{},torch.device('cpu'),'wrong',feature_reader=reader)


if __name__=='__main__':
    unittest.main()
