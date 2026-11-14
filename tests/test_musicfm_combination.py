import unittest
import numpy as np
from deployment.evaluate_cached_combination import check_same_inventory
from deployment.prepare_musicfm_iphone import fixture_inputs
from deployment.iphone_project import project_text


class CombinationTests(unittest.TestCase):
    def test_combination_rejects_changed_labels_split_or_checkpoint(self):
        original=dict(plan_id='plan',provenance_id='source',annotations_id='labels',checkpoint_sha256='weights',tracks=32)
        check_same_inventory(original,original)
        for key in original:
            with self.assertRaisesRegex(ValueError,key):
                check_same_inventory(original,{**original,key:'different'})

    def test_combination_runs_without_audio_or_encoders(self):
        import json,sys,tempfile
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from finetuning.common import fingerprint,write
        import deployment.evaluate_cached_combination as combo
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            row={'id':'fixture','track_id':'missing.mp3','audio_path':'/not/available.mp3',
                 'audio_sha256':'audio','approved':True,'split':'val',
                 'sections':[{'start':0,'end':.24,'label':'verse'}]}
            provenance={'test':'fixture'}
            identity=fingerprint(provenance)
            cache=root/'cache'
            chunk=cache/'fixture'
            chunk.mkdir(parents=True)
            np.save(chunk/'chunk.npy',np.ones((6,4096),np.float32))
            write(chunk/'metadata.json',{'candidate_encoder_id':identity,'audio_sha256':'audio',
                'chunks':[{'file':'chunk.npy','offset':0}]})
            write(cache/'encoder_provenance.json',{'id':identity,**provenance})
            inventory=dict(full_validation=True,tracks=1,plan_id='plan',provenance_id='source',
                annotations_id=fingerprint([(row['track_id'],row['sections'])]),checkpoint_sha256='weights')
            package=root/'head.mlpackage'
            package.mkdir()
            pipeline={**inventory,'diagnostic_cache':str(cache),'candidate_encoder_id':identity,
                'coreml_encoders_with_fp32_head':{'macro_f1_present':1.0},
                'musicfm_override':{'compression_scheme':'palette6'}}
            head={**inventory,'package':str(package),'compression_scheme':'palette6'}
            compression={'scheme':'palette6','package':str(package),'checkpoint_sha256':'weights'}
            for name,value in [('pipeline',pipeline),('head',head),('compression',compression)]:
                write(root/f'{name}.json',value)
            def predict(batch):
                out=np.zeros((1,batch['input_embeddings'].shape[1]//3,128),np.float32)
                out[:,:,1]=2
                return {'function_logits':out}
            model=NS(user_defined_metadata={'checkpoint_sha256':'weights','compression_scheme':'palette6'},predict=predict)
            ct=NS(models=NS(MLModel=lambda *a,**k:model),ComputeUnit=NS(CPU_ONLY='cpu'))
            output=root/'result.json'
            argv=['combo','--encoder-pipeline-report',str(root/'pipeline.json'),
                  '--head-validation-report',str(root/'head.json'),
                  '--head-compression-report',str(root/'compression.json'),'--output',str(output)]
            with patch.dict(sys.modules,{'coremltools':ct}), \
                 patch.object(combo.platform,'system',return_value='Darwin'), \
                 patch.object(combo,'config',return_value={}), \
                 patch.object(combo,'manifest',return_value=({'records':[row],'plan_id':'plan'},'source')), \
                 patch.object(sys,'argv',argv):
                combo.main()
            result=json.loads(output.read_text())
            self.assertFalse(result['encoders_executed'])
            self.assertEqual(result['metrics']['section_accuracy'],1)
            np.testing.assert_array_equal(np.load(chunk/'chunk.npy'),np.ones((6,4096),np.float32))

    def test_rotary_prefix_preserves_both_planes_and_owns_buffers(self):
        conv=np.arange(1200*1024,dtype=np.float32).reshape(1,1200,1024)
        rotary=np.arange(2*1200*64,dtype=np.float32).reshape(2,1200,1,1,64)
        result=fixture_inputs(conv,rotary,750)
        np.testing.assert_array_equal(result['rotary'],rotary[:,:750])
        self.assertEqual(result['conv_in'].shape,(1,750,1024))
        self.assertTrue(all(v.flags.owndata and v.flags.writeable and v.flags.c_contiguous for v in result.values()))
        with self.assertRaises(ValueError):
            fixture_inputs(conv,rotary,1201)

    def test_musicfm_project_has_custom_assets_without_old_head_resources(self):
        resources=['MusicFMINT8.mlmodelc','MusicFMPalette6.mlmodelc','conv_short.f32','rotary_short.f32','reference_short.f32',
            'conv_failed_full.f32','rotary_failed_full.f32','reference_failed_full.f32','benchmark_manifest.json']
        text=project_text(resource_names=resources)
        for resource in resources:
            self.assertIn(resource,text)
        self.assertNotIn('HeadFP32.mlmodelc',text)
        self.assertEqual(text.count('{'),text.count('}'))
        with self.assertRaises(ValueError):
            project_text(resource_names=['bad;injection'])


if __name__=='__main__':
    unittest.main()
