import tempfile
from pathlib import Path
import unittest
import numpy as np
from deployment.coreml_features import validate_musicfm_proof
from deployment.compress_musicfm import validate_baseline
from deployment.musicfm_candidates import reuse_muq_chunks, validate_muq_provenance
from finetuning.common import sha_file, fingerprint


class MusicFMCompressionTests(unittest.TestCase):
    def test_compressed_candidate_requires_finite_smoke_and_verified_source(self):
        proof={'compute_precision':'float32','compression_scheme':'palette6',
               'source_exact_window_passed':True,'window_finite_passed':True,'source_full_validation':True,
               'musicfm_weights_sha256':'weights','musicfm_stats_sha256':'stats',
               'exact_window_passed':False}
        # Lossy numerical mismatch is recorded, not confused with non-finite output.
        validate_musicfm_proof(proof)
        for key,value in [('window_finite_passed',False),('source_full_validation',False),
                          ('source_exact_window_passed',False),('compression_scheme','bad')]:
            with self.assertRaises(ValueError):
                validate_musicfm_proof({**proof,key:value})

    def test_baseline_full_validation_matches_source_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'repair.json'
            path.write_text('fixture')
            repair={'compute_precision':'float32','exact_window_passed':True,
                    'musicfm_weights_sha256':'weights','musicfm_stats_sha256':'stats'}
            pipeline={'full_validation':True,'tracks':32,'musicfm_override':{
                'report_sha256':sha_file(path),'rotary_mode':'original_app_values_unchanged',
                'musicfm_weights_sha256':'weights','musicfm_stats_sha256':'stats'},
                'coreml_encoders_with_fp32_head':{'sections':451,'low_coverage_sections':0}}
            validate_baseline(repair,pipeline,path)
            with self.assertRaises(ValueError):
                validate_baseline(repair,{**pipeline,'full_validation':False},path)
            bad={**pipeline,'musicfm_override':{**pipeline['musicfm_override'],'report_sha256':'wrong'}}
            with self.assertRaises(ValueError):
                validate_baseline(repair,bad,path)

    def test_muq_provenance_rejects_runtime_profile_and_unchanged_model_changes(self):
        import copy
        base={'coremltools':'8.3','librosa':'0.11','platform':'mac','compute_units':'CPU_ONLY',
              'feature_settings':{'profile':'30/120'},'app_manifest_sha256':'manifest',
              'packages':{r:r for r in ('log_mel','musicfm_conv','musicfm','muq_conv','muq')}}
        identity=fingerprint(base)
        recorded={'id':identity,**base}
        candidate=copy.deepcopy(base)
        candidate['packages']['musicfm']='compressed'
        self.assertEqual(validate_muq_provenance(recorded,candidate),identity)
        for key in ('librosa','coremltools','platform','app_manifest_sha256'):
            with self.assertRaisesRegex(ValueError,'differs'):
                validate_muq_provenance(recorded,{**candidate,key:'different'})
        wrong=copy.deepcopy(candidate)
        wrong['packages']['muq']='different'
        with self.assertRaisesRegex(ValueError,'package differs'):
            validate_muq_provenance(recorded,wrong)
        with self.assertRaisesRegex(ValueError,'identity invalid'):
            validate_muq_provenance({**recorded,'id':'wrong'},candidate)

    def test_muq_reuse_changes_only_musicfm_columns_and_preserves_source(self):
        baseline=np.zeros((24,4096),np.float32)
        baseline[:,1024:2048]=7
        baseline[:,3072:4096]=9
        source=baseline.copy()
        calls=[]
        def encode(audio):
            calls.append(1)
            return np.full((25,1024),len(calls),np.float32)
        settings={'sample_rate':24000,'head_window_seconds':1,'long_window_seconds':120}
        outputs=list(reuse_muq_chunks(np.zeros(24000,np.float32),encode,settings,[(baseline,0,'base')]))
        result=outputs[0][1]
        self.assertEqual(len(calls),2)
        self.assertEqual([result[0,i] for i in (0,1024,2048,3072)],[1,7,2,9])
        np.testing.assert_array_equal(baseline,source)
        with self.assertRaisesRegex(ValueError,'inventories'):
            list(reuse_muq_chunks(np.zeros(24000,np.float32),encode,settings,[]))
        with self.assertRaisesRegex(ValueError,'offsets'):
            list(reuse_muq_chunks(np.zeros(24000,np.float32),encode,settings,[(baseline,1,'base')]))


if __name__=='__main__':
    unittest.main()
