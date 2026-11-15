import unittest
import numpy as np
from deployment.evaluate_muq import replace_muq, validate_source


class MuQEvaluationTests(unittest.TestCase):
    def test_replaces_only_muq_and_preserves_source(self):
        source=np.arange(6*4096,dtype=np.float32).reshape(6,4096)
        before=source.copy()
        short=np.full((8,1024),-3,dtype=np.float32)
        long=np.full((7,1024),-4,dtype=np.float32)
        result=list(replace_muq([(0,short,long)],[(source,0,'baseline')]))[0][1]
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(result[:,:1024],source[:,:1024])
        np.testing.assert_array_equal(result[:,2048:3072],source[:,2048:3072])
        self.assertTrue((result[:,1024:2048]==-3).all())
        self.assertTrue((result[:,3072:4096]==-4).all())

    def test_rejects_offset_missing_chunks_bad_frames_and_nan(self):
        source=np.zeros((6,4096),np.float32)
        good=np.zeros((6,1024),np.float32)
        bad=good.copy();bad[0,0]=np.nan
        for produced,baseline in [([(1,good,good)],[(source,0,'x')]),
                ([],[(source,0,'x')]), ([(0,good,good)],[]),
                ([(0,good[:3],good)],[(source,0,'x')]),
                ([(0,bad,good)],[(source,0,'x')])]:
            with self.subTest(produced=len(produced),baseline=len(baseline)):
                with self.assertRaises(ValueError):list(replace_muq(produced,baseline))

    def test_source_requires_training_snapshot_and_frontend_identity(self):
        old={'muq_repo':'repo','muq_revision':'commit'}
        baseline={'app_manifest_sha256':'manifest','packages':{'log_mel':'mel','muq_conv':'conv'}}
        proof={'source_kind':'original_training_snapshot','compute_precision':'float32',
            'source_window_passed':True,'muq_repo':'repo','muq_revision':'commit',
            'muq_snapshot_digest':'snapshot','app_manifest_sha256':'manifest',
            'frontend_packages':{'log_mel':'mel','muq_conv':'conv'}}
        validate_source(proof,old,baseline)
        for key,value in [('source_window_passed',False),('muq_revision','other'),
                ('muq_snapshot_digest',''),('app_manifest_sha256','other')]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_source({**proof,key:value},old,baseline)
        with self.assertRaises(ValueError):
            validate_source({**proof,'frontend_packages':{'log_mel':'different','muq_conv':'conv'}},old,baseline)
