import tempfile
from pathlib import Path
import unittest
from deployment.compress_muq import validate_baseline, package_digest
from deployment.evaluate_muq import validate_source
from finetuning.common import sha_file


class MuQCompressionTests(unittest.TestCase):
    def test_baseline_requires_full_uncompressed_matching_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'export.json';path.write_text('{}')
            proof={'source_kind':'original_training_snapshot','compute_precision':'float32',
                'source_window_passed':True,'plan_id':'plan','provenance_id':'provenance'}
            validation={'full_validation':True,'muq_scheme':'fp32','head_scheme':'palette6',
                'reused_musicfm':True,'muq_export_report_sha256':sha_file(path),
                'plan_id':'plan','provenance_id':'provenance','metrics':{'sections':451,'low_coverage_sections':0}}
            validate_baseline(proof,validation,path)
            for changed in ({'full_validation':False},{'muq_scheme':'int8'},
                    {'muq_export_report_sha256':'wrong'},{'plan_id':'different'},
                    {'metrics':{'sections':451,'low_coverage_sections':1}}):
                with self.assertRaises(ValueError):validate_baseline(proof,{**validation,**changed},path)
            with self.assertRaises(ValueError):validate_baseline({**proof,'compression_scheme':'int8'},validation,path)

    def test_compressed_source_requires_finite_fixture_and_validated_source(self):
        proof={'source_kind':'original_training_snapshot','compute_precision':'float32',
            'source_window_passed':True,'compression_scheme':'palette6','source_full_validation':True,
            'window_finite_passed':True,'muq_repo':'r','muq_revision':'c','muq_snapshot_digest':'s',
            'app_manifest_sha256':'app','frontend_packages':{'log_mel':'mel','muq_conv':'conv'}}
        baseline={'app_manifest_sha256':'app','packages':proof['frontend_packages']}
        original={'muq_repo':'r','muq_revision':'c'}
        validate_source(proof,original,baseline)
        for change in ({'window_finite_passed':False},{'source_full_validation':False},{'compression_scheme':'unknown'}):
            with self.assertRaises(ValueError):validate_source({**proof,**change},original,baseline)

    def test_digest_tracks_package_content_without_absolute_location(self):
        with tempfile.TemporaryDirectory() as temp:
            first=Path(temp)/'a';second=Path(temp)/'b'
            first.mkdir();second.mkdir()
            (first/'weights.bin').write_bytes(b'original')
            (second/'weights.bin').write_bytes(b'original')
            self.assertEqual(package_digest(first),package_digest(second))
            (second/'weights.bin').write_bytes(b'changed')
            self.assertNotEqual(package_digest(first),package_digest(second))
