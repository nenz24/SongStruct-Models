import copy
import unittest
from deployment.summarize_pipeline_batch import summarize


def completed(index):
    return {'run_index':index,'state':'completed','report':{'track_id':'song','source_validation_sha256':'proof',
        'requested_compute_units':'hybrid','pcm_pipeline_seconds':80,'section_agreement_with_mac':1,
        'max_section_probability_error':.000001,'all_predictions_finite':True,'low_coverage_sections':0,
        'process_footprint_snapshots_bytes':[100,200,150]}}


class BatchSummaryTests(unittest.TestCase):
    def test_three_completed_runs_report_success_without_integrating(self):
        batch={'requested_runs':3,'completed_runs':3,'state':'completed','runs':[completed(i) for i in (1,2,3)]}
        result=summarize(batch)
        self.assertTrue(result['all_requested_runs_completed'])
        self.assertFalse(result['ready_for_app_integration'])
        self.assertEqual(result['runs'][0]['largest_footprint_sample_bytes'],200)

    def test_partial_or_failed_batches_are_not_counted_as_complete(self):
        batch={'requested_runs':3,'completed_runs':1,'state':'running','runs':[completed(1)]}
        self.assertFalse(summarize(batch)['all_requested_runs_completed'])
        batch['state']='failed';batch['runs'].append({'run_index':2,'state':'failed','error':'BNNS'})
        self.assertFalse(summarize(batch)['all_requested_runs_completed'])
        batch['completed_runs']=2
        with self.assertRaises(ValueError):summarize(batch)

    def test_rejects_mixed_source_or_run_sequence(self):
        batch={'requested_runs':2,'completed_runs':2,'state':'completed','runs':[completed(1),completed(2)]}
        other=copy.deepcopy(batch);other['runs'][1]['report']['source_validation_sha256']='changed'
        with self.assertRaises(ValueError):summarize(other)
        other=copy.deepcopy(batch);other['runs'][1]['run_index']=1
        with self.assertRaises(ValueError):summarize(other)
