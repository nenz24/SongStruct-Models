import copy
import unittest
from deployment.compare_pipeline_stages import compare


def fixture():
    rows=[]
    for role in ('log_mel','musicfm_conv'):
        for direction,name in [('input','x'),('output','y')]:
            rows.append({'key':f'{role}/1/{direction}/{name}','role':role,'direction':direction,
                'shape':[1,2],'count':2,'sample_indices':[0,1],'samples':[1.,2.],'sha256_float32':'same'})
    return {'stage_diagnostics':{'identity':'manifest','stages':rows}}


class StageComparisonTests(unittest.TestCase):
    def test_locates_origin_and_distinguishes_propagated_input_changes(self):
        left=fixture();right=copy.deepcopy(left)
        rows=right['stage_diagnostics']['stages']
        for i in (1,2,3):rows[i]['sha256_float32']='changed';rows[i]['samples'][1]=2.5
        result=compare(left,right)
        self.assertEqual(result['first_changed_output']['role'],'log_mel')
        self.assertEqual(len(result['same_input_changed_outputs']),1)
        self.assertEqual(result['same_input_changed_outputs'][0]['sample_max_abs_error'],.5)

    def test_identical_reports_and_rejects_mixed_manifest_or_inventory(self):
        left=fixture();self.assertEqual(compare(left,copy.deepcopy(left))['changed_stages'],0)
        other=fixture();other['stage_diagnostics']['identity']='different'
        with self.assertRaises(ValueError):compare(left,other)
        other=fixture();other['stage_diagnostics']['stages'].pop()
        with self.assertRaises(ValueError):compare(left,other)
