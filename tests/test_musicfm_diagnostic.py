import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from deployment.diagnose_musicfm import exact_window, invalid_frames, repeat_predictions


class DiagnosticTests(unittest.TestCase):
    def test_exact_sample_window_preserves_values_outside_unit_range(self):
        audio=np.linspace(-1.3,1.4,100,dtype=np.float32)
        result=exact_window(audio,dict(start_sample=0,end_sample=100,sample_rate=24000))
        np.testing.assert_array_equal(result,audio)
        self.assertGreater(result.max(),1)
        self.assertLess(result.min(),-1)
        self.assertFalse(np.shares_memory(result,audio))
        with self.assertRaises(ValueError):
            exact_window(audio,dict(start_sample=20,end_sample=101,sample_rate=24000))

    def test_invalid_frames_distinguishes_full_and_partial_corruption(self):
        output=np.zeros((1,5,1024),np.float32)
        output[0,2]=np.nan
        output[0,4,0]=np.inf
        result=invalid_frames(output)
        self.assertEqual(result['bad_frame_indices'],[2,4])
        self.assertEqual(result['fully_invalid_frame_indices'],[2])
        self.assertEqual(result['invalid_channels_per_bad_frame'],[1024,1])
        self.assertEqual(result['nan_count'],1024)
        self.assertEqual(result['inf_count'],1)

    def test_replay_owns_inputs_and_preserves_nan_outputs(self):
        inputs={'conv_in':np.ones((1,3,1024),np.float32)}
        inputs['conv_in'].flags.writeable=False
        calls=[]
        def predict(batch):
            value=batch['conv_in']
            self.assertTrue(value.flags.owndata)
            self.assertTrue(value.flags.writeable)
            np.testing.assert_array_equal(value,inputs['conv_in'])
            calls.append(value.copy())
            result=np.zeros((1,3,1024),np.float32)
            result[0,1]=np.nan
            return {'embedding':result}
        with tempfile.TemporaryDirectory() as folder:
            results=repeat_predictions(NS(predict=predict),inputs,3,Path(folder))
            self.assertEqual(len(calls),3)
            self.assertTrue(all(r['bad_frame_indices']==[1] for r in results))
            saved=np.load(Path(folder)/results[0]['output_file'])
            self.assertTrue(np.isnan(saved[0,1]).all())
            self.assertFalse(inputs['conv_in'].flags.writeable)


if __name__=='__main__':
    unittest.main()
