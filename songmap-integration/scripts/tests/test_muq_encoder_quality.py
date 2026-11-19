import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('muq_encoder_quality', SCRIPTS / 'muq_encoder_quality.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from prepare_encoder_pipeline_iphone import pcm_chunks
import encoder_stream_ablation as ablation

SONGSTRUCT = Path(os.environ.get('SONGSTRUCT_ROOT', '/home/user/nenz24/songstruct-models'))
HAS_ENGINE = (SONGSTRUCT / 'finetuning/engine.py').is_file()
SETTINGS = {'sample_rate': 24000, 'head_window_seconds': 420}


class FakeNative:
    def __init__(self, fn): self.fn = fn
    def predict(self, inputs): return self.fn(inputs)


def fake_encode(audio, seconds):
    """Deterministic 25-Hz 'features' from the samples; same frame count rule as the encoders."""
    frames = len(audio) // 960  # 25 Hz at 24 kHz, like the real encoders (30 s -> 750 frames)
    base = np.float32(audio[::960][:frames].mean() if len(audio) else 0.)
    return np.full((frames, 1024), base + seconds, dtype=np.float32) + np.arange(frames, dtype=np.float32)[:, None] * 1e-3


class MuQEncoderQualityTests(unittest.TestCase):
    def test_function_selection_matches_canonical_dispatch(self):
        self.assertEqual(module.function_for(30 * 24000, 30), 'short_30')
        self.assertEqual(module.function_for(120 * 24000, 120), 'long_120')
        self.assertEqual(module.function_for(30 * 24000, 120), 'partial')  # 30 s tail of the 120 s stream
        self.assertEqual(module.function_for(29 * 24000, 30), 'partial')
        with self.assertRaises(ValueError): module.function_for(60 * 24000, 60)

    @unittest.skipUnless((SONGSTRUCT / 'deployment/verify_encoders.py').is_file(), 'needs SongStruct-Models clone')
    def test_rotary_matches_songstruct(self):
        sys.path.insert(0, str(SONGSTRUCT))
        try:
            from deployment.verify_encoders import rotary
        except ImportError as error:
            self.skipTest(str(error))
        frequencies = np.linspace(.001, 1, 32)
        np.testing.assert_array_equal(module.rotary(37, frequencies), rotary(37, frequencies))
        with self.assertRaises(ValueError): module.rotary(3, frequencies[:31])

    def test_muq_chunks_equal_muq_columns_of_the_two_encoder_fusion(self):
        rng = np.random.default_rng(0)
        waveform = rng.normal(size=int(24000 * 457.3)).astype(np.float32)  # two head chunks, partial tails
        muq_only = list(module.muq_chunks(waveform, fake_encode, SETTINGS))
        both = list(pcm_chunks(waveform, lambda a, s: [fake_encode(a, s) + 7, fake_encode(a, s)], SETTINGS))
        self.assertEqual([o for o, _ in muq_only], [o for o, _ in both])
        self.assertEqual([o for o, _ in muq_only], [0., 420.])
        for (_, mine), (_, fused) in zip(muq_only, both):
            np.testing.assert_array_equal(mine, fused[:, ablation.columns('muq')])
            self.assertEqual(mine.shape[1], 2048); self.assertTrue(mine.flags['C_CONTIGUOUS'])

    def test_feature_drift_and_gate(self):
        a = [(np.zeros((6, 2048), np.float32), 0.)]
        self.assertEqual(module.feature_drift(a, [(np.full((6, 2048), .5, np.float32), 0.)])['max_abs'], .5)
        with self.assertRaises(ValueError): module.feature_drift(a, [(np.zeros((9, 2048), np.float32), 0.)])
        ref = {'section_accuracy': .80, 'macro_f1_present': .76}
        self.assertTrue(module.decide(ref, {'section_accuracy': .791, 'macro_f1_present': .751}))
        self.assertFalse(module.decide(ref, {'section_accuracy': .80, 'macro_f1_present': .749}))

    def test_bank_report_validation(self):
        good = {'encoder': 'muq', 'native_cpu_passed': True, 'candidate_for_quality_validation': True,
                'weights_requantized': False, 'retrained': False, 'compiled_digest': 'x'}
        for change in ({'encoder': 'musicfm'}, {'native_cpu_passed': False}, {'error': 'boom'}, {'retrained': True}):
            with self.assertRaises(ValueError): module.validate_bank({**good, **change}, Path('/nonexistent'))
        with self.assertRaises(ValueError): module.validate_bank(good, Path('/nonexistent'))

    @unittest.skipUnless(HAS_ENGINE, 'needs SongStruct-Models finetuning.engine')
    def test_song_pipeline_scores_eligible_targets_with_context_over_all_sections(self):
        sys.path.insert(0, str(SONGSTRUCT))
        row = {'track_id': 't', 'duration': 30., 'sections': [{'start': 0., 'end': 10., 'label': 'verse'},
                                                              {'start': 20., 'end': 30., 'label': 'chorus'}]}
        source = {'track_id': 't', 'duration': 30., 'sections': [{'start': 0., 'end': 10.}, {'start': 10., 'end': 20.},
                                                                  {'start': 20., 'end': 30.}]}

        def head_logits(inputs):
            frames = inputs['input_embeddings'].shape[1] // 3
            out = np.zeros((1, frames, 128), np.float32)
            out[0, :frames // 2, 1] = 5.; out[0, frames // 2:, 2] = 5.   # verse then chorus
            return {'function_logits': out}
        seen = {}

        def residual(inputs):
            seen['n'] = len(inputs['section_audio'])
            r = np.zeros((seen['n'], 8), np.float32); r[2, 3] = 20.  # push the last section to bridge
            return {'context_residual': r}
        head = module.CoreMLHead(FakeNative(head_logits))
        chunks = [(np.random.default_rng(1).normal(size=(750, 2048)).astype(np.float32), 0.)]
        details, offsets, coverage = module.song_pipeline(row, chunks, head, {'source': source, 'native': FakeNative(residual)})
        self.assertEqual(seen['n'], 3)  # context sees ALL original sections
        self.assertEqual([d['prediction'] for d in details], ['verse', 'bridge'])
        self.assertEqual([d['label_manual'] for d in details], ['verse', 'chorus'])
        self.assertEqual(offsets, [0.]); self.assertAlmostEqual(coverage, 1., places=5)
        with self.assertRaises(ValueError):
            module.song_pipeline(row, chunks, head, {'source': source, 'native': FakeNative(residual)}, offsets=[5.])


if __name__ == '__main__':
    unittest.main()
