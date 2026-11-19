import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'encoder_stream_ablation.py'
spec = importlib.util.spec_from_file_location('encoder_stream_ablation', SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
LABELS = ('intro', 'verse', 'chorus', 'bridge', 'inst', 'outro', 'silence', 'pre-chorus')


class Toy(torch.nn.Module):
    """Same parameter names as SongFormer's label path, tiny widths."""
    def __init__(self):
        super().__init__()
        self.mixed_win_downsample = torch.nn.Linear(4096, 8)
        self.input_norm = torch.nn.LayerNorm(8)
        self.transformer = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(.5))
        self.boundary_head = torch.nn.Linear(8, 1)
        self.function_head = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.function_head(self.transformer(self.input_norm(self.mixed_win_downsample(x))))


def item(track, start, manual, predicted):
    return dict(track_id=track, start=start, end=start + 10., label_manual=manual,
                prediction=predicted, correct=manual == predicted)


class StreamAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_columns_follow_cached_stream_order(self):
        self.assertEqual(module.STREAMS, ('musicfm30', 'muq30', 'musicfm120', 'muq120'))
        self.assertEqual(module.columns('all4'), list(range(4096)))
        self.assertEqual(module.columns('musicfm'), list(range(1024)) + list(range(2048, 3072)))
        self.assertEqual(module.columns('muq'), list(range(1024, 2048)) + list(range(3072, 4096)))
        self.assertEqual(module.columns('short30'), list(range(2048)))
        self.assertEqual(module.columns('muq30'), list(range(1024, 2048)))
        with self.assertRaises(ValueError): module.columns('musicfm60')

    def test_implications_weight_real_bank_sizes(self):
        self.assertEqual(module.implications('musicfm')['encoder_bytes_fraction'], .5)
        self.assertEqual(module.implications('short30')['encoder_bytes_fraction'], 1.)
        self.assertEqual(module.implications('short30')['encoder_passes_fraction'], .5)
        self.assertEqual(module.implications('muq30')['encoder_passes_fraction'], .25)
        weighted = module.implications('muq', {'musicfm': 300, 'muq': 100})
        self.assertEqual((weighted['encoder_bytes'], weighted['encoder_bytes_fraction']), (100, .25))
        self.assertEqual(module.implications('muq')['input_dim_raw'], 2048)
        for bad in ({'musicfm': 1}, {'musicfm': 1, 'muq': 0}):
            with self.assertRaises(ValueError): module.implications('muq', bad)

    def test_reduced_projection_equals_original_with_dropped_streams_zeroed(self):
        torch.manual_seed(0)
        for variant in ('musicfm', 'muq', 'short30', 'muq30'):
            model = Toy(); original = copy.deepcopy(model.mixed_win_downsample)
            cols = module.columns(variant); module.reduce_projection(model, cols)
            self.assertEqual(model.mixed_win_downsample.in_features, len(cols))
            x = torch.randn(5, 4096); zeroed = torch.zeros_like(x); zeroed[:, cols] = x[:, cols]
            torch.testing.assert_close(model.mixed_win_downsample(x[:, cols]), original(zeroed), rtol=1e-5, atol=1e-5)
        model = Toy(); layer = model.mixed_win_downsample
        module.reduce_projection(model, module.columns('all4'))
        self.assertIs(model.mixed_win_downsample, layer)
        model.mixed_win_downsample = torch.nn.Linear(2048, 8)
        with self.assertRaises(ValueError): module.reduce_projection(model, module.columns('muq'))

    def test_sliced_reader_returns_owned_kept_columns(self):
        values = np.arange(6 * 4096, dtype=np.float32).reshape(6, 4096)
        reader = lambda cfg, row: iter([(values, 4.5, 'sig')])
        cols = module.columns('muq')
        (out, offset, signature), = list(module.sliced(reader, cols)(None, None))
        self.assertEqual((offset, signature, out.dtype, out.shape), (4.5, 'sig', np.float32, (6, 2048)))
        self.assertTrue(out.flags['C_CONTIGUOUS'])
        np.testing.assert_array_equal(out, values[:, cols])
        (same, _, _), = list(module.sliced(reader, module.columns('all4'))(None, None))
        self.assertIs(same, values)
        bad = lambda cfg, row: iter([(values[:, :2048], 0., 'sig')])
        with self.assertRaises(ValueError): list(module.sliced(bad, cols)(None, None))

    def test_trainable_stages_never_touch_boundary_head(self):
        model = Toy()
        params = module.configure(model, 'projection_head')
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(names, {'mixed_win_downsample.weight', 'mixed_win_downsample.bias', 'input_norm.weight',
                                 'input_norm.bias', 'function_head.weight', 'function_head.bias'})
        self.assertEqual(len(params), 6)
        module.configure(model, 'label_backbone')
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertNotIn('boundary_head.weight', names); self.assertIn('transformer.0.weight', names)
        with self.assertRaises(ValueError): module.configure(model, 'encoders')
        module.train_mode(model, 'projection_head')
        self.assertFalse(model.transformer.training); self.assertTrue(model.function_head.training)
        module.train_mode(model, 'label_backbone')
        self.assertTrue(model.transformer.training); self.assertFalse(model.boundary_head.training)

    def test_section_scores_match_engine_formula(self):
        items = [item('a', 0, 'verse', 'verse'), item('a', 10, 'chorus', 'verse'), item('b', 0, 'chorus', 'chorus')]
        score = module.section_scores(items, LABELS)
        # verse F1 = 2*1/(1+2), chorus F1 = 2*1/(2+1); only present manual classes are averaged.
        self.assertAlmostEqual(score['section_accuracy'], 2 / 3)
        self.assertAlmostEqual(score['macro_f1_present'], 2 / 3)

    def test_paired_comparison_requires_identical_targets_and_applies_fixed_rule(self):
        control = [item(t, s, 'chorus' if s else 'verse', 'chorus' if s else 'verse') for t in 'abcd' for s in (0, 10)]
        same = module.paired_comparison(control, copy.deepcopy(control), LABELS)
        self.assertEqual((same['delta_section_accuracy'], same['verdict'], same['changed_sections']), (0, 'candidate', []))
        self.assertEqual(same['bootstrap_interval']['section_accuracy'], [0., 0.])
        worse = copy.deepcopy(control); worse[1].update(prediction='bridge', correct=False)
        result = module.paired_comparison(control, worse, LABELS)
        self.assertAlmostEqual(result['delta_section_accuracy'], -1 / 8)
        self.assertEqual((result['verdict'], result['only_control_correct'], result['only_variant_correct']), ('rejected', 1, 0))
        self.assertEqual(result, module.paired_comparison(control, worse, LABELS))
        with self.assertRaises(ValueError): module.paired_comparison(control, worse[:-1], LABELS)
        relabeled = copy.deepcopy(control); relabeled[0]['label_manual'] = 'bridge'
        with self.assertRaises(ValueError): module.paired_comparison(control, relabeled, LABELS)
        with self.assertRaises(ValueError): module.paired_comparison(control + control[:1], control + control[:1], LABELS)

    def test_training_loop_saves_start_selects_best_and_stops_early(self):
        torch.manual_seed(1)
        model = Toy(); module.reduce_projection(model, module.columns('musicfm'))
        rows = [{'id': i} for i in range(4)]
        data = {i: (torch.randn(6, 2048), torch.randint(0, 3, (6,))) for i in range(4)}
        read = lambda row: iter([(data[row['id']][0], 0., 'sig')])
        def step_loss(model, values, offset, row):
            yield torch.nn.functional.cross_entropy(model(values), data[row['id']][1])
        calls, saved = [], []
        def validate(model):
            calls.append(1)
            f1 = [.5, .6, .7, .65, .66, .64][min(len(calls) - 1, 5)]
            return {'section_accuracy': f1, 'macro_f1_present': f1, 'sections': 2}, [{'epoch_call': len(calls)}]
        settings = dict(module.SETTINGS, epochs=10, patience=3, learning_rate=1e-2, gradient_accumulation=3, seed=42)
        summary = module.train_variant(model, 'projection_head', settings, rows, read, step_loss, validate,
                                       lambda tag, epoch, m, d: saved.append((tag, epoch, m['macro_f1_present'])),
                                       log=lambda text: None)
        self.assertEqual(saved, [('best', 0, .5), ('best', 1, .6), ('best', 2, .7)])
        self.assertEqual((summary['best_epoch'], summary['best_macro_f1_present']), (2, .7))
        self.assertEqual([h['epoch'] for h in summary['history']], [0, 1, 2, 3, 4, 5])
        self.assertTrue(all(np.isfinite(h.get('train_loss', 0)) for h in summary['history']))
        self.assertFalse(model.boundary_head.weight.requires_grad)

    def test_training_never_uses_full_420_second_chunks_by_default(self):
        recipe = {'gradient_accumulation': 4}  # worship-v3: function head only, full chunks
        settings = module.resolve_settings(recipe, 'projection_head', 42)
        # Same audio per optimizer step as 4 full 420-second chunks.
        self.assertEqual((settings['window_seconds'], settings['learning_rate'], settings['gradient_accumulation']),
                         (30., 1e-5, 56))
        self.assertEqual(module.resolve_settings(recipe, 'projection_head', 42, window_seconds=120)['gradient_accumulation'], 14)
        self.assertEqual(module.resolve_settings(recipe, 'projection_head', 42, gradient_accumulation=8)['gradient_accumulation'], 8)
        self.assertEqual(module.resolve_settings({**recipe, 'window_seconds': 30}, 'projection_head', 42)['gradient_accumulation'], 4)
        self.assertEqual(module.resolve_settings(recipe, 'label_backbone', 42)['learning_rate'], 1e-5)
        self.assertEqual(module.resolve_settings({**recipe, 'window_seconds': 60}, 'projection_head', 42)['window_seconds'], 60.)
        self.assertEqual(module.resolve_settings(recipe, 'projection_head', 42, window_seconds=120)['window_seconds'], 120.)
        for bad in (.1, 421., float('nan')):
            with self.assertRaises(ValueError): module.resolve_settings(recipe, 'projection_head', 42, window_seconds=bad)

    def test_initial_reproduction_check(self):
        stored = {'sections': 451, 'section_accuracy': .76, 'macro_f1_present': .72}
        self.assertTrue(module.reproduces(dict(stored), stored))
        self.assertFalse(module.reproduces({**stored, 'sections': 450}, stored))
        self.assertFalse(module.reproduces(stored, None))

    def test_compare_cli_requires_control_and_identical_recipe(self):
        control = [item(t, s, 'verse', 'verse') for t in 'ab' for s in (0, 10)]
        meta = dict(plan_id='p', provenance_id='v', annotations_id='a', init_checkpoint_sha256='c',
                    trainable='projection_head', settings={'epochs': 20}, labels=list(LABELS))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(variant, details, **change):
                (root / variant).mkdir(exist_ok=True)
                (root / variant / 'run.json').write_text(json.dumps({**meta, **change, 'variant': variant}))
                (root / variant / 'best_val_sections.json').write_text(json.dumps({'sections': details}))
                (root / variant / 'summary.json').write_text(json.dumps({'best_epoch': 1}))
            cli = [sys.executable, str(SCRIPT), 'compare', '--output-dir', folder,
                   '--encoder-bytes', 'musicfm=200,muq=100']
            write('muq', control)
            self.assertNotEqual(subprocess.run(cli, capture_output=True).returncode, 0)
            write('all4', control)
            subprocess.run(cli, check=True, capture_output=True)
            report = json.loads((root / 'comparison.json').read_text())
            self.assertEqual(report['variants']['muq']['comparison']['verdict'], 'candidate')
            self.assertAlmostEqual(report['variants']['muq']['implications']['encoder_bytes_fraction'], 1 / 3)
            write('muq', control, settings={'epochs': 5})
            self.assertNotEqual(subprocess.run(cli, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
