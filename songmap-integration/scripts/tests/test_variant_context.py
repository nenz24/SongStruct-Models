import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('train_variant_context', SCRIPTS / 'train_variant_context.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
import train_section_context as context
import encoder_stream_ablation as ablation

LABELS = module.LABELS


def sample(track, split, n=4, width=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 3, (n,), generator=g)
    probabilities = torch.full((n, 8), .02); probabilities[:, 1] = .86  # head always says verse
    details = [dict(track_id=track, start=float(i), end=float(i + 1), label_manual=LABELS[int(labels[i])],
                    prediction='verse') for i in range(n)]
    return {'track_id': track, 'split': split, 'audio': torch.randn(n, width, generator=g) + labels[:, None].float(),
            'timing': torch.rand(n, 6, generator=g), 'indices': torch.arange(n), 'probabilities': probabilities,
            'labels': labels, 'details': details}


def item(track, start, manual, predicted):
    return dict(track_id=track, start=start, end=start + 10., label_manual=manual, prediction=predicted,
                correct=manual == predicted)


class VariantContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_sliced_pooled_means_equal_pooling_sliced_features(self):
        rng = np.random.default_rng(3)
        chunks = [(rng.normal(size=(120, 4096)).astype(np.float32), 0., 'p'),
                  (rng.normal(size=(60, 4096)).astype(np.float32), 120 / 25, 'p')]
        sections = [{'start': .5, 'end': 3.1}, {'start': 3.1, 'end': 7.0}]
        cols = ablation.columns('muq')
        full, coverage = context.pool_audio(iter(chunks), sections, 'p')
        sliced, sliced_coverage = context.pool_audio(iter([(v[:, cols], o, s) for v, o, s in chunks]), sections, 'p',
                                                     width=len(cols))
        np.testing.assert_allclose(full[:, cols], sliced, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(coverage, sliced_coverage)
        samples = [{'audio': torch.from_numpy(full), 'track_id': 't'}]
        out = module.slice_samples(samples, cols)
        torch.testing.assert_close(out[0]['audio'], torch.from_numpy(full[:, cols]))
        self.assertIs(module.slice_samples(samples, ablation.columns('all4')), samples)
        with self.assertRaises(ValueError): module.slice_samples(out, cols)

    def test_context_head_width_keeps_default_and_zero_residual(self):
        self.assertEqual(context.SectionContextHead().audio[1].in_features, 4096)
        model = context.SectionContextHead(width=2048).eval()
        self.assertEqual(model.audio[1].in_features, 2048)
        out = model(torch.randn(5, 2048), torch.rand(5, 6))
        self.assertEqual(tuple(out.shape), (5, 8)); self.assertEqual(float(out.detach().abs().max()), 0.)

    def test_fit_context_starts_from_head_and_keeps_best_validation_epoch(self):
        torch.manual_seed(0)
        train = [sample(f'train{i}', 'train', seed=i) for i in range(6)]
        val = [sample('val0', 'val', seed=99)]
        rows = [{'track_id': 'val0', 'split': 'val', 'approved': True}]
        model = context.SectionContextHead(16, 0., width=2048)
        settings = {**context.SETTINGS, 'hidden': 16, 'dropout': 0., 'epochs': 6, 'patience': 2, 'learning_rate': 3e-3}
        best_state, best_epoch, history = module.fit_context(model, train, val, rows, torch.device('cpu'), settings,
                                                             log=lambda text: None)
        head_only = history[0]['val']['macro_f1_present']
        scores = [h['val']['macro_f1_present'] for h in history]
        self.assertEqual(history[0]['epoch'], 0)
        self.assertEqual(best_epoch, 0 if max(scores) <= head_only else scores.index(max(scores)))
        for key, value in best_state.items():
            torch.testing.assert_close(model.state_dict()[key].cpu(), value)
        with self.assertRaises(ValueError):  # validation samples can never produce gradients
            context.supervised_loss(model, val[0], torch.device('cpu'), .01)

    def test_decision_needs_paired_and_production_non_inferiority(self):
        def songs(correct):
            return [item(f't{i}', 0., 'verse', 'verse' if i < correct else 'chorus') for i in range(100)]
        control = songs(80)
        self.assertEqual(module.decide(control, songs(80), rule={**module.DECISION_RULE,
            'production_reference': {'section_accuracy': .80, 'macro_f1_present': 0.}})['verdict'], 'candidate')
        low_production = {**module.DECISION_RULE, 'production_reference': {'section_accuracy': .85, 'macro_f1_present': 0.}}
        result = module.decide(control, songs(80), rule=low_production)
        self.assertEqual((result['paired_verdict'], result['production_verdict'], result['verdict']),
                         ('candidate', 'fail', 'rejected'))
        worse = module.decide(control, songs(75), rule={**module.DECISION_RULE,
            'production_reference': {'section_accuracy': .70, 'macro_f1_present': 0.}})
        self.assertEqual((worse['paired_verdict'], worse['verdict']), ('rejected', 'rejected'))

    def test_compare_cli_requires_all4_control_and_same_recipe(self):
        sections = [item('a', float(s), 'verse', 'verse') for s in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(name, variant, settings=context.SETTINGS):
                (root / name).mkdir()
                (root / name / 'summary.json').write_text(json.dumps({'variant': variant, 'settings': settings}))
                (root / name / 'val_sections.json').write_text(json.dumps({'sections': sections}))
            write('all4', 'all4'); write('muq', 'muq'); write('bad', 'muq', {**context.SETTINGS, 'epochs': 1})
            cli = lambda control, variant: subprocess.run(
                [sys.executable, str(SCRIPTS / 'train_variant_context.py'), 'compare',
                 '--control', str(root / control), '--variant', str(root / variant)], capture_output=True, text=True)
            ok = cli('all4', 'muq')
            self.assertEqual(ok.returncode, 0, ok.stderr)
            report = json.loads((root / 'muq' / 'comparison.json').read_text())
            self.assertIn(report['comparison']['verdict'], ('candidate', 'rejected'))
            self.assertEqual(report['decision_rule']['production_reference']['section_accuracy'], .793792)
            self.assertNotEqual(cli('muq', 'muq').returncode, 0)
            self.assertNotEqual(cli('all4', 'bad').returncode, 0)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixed_win_downsample = torch.nn.Linear(4096, 4)
        self.input_norm = torch.nn.LayerNorm(4)
        self.boundary_head = torch.nn.Linear(4, 1)
        self.function_head = torch.nn.Linear(4, 3)


class VariantContextTrainFlowTests(unittest.TestCase):
    def test_train_flow_reproduces_head_caches_predictions_and_writes_reports(self):
        rows = ([{'id': f'tr{i}', 'track_id': f'tr{i}', 'split': 'train', 'approved': True, 'group_id': f'g{i}',
                  'audio_sha256': f'a{i}', 'annotation_sha256': 'x'} for i in range(4)]
                + [{'id': 'v0', 'track_id': 'v0', 'split': 'val', 'approved': True, 'group_id': 'gv',
                    'audio_sha256': 'av', 'annotation_sha256': 'x'}])
        calls = []

        def evaluate(model, records, cfg, target, provenance, details=None, feature_reader=None):
            calls.append([r['split'] for r in records])
            self.assertIsNotNone(feature_reader)
            for r in records:
                for s in range(3):
                    manual = ('verse', 'chorus', 'bridge')[s]
                    p = dict.fromkeys(LABELS, .02); p['verse'] = .86
                    details.append(dict(track_id=r['track_id'], start=float(s), end=float(s + 1), label_manual=manual,
                                        prediction='verse', correct=manual == 'verse', probabilities=p))
            return {'sections': 3 * len(records), 'section_accuracy': 1 / 3, 'macro_f1_present': 1 / 3 * 2 / 4 * 1 / 1}

        def prepare(rows_, details, cfg, provenance, cache, helpers):
            out = []
            for r in rows_:
                d = [x for x in details if x['track_id'] == r['track_id']]
                out.append({'track_id': r['track_id'], 'split': r['split'], 'identity': r['id'],
                            'audio': torch.randn(3, 4096), 'timing': torch.rand(3, 6), 'indices': torch.arange(3),
                            'probabilities': torch.tensor([[x['probabilities'][l] for l in LABELS] for x in d]),
                            'labels': torch.tensor([LABELS.index(x['label_manual']) for x in d]),
                            'details': d, 'coverage_diagnostics': []})
            return out

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'finetuning').mkdir(); (root / 'finetuning/config.json').write_text('{}')
            head = Toy(); ablation.reduce_projection(head, ablation.columns('muq'))
            val_metrics = evaluate(None, rows[-1:], None, None, None, details=[], feature_reader=lambda *a: None)
            calls.clear()
            torch.save({'model': head.state_dict(), 'protocol': ablation.PROTOCOL, 'variant': 'muq',
                        'columns': ablation.columns('muq'), 'plan_id': 'plan', 'provenance_id': 'prov',
                        'validation': val_metrics}, root / 'muq.pt')
            common = types.ModuleType('finetuning.common'); engine = types.ModuleType('finetuning.engine')
            common.config = lambda path: {'work_dir': str(root)}
            common.device = lambda name: torch.device('cpu')
            common.fingerprint = lambda value: json.dumps(value, sort_keys=True, default=str)
            common.iter_features = lambda cfg, row: iter(())
            common.load_head = lambda cfg, target, checkpoint=None: (Toy(), None)
            common.read = lambda path: json.loads(Path(path).read_text())
            common.write = lambda path, value: Path(path).write_text(json.dumps(value))
            common.sha_file = ablation.sha_file
            engine.evaluate = evaluate
            engine.manifest = lambda cfg: ({'records': rows, 'plan_id': 'plan'}, 'prov')
            parent = types.ModuleType('finetuning'); parent.__path__ = []
            args = types.SimpleNamespace(songstruct_root=root, head_checkpoint=root / 'muq.pt',
                                         output_dir=root / 'out', cache_dir=root / 'cache', device='cpu', smoke_test=False)
            settings = {**context.SETTINGS, 'epochs': 2, 'patience': 1}
            with patch.dict(sys.modules, {'finetuning': parent, 'finetuning.common': common, 'finetuning.engine': engine}), \
                    patch.object(module, 'prepare_samples', prepare), patch.object(module, 'SETTINGS', settings):
                module.run_train(args)
                self.assertEqual(calls, [['val'], ['train'] * 4])
                summary = json.loads((root / 'out/summary.json').read_text())
                self.assertEqual((summary['variant'], summary['columns_count'], summary['test_used']), ('muq', 2048, False))
                self.assertAlmostEqual(summary['head_only']['section_accuracy'], 1 / 3)
                saved = torch.load(root / 'out/best.pt', weights_only=False)
                self.assertEqual((saved['width'], saved['model']['audio.1.weight'].shape[1]), (2048, 2048))
                self.assertEqual(len(json.loads((root / 'out/val_sections.json').read_text())['sections']), 3)
                calls.clear()
                (root / 'out/summary.json').unlink()
                module.run_train(args)  # head predictions are reused from the per-head cache
                self.assertEqual(calls, [])
                bad = {**torch.load(root / 'muq.pt', weights_only=False), 'validation': {**val_metrics, 'sections': 99}}
                torch.save(bad, root / 'bad.pt')
                with self.assertRaises(ValueError):
                    module.run_train(types.SimpleNamespace(**{**vars(args), 'head_checkpoint': root / 'bad.pt',
                                                              'output_dir': root / 'out2'}))
                # Relative notebook paths must not follow load_head's chdir into the upstream folder.
                with tempfile.TemporaryDirectory() as elsewhere:
                    previous = os.getcwd(); os.chdir(root)
                    try:
                        common.load_head = lambda cfg, target, checkpoint=None: (os.chdir(elsewhere), (Toy(), None))[1]
                        module.run_train(types.SimpleNamespace(songstruct_root=Path('.'), head_checkpoint=Path('muq.pt'),
                                                               output_dir=Path('rel-out'), cache_dir=Path('rel-cache'),
                                                               device='cpu', smoke_test=False))
                    finally:
                        os.chdir(previous)
                    self.assertTrue((root / 'rel-out/summary.json').is_file())
                    self.assertTrue((root / 'rel-cache/muq').is_dir())
                    self.assertFalse(Path(elsewhere, 'rel-out').exists())


if __name__ == '__main__':
    unittest.main()
