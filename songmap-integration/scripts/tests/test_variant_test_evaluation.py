import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('evaluate_variant_test', SCRIPTS / 'evaluate_variant_test.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
import encoder_stream_ablation as ablation
import train_section_context as context
from calibrate_bridge_chorus import metrics

LABELS = module.LABELS


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixed_win_downsample = torch.nn.Linear(4096, 4)
        self.input_norm = torch.nn.LayerNorm(4)
        self.boundary_head = torch.nn.Linear(4, 1)
        self.function_head = torch.nn.Linear(4, 3)


def item(track, start, manual, predicted):
    return dict(track_id=track, start=start, end=start + 1., label_manual=manual, prediction=predicted,
                correct=manual == predicted)


class VariantTestEvaluationTests(unittest.TestCase):
    def test_decision_is_muq_against_production_on_identical_sections(self):
        production = [item(f't{i}', 0., 'verse', 'verse' if i < 80 else 'chorus') for i in range(100)]
        same = {'production': {'sections': production}, 'all4': {'sections': production},
                'muq': {'sections': [dict(x) for x in production]}}
        self.assertEqual(module.decide_test(same)['decision'], 'candidate')
        worse = [dict(x) for x in production]
        for x in worse[:5]:
            x.update(prediction='chorus', correct=False)
        result = module.decide_test({**same, 'muq': {'sections': worse}})
        self.assertEqual((result['decision'], result['all4']['verdict']), ('rejected', 'candidate'))
        self.assertAlmostEqual(result['muq']['delta_section_accuracy'], -.05)
        with self.assertRaises(ValueError):
            module.decide_test({**same, 'muq': {'sections': worse[:-1]}})

    def test_scores_and_lock(self):
        a = {'sections': 10, 'section_accuracy': .5, 'macro_f1_present': .4}
        self.assertTrue(module.same_scores({**a, 'section_accuracy': .5 + 1e-8}, a))
        self.assertFalse(module.same_scores({**a, 'sections': 11}, a))
        with tempfile.TemporaryDirectory() as folder:
            read = lambda p: json.loads(Path(p).read_text()); write = lambda p, v: Path(p).write_text(json.dumps(v))
            fp = lambda v: json.dumps(v, sort_keys=True)
            first = module.lock(Path(folder), {'x': 1}, fp, read, write)
            self.assertEqual(module.lock(Path(folder), {'x': 1}, fp, read, write), first)
            with self.assertRaises(ValueError): module.lock(Path(folder), {'x': 2}, fp, read, write)

    def test_full_flow_reproduces_validation_locks_then_scores_test(self):
        splits = {'train': 2, 'val': 2, 'test': 3}
        rows = [{'id': f'{s}{i}', 'track_id': f'{s}{i}', 'split': s, 'approved': True, 'group_id': f'g{s}{i}',
                 'audio_sha256': f'a{s}{i}', 'annotation_sha256': 'x', 'source_json': 'unused'}
                for s, n in splits.items() for i in range(n)]
        evaluated = []

        def details_for(records, predicted):
            out = []
            for r in records:
                for s, manual in enumerate(('verse', 'chorus', 'bridge')):
                    p = dict.fromkeys(LABELS, .02); p[predicted] = .86
                    out.append(dict(track_id=r['track_id'], start=float(s), end=float(s + 1), label_manual=manual,
                                    probabilities=p, prediction=predicted, correct=manual == predicted))
            return out

        def evaluate(model, records, cfg, target, provenance, details=None, feature_reader=None):
            evaluated.append((model.mixed_win_downsample.in_features, records[0]['split']))
            details.extend(details_for(records, 'verse' if model.mixed_win_downsample.in_features == 4096 else 'intro'))
            return metrics(details, {r['track_id']: r for r in records}, {'kind': 'identity'})

        def prepare(rows_, details, cfg, provenance, cache, helpers):
            return [{'track_id': r['track_id'], 'split': r['split'], 'identity': r['id'], 'audio': torch.randn(3, 4096),
                     'timing': torch.rand(3, 6), 'indices': torch.arange(3),
                     'probabilities': torch.tensor([[d['probabilities'][l] for l in LABELS] for d in details if d['track_id'] == r['track_id']]),
                     'labels': torch.tensor([LABELS.index(d['label_manual']) for d in details if d['track_id'] == r['track_id']]),
                     'details': [d for d in details if d['track_id'] == r['track_id']], 'coverage_diagnostics': []} for r in rows_]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'ss/finetuning').mkdir(parents=True); (root / 'ss/finetuning/config.json').write_text('{}')
            val_rows = [r for r in rows if r['split'] == 'val']
            prod_val = metrics(details_for(val_rows, 'verse'), {r['track_id']: r for r in val_rows}, {'kind': 'identity'})
            var_val = metrics(details_for(val_rows, 'intro'), {r['track_id']: r for r in val_rows}, {'kind': 'identity'})
            torch.save({'model': Toy().state_dict(), 'plan_id': 'plan', 'provenance_id': 'prov'}, root / 'v3.pt')
            run = root / 'prod-run'; run.mkdir()
            (run / 'comparison.json').write_text(json.dumps({'best_epoch': 5, 'test_used': False, 'candidate': prod_val}))
            (run / 'run.json').write_text(json.dumps({'device': 'cpu'}))
            torch.save({'model': context.SectionContextHead().state_dict(), 'settings': context.SETTINGS,
                        'plan_id': 'plan', 'provenance_id': 'prov'}, run / 'best.pt')
            for name in ('all4', 'muq'):
                head = Toy(); ablation.reduce_projection(head, ablation.columns(name))
                torch.save({'model': head.state_dict(), 'protocol': ablation.PROTOCOL, 'variant': name,
                            'columns': ablation.columns(name), 'plan_id': 'plan', 'provenance_id': 'prov',
                            'validation': {}}, root / f'{name}.pt')
                folder_ = root / 'ctx' / name; folder_.mkdir(parents=True)
                expected = prod_val if name == 'all4' else var_val
                (folder_ / 'summary.json').write_text(json.dumps({'variant': name, 'test_used': False,
                    'head_checkpoint': str(root / f'{name}.pt'), 'pipeline': expected, 'device': 'cpu'}))
                width = len(ablation.columns(name))
                torch.save({'model': context.SectionContextHead(width=width).state_dict(), 'settings': context.SETTINGS,
                            'width': width, 'columns': ablation.columns(name), 'head_sha256': ablation.sha_file(root / f'{name}.pt'),
                            'plan_id': 'plan', 'provenance_id': 'prov'}, folder_ / 'best.pt')
            common = types.ModuleType('finetuning.common'); engine = types.ModuleType('finetuning.engine')
            common.config = lambda path: {}
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
            argv = ['evaluate_variant_test.py', '--songstruct-root', str(root / 'ss'), '--production-head', str(root / 'v3.pt'),
                    '--production-context-run', str(run), '--variant-context-dir', str(root / 'ctx'),
                    '--cache-dir', str(root / 'cache'), '--output-dir', str(root / 'out'), '--device', 'cpu']
            boundaries = lambda document, row: ([{'start': 0., 'end': 3.}], [0])
            read_json = common.read
            common.read = lambda path: {} if str(path) == 'unused' else read_json(path)  # source_json stub
            with patch.dict(sys.modules, {'finetuning': parent, 'finetuning.common': common, 'finetuning.engine': engine}), \
                    patch.object(module, 'prepare_samples', prepare), patch.object(module, 'source_sections', boundaries), \
                    patch.object(sys, 'argv', argv):
                module.main()
                self.assertEqual([s for _, s in evaluated], ['val'] * 3 + ['test'] * 3)
                report = json.loads((root / 'out/test-comparison.json').read_text())
                self.assertEqual(report['decision'], 'rejected')  # MuQ stub always predicts intro (wrong)
                self.assertEqual(report['comparisons']['all4']['verdict'], 'candidate')
                self.assertTrue((root / 'out/selection.json').is_file())
                # A changed expected validation stops before any new test inference.
                bad = json.loads((root / 'ctx/muq/summary.json').read_text()); bad['pipeline']['sections'] = 99
                (root / 'ctx/muq/summary.json').write_text(json.dumps(bad))
                evaluated.clear()
                with self.assertRaises(ValueError): module.main()
                self.assertNotIn('test', [s for _, s in evaluated])


if __name__ == '__main__':
    unittest.main()
