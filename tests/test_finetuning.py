import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from finetuning.common import CLASS_IDS, LABELS, config, write
from finetuning.prepare import allocate, group_suggestion, validate_review

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:
    torch = None


class DataTests(unittest.TestCase):
    def test_time_validation_roundoff_and_diagnostics(self):
        import copy
        source = {'duration': 20, 'sections': [
            {'start': 0, 'end': 10.0000000001, 'label_initial': 'verse'},
            {'start': 10, 'end': 20, 'label_initial': 'chorus'}]}
        corrected = copy.deepcopy(source)
        corrected['review_status'] = 'approved'
        for section in corrected['sections']:
            section['label_manual'] = section['label_initial']
        self.assertEqual(len(validate_review(source, corrected)), 2)
        for end, expected in [(10.1, 'Section 2'), (21, 'Section 1')]:
            bad = copy.deepcopy(source)
            bad['sections'][0]['end'] = end
            with self.assertRaisesRegex(ValueError, expected):
                bad_corrected = copy.deepcopy(corrected)
                bad_corrected['sections'][0]['end'] = end
                validate_review(bad, bad_corrected)


    def test_notes_are_held_until_reviewed_without_losing_clean_sections(self):
        import csv
        from finetuning.audit import audit
        source = {'track_id': 'song.mp3', 'duration': 20, 'sections': [
            {'start': 0, 'end': 10, 'label_initial': 'verse'},
            {'start': 10, 'end': 20, 'label_initial': 'chorus'}]}
        corrected = {**source, 'review_status': 'approved', 'sections': [
            {**source['sections'][0], 'label_manual': 'verse', 'note': ''},
            {**source['sections'][1], 'label_manual': 'chorus', 'note': 'chorus start di 0:12'}]}
        with tempfile.TemporaryDirectory() as folder:
            decisions, stats = audit([corrected], folder)
            self.assertEqual(stats['eligible_sections'], 1)
            self.assertEqual(stats['boundary_or_timing_hints'], 1)
            self.assertEqual(len(validate_review(source, corrected, decisions)), 1)
            path = Path(folder) / 'section_review.csv'
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            rows[0]['decision'], rows[1]['decision'] = 'exclude', 'include'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            # Partial/empty exports must not erase review decisions for other tracks.
            audit([], folder)
            decisions, _ = audit([corrected], folder)
            self.assertEqual(validate_review(source, corrected, decisions), [
                {'start': 10, 'end': 20, 'label': 'chorus'}])
            # A changed note invalidates the prior include decision.
            corrected['sections'][1]['note'] = 'chorus start di 0:13'
            decisions, _ = audit([corrected], folder)
            self.assertEqual(validate_review(source, corrected, decisions), [])

    def test_explicit_boundary_issue_cannot_be_included(self):
        from finetuning.audit import allowed, section_digest
        section = {'start': 0, 'end': 1, 'label_manual': 'verse', 'note': '', 'boundary_issue': True}
        decision = {('song', '1', section_digest('song', 1, section)): 'include'}
        self.assertFalse(allowed('song', 1, section, decision))

    def test_prepare_before_and_after_approval_preserves_frozen_plan(self):
        import finetuning.prepare as prepare
        import check_dataset
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg.update(work_dir=str(root / 'work'), labeled_dir=str(root / 'labels'), reviewed_dir=str(root / 'reviewed'))
            audio = root / 'audio'
            documents = []
            for i in range(4):
                key = f'Song {i}/{i}.mp3'
                file = audio / key
                file.parent.mkdir(parents=True)
                file.write_bytes(f'audio-{i}'.encode())
                doc = {'track_id': key, 'duration': 10, 'label_source': 'ASLP-lab/SongFormer',
                       'sections': [{'start': .1, 'end': 10, 'label_initial': 'verse'}]}
                write(Path(cfg['labeled_dir']) / f'{i}.json', doc)
                documents.append(doc)
            with patch.object(prepare, 'config', return_value=cfg), patch.object(check_dataset, 'DATASET_PATH', audio):
                with patch.object(sys, 'argv', ['prepare']):
                    prepare.main()
                self.assertFalse(json.loads((root / 'work/manifest.json').read_text())['groups_confirmed'])
                with patch.object(sys, 'argv', ['prepare', '--confirm-groups']):
                    prepare.main()
                frozen = (root / 'work/split_plan.json').read_bytes()
                doc = documents[0]
                corrected = {**doc, 'review_status': 'approved', 'sections': [
                    {**doc['sections'][0], 'label_manual': 'chorus'}]}
                write(root / 'reviewed/0.json', corrected)
                with patch.object(sys, 'argv', ['prepare']):
                    prepare.main()
                self.assertEqual((root / 'work/split_plan.json').read_bytes(), frozen)
                manifest = json.loads((root / 'work/manifest.json').read_text())
                self.assertEqual(sum(r['approved'] for r in manifest['records']), 1)
                self.assertEqual(manifest['records'][0]['sections'][0]['label'], 'chorus')

    def test_manual_approval_required_and_boundaries_immutable(self):
        raw = {'duration': 10, 'sections': [{'start': 0.1, 'end': 10, 'label_initial': 'chorus'}]}
        self.assertIsNone(validate_review(raw, {'review_status': 'pending'}))
        corrected = {'duration': 10, 'review_status': 'approved', 'sections': [
            {'start': 0.1, 'end': 10, 'label_initial': 'chorus', 'label_manual': 'bridge'}]}
        self.assertEqual(validate_review(raw, corrected)[0]['label'], 'bridge')
        corrected['sections'][0]['label_manual'] = None
        with self.assertRaises(ValueError):
            validate_review(raw, corrected)
        corrected['sections'][0].update(label_manual='bridge', start=0)
        with self.assertRaises(ValueError):
            validate_review(raw, corrected)

    def test_group_split_reproducible_and_grouped(self):
        groups = ['song-' + str(i) for i in range(20)]
        a = allocate(groups, [.8, .1, .1], 42)
        self.assertEqual(a, allocate(list(reversed(groups)), [.8, .1, .1], 42))
        self.assertEqual([list(a.values()).count(s) for s in ('train', 'val', 'test')], [16, 2, 2])
        self.assertEqual(group_suggestion('Abide/0302-Abide.mp3'), group_suggestion('Abide/0303-Abide-live.mp3'))


@unittest.skipIf(torch is None, 'PyTorch/numpy not installed')
class GradientTests(unittest.TestCase):
    def test_overlap_pooling_and_gradient(self):
        from finetuning.engine import label_loss, pooled_sections
        scores = torch.zeros(2, 8, requires_grad=True)
        sections = [{'start': .06, 'end': .18, 'label': 'bridge'}]
        pooled = pooled_sections(scores, sections, 0)
        self.assertAlmostEqual(pooled[0][2], .12, places=5)
        loss = label_loss(scores, sections, 0)
        loss.backward()
        self.assertLess(scores.grad[0, LABELS.index('bridge')].item(), 0)
        self.assertLess(scores.grad[1, LABELS.index('bridge')].item(), 0)

    def model(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.mixed_win_downsample = nn.Linear(4096, 4)
                self.function_head = nn.Linear(4, 128)
                self.boundary_head = nn.Linear(4, 1)

            def forward_func(self, batch):
                x = self.mixed_win_downsample(batch['input_embeddings'])
                x = x.reshape(x.shape[0], -1, 3, 4).mean(2)
                return {'function_logits': self.function_head(x), 'boundary_logits': self.boundary_head(x)}
        return Tiny()

    def test_only_label_head_changes(self):
        from finetuning.engine import configure_training, logits, label_loss, train_mode
        torch.manual_seed(4)
        model = self.model()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        params = configure_training(model, 'function_head')
        train_mode(model, 'function_head')
        optimizer = torch.optim.SGD(params, lr=.1)
        scores = logits(model, np.ones((12, 4096), dtype=np.float32), torch.device('cpu'))
        loss = label_loss(scores, [{'start': 0, 'end': .48, 'label': 'verse'}], 0)
        loss.backward()
        optimizer.step()
        for name, value in model.state_dict().items():
            if not name.startswith('function_head.'):
                self.assertTrue(torch.equal(value, before[name]), name)
        self.assertFalse(torch.equal(model.function_head.weight, before['function_head.weight']))
        self.assertFalse(model.training)

    def test_training_loop_and_ema_checkpoint_roundtrip(self):
        try:
            from safetensors.torch import load_file
            from ema_pytorch import EMA
        except ImportError:
            self.skipTest('EMA/safetensors unavailable')
        import finetuning.train as trainer
        cfg = config()
        cfg['training'].update(epochs=2, patience=2, gradient_accumulation=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg['work_dir'] = str(root)
            cfg['checkpoint'] = str(root / 'pretrained.placeholder')
            Path(cfg['checkpoint']).write_bytes(b'fixture')
            records = []
            for index, split in enumerate(['train', 'train', 'val', 'test']):
                row = {'id': str(index), 'track_id': f'song{index}.mp3', 'audio_sha256': 'fixture',
                       'approved': True, 'split': split, 'annotation_sha256': 'manual',
                       'sections': [{'start': .03, 'end': .45, 'label': 'verse'}]}
                records.append(row)
                cache = root / 'features' / row['id']
                cache.mkdir(parents=True)
                np.save(cache / 'chunk.npy', np.ones((12, 4096), dtype=np.float32))
                write(cache / 'metadata.json', {'audio_sha256': 'fixture', 'features': cfg['features'],
                    'provenance_id': 'test-provenance', 'chunks': [{'file': 'chunk.npy', 'offset': 0}]})
            with patch.object(trainer, 'config', return_value=cfg), patch.object(trainer, 'manifest',
                    return_value=({'records': records, 'plan_id': 'fixture'}, 'test-provenance')), \
                    patch.object(trainer, 'load_head', side_effect=lambda *a: (self.model(), None)), \
                    patch.object(sys, 'argv', ['train', '--run-name', 'smoke', '--device', 'cpu']):
                trainer.main()
            run = root / 'runs/smoke'
            self.assertTrue((run / 'baseline_val.json').is_file())
            self.assertEqual(len((run / 'history.jsonl').read_text().splitlines()), 2)
            checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            fresh = self.model()
            ema = EMA(fresh, include_online_model=False)
            ema.load_state_dict(load_file(str(run / 'best.safetensors')), strict=True)
            for name, value in checkpoint['model'].items():
                self.assertTrue(torch.equal(value, ema.ema_model.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
