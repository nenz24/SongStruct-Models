import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('section_context', Path(__file__).resolve().parents[1] / 'train_section_context.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def sample(split='train'):
    return dict(split=split, audio=torch.randn(3, 4096), timing=torch.randn(3, 6),
        indices=torch.tensor([0, 2]), probabilities=torch.tensor([[.1, .1, .3, .1, .1, .1, .1, .1],
        [.1, .1, .1, .3, .1, .1, .1, .1]]), labels=torch.tensor([2, 3]))


class SectionContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_overlap_pooling_across_chunks_preserves_fractional_frames(self):
        sections = [dict(start=.02, end=.10), dict(start=.10, end=.22)]
        values = np.asarray([[0., 10.], [2., 20.], [4., 30.], [6., 40.], [8., 50.], [10., 60.]], dtype=np.float32)
        means, coverage = module.pool_audio([(values[:3], 0., 'p'), (values[3:], .12, 'p')], sections, 'p', width=2)
        expected = np.asarray([(values[0]*.02+values[1]*.04+values[2]*.02)/.08,
                               (values[2]*.02+values[3]*.04+values[4]*.04+values[5]*.02)/.12])
        np.testing.assert_allclose(means, expected, atol=1e-5)
        np.testing.assert_allclose(coverage, [.08, .12])

    def test_pooling_rejects_overlap_mixed_or_nonfinite_features(self):
        values = np.ones((3, 2), dtype=np.float32); sections = [dict(start=0., end=.12)]
        for chunks in ([(values, 0., 'other')], [(values, 0., 'p'), (values, .08, 'p')],
                       [(values * float('nan'), 0., 'p')], [(values, .20, 'p')]):
            with self.assertRaises(ValueError): module.pool_audio(chunks, sections, 'p', width=2)

    def test_short_final_target_accepts_only_metadata_explained_cache_tail(self):
        record = dict(track_id='Bagai Anak Panah', split='train', duration=263.723)
        sections = [dict(start=261.902, end=263.723)]
        values = np.ones((6591, 2), dtype=np.float32)
        means, coverage = module.pool_audio([(values, 0., 'p')], sections, 'p', width=2)
        meta = {'chunks': [dict(offset=0., frames=6591)]} # Cache ends at 263.64s, a valid 83ms tail.
        review = module.coverage_review(record, sections, [0], coverage, meta)
        np.testing.assert_allclose(means, [[1.,1.]])
        self.assertEqual(len(review),1)
        self.assertTrue(.95 < review[0]['coverage_fraction'] < .96)
        self.assertAlmostEqual(review[0]['missing_seconds'],.083)
        with self.assertRaisesRegex(ValueError,'not explained'):
            module.coverage_review(record, sections, [0], coverage-.01, meta)
        with self.assertRaisesRegex(ValueError,'not explained'):
            module.coverage_review(record, sections, [0], np.asarray([263.4-261.902]),
                                   {'chunks':[dict(offset=0.,frames=6585)]}) # 323ms is too large.

    def test_missing_middle_chunk_is_not_treated_as_rounding_at_a_tail(self):
        record = dict(track_id='song',split='train',duration=840.2)
        sections = [dict(start=419.8,end=420.2)]
        meta = {'chunks':[dict(offset=0.,frames=10497),dict(offset=840.,frames=3)]}
        with self.assertRaisesRegex(ValueError,'not explained'):
            module.coverage_review(record,sections,[0],np.asarray([.08]),meta)
        # Two real adjacent chunk tails are bounded independently, with no missing chunk.
        record['duration']=420.2
        meta['chunks'][1]['offset']=420.
        review=module.coverage_review(record,sections,[0],np.asarray([.20]),meta)
        self.assertEqual(len(review[0]['tails']),2)

    def test_existing_pooled_cache_is_reused_without_changing_manual_target(self):
        fingerprint=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
        sha_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        read=lambda p:json.loads(Path(p).read_text())
        def write(p,value):
            p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'pooled';cache.mkdir()
            row=dict(id='song',track_id='song',split='train',duration=263.723,
                source_json=str(root/'source.json'),annotation_sha256='ann',audio_sha256='audio',
                sections=[dict(start=261.902,end=263.723,label='bridge')])
            write(row['source_json'],dict(track_id='song',duration=263.723,
                sections=[dict(start=0.,end=261.902),dict(start=261.902,end=263.723)]))
            cfg={'work_dir':str(root),'features':{'profile':'frozen'}}
            write(root/'features/song/metadata.json',dict(audio_sha256='audio',features=cfg['features'],
                provenance_id='p',chunks=[dict(offset=0.,frames=6591)]))
            probabilities=dict.fromkeys(module.LABELS,.01);probabilities['bridge']=.93
            details=[dict(track_id='song',start=261.902,end=263.723,label_manual='bridge',
                          prediction='bridge',probabilities=probabilities)]
            before=copy.deepcopy((row,details))
            helpers=(read,write,fingerprint,sha_file,lambda cfg,row: [])
            with patch.object(module,'pool_audio',return_value=(np.ones((2,4096),dtype=np.float32),
                              np.asarray([261.902,263.64-261.902]))) as pooling:
                first=module.prepare_samples([row],details,cfg,'p',cache,helpers)
                self.assertEqual(pooling.call_count,1)
                content_hash=sha_file(cache/'song.npz')
                second=module.prepare_samples([row],details,cfg,'p',cache,helpers)
                self.assertEqual(pooling.call_count,1) # Same old .npz/.json need no rebuilding.
            self.assertEqual(sha_file(cache/'song.npz'),content_hash)
            self.assertEqual(first[0]['indices'].tolist(),[1])
            self.assertEqual(second[0]['labels'].tolist(),[module.LABELS.index('bridge')])
            self.assertEqual(len(second[0]['coverage_diagnostics']),1)
            self.assertEqual((row,details),before)

    def test_held_sections_supply_context_without_becoming_targets(self):
        record = dict(track_id='song', duration=.24,
                      sections=[dict(start=0., end=.08, label='chorus'), dict(start=.16, end=.24, label='bridge')])
        source = dict(track_id='song', duration=.24, sections=[dict(start=0., end=.08, label_manual='wrong'),
            dict(start=.08, end=.16, label_manual='not_a_target'), dict(start=.16, end=.24, label_manual='wrong')])
        sections, targets = module.source_sections(source, record)
        self.assertEqual(len(sections), 3); self.assertEqual(targets, [0, 2])
        self.assertTrue(all(set(s) == {'start', 'end'} for s in sections))
        changed = copy.deepcopy(record); changed['sections'][0]['label'] = 'verse'
        self.assertEqual(module.source_sections(source, changed), (sections, targets))
        source['sections'][-1]['end'] = .25
        self.assertEqual(module.source_sections(source, record)[0][-1]['end'], .24)
        source['sections'][-1]['end'] = .27
        with self.assertRaises(ValueError): module.source_sections(source, record)

    def test_zero_residual_reproduces_baseline_and_labels_are_not_features(self):
        torch.manual_seed(42); model = module.SectionContextHead(); model.eval(); data = sample()
        with torch.no_grad():
            scores, residual, _ = module.forward_targets(model, data, torch.device('cpu'))
            self.assertEqual(scores.argmax(-1).tolist(), [2, 3])
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
            changed = {**data, 'labels': torch.tensor([0, 0])}
            self.assertTrue(torch.equal(scores, module.forward_targets(model, changed, torch.device('cpu'))[0]))

    def test_attention_uses_other_section_audio_and_training_has_gradients(self):
        torch.manual_seed(7); model = module.SectionContextHead(dropout=0.); data = sample()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(); loss = module.supervised_loss(model, data, torch.device('cpu'), .01)
            loss.backward(); self.assertTrue(torch.isfinite(loss)); optimizer.step()
        self.assertGreater(model.audio[1].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            before = model(data['audio'], data['timing'])[0]
            altered = data['audio'].clone(); altered[1] = torch.randn(4096) * 5
            after = model(altered, data['timing'])[0]
            self.assertGreater((before-after).abs().max().item(), 1e-7)

    def test_holdouts_cannot_receive_training_gradients_or_leak_groups(self):
        model = module.SectionContextHead()
        for split in ('val', 'test'):
            with self.assertRaises(ValueError): module.supervised_loss(model, sample(split), torch.device('cpu'), .01)
        train = [dict(track_id='a', group_id='g', audio_sha256='hash-a', approved=True, split='train')]
        val = [dict(track_id='b', group_id='other', audio_sha256='hash-b', approved=True, split='val')]
        module.validate_splits(train, val)
        for key in ('track_id', 'group_id', 'audio_sha256'):
            changed = copy.deepcopy(val); changed[0][key] = train[0][key]
            with self.assertRaises(ValueError): module.validate_splits(train, changed)

    def test_cli_smoke_baseline_retention_and_resume_with_real_torch(self):
        # Real tiny head training; only upstream manifest/feature I/O is stubbed.
        fingerprint = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        sha_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        read = lambda path: json.loads(Path(path).read_text())
        def write(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, allow_nan=False))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); records = []; details = []
            for i, split in enumerate(('train', 'train', 'val', 'test')):
                sections = [dict(start=0., end=.08, label='chorus'), dict(start=.16, end=.24, label='bridge')]
                row = dict(id=f'id-{i}', track_id=f'song-{i}', group_id=f'group-{i}', split=split,
                    approved=True, duration=.24, audio_sha256=f'audio-{i}', annotation_sha256=f'ann-{i}',
                    sections=sections, source_json=str(root / f'source-{i}.json'))
                records.append(row)
                write(row['source_json'], dict(track_id=row['track_id'], duration=.24,
                    sections=[dict(start=0., end=.08), dict(start=.08, end=.16), dict(start=.16, end=.24)]))
                for s in sections:
                    probabilities = dict.fromkeys(module.LABELS, .01); probabilities[s['label']] = .93
                    details.append(dict(track_id=row['track_id'], start=s['start'], end=s['end'],
                        label_manual=s['label'], prediction=s['label'], probabilities=probabilities))
                write(root / 'features' / row['id'] / 'metadata.json', dict(audio_sha256=row['audio_sha256'],
                    features={'profile': 'frozen'}, provenance_id='provenance'))
            checkpoint = root / 'songformer.pt'; checkpoint.write_bytes(b'untouched original')
            audit_path = root / 'audit/train-all.json'
            write(audit_path, dict(full_train=True, split='train', tracks=2, plan_id='plan', provenance_id='provenance',
                checkpoint_sha256=sha_file(checkpoint), audit_identity='train-cache'))
            for row in records[:2]:
                write(audit_path.parent / 'per-track' / (row['id']+'.json'), dict(identity='train-cache',
                    audio_sha256=row['audio_sha256'], sections=[s for s in details if s['track_id']==row['track_id']]))
            calibration = root / 'calibration/comparison.json'; adapter = {'kind': 'identity'}
            write(calibration.parent / 'adapter.json', {'adapter': adapter})
            validation_identity = fingerprint(dict(protocol='pair-calibration-v1', train_audit_sha256=sha_file(audit_path),
                checkpoint_sha256=sha_file(checkpoint), plan_id='plan', provenance_id='provenance',
                val_annotations=[(records[2]['id'], records[2]['annotation_sha256'])], selected_adapter=adapter,
                device='cpu', torch=str(torch.__version__)))
            val_details = [s for s in details if s['track_id']==records[2]['track_id']]
            baseline = module.evaluate_validation([records[2]], val_details, {'kind':'identity'})
            write(calibration, dict(split='val', full_validation=True, tracks=1, plan_id='plan', provenance_id='provenance',
                checkpoint=str(checkpoint), checkpoint_sha256=sha_file(checkpoint), train_audit=str(audit_path),
                experiment_identity=validation_identity, test_used=False, baseline=baseline, candidate=baseline))
            write(calibration.parent / 'val-predictions' / (records[2]['id']+'.json'), dict(identity=validation_identity, sections=val_details))
            common = types.ModuleType('finetuning.common'); engine = types.ModuleType('finetuning.engine')
            seen = []
            def iter_features(cfg, row):
                self.assertIn(row['split'], ('train','val')); seen.append(row['id'])
                yield np.arange(6*4096, dtype=np.float32).reshape(6,4096) / 4096, 0., 'provenance'
            cfg = dict(work_dir=str(root), features={'profile':'frozen'})
            for name, value in dict(config=lambda p: cfg, read=read, write=write, sha_file=sha_file,
                    fingerprint=fingerprint, iter_features=iter_features, device=torch.device).items(): setattr(common,name,value)
            engine.manifest = lambda cfg: ({'plan_id':'plan','records':records}, 'provenance')
            parent = types.ModuleType('finetuning'); parent.__path__=[]
            args = ['train_section_context.py', '--songstruct-root', str(root), '--calibration',str(calibration),
                    '--output-dir',str(root/'run'), '--cache-dir',str(root/'pooled'), '--device','cpu']
            previous_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {'finetuning':parent,'finetuning.common':common,'finetuning.engine':engine}), \
                        patch.dict(module.SETTINGS, {'epochs':2,'patience':1}):
                    with patch.object(sys,'argv',args+['--smoke-test']): module.main()
                    self.assertFalse((root/'run').exists())
                    with patch.object(sys,'argv',args): module.main()
                    report = read(root/'run/comparison.json')
                    self.assertEqual(report['best_epoch'],0)
                    self.assertEqual(report['candidate']['confusion_matrix'],baseline['confusion_matrix'])
                    self.assertFalse(report['test_used']); self.assertFalse(report['encoder_executed'])
                    first_seen = list(seen)
                    with patch.object(sys,'argv',args+['--resume']): module.main()
                    self.assertEqual(seen,first_seen) # All pooling came from verified compact caches.
                    state = torch.load(root/'run/latest.pt',map_location='cpu',weights_only=False)
                    self.assertEqual(len(state['history']),1)
                    self.assertEqual(state['epoch'],1)
                    # A real interrupted/resumed optimization must match uninterrupted CPU training.
                    uninterrupted = list(args); interrupted = list(args)
                    output_arg = args.index('--output-dir') + 1
                    uninterrupted[output_arg] = str(root/'uninterrupted')
                    interrupted[output_arg] = str(root/'interrupted')
                    with patch.dict(module.SETTINGS, {'epochs':3,'patience':5}):
                        with patch.object(sys,'argv',uninterrupted): module.main()
                        real_save = module.atomic_torch_save
                        def stop_after_epoch_one(path, payload):
                            real_save(path,payload)
                            if path.name=='latest.pt' and payload.get('epoch')==1:
                                raise KeyboardInterrupt('Simulated interruption after durable epoch snapshot')
                        with patch.object(sys,'argv',interrupted), patch.object(module,'atomic_torch_save',side_effect=stop_after_epoch_one), self.assertRaises(KeyboardInterrupt):
                            module.main()
                        with patch.object(sys,'argv',interrupted+['--resume']): module.main()
                    full = torch.load(root/'uninterrupted/latest.pt',map_location='cpu',weights_only=False)
                    resumed = torch.load(root/'interrupted/latest.pt',map_location='cpu',weights_only=False)
                    for key, tensor in full['model'].items():
                        torch.testing.assert_close(tensor,resumed['model'][key],rtol=0,atol=0)
                    self.assertEqual(len(resumed['history']),3)
                    # Source cache corruption must not be reused.
                    (root/'pooled/id-0.npz').write_bytes(b'corruption')
                    with patch.object(sys,'argv',args+['--resume']), self.assertRaisesRegex(ValueError,'Section cache changed'):
                        module.main()
            finally:
                sys.path[:]=previous_path


if __name__ == '__main__': unittest.main()
