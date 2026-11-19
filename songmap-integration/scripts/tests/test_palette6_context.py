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

spec=importlib.util.spec_from_file_location('palette_context',Path(__file__).resolve().parents[1]/'evaluate_palette6_context.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

fingerprint=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
sha_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,allow_nan=False))


def fixture():
    rows=[dict(track_id='song',split='val',approved=True,sections=[dict(start=0.,end=1.,label='bridge')])]
    p=dict.fromkeys(module.LABELS,.01);p['bridge']=.93
    details=[dict(track_id='song',start=0.,end=1.,label_manual='bridge',prediction='bridge',probabilities=p)]
    proof=dict(full_validation=True,muq_scheme='palette6',head_scheme='palette6',reused_musicfm=True,
        tracks=1,plan_id='plan',provenance_id='provenance',checkpoint_sha256='checkpoint',
        annotations_id=fingerprint([(r['track_id'],r['sections']) for r in rows]),
        sections=details,metrics=module.probability_metrics(details))
    return rows,details,proof


class Palette6ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_requires_full_same_inventory_palette6_baseline(self):
        rows,details,proof=fixture()
        result=module.validate_palette_report(proof,rows,'plan','provenance','checkpoint',fingerprint)
        self.assertEqual(result,details)
        for key,value in (('full_validation',False),('muq_scheme','int8'),('head_scheme','fp32'),
                          ('reused_musicfm',False),('annotations_id','changed'),('checkpoint_sha256','changed')):
            with self.assertRaises(ValueError):
                module.validate_palette_report({**proof,key:value},rows,'plan','provenance','checkpoint',fingerprint)
        changed=copy.deepcopy(proof);changed['metrics']['section_nll']+=.1
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            module.validate_palette_report(changed,rows,'plan','provenance','checkpoint',fingerprint)

    def test_detail_order_is_normalized_without_changing_manual_targets(self):
        rows,details,_=fixture()
        second={**rows[0],'track_id':'other'};rows.append(second)
        other={**details[0],'track_id':'other'}
        before=copy.deepcopy(rows)
        result=module.ordered_details(rows,[other,details[0]])
        self.assertEqual([s['track_id'] for s in result],['song','other'])
        self.assertEqual(rows,before)
        with self.assertRaises(ValueError):module.ordered_details(rows,[other])

    def test_palette_reader_rejects_bad_identity_shape_and_escaping_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row={'id':'song','audio_sha256':'audio'};folder=root/'song';folder.mkdir()
            np.save(folder/'chunk.npy',np.ones((6,4096),dtype=np.float32),allow_pickle=False)
            meta=dict(audio_sha256='audio',candidate_encoder_id='encoder',chunks=[dict(file='chunk.npy',offset=0.,frames=6)])
            write(folder/'metadata.json',meta)
            self.assertEqual(list(module.palette_features(root,row,'encoder',read))[0][0].shape,(6,4096))
            with self.assertRaises(ValueError):list(module.palette_features(root,row,'other',read))
            for chunk in (dict(file='chunk.npy',offset=0.,frames=3),dict(file='../../outside.npy',offset=0.,frames=6)):
                write(folder/'metadata.json',{**meta,'chunks':[chunk]})
                with self.assertRaises(ValueError):list(module.palette_features(root,row,'encoder',read))

    def test_cli_reuses_palette_features_and_probabilities_without_training_or_coreml(self):
        import train_section_context as training
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[];details=[];cfg={'work_dir':str(root),'features':{'profile':'frozen'}}
            for i in range(2):
                sections=[dict(start=0.,end=.08,label='chorus'),dict(start=.16,end=.24,label='bridge')]
                row=dict(id=f'id-{i}',track_id=f'song-{i}',split='val',approved=True,duration=.24,
                    audio_sha256=f'audio-{i}',annotation_sha256=fingerprint(sections),
                    source_json=str(root/f'source-{i}.json'),sections=sections)
                rows.append(row)
                write(row['source_json'],dict(track_id=row['track_id'],duration=.24,
                    sections=[dict(start=0.,end=.08),dict(start=.08,end=.16),dict(start=.16,end=.24)]))
                folder=root/'features'/row['id'];folder.mkdir(parents=True)
                np.save(folder/'chunk.npy',np.random.RandomState(i).randn(6,4096).astype(np.float32),allow_pickle=False)
                write(folder/'metadata.json',dict(audio_sha256=row['audio_sha256'],features=cfg['features'],
                    provenance_id='provenance',chunks=[dict(file='chunk.npy',offset=0.,frames=6)]))
                for section in sections:
                    p=dict.fromkeys(module.LABELS,.01);p[section['label']]=.93
                    details.append(dict(track_id=row['track_id'],start=section['start'],end=section['end'],
                        label_manual=section['label'],prediction=section['label'],probabilities=p))
            original_cache=root/'original-pooled';original_cache.mkdir()
            def iter_features(cfg,row):
                yield np.load(root/'features'/row['id']/'chunk.npy',allow_pickle=False),0.,'provenance'
            helpers=read,write,fingerprint,sha_file,iter_features
            original=module.prepare_samples(rows,details,cfg,'provenance',original_cache,helpers)
            torch.manual_seed(7);model=module.SectionContextHead()
            torch.nn.init.normal_(model.output.weight,std=.01)
            original_metrics,_=module.evaluate_context(model,original,rows,torch.device('cpu'))
            context=root/'context';context.mkdir()
            run=dict(protocol=module.PROTOCOL,device='cpu',settings=dict(training.SETTINGS),
                coverage_policy=module.COVERAGE_POLICY,sample_identities=[(s['track_id'],s['identity']) for s in original])
            run_id=fingerprint(run)
            write(context/'run.json',{'identity':run_id,**run})
            write(context/'comparison.json',dict(protocol=module.PROTOCOL,split='val',tracks=2,full_validation=True,
                best_epoch=5,experiment_identity=run_id,baseline=module.probability_metrics(details),candidate=original_metrics))
            calibration=root/'calibration/comparison.json'
            write(calibration,dict(experiment_identity='original-val-cache'))
            write(calibration.parent/'adapter.json',{'adapter':{'kind':'identity'}})
            for row in rows:
                write(calibration.parent/'val-predictions'/(row['id']+'.json'),dict(identity='original-val-cache',
                    sections=[s for s in details if s['track_id']==row['track_id']]))
            frozen=root/'frozen';frozen.mkdir()
            torch.save(dict(identity=run_id,architecture=module.PROTOCOL,best_epoch=5,settings=dict(training.SETTINGS),
                model=model.state_dict(),plan_id='plan',provenance_id='provenance'),frozen/'frozen-context.pt')
            lock=dict(protocol='frozen-section-context-test-v1',context_epoch=5,
                context_checkpoint_sha256=sha_file(frozen/'frozen-context.pt'),plan_id='plan',provenance_id='provenance',
                feature_settings=cfg['features'],torch=str(torch.__version__),coverage_policy=module.COVERAGE_POLICY,
                training_run_sha256=sha_file(context/'run.json'),selected_validation_report_sha256=sha_file(context/'comparison.json'),
                calibration_report_sha256=sha_file(calibration),calibration_adapter_sha256=sha_file(calibration.parent/'adapter.json'),
                checkpoint_sha256='original-head-checkpoint')
            write(frozen/'selection.json',{'identity':fingerprint(lock),**lock})
            # No test metric file is needed or allowed to be read by this validation experiment.
            palette=root/'palette';palette.mkdir()
            export=root/'export.json';write(export,dict(compression_scheme='palette6',source_full_validation=True,
                window_finite_passed=True,package_digest='muq-package'))
            pipeline=root/'pipeline.json';write(pipeline,dict(candidate_encoder_id='musicfm-pal',
                musicfm_override={'compression_scheme':'palette6'}))
            shared=dict(plan_id='plan',provenance_id='provenance',annotations_id=fingerprint([(r['track_id'],r['sections']) for r in rows]),
                checkpoint_sha256='original-head-checkpoint',tracks=2)
            base=root/'base.json';write(base,dict(**shared,full_validation=True,head_scheme='palette6',
                musicfm_override={'compression_scheme':'palette6'},candidate_encoder_id='musicfm-pal',
                encoder_pipeline_report=str(pipeline),encoder_pipeline_report_sha256=sha_file(pipeline)))
            header=dict(baseline_encoder_id='musicfm-pal',reused_musicfm=True,muq_export_report_sha256=sha_file(export),
                muq_package_digest='muq-package',environment={'feature_settings':cfg['features']})
            encoder_id=fingerprint(header);write(palette/'encoder_provenance.json',{'id':encoder_id,**header})
            palette_details=[]
            for row in rows:
                folder=palette/row['id'];folder.mkdir()
                values=np.load(root/'features'/row['id']/'chunk.npy',allow_pickle=False)+.05
                np.save(folder/'chunk.npy',values,allow_pickle=False)
                write(folder/'metadata.json',dict(audio_sha256=row['audio_sha256'],candidate_encoder_id=encoder_id,
                    chunks=[dict(file='chunk.npy',offset=0.,frames=6)]))
            for item in details:
                p=dict.fromkeys(module.LABELS,.02);p[item['label_manual']]=.86
                palette_details.append({**item,'probabilities':p})
            proof=root/'palette-validation.json';write(proof,dict(**shared,full_validation=True,muq_scheme='palette6',
                head_scheme='palette6',reused_musicfm=True,sections=palette_details,metrics=module.probability_metrics(palette_details),
                baseline_combination=str(base),baseline_combination_sha256=sha_file(base),muq_export_report=str(export),
                muq_export_report_sha256=sha_file(export),candidate_encoder_id=encoder_id,diagnostic_cache=str(palette)))
            out=root/'output';candidate_cache=root/'palette-pooled'
            common=types.ModuleType('finetuning.common');engine=types.ModuleType('finetuning.engine')
            def checked_read(p):
                if Path(p).name=='test-all.json':raise AssertionError('No test metrics in validation')
                return read(p)
            for key,value in dict(config=lambda p:cfg,read=checked_read,write=write,fingerprint=fingerprint,sha_file=sha_file,
                    iter_features=iter_features,device=torch.device).items():setattr(common,key,value)
            engine.manifest=lambda cfg:({'plan_id':'plan','records':rows},'provenance')
            parent=types.ModuleType('finetuning');parent.__path__=[]
            args=['evaluate_palette6_context.py','--songstruct-root',str(root),'--frozen-evaluation',str(frozen),
                '--context-run',str(context),'--calibration',str(calibration),'--palette6-report',str(proof),
                '--original-cache-dir',str(original_cache),'--cache-dir',str(candidate_cache),'--output-dir',str(out)]
            old_path=list(sys.path);state_hash=sha_file(frozen/'frozen-context.pt')
            try:
                with patch.dict(sys.modules,{'finetuning':parent,'finetuning.common':common,'finetuning.engine':engine}), \
                        patch.object(torch.optim,'AdamW',side_effect=AssertionError('No fitting')), \
                        patch.dict(sys.modules,{'coremltools':None}):
                    with patch.object(sys,'argv',args+['--limit','1']):module.main()
                    self.assertFalse(read(out/'validation-first-1.json')['full_validation'])
                    with patch.object(sys,'argv',args):module.main()
                    report=read(out/'validation-all.json')
                    self.assertTrue(report['full_validation']);self.assertEqual(report['tracks'],2)
                    self.assertTrue(report['original_context_validation_reproduced'])
                    self.assertFalse(report['weights_updated']);self.assertFalse(report['coreml_head_executed'])
                    self.assertFalse(report['test_metrics_used']);self.assertFalse(report['encoder_executed'])
                    original_array=np.load(original_cache/'id-0.npz',allow_pickle=False)['audio']
                    candidate_array=np.load(candidate_cache/'id-0.npz',allow_pickle=False)['audio']
                    self.assertFalse(np.array_equal(original_array,candidate_array))
                    # Reuse separate checksummed candidate means without re-reading .npy source arrays.
                    with patch.object(module,'palette_features',side_effect=AssertionError('Cached means should be reused')):
                        with patch.object(sys,'argv',args):module.main()
                    self.assertEqual(sha_file(frozen/'frozen-context.pt'),state_hash)
                    broken=read(proof);broken['metrics']['section_nll']+=.1;write(proof,broken)
                    with patch.object(sys,'argv',args),self.assertRaisesRegex(ValueError,'does not reproduce'):
                        module.main()
            finally:sys.path[:]=old_path


if __name__=='__main__':unittest.main()
