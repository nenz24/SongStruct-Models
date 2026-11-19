import argparse
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

spec=importlib.util.spec_from_file_location('context_ml',Path(__file__).resolve().parents[1]/'context_coreml.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

fingerprint=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
sha_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,allow_nan=False))


def model_and_samples():
    torch.manual_seed(7);model=module.SectionContextHead().eval()
    torch.nn.init.normal_(model.output.weight,std=.03)
    samples=[];rows=[]
    for i,n in enumerate((3,7)):
        details=[];sections=[]
        for index,label in enumerate(('chorus','bridge')):
            p=dict.fromkeys(module.LABELS,.01);p[label]=.93
            sections.append(dict(start=index*2.,end=index*2.+1.,label=label))
            details.append(dict(track_id=f'song-{i}',start=index*2.,end=index*2.+1.,
                label_manual=label,prediction=label,probabilities=p))
        rows.append(dict(id=f'id-{i}',track_id=f'song-{i}',approved=True,split='val',sections=sections))
        samples.append(dict(track_id=f'song-{i}',split='val',identity=f'sample-{i}',
            audio=torch.randn(n,4096),timing=torch.rand(n,6),indices=torch.tensor([0,n-1]),
            labels=torch.tensor([2,3]),probabilities=torch.tensor([[d['probabilities'][k] for k in module.LABELS] for d in details]),
            details=details))
    return model,samples,rows


class ContextCoreMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_explicit_attention_and_trace_preserve_weights_and_variable_lengths(self):
        model,_,_=model_and_samples();before={k:v.clone() for k,v in model.state_dict().items()}
        wrapper=module.ExportableContext(model).eval()
        trace=torch.jit.trace(wrapper,(torch.randn(16,4096),torch.rand(16,6)),check_trace=False,strict=True)
        with torch.inference_mode():
            for n in (1,2,7,37,129,512):
                audio=torch.randn(n,4096);timing=torch.rand(n,6)
                reference=model(audio,timing).numpy()
                self.assertTrue(module.numerical(reference,wrapper(audio,timing).numpy(),1e-5,1e-5)['passed'])
                self.assertTrue(module.numerical(reference,trace(audio,timing).numpy(),1e-5,1e-5)['passed'])
        for key,value in before.items():torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)

    @unittest.skipUnless(importlib.util.find_spec('coremltools'),'coremltools needed for actual MIL conversion')
    def test_actual_mil_conversion_has_flexible_float32_contract(self):
        import coremltools as ct
        model,_,_=model_and_samples();wrapper=module.ExportableContext(model).eval()
        traced=torch.jit.trace(wrapper,(torch.randn(16,4096),torch.rand(16,6)),check_trace=False,strict=True)
        converted=module.convert_context(ct,traced,65,skip_model_load=True)
        counts=module.check_contract(converted.get_spec(),65)
        self.assertGreater(counts.get('FLOAT32',0),0);self.assertEqual(counts.get('FLOAT16',0),0)
        with tempfile.TemporaryDirectory() as folder:
            package=Path(folder)/'Context.mlpackage';converted.save(str(package))
            self.assertTrue((package/'Manifest.json').is_file())

    def test_native_inputs_are_owned_float32_and_reject_mismatched_or_nonfinite_shapes(self):
        values=module.native_inputs(np.ones((3,8192))[:,::2],np.zeros((3,6),dtype=np.float16),512)
        for v in values.values():
            self.assertEqual(v.dtype,np.float32);self.assertTrue(v.flags.c_contiguous);self.assertTrue(v.flags.owndata)
        for audio,timing in ((np.zeros((3,4096)),np.zeros((2,6))),
                              (np.zeros((0,4096)),np.zeros((0,6))),
                              (np.full((3,4096),float('nan')),np.zeros((3,6)))):
            with self.assertRaises(ValueError):module.native_inputs(audio,timing,512)
        with self.assertRaises(ValueError):module.native_inputs(np.zeros((4,4096)),np.zeros((4,6)),3)

    def test_native_parity_checks_all_residuals_and_final_probabilities(self):
        model,samples,_=model_and_samples();sample=samples[0]
        def predict(inputs):
            with torch.inference_mode():result=model(torch.from_numpy(inputs['section_audio']),torch.from_numpy(inputs['section_timing'])).numpy()
            return {'context_residual':result}
        checks,details=module.verify_track(model,sample,predict,512)
        self.assertTrue(checks['passed']);self.assertEqual(len(details),2) # Middle context-only token is not a target.
        def shifted(inputs):return {'context_residual':predict(inputs)['context_residual']+.01}
        failed,_=module.verify_track(model,sample,shifted,512)
        self.assertFalse(failed['passed']);self.assertEqual(failed['section_argmax_agreement'],1.)
        with self.assertRaisesRegex(ValueError,'Float32'):
            module.verify_track(model,sample,lambda x:{'context_residual':predict(x)['context_residual'].astype(np.float16)},512)

    def test_loader_binds_frozen_weights_reports_and_palette_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={'work_dir':str(root),'features':{'profile':'frozen'}}
            row=dict(id='song',track_id='song',duration=.24,approved=True,split='val',audio_sha256='audio',
                source_json=str(root/'source.json'),sections=[dict(start=0.,end=.08,label='chorus'),dict(start=.16,end=.24,label='bridge')])
            row['annotation_sha256']=fingerprint(row['sections'])
            write(row['source_json'],dict(track_id='song',duration=.24,
                sections=[dict(start=0.,end=.08),dict(start=.08,end=.16),dict(start=.16,end=.24)]))
            header={'environment':{'feature_settings':cfg['features']}};encoder_id=fingerprint(header)
            raw=root/'raw';write(raw/'encoder_provenance.json',{'id':encoder_id,**header})
            (raw/'song').mkdir();np.save(raw/'song/chunk.npy',np.random.RandomState(7).randn(6,4096).astype(np.float32),allow_pickle=False)
            write(raw/'song/metadata.json',dict(audio_sha256='audio',candidate_encoder_id=encoder_id,
                chunks=[dict(file='chunk.npy',offset=0.,frames=6)]))
            details=[]
            for s in row['sections']:
                p=dict.fromkeys(module.LABELS,.01);p[s['label']]=.93
                details.append(dict(track_id='song',start=s['start'],end=s['end'],label_manual=s['label'],prediction=s['label'],probabilities=p))
            annotations=fingerprint([(row['track_id'],row['sections'])])
            proof=root/'proof.json';write(proof,dict(full_validation=True,muq_scheme='palette6',head_scheme='palette6',reused_musicfm=True,
                tracks=1,plan_id='plan',provenance_id='provenance',checkpoint_sha256='base',annotations_id=annotations,
                candidate_encoder_id=encoder_id,diagnostic_cache=str(raw),sections=details,metrics=module.probability_metrics(details)))
            cache=root/'pooled';cache.mkdir()
            helpers=read,write,fingerprint,sha_file,lambda cfg,r:module.palette_features(raw,r,encoder_id,read)
            samples=module.prepare_samples([row],details,cfg,encoder_id,cache,helpers,feature_cache_root=raw)
            model=module.SectionContextHead().eval();torch.nn.init.normal_(model.output.weight,std=.01)
            reference,_=module.evaluate_context(model,samples,[row],torch.device('cpu'))
            frozen=root/'frozen';frozen.mkdir()
            torch.save(dict(model=model.state_dict(),architecture=module.PROTOCOL,best_epoch=5,plan_id='plan',provenance_id='provenance',
                settings={'hidden':64,'dropout':.25}),frozen/'frozen-context.pt')
            lock=dict(protocol='frozen-section-context-test-v1',context_epoch=5,context_checkpoint_sha256=sha_file(frozen/'frozen-context.pt'),plan_id='plan',provenance_id='provenance',
                feature_settings=cfg['features'],checkpoint_sha256='base')
            write(frozen/'selection.json',{'identity':fingerprint(lock),**lock})
            selection=dict(protocol='palette6-context-validation-v1',plan_id='plan',provenance_id='provenance',annotations_id=annotations,
                frozen_selection_identity=fingerprint(lock),context_checkpoint_sha256=sha_file(frozen/'frozen-context.pt'),
                palette6_report_sha256=sha_file(proof),candidate_encoder_id=encoder_id,encoder_header_sha256=sha_file(raw/'encoder_provenance.json'),
                torch=str(torch.__version__),coverage_policy=module.COVERAGE_POLICY)
            report=root/'validation/validation-all.json'
            write(report.parent/'selection.json',{'identity':fingerprint(selection),**selection})
            write(report,dict(protocol=selection['protocol'],split='val',full_validation=True,tracks=1,selected_epoch=5,
                selection_identity=fingerprint(selection),plan_id='plan',provenance_id='provenance',weights_updated=False,
                context_runtime='PyTorch',original_context_validation_reproduced=True,saved_palette6_baseline_metrics_reproduced=True,
                palette6_features_with_context=reference))
            common=types.ModuleType('finetuning.common');engine=types.ModuleType('finetuning.engine')
            for k,v in dict(config=lambda p:cfg,read=read,write=write,fingerprint=fingerprint,sha_file=sha_file).items():setattr(common,k,v)
            engine.manifest=lambda cfg:({'plan_id':'plan','records':[row]},'provenance')
            parent=types.ModuleType('finetuning');parent.__path__=[]
            args=argparse.Namespace(songstruct_root=root,palette_validation=report,frozen_evaluation=frozen,palette6_report=proof,cache_dir=cache)
            previous=list(sys.path)
            try:
                with patch.dict(sys.modules,{'finetuning':parent,'finetuning.common':common,'finetuning.engine':engine}):
                    loaded,loaded_samples,rows,identity,_=module.load_inputs(args)
                    self.assertEqual(identity['samples'],[['song',samples[0]['identity']]])
                    self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                    self.assertEqual(loaded_samples[0]['indices'].tolist(),[0,2])
                    (frozen/'frozen-context.pt').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'identity changed'):module.load_inputs(args)
            finally:sys.path[:]=previous

    def test_export_verify_orchestration_preserves_failed_numerical_diagnostics_and_reuses_cache(self):
        model,samples,rows=model_and_samples()
        identity=dict(protocol=module.EXPORT_PROTOCOL,context_checkpoint_sha256='frozen',
            samples=[[s['track_id'],s['identity']] for s in samples])
        loaded=(model,samples,rows,identity,(read,write,fingerprint,sha_file))
        calls=[]
        class FakeNative:
            user_defined_metadata={}
            short_description=''
            offset=0.
            def save(self,path):
                p=Path(path);p.mkdir(parents=True);(p/'Manifest.json').write_text('test runtime stub')
            def get_spec(self):return object()
            def predict(self,inputs):
                calls.append(len(inputs['section_audio']))
                with torch.inference_mode():out=model(torch.from_numpy(inputs['section_audio']),torch.from_numpy(inputs['section_timing'])).numpy()
                return {'context_residual':out+self.offset}
        fake=FakeNative();ct=types.SimpleNamespace(__version__='8.3.0',models=types.SimpleNamespace(MLModel=lambda *a,**k:fake),
            ComputeUnit=types.SimpleNamespace(CPU_ONLY='cpu'))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);export=root/'export';out=root/'verify'
            with patch.object(module,'load_inputs',return_value=loaded),patch.dict(sys.modules,{'coremltools':ct}), \
                    patch.object(module,'convert_context',return_value=fake),patch.object(module,'check_contract',return_value={'FLOAT32':1}), \
                    patch.object(torch.optim,'AdamW',side_effect=AssertionError('No optimizer during conversion')):
                module.export_model(argparse.Namespace(output_dir=export,max_sections=65))
                exported=read(export/'export_report.json')
                self.assertTrue(exported['native_synthetic_parity_passed'])
                args=argparse.Namespace(export_dir=export,output_dir=out,limit=0)
                module.verify_model(args)
                report=read(out/'validation-all.json')
                self.assertTrue(report['passed']);self.assertEqual(report['section_agreement'],1.)
                n=len(calls);module.verify_model(args);self.assertEqual(len(calls),n)
                self.assertFalse(report['ready_for_app_integration'])
                fake.offset=.01
                args.output_dir=root/'failed'
                with self.assertRaisesRegex(RuntimeError,'parity failed'):module.verify_model(args)
                failure=read(args.output_dir/'validation-all.json')
                self.assertFalse(failure['passed']);self.assertEqual(failure['section_agreement'],1.)
                self.assertEqual(failure['macro_f1_delta'],0.)


if __name__=='__main__':unittest.main()
