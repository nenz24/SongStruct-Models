import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('context_install',Path(__file__).resolve().parents[1]/'install_section_context.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class ContextInstallTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder)/'SongMap';struct=Path(folder)/'Section-Annotations'
        raw=struct/'outputs.local/muq-compression-fixture/palette6-all-validation.json'
        write(raw,dict(full_validation=True,head_scheme='palette6',muq_scheme='palette6',
            plan_id='plan',provenance_id='provenance',checkpoint_sha256='b'*64))
        source=root/'outputs.local/context-palette6-validation-fixture/validation-all.json'
        write(source,dict(context_runtime='PyTorch'))
        inputs=dict(palette6_report_sha256=module.sha_file(raw))
        write(source.parent/'selection.json',{'identity':module.fingerprint(inputs),**inputs})
        export=root/'outputs.local/context-coreml-fp32-fixture/export'
        package=export/'WorshipSectionContextFP32.mlpackage';package.mkdir(parents=True)
        (package/'Manifest.json').write_text('fixture model')
        common=dict(protocol='section-context-coreml-fp32-v1',context_checkpoint_sha256='a'*64,
            palette_validation_sha256=module.sha_file(source),palette_selection_identity=module.fingerprint(inputs),
            plan_id='plan',provenance_id='provenance',selected_epoch=5,torch='2.4.0',samples=[['song','sample']],
            package_digest=module.package_digest(package))
        contract=dict(dtype='float32',inputs={'section_audio':['N',4096],'section_timing':['N',6]},
            output={'context_residual':['N',8]},label_order=module.LABELS,all_original_sections_required=True)
        write(export/'export_report.json',dict(**common,package=str(package),input_contract=contract,precision='fp32',
            native_synthetic_parity_passed=True,attention_parity_passed=True,trace_parity_passed=True,max_sections=512,
            declared_tensor_dtypes={'FLOAT32':54}))
        parity=export.parent/'parity/validation-all.json'
        write(parity,dict(**common,split='val',full_validation=True,passed=True,status='numerically_consistent',
            precision='fp32',section_agreement=1.,macro_f1_delta=0.,tracks=1,checks=[dict(passed=True,
                residual={'passed':True},final_logits={'passed':True},probabilities={'passed':True})]))
        base=root/'SongMap/Resources/MLModels/SongFormerWorship'
        roles=['log_mel','musicfm_conv','musicfm','muq_conv','muq','head']
        static=['musicfm_750','musicfm_3000','muq_750','muq_3000']
        manifest=dict(sample_rate=24000,encoder_window_seconds=120,head_max_seconds=420,
            head={'dataset_id':5,'allowed_label_ids':[0,1,2,3,4,5,6,26]},
            models={r:'Worship_'+r+'.mlmodelc' for r in roles},worship_static={r:'Worship_'+r+'.mlmodelc' for r in static},
            installation_provenance={'source_validation_sha256':module.sha_file(raw)})
        write(base/'songformer_worship_manifest.json',manifest)
        for name in roles+static:
            model=base/('Worship_'+name+'.mlmodelc');model.mkdir();(model/'model.bin').write_bytes(b'base model')
        calls=[]
        def compile_stub(source,output):
            calls.append(source)
            target=output/(source.stem+'.mlmodelc');target.mkdir();(target/'model.bin').write_bytes(b'compiled context')
        return root,parity,base,calls,compile_stub

    def test_preserves_worship_installs_atomically_and_reuses_identical_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            root,parity,base,calls,compiler=self.fixture(folder)
            before=module.file_hashes(base)
            destination=module.install(root,parity,compiler=compiler)
            self.assertEqual(module.file_hashes(base),before)
            receipt=module.read(destination/'songformer_section_context_manifest.json')
            self.assertEqual(receipt['selectedEpoch'],5);self.assertEqual(receipt['baseCheckpointSHA256'],'b'*64)
            self.assertEqual(receipt['maxSections'],512)
            self.assertEqual(module.install(root,parity,compiler=compiler),destination)
            self.assertEqual(len(calls),1)
            (destination/receipt['model']/'model.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'differs/corrupt'):module.install(root,parity,compiler=compiler)

    def test_failed_parity_or_changed_base_profile_never_compiles_or_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            root,parity,base,calls,compiler=self.fixture(folder)
            original=module.read(parity)
            write(parity,{**original,'passed':False})
            with self.assertRaises(ValueError):module.install(root,parity,compiler=compiler)
            write(parity,original)
            manifest=module.read(base/'songformer_worship_manifest.json')
            manifest['installation_provenance']['source_validation_sha256']='wrong'
            write(base/'songformer_worship_manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Installed worship'):module.install(root,parity,compiler=compiler)
            self.assertEqual(calls,[])
            self.assertFalse((base.parent/'SongFormerContext').exists())

    def test_changed_export_or_incomplete_compilation_leaves_existing_resources_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root,parity,base,calls,compiler=self.fixture(folder)
            before=module.file_hashes(base)
            def broken_compiler(source,output):raise RuntimeError('compiler failed')
            with self.assertRaises(RuntimeError):module.install(root,parity,compiler=broken_compiler)
            self.assertEqual(module.file_hashes(base),before)
            self.assertFalse((base.parent/'SongFormerContext').exists())
            package=Path(module.read(parity.parent.parent/'export/export_report.json')['package'])
            (package/'Manifest.json').write_text('changed package')
            with self.assertRaisesRegex(ValueError,'package/input contract changed'):module.install(root,parity,compiler=compiler)
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
