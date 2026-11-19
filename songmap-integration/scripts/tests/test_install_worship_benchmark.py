import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('install',Path(__file__).resolve().parents[1]/'install_worship_benchmark.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class InstallTests(unittest.TestCase):
    def setup_files(self,temp):
        root=Path(temp)/'app';bench=Path(temp)/'bench'
        manifest=root/'SongMap/Resources/MLModels/SongFormer/songformer_coreml_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'sample_rate':24000,'head':{'dataset_id':5},'rotary_inv_freq':{'muq':[1],'musicfm':[2]}}))
        assets=bench/'Assets';assets.mkdir(parents=True)
        (assets/'benchmark_manifest.json').write_text(json.dumps({'hybrid':True,'class_ids':[0,1,2,3,4,5,6,26],
            'labels':['intro','verse','chorus','bridge','inst','outro','silence','pre-chorus'],
            'rotary_inv_freq':{'muq':[1],'musicfm':[2]},'source_validation_sha256':'proof','package_digests':{}}))
        for role in ['log_mel','musicfm_conv','musicfm','muq_conv','muq','head','musicfm_750','musicfm_3000','muq_750','muq_3000']:
            d=assets/(role+'.mlmodelc');d.mkdir();(d/'model.bin').write_bytes(b'weights')
        return root,bench,manifest

    def test_keeps_legacy_and_requires_explicit_new_destination(self):
        with tempfile.TemporaryDirectory() as t:
            root,bench,manifest=self.setup_files(t);before=manifest.read_bytes()
            dest=module.install(root,bench)
            self.assertEqual(before,manifest.read_bytes())
            new=json.loads((dest/'songformer_worship_manifest.json').read_text())
            self.assertEqual(new['models']['head'],'Worship_head.mlmodelc')
            self.assertEqual(len(list(dest.glob('*.mlmodelc'))),10)
            with self.assertRaises(ValueError):module.install(root,bench)

    def test_missing_asset_fails_before_installing_anything(self):
        with tempfile.TemporaryDirectory() as t:
            root,bench,manifest=self.setup_files(t)
            (bench/'Assets/muq_750.mlmodelc/model.bin').unlink()
            with self.assertRaises(ValueError):module.install(root,bench)
            self.assertFalse((manifest.parent.parent/'SongFormerWorship').exists())
