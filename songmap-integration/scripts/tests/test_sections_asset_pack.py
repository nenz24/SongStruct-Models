"""The MuQ-only section models ship in the "sections" asset pack: these lists must agree."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PACKS = load('build_asset_packs')
INSTALL = load('install_muq_pipeline')
LOCATOR = (ROOT / 'SongMap/Data/SongAnalysis/ModelLocator.swift').read_text()
TEMPLATE = (ROOT / 'config/xcode/project.pbxproj.template').read_text()
DOWNLOADER = (ROOT / 'AssetDownloader/AssetDownloader.swift').read_text()
PIPELINE = (ROOT / 'SongMap/Data/SongAnalysis/Structure/SongFormerMuQPipeline.swift').read_text()
ANALYZER = (ROOT / 'SongMap/Data/SongAnalysis/Structure/SongFormerAnalyzer.swift').read_text()


def swift_packs():
    return {name: pack for name, pack in re.findall(r'\.model\("([^"]+)"\): \.(\w+)', LOCATOR)}


class SectionsAssetPackTests(unittest.TestCase):
    def test_pack_builder_app_locator_and_installer_agree(self):
        installed = {name + '.mlmodelc' for name in INSTALL.RESOURCES.values()}
        built = {Path(p).name for p in PACKS.PACKS['sections'][0]}
        self.assertEqual(built, installed)
        self.assertTrue(all(p.startswith(Path(INSTALL.DESTINATION).name + '/') for p in PACKS.PACKS['sections'][0]))
        swift = {name + '.mlmodelc' for name, pack in swift_packs().items() if pack == 'sections'}
        self.assertEqual(swift, installed)
        for pack, (models, files) in PACKS.PACKS.items():
            self.assertEqual({Path(p).stem for p in models} | {Path(f).stem for f in files},
                             {n for n, p in swift_packs().items() if p == pack} | ({'ggml-small-q5_0'} if pack == 'lyrics' else set()))

    def test_release_leaves_the_pack_models_out_of_the_app_but_keeps_the_manifest(self):
        lists = re.findall(r'SONGMAP_ASSET_PACK_FILES = "([^"]+)";', TEMPLATE)
        self.assertEqual(len(lists), 2)
        for value in lists:
            names = set(value.split())
            self.assertTrue({name + '.mlmodelc' for name in INSTALL.RESOURCES.values()} <= names)
            self.assertNotIn(INSTALL.MANIFEST, names)

    def test_app_downloads_sections_not_the_legacy_structure_pack(self):
        self.assertIn('static let songAnalysis = packs.keys.filter { $0.pack != .structure }', LOCATOR)
        self.assertIn('assetPack.id != "structure"', DOWNLOADER)
        self.assertIn('ModelLocator.modelURL(named: name)', PIPELINE)
        self.assertNotIn('Bundle.main.url(forResource: name, withExtension: "mlmodelc")', PIPELINE + ANALYZER)
        self.assertIn('muqOnly = true; worship = false', ANALYZER)

    def test_precompiled_models_are_packed_only_when_they_match_the_validated_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'SongFormerMuQ_Head.mlmodelc'; (model / 'weights').mkdir(parents=True)
            (model / 'model.mil').write_text('mil'); (model / 'weights/weight.bin').write_bytes(b'w')
            manifest = Path(folder) / 'songformer_muq_manifest.json'
            hashes = {'model.mil': PACKS.sha256(model / 'model.mil'), 'weights/weight.bin': PACKS.sha256(model / 'weights/weight.bin')}
            manifest.write_text(json.dumps({'compiledFiles': {model.name: hashes}}))
            PACKS.check_compiled(model)
            (model / 'model.mil').write_text('changed')
            with self.assertRaises(SystemExit):
                PACKS.check_compiled(model)
            manifest.unlink()
            with self.assertRaises(SystemExit):
                PACKS.check_compiled(model)

    def test_sections_pack_manifest_copies_compiled_models_without_compiling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / PACKS.MODELS / 'SongFormerMuQ'; target.mkdir(parents=True)
            compiled = {}
            for name in INSTALL.RESOURCES.values():
                (target / f'{name}.mlmodelc').mkdir(); (target / f'{name}.mlmodelc/model.mil').write_text(name)
                compiled[f'{name}.mlmodelc'] = {'model.mil': PACKS.sha256(target / f'{name}.mlmodelc/model.mil')}
            (target / INSTALL.MANIFEST).write_text(json.dumps({'compiledFiles': compiled}))
            (root / 'out').mkdir(); (root / 'out/sections.aar').write_text('earlier run')
            with mock.patch.object(PACKS.subprocess, 'run') as run:
                PACKS.build_pack(root, 'sections', root / 'out')
            self.assertFalse((root / 'out/sections.aar').exists())  # ba-package can't overwrite
            self.assertEqual(run.call_count, 1)  # ba-package only; nothing compiled
            self.assertEqual(run.call_args.args[0][:3], ['xcrun', 'ba-package', 'package'])
            manifest = json.loads((root / 'out/sections/Manifest.json').read_text())
            self.assertEqual(manifest['assetPackID'], 'sections')
            self.assertEqual(sorted(s['directoryDestination'] for s in manifest['fileSelectors']),
                             sorted(f'Models/{n}.mlmodelc' for n in INSTALL.RESOURCES.values()))


if __name__ == '__main__':
    unittest.main()
