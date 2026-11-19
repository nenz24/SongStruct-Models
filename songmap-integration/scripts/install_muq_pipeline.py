#!/usr/bin/env python3
"""Install the iPhone-validated MuQ-only SongFormer resources: the app's section-label models.

Source: a prepared `muq_pipeline_iphone.ipynb` probe folder (outputs.local/muq-pipeline-iphone-*), whose five
compiled models already passed the Mac reference and the phone CPU gates. Every compiled file is checked
against the probe manifest before copying. Destination: SongMap/Resources/MLModels/SongFormerMuQ/, tracked
with the compiled models in Git LFS, and songformer_muq_manifest.json read by SongFormerMuQPipeline.
Debug builds carry the models in the app; Release builds leave them out and download them in the
"sections" asset pack (scripts/build_asset_packs.py --pack sections, uploaded to App Store Connect).
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from install_section_context import file_hashes, read, sha_file

DESTINATION = 'SongMap/Resources/MLModels/SongFormerMuQ'
MANIFEST = 'songformer_muq_manifest.json'
# Probe asset -> unique bundle resource name (the bundle is flat; avoid clashing with legacy MuQ_* packages).
RESOURCES = {'log_mel': 'SongFormerMuQ_LogMel', 'muq_conv': 'SongFormerMuQ_Conv', 'muq': 'SongFormerMuQ_Encoder',
             'head': 'SongFormerMuQ_Head', 'context': 'SongFormerMuQ_Context'}
LABELS = ['intro', 'verse', 'chorus', 'bridge', 'inst', 'outro', 'silence', 'pre-chorus']
FUNCTIONS = {'750': 'short_30', '3000': 'long_120', 'partial': 'partial'}


def validate_probe(probe, phone_batch=None):
    """Return the probe manifest after checking preparation, manifest and every compiled file."""
    preparation = read(probe / 'preparation.json')
    assets = probe / 'Assets'
    manifest_path = assets / 'benchmark_manifest.json'
    if (preparation.get('protocol') != 'muq-pipeline-phone-preparation-v1' or not preparation.get('native_swift_checks_passed')
            or not preparation.get('native_typecheck_passed') or not preparation.get('mac_reference_passed')
            or preparation.get('musicfm_bundled') is not False or preparation.get('encoder_roles') != ['muq']
            or sha_file(manifest_path) != preparation.get('manifest_sha256')):
        raise ValueError('Use a completed MuQ-only probe preparation (muq_pipeline_iphone.ipynb).')
    manifest = read(manifest_path)
    if (manifest.get('encoder_roles') != ['muq'] or manifest.get('canonical_encoder_functions') != {'muq': FUNCTIONS}
            or set(manifest.get('files', {})) != set(RESOURCES) or len(manifest.get('rotary', {}).get('muq', [])) != 32):
        raise ValueError('Probe manifest is not the validated MuQ-only contract.')
    for name, expected in manifest['files'].items():
        if file_hashes(assets / (name + '.mlmodelc')) != expected:
            raise ValueError(f'Compiled model changed since the probe was prepared: {name}')
    evidence = None
    if phone_batch is not None:
        batch = read(phone_batch)
        runs = [run['report'] for run in batch.get('runs', [])]
        if (batch.get('state') != 'completed' or batch.get('mode') != 'cpuOnly' or not runs
                or batch.get('completed_runs') != batch.get('requested_runs')
                or any(r.get('manifest_sha256') != preparation['manifest_sha256'] or r.get('label_parity_passed') is not True
                       for r in runs)):
            raise ValueError('The phone batch must be a completed CPU batch of this exact probe with every parity gate passed.')
        evidence = {'phone_batch_sha256': sha_file(phone_batch), 'phone_runs': len(runs),
                    'phone_hardware': sorted({r.get('hardware_identifier') for r in runs})}
    return preparation, manifest, evidence


def find_phone_batch(probe, folders):
    """Newest completed CPU batch of this exact probe among pipeline-batch-latest*.json files, else None."""
    found = []
    for folder in folders:
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            continue
        for path in folder.rglob('pipeline-batch-latest*.json'):
            try:
                validate_probe(probe, path)
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
                continue
            found.append(path)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def app_manifest(preparation, manifest, evidence, compiled):
    return {'schemaVersion': 1, 'pipeline': 'muq-only-v1',
            'models': {role: name + '.mlmodelc' for role, name in RESOURCES.items()},
            'canonicalFunctions': FUNCTIONS, 'rotaryInverseFrequencies': manifest['rotary']['muq'],
            'headCheckpointSHA256': manifest['base_checkpoint'], 'contextCheckpointSHA256': manifest['context_checkpoint'],
            'labelOrder': LABELS, 'allowedLabelIDs': [0, 1, 2, 3, 4, 5, 6, 26], 'maxSections': 512,
            'featureWidth': 2048, 'sampleRate': 24000, 'shortWindowSeconds': 30, 'longWindowSeconds': 120,
            'headWindowSeconds': 420,
            'provenance': {'probeManifestSHA256': preparation['manifest_sha256'], 'qualitySHA256': preparation['quality_sha256'],
                           'compiledEncoderBytes': manifest['compiled_encoder_bytes'], **(evidence or {})},
            'compiledFiles': compiled}


def install(root, probe, phone_batch=None):
    root, probe = Path(root).resolve(), Path(probe).resolve()
    preparation, manifest, evidence = validate_probe(probe, phone_batch)
    destination = root / DESTINATION
    compiled = {RESOURCES[name] + '.mlmodelc': files for name, files in manifest['files'].items()}
    expected = app_manifest(preparation, manifest, evidence, compiled)
    if destination.exists():
        if read(destination / MANIFEST) != expected or any(
                file_hashes(destination / name) != files for name, files in compiled.items()):
            raise ValueError('A different SongFormerMuQ installation exists; move it aside first.')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.muq-stage-') as temp:
        staged = Path(temp) / 'SongFormerMuQ'
        staged.mkdir()
        for name, resource in RESOURCES.items():
            shutil.copytree(probe / 'Assets' / (name + '.mlmodelc'), staged / (resource + '.mlmodelc'))
            if file_hashes(staged / (resource + '.mlmodelc')) != manifest['files'][name]:
                raise ValueError('Copy changed a compiled model: ' + name)
        (staged / MANIFEST).write_text(json.dumps(expected, indent=2, allow_nan=False) + '\n')
        staged.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, help='Default: newest prepared outputs.local/muq-pipeline-iphone-*')
    parser.add_argument('--phone-batch', type=Path, help='Completed CPU pipeline-batch-latest.json from the phone '
                        '(default: searched in ~/Downloads, ~/Desktop, ~/Documents and outputs.local)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    probe = args.probe_dir
    if probe is None:
        candidates = sorted(p for p in (root / 'outputs.local').glob('muq-pipeline-iphone-*') if (p / 'preparation.json').is_file())
        if not candidates:
            parser.error('No prepared MuQ-only probe found; run muq_pipeline_iphone.ipynb first.')
        probe = candidates[-1]
    print('Probe:', probe)
    batch = args.phone_batch
    if batch is not None and not batch.expanduser().is_file():
        parser.error(f'Phone batch not found: {batch}. Omit --phone-batch to search the usual folders.')
    if batch is None:
        batch = find_phone_batch(probe, [Path.home() / 'Downloads', Path.home() / 'Desktop', Path.home() / 'Documents',
                                         root / 'outputs.local'])
        print('Phone CPU batch:', batch or 'not found (installing with Mac/probe evidence only)')
    print('Installed locally:', install(root, probe, batch))
    print('Next: commit SongMap/Resources/MLModels/SongFormerMuQ (Git LFS), then build the asset pack with '
          'python3 scripts/build_asset_packs.py --pack sections and upload it to App Store Connect')


if __name__ == '__main__':
    main()
