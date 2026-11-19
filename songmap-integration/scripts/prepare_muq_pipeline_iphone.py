#!/usr/bin/env python3
"""Stage an isolated physical-iPhone probe of the MuQ-only label pipeline (no MusicFM).

Bundles exactly five compiled models: log_mel, muq_conv (unchanged app frontends), muq (the
existing canonical MuQ multi-function bank, copied by digest), head (2048-input palette6) and
context (2048-input FP32). The probe runner is the canonical one with `encoder_roles=["muq"]`:
serial role schedule, owned FP32 transfers, CPU frontends and partials, CPU+GPU selectable only
for full MuQ windows, and the unchanged phone gates (identical labels, base and final
probabilities within 1e-4, no unexplained low coverage).

Before staging, on the Mac: native Swift checks (owned tensors, tensor bridge incl. two-stream
fusion, section-context math incl. 2048-wide pooling), an iPhone SDK typecheck, and a fresh Mac
Core ML reference for the two bundled songs that must reproduce the completed 32-song
MuQ encoder validation (`muq_encoder_quality.py`): identical labels, final probabilities <= 1e-4.
"""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_muq_pipeline import CONTEXT, HEAD_PALETTE6, CoreMLHead
from install_section_context import file_hashes, package_digest
from muq_encoder_quality import PROTOCOL as QUALITY_PROTOCOL, MuQEncoder, muq_chunks, song_pipeline, validate_bank
from prepare_canonical_pipeline_iphone import FUNCTIONS, POLICY, SCHEDULE, TRANSFER
from prepare_encoder_pipeline_iphone import choose_tracks
from train_section_context import LABELS

PROTOCOL = 'muq-pipeline-phone-preparation-v1'
PREFLIGHT = 'muq-functional-v1'
ROLES = ['muq']
BUNDLE = 'com.danendra.MuQOnlyPipelineProbe'
RESOURCES = {'log_mel', 'muq_conv', 'muq', 'head', 'context'}
TOLERANCE = 1e-4


def validate_quality(q, selection, fingerprint):
    if (q.get('protocol') != QUALITY_PROTOCOL or q.get('split') != 'val' or not q.get('full_validation')
            or not q.get('passed') or not q.get('reference_reproduces_verification') or q.get('musicfm_loaded') is not False
            or q.get('test_used') is not False or q.get('selection_identity') != selection.get('identity')
            or selection.get('identity') != fingerprint({k: v for k, v in selection.items() if k != 'identity'})):
        raise ValueError('Use the completed, passed MuQ encoder validation and its exact selection.')


def reproduce(reference, actual, tolerance=TOLERANCE):
    """Mac self-consistency: the fresh reference must match the validated per-track result."""
    key = lambda d: (round(float(d['start']), 6), round(float(d['end']), 6))
    expected = {key(d): d for d in reference}
    if len(expected) != len(reference) or {key(d) for d in actual} != set(expected) or len(actual) != len(reference):
        raise ValueError('Section inventory differs from the validated result.')
    changed = [key(d) for d in actual if d['prediction'] != expected[key(d)]['prediction']]
    error = max(abs(d['probabilities'][k] - expected[key(d)]['probabilities'][k]) for d in actual for k in LABELS)
    return {'passed': not changed and error <= tolerance, 'changed_sections': changed, 'max_final_probability_error': error}


def bundle_targets(details, arrays):
    if len(details) != len(arrays['indices']) or arrays['priors'].shape != (len(details), 8):
        raise ValueError('Target/prior inventory differs.')
    return [dict(index=int(index), label=d['label_manual'], prediction=d['prediction'],
                 probabilities=[d['probabilities'][label] for label in LABELS], base_probabilities=[float(v) for v in prior])
            for index, d, prior in zip(arrays['indices'], details, arrays['priors'])]


def manifest_data(*, quality_sha, bank, bank_report_sha, export, rotary, files, tracks, probe_source_sha):
    if set(files) != RESOURCES:
        raise ValueError('Unexpected model resources: ' + ', '.join(sorted(files)))
    return dict(version='encoder-pipeline-phone-v2', quality_sha256=quality_sha, static_graph_policy='split-before-conv-v2',
                frontend_route_probe=False, frontend_gpu_policy='disabled', graph_repair_sha256=bank['evidence']['repair_sha256'],
                mac_preflight_policy=PREFLIGHT, encoder_roles=ROLES,
                canonical_encoder_policy=POLICY, canonical_encoder_functions={'muq': FUNCTIONS},
                canonical_encoder_reports={'muq': bank_report_sha}, compiled_encoder_bytes=bank['compiled_bytes'],
                canonical_encoder_schedule=SCHEDULE, tensor_transfer_policy=TRANSFER, probe_source_sha256=probe_source_sha,
                graph_states={'muq': bank['evidence']['source_state_sha256']},
                base_checkpoint=export['head_checkpoint_sha256'], context_checkpoint=export['context_checkpoint_sha256'],
                rotary=rotary, files=files, tracks=tracks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('songstruct-root', 'songmap-root', 'quality-report', 'export-dir', 'muq-report', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--track-id', action='append')
    a = p.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Prepare on the Mac with Xcode and the SongFormer environment.')
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.expanduser().resolve())
    out, songmap, source = a.output_dir, a.songmap_root, Path(__file__).resolve().parent
    if out.exists() or (songmap / 'outputs.local') not in out.parents:
        raise ValueError('Use a fresh output directory under outputs.local.')
    sys.path.insert(0, str(a.songstruct_root))
    from finetuning.common import config, fingerprint, read, sha_file, write
    from finetuning.engine import manifest
    from deployment.iphone_project import project_text
    import coremltools as ct
    import librosa
    cfg = config(a.songstruct_root / 'finetuning/config.json'); data, provenance = manifest(cfg)
    q = read(a.quality_report); selection = read(a.quality_report.parent / 'selection.json')
    validate_quality(q, selection, fingerprint)
    if selection['plan_id'] != data['plan_id'] or selection['provenance_id'] != provenance:
        raise ValueError('Dataset identity changed.')
    if selection['librosa'] != librosa.__version__ or selection['coremltools'] != ct.__version__:
        raise ValueError('Decoder/Core ML runtime differs from the validated run.')
    export = read(a.export_dir / 'export_report.json')
    head_package, context_package = a.export_dir / HEAD_PALETTE6, a.export_dir / CONTEXT
    bank = read(a.muq_report); compiled_bank = a.muq_report.parent / 'Compiled' / 'muq_Canonical.mlmodelc'
    validate_bank(bank, compiled_bank)
    app_root = songmap / 'SongMap/Resources/MLModels/SongFormer'; app = read(app_root / 'songformer_coreml_manifest.json')
    frontends = {role: app_root / app['models'][role] for role in ('log_mel', 'muq_conv')}
    expected = {'head_digest': package_digest(head_package), 'context_digest': package_digest(context_package),
                'muq_bank_compiled_digest': bank['compiled_digest'], 'muq_bank_compiled_bytes': bank['compiled_bytes'],
                'frontends': {k: package_digest(v) for k, v in frontends.items()}}
    if any(selection[k] != v for k, v in expected.items()):
        raise ValueError('Head, context, MuQ bank or frontends differ from the validated selection.')
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    chosen = choose_tracks(rows, a.track_id)
    math_source = songmap / 'SongMap/Data/SongAnalysis/Structure/SongFormerContextMath.swift'
    bridge_source = songmap / 'SongMap/Data/SongAnalysis/Structure/SongFormerTensorBridge.swift'
    copy_source = source / 'encoder_pipeline_iphone/ProbeOwnedTensorCopy.swift'
    out.mkdir(parents=True); assets = out / 'Assets'; assets.mkdir()
    shutil.copyfile(source / 'encoder_pipeline_iphone/BenchmarkApp.swift', out / 'BenchmarkApp.swift')
    (out / 'BenchmarkRunner.swift').write_text('\n'.join(path.read_text() for path in
        (source / 'encoder_pipeline_iphone/BenchmarkRunner.swift', copy_source, math_source, bridge_source)))
    # Native checks of the exact Swift sources before any large asset is staged.
    test_source = out / 'OwnedTensorNativeTests.swift'
    test_source.write_text(copy_source.read_text() + '\n' + bridge_source.read_text() +
        '\n@main struct OwnedTensorNativeTests { static func main() throws { try ProbeOwnedTensorCopy.selfTest(); print("Native tensor ownership tests passed") } }\n')
    mac_sdk = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()
    subprocess.run(['xcrun', 'swiftc', '-parse-as-library', '-sdk', mac_sdk, str(test_source), '-o', str(out / 'OwnedTensorNativeTests')], check=True)
    subprocess.run([str(out / 'OwnedTensorNativeTests')], check=True)
    subprocess.run(['bash', str(source / 'test_songformer_tensor_bridge.sh')], check=True)
    subprocess.run(['bash', str(source / 'test_section_context.sh')], check=True, env={**os.environ, 'SECTION_CONTEXT_PYTHON': sys.executable})
    sdk = subprocess.check_output(['xcrun', '--sdk', 'iphoneos', '--show-sdk-path'], text=True).strip()
    subprocess.run(['xcrun', 'swiftc', '-typecheck', '-parse-as-library', '-target', 'arm64-apple-ios18.0', '-sdk', sdk,
                    str(out / 'BenchmarkApp.swift'), str(out / 'BenchmarkRunner.swift')], check=True)
    for role, package in {**frontends, 'head': head_package, 'context': context_package}.items():
        print('Compiling unchanged model:', role, flush=True)
        work = out / 'CompileWork' / role; work.mkdir(parents=True)
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(package), str(work)], check=True)
        shutil.move(str(work / (package.stem + '.mlmodelc')), assets / (role + '.mlmodelc'))
    shutil.copytree(compiled_bank, assets / 'muq.mlmodelc')
    if package_digest(assets / 'muq.mlmodelc') != bank['compiled_digest']:
        raise ValueError('Copied canonical MuQ bank changed.')
    # Fresh Mac Core ML reference for the bundled songs (same packages as the validated run).
    head = CoreMLHead(ct.models.MLModel(str(head_package), compute_units=ct.ComputeUnit.CPU_ONLY))
    native_context = ct.models.MLModel(str(context_package), compute_units=ct.ComputeUnit.CPU_ONLY)
    encoder = MuQEncoder(ct, frontends, compiled_bank, app['rotary_inv_freq']['muq'])
    tracks, checks = [], []
    for number, row in enumerate(chosen):
        print('MuQ-only Mac reference:', row['track_id'], flush=True)
        if row['annotation_sha256'] != fingerprint(row['sections']) or sha_file(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Audio/annotation changed.')
        validated = read(a.quality_report.parent / 'per-track' / (row['id'] + '.json'))['candidate']
        waveform, _ = librosa.load(row['audio_path'], sr=24000, mono=True)
        chunks = [(values, offset) for offset, values in muq_chunks(waveform, encoder.encode, cfg['features'])]
        arrays = {}
        details, _, _ = song_pipeline(row, chunks, head, {'source': read(row['source_json']), 'native': native_context}, arrays=arrays)
        encoder.release()
        check = {'track_id': row['track_id'], **reproduce(validated, details)}
        checks.append(check)
        write(out / 'mac-preflight.json', {'policy': PREFLIGHT, 'checks': checks, 'passed': all(c['passed'] for c in checks)})
        if not check['passed']:
            raise ValueError(f'Mac reference does not reproduce the validated MuQ result; see {out / "mac-preflight.json"}')
        pcm, means = f'track{number}_pcm', f'track{number}_means'
        np.asarray(waveform, dtype='<f4').tofile(assets / (pcm + '.f32'))
        np.asarray(arrays['audio'], dtype='<f4').tofile(assets / (means + '.f32'))
        tracks.append(dict(id=row['track_id'], title=Path(row['track_id']).stem, duration=row['duration'], samples=len(waveform),
                           pcm=pcm, pcm_sha256=sha_file(assets / (pcm + '.f32')), means=means, means_sha256=sha_file(assets / (means + '.f32')),
                           intervals=arrays['intervals'], targets=bundle_targets(details, arrays)))
    files = {path.stem: file_hashes(path) for path in assets.glob('*.mlmodelc')}
    probe_source = fingerprint(dict(runner=sha_file(out / 'BenchmarkRunner.swift'), view=sha_file(out / 'BenchmarkApp.swift')))
    manifest_values = manifest_data(quality_sha=sha_file(a.quality_report), bank=bank, bank_report_sha=sha_file(a.muq_report),
                                    export=export, rotary=app['rotary_inv_freq'], files=files, tracks=tracks, probe_source_sha=probe_source)
    write(assets / 'benchmark_manifest.json', manifest_values)
    project = out / 'Benchmark.xcodeproj'; project.mkdir()
    (project / 'project.pbxproj').write_text(project_text(BUNDLE, resource_names=[p.name for p in assets.iterdir()]).replace(
        'GENERATE_INFOPLIST_FILE = YES;', 'GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_CFBundleDisplayName = "MuQ Only";'))
    model_bytes = {path.stem: sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) for path in assets.glob('*.mlmodelc')}
    write(out / 'preparation.json', dict(
        protocol=PROTOCOL, quality_sha256=sha_file(a.quality_report), manifest_sha256=sha_file(assets / 'benchmark_manifest.json'),
        preparation_script_sha256=sha_file(Path(__file__)), runner_sha256=sha_file(out / 'BenchmarkRunner.swift'),
        app_math_sha256=sha_file(math_source), app_bridge_sha256=sha_file(bridge_source), bundle_identifier=BUNDLE,
        native_swift_checks_passed=True, native_typecheck_passed=True, mac_reference_passed=True, checks=checks,
        encoder_roles=ROLES, mac_preflight_policy=PREFLIGHT, compiled_model_bytes=model_bytes,
        temporary_model_bytes=sum(model_bytes.values()), compiled_encoder_bytes=bank['compiled_bytes'],
        musicfm_bundled=False, ready_for_app_integration=False,
        notes=['Compare against the canonical two-encoder probe on the same two songs, same mode order (CPU first, then CPU+GPU).',
               'Sampled process footprint is not true peak memory; timings start from bundled PCM and exclude MP3 decoding, MusicUnderstanding, beats, chords and lyrics.']))
    print('Model MiB:', {k: round(v / 2**20, 1) for k, v in model_bytes.items()}, flush=True)
    print('Prepared:', project, flush=True)


if __name__ == '__main__':
    main()
