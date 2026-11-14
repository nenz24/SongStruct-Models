"""Prepare a standalone Xcode app with three head candidates and identical real features."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import numpy as np
from deployment.audit import package_inventory
from deployment.compress_coreml import package_bytes, validate_source_report
from deployment.iphone_project import project_text
from finetuning.common import config, iter_features, read, sha_file, write
from finetuning.engine import manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True, help='coreml-compression-*/comparison.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--songmap-root', type=Path, help='Optional local encoder inventory only')
    parser.add_argument('--bundle-id', default='com.danendra.SongFormerBenchmark')
    parser.add_argument('--team-id')
    parser.add_argument('--seconds', type=int, choices=[30,120,420], default=30)
    parser.add_argument('--config')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Prepare on macOS with Xcode and coremltools installed.')
    import coremltools as ct
    output, comparison = args.output_dir.expanduser().resolve(), args.comparison.expanduser().resolve()
    if output.exists():
        raise ValueError('Choose a new output folder.')
    project = project_text(args.bundle_id, args.team_id)
    summary = read(comparison)
    proof_path = Path(summary['fp32_validation_report']).expanduser().resolve()
    proof = read(proof_path)
    source = Path(proof['package']).resolve()
    validate_source_report(proof, source, proof.get('checkpoint_sha256'))
    packages = [('fp32', 'HeadFP32', source)]
    for scheme, resource in [('int8','HeadINT8'),('palette6','HeadPalette6')]:
        report = read(comparison.parent / f'{scheme}-validation.json')
        if not report['full_validation']:
            raise ValueError('Complete candidate validation before iPhone benchmark.')
        for key in ('plan_id', 'provenance_id', 'annotations_id', 'checkpoint_sha256', 'tracks'):
            if report[key] != proof[key]:
                raise ValueError(f'Candidate validation does not match FP32: {key}')
        packages.append((scheme, resource, Path(report['package']).resolve()))
    cfg = config(args.config)
    data, provenance = manifest(cfg)
    if data['plan_id'] != proof['plan_id'] or provenance != proof['provenance_id']:
        raise ValueError('Local cache/split differs from validated models.')
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    values, _, signature = next(iter_features(cfg, rows[0]))
    if signature != provenance:
        raise ValueError('Cache provenance mismatch.')
    frames = min(args.seconds * 25, len(values)) // 3 * 3
    if frames < 3:
        raise ValueError('Fixture too short.')
    values = np.asarray(values[:frames], dtype='<f4')
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'Assets'
    assets.mkdir()
    values.tofile(assets / 'features.f32')
    model = ct.models.MLModel(str(source), compute_units=ct.ComputeUnit.CPU_ONLY)
    reference = np.asarray(model.predict({'input_embeddings': values[None]})['function_logits'], dtype='<f4')
    if reference.shape != (1, frames//3, 128) or not np.isfinite(reference).all():
        raise ValueError('Invalid FP32 reference fixture output.')
    reference.tofile(assets / 'reference_logits.f32')
    del model
    candidate_metadata = []
    for name, resource, package in packages:
        print('Compiling:', name, flush=True)
        meta = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY).user_defined_metadata
        if meta.get('checkpoint_sha256') != proof['checkpoint_sha256']:
            raise ValueError('Package checkpoint metadata mismatch.')
        if name != 'fp32' and meta.get('compression_scheme') != name:
            raise ValueError('Compression metadata mismatch.')
        # Rename a temporary copy so coremlcompiler produces distinct resource names.
        with tempfile.TemporaryDirectory() as temporary:
            renamed = Path(temporary) / f'{resource}.mlpackage'
            shutil.copytree(package, renamed)
            subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(renamed), str(assets)], check=True)
        if not (assets / f'{resource}.mlmodelc').is_dir():
            raise RuntimeError('Compiled model folder was not produced.')
        candidate_metadata.append({'name': name, 'resource': resource, 'package_bytes': package_bytes(package)})
    write(assets / 'benchmark_manifest.json', {'frames': frames, 'feature_width': 4096,
        'reference_label': 'Mac Core ML FP32 on identical validation cache fixture, not ground-truth accuracy',
        'models': candidate_metadata, 'plan_id': proof['plan_id'], 'provenance_id': provenance,
        'track_id': rows[0]['track_id'], 'checkpoint_sha256': proof['checkpoint_sha256'],
        'features_sha256': sha_file(assets / 'features.f32')})
    template = Path(__file__).resolve().parent / 'iphone'
    for name in ('BenchmarkApp.swift','BenchmarkRunner.swift'):
        shutil.copyfile(template / name, output / name)
    project_dir = output / 'Benchmark.xcodeproj'
    project_dir.mkdir()
    (project_dir / 'project.pbxproj').write_text(project)
    if args.songmap_root:
        root = args.songmap_root.expanduser().resolve() / 'SongMap/Resources/MLModels/SongFormer'
        app_manifest, inventory = package_inventory(root)
        write(output / 'encoder_inventory.json', {'assets': inventory, 'manifest': app_manifest,
            'status': 'inventory only; no encoder numerical parity or compression performed'})
    print('Open:', project_dir)
    print('Select your signing team and physical iPhone, then Run. Share the JSON from the benchmark app.')


if __name__ == '__main__':
    main()
