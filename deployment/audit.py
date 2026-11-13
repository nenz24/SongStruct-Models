"""Inventory SongMap Core ML assets; optionally benchmark PyTorch head on cached features."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time


def package_inventory(root):
    manifest_path = root / 'songformer_coreml_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    result = []
    for role in ('log_mel', 'muq_conv', 'muq', 'musicfm_conv', 'musicfm', 'head'):
        name = manifest['models'][role]
        package = (root / name).resolve()
        package.relative_to(root.resolve())
        files = sorted(p for p in package.rglob('*') if p.is_file()) if package.is_dir() else []
        pointers = []
        for path in files:
            if path.stat().st_size <= 1024:
                with path.open('rb') as handle:
                    if handle.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                        pointers.append(str(path.relative_to(root)))
        result.append({'role': role, 'name': name, 'exists': package.is_dir(),
                       'bytes_on_disk': sum(p.stat().st_size for p in files),
                       'git_lfs_pointers': pointers,
                       'complete_for_size_measurement': bool(files) and not pointers})
    return manifest, result


def timed_calls(call, sync, repeats):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    call()  # warmup excluded
    sync()
    elapsed = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        call()
        sync()
        elapsed.append(time.perf_counter() - start)
    return {'repeats': repeats, 'seconds': elapsed, 'median_seconds': statistics.median(elapsed)}


def benchmark(cfg, checkpoint, device_name, seconds, repeats):
    import torch
    import numpy as np
    from finetuning.common import device, iter_features, load_head
    from finetuning.engine import logits, manifest
    data, provenance = manifest(cfg)
    rows = [r for r in data['records'] if r['approved'] and r['split'] == 'val']
    if not rows:
        raise ValueError('No eligible validation track.')
    if seconds <= 0 or seconds > 420:
        raise ValueError('seconds must be in (0, 420].')
    frames = int(seconds * 25 / 3) * 3
    if frames < 3:
        raise ValueError('At least 0.12 seconds required.')
    values, _, signature = next(iter_features(cfg, rows[0]))
    if signature != provenance:
        raise ValueError('Cache provenance mismatch.')
    values = np.array(values[:frames], copy=True)
    target = device(device_name)
    sync = (torch.mps.synchronize if target.type == 'mps' else
            torch.cuda.synchronize if target.type == 'cuda' else lambda: None)
    sync()
    start = time.perf_counter()
    model, _ = load_head(cfg, target, checkpoint)
    model.eval()
    sync()
    load_seconds = time.perf_counter() - start
    # Input copying/host-to-device and head execution included; encoders excluded.
    with torch.inference_mode():
        timing = timed_calls(lambda: logits(model, values, target), sync, repeats)
    params = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    report = {'scope': 'PyTorch cached-feature head only on this computer; not Core ML or iPhone',
              'device': str(target), 'track_id': rows[0]['track_id'],
              'audio_seconds_represented': len(values) / 25, 'input_frames': len(values),
              'load_seconds': load_seconds, 'timing': timing,
              'real_time_factor_head_only': timing['median_seconds'] / (len(values) / 25),
              'parameters': params, 'parameter_bytes_current_dtype': parameter_bytes,
              'ideal_6bit_parameter_payload_bytes': (params * 6 + 7) // 8,
              'compression_note': 'Ideal payload only, excludes LUTs, scales, uncompressed ops, activations and encoders.',
              'plan_id': data['plan_id'], 'provenance_id': provenance}
    if target.type == 'mps':
        report['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        report['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        report['memory_note'] = 'Current allocations after benchmark, not peak memory.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songmap-root', type=Path, help='Local SongMapApp checkout (optional)')
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs.local/finetuning/runs/worship-v3/best.pt'))
    parser.add_argument('--benchmark', action='store_true', help='Load trusted local checkpoint and time cached-feature head')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new report name.')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    report = {'platform': platform.platform(), 'python': platform.python_version(),
              'checkpoint': str(checkpoint), 'checkpoint_file_bytes': checkpoint.stat().st_size,
              'iphone_measurements': None,
              'limitations': ['No iPhone latency or peak-memory measurement.',
                             'No quantization or Core ML conversion performed.',
                             'Swift frame argmax is not the evaluated Apple-section probability pooling.']}
    if args.songmap_root:
        root = args.songmap_root.expanduser().resolve() / 'SongMap/Resources/MLModels/SongFormer'
        manifest, assets = package_inventory(root)
        report.update(coreml_assets=assets,
                      asset_size_complete=all(a['complete_for_size_measurement'] for a in assets),
                      total_asset_bytes=sum(a['bytes_on_disk'] for a in assets),
                      app_profile={k: manifest.get(k) for k in ('sample_rate', 'encoder_layer', 'encoder_window_seconds',
                                                               'head_max_seconds', 'stream_order')})
    if args.benchmark:
        from finetuning.common import config
        cfg = config(args.config)
        # Resolve before load_head changes working directory to upstream source.
        report['head_benchmark'] = benchmark(cfg, checkpoint, args.device, args.seconds, args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Report:', output)
    print('Checkpoint file: %.1f MiB' % (report['checkpoint_file_bytes'] / 1024**2))
    if 'coreml_assets' in report:
        for asset in report['coreml_assets']:
            print('%s: %.1f MiB; size complete=%s' % (asset['role'], asset['bytes_on_disk']/1024**2,
                                                      asset['complete_for_size_measurement']))
    if 'head_benchmark' in report:
        b = report['head_benchmark']
        print('PyTorch head only: %.3fs per %.2fs features' % (b['timing']['median_seconds'], b['audio_seconds_represented']))
    print('This is not an iPhone or end-to-end audio benchmark.')


if __name__ == '__main__':
    main()
