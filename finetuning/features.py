"""Cache frozen MuQ/MusicFM features independently of annotation completion."""
import argparse
import os
from pathlib import Path
import tempfile
from .common import config, device, feature_path, fingerprint, read, revision, sha_file, upstream, write


def extract_chunks(wav, muq, musicfm, target, settings):
    import torch
    sample_rate = settings['sample_rate']

    def encode(chunk):
        audio = torch.as_tensor(chunk, device=target).unsqueeze(0)
        muq_values = muq(audio, output_hidden_states=True)['hidden_states'][10]
        fm_values = musicfm.get_predictions(audio)[1][10]
        return fm_values.detach().cpu(), muq_values.detach().cpu()

    def streams(chunk, seconds):
        parts = [encode(chunk[i:i + seconds * sample_rate])
                 for i in range(0, len(chunk), seconds * sample_rate)
                 if len(chunk[i:i + seconds * sample_rate]) > 2048]
        if not parts:
            raise ValueError('Tidak ada encoder frame.')
        return [torch.cat([p[i] for p in parts], dim=1) for i in (0, 1)]

    with torch.inference_mode():
        step = settings['head_window_seconds'] * sample_rate
        for offset in range(0, len(wav), step):
            chunk = wav[offset:offset + step]
            if len(chunk) <= 2048:
                break
            short = streams(chunk, 30)
            long = streams(chunk, settings['long_window_seconds'])
            parts = short + long
            lengths = [p.shape[1] for p in parts]
            if max(lengths) - min(lengths) > 4:
                raise ValueError(f'Encoder streams tidak sejajar: {lengths}')
            count = min(lengths) // 3 * 3
            if count < 3:
                raise ValueError('Encoder frame terlalu pendek.')
            fused = torch.cat([p[:, :count] for p in parts], dim=-1)[0].float().numpy()
            yield offset / sample_rate, fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    cfg = config(args.config)
    records = read(Path(cfg['work_dir']) / 'manifest.json')['records']
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError('Inventory kosong; jalankan prepare dahulu.')
    target = device(args.device)
    folder = upstream(cfg)
    import librosa
    import numpy as np
    from huggingface_hub import snapshot_download
    from muq import MuQ
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    provenance_path = Path(cfg['work_dir']) / 'encoder_provenance.json'
    provenance = read(provenance_path) if provenance_path.exists() else None
    desired_revision = provenance['muq_revision'] if provenance else 'main'
    kwargs = dict(repo_id=cfg['features']['muq_repo'], revision=desired_revision)
    try:
        snapshot = Path(snapshot_download(**kwargs, local_files_only=True))
    except Exception:
        snapshot = Path(snapshot_download(**kwargs))
    musicfm_weights = folder / 'ckpts/MusicFM/pretrained_msd.pt'
    stats = folder / 'ckpts/MusicFM/msd_stats.json'
    resolved = {'muq_repo': cfg['features']['muq_repo'], 'muq_revision': snapshot.name,
                'musicfm_sha256': sha_file(musicfm_weights), 'stats_sha256': sha_file(stats),
                'upstream_revision': revision(cfg)}
    resolved['id'] = fingerprint(resolved)
    if provenance and resolved != provenance:
        raise ValueError('Encoder/source provenance changed. Use a new work_dir to avoid mixing caches.')
    if not provenance:
        write(provenance_path, resolved)
    pending = []
    for record in records:
        cache = feature_path(cfg, record) / 'metadata.json'
        if cache.exists():
            meta = read(cache)
            if (meta['features'] != cfg['features'] or meta['audio_sha256'] != record['audio_sha256']
                    or meta['provenance_id'] != resolved['id']):
                raise ValueError(f"Cache incompatible: {record['track_id']}")
            continue
        pending.append(record)
    if not pending:
        print('Semua cache terverifikasi. Tidak menjalankan encoder ulang.')
        return
    import shutil
    required = sum(float(r['duration']) for r in pending) * 25 * 4096 * 4
    free = shutil.disk_usage(cfg['work_dir']).free
    if free < required + 1024 ** 3:
        raise ValueError(f'Cache membutuhkan sekitar {required / 1024**3:.1f} GiB + ruang cadangan; hanya {free / 1024**3:.1f} GiB tersedia. Pindahkan work_dir atau gunakan --limit.')
    print(f'Loading frozen encoders on {target}; {len(pending)} audio pending.', flush=True)
    muq = MuQ.from_pretrained(str(snapshot)).to(target).eval()
    musicfm = MusicFM25Hz(is_flash=False, stat_path=str(stats), model_path=str(musicfm_weights)).to(target).eval()
    for index, record in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {record['track_id']}", flush=True)
        if sha_file(record['audio_path']) != record['audio_sha256']:
            raise ValueError('Audio changed since prepare; rebuild manifest first.')
        wav, _ = librosa.load(record['audio_path'], sr=24000, mono=True)
        root = feature_path(cfg, record)
        root.mkdir(parents=True, exist_ok=True)
        chunks = []
        for offset, values in extract_chunks(wav, muq, musicfm, target, cfg['features']):
            name = f'chunk_{int(offset):06d}.npy'
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
                temp = Path(handle.name)
                np.save(handle, values, allow_pickle=False)
            temp.replace(root / name)
            chunks.append({'file': name, 'offset': offset, 'frames': values.shape[0]})
            print(f"  {offset:.0f}s: {values.shape[0]} frames", flush=True)
        if not chunks:
            raise ValueError('Audio tidak menghasilkan fitur.')
        write(root / 'metadata.json', {'audio_sha256': record['audio_sha256'], 'features': cfg['features'],
                                      'provenance_id': resolved['id'], 'chunks': chunks})
    print('Cache selesai. Encoder tidak perlu dilatih atau dijalankan setiap epoch.')


if __name__ == '__main__':
    main()
