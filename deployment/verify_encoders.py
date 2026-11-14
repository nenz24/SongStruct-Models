"""Diagnose existing Core ML encoders against corresponding frozen cache stream prefixes."""
import argparse
import gc
from pathlib import Path
import platform
import numpy as np
from deployment.audit import package_inventory
from finetuning.common import config, iter_features, write
from finetuning.engine import manifest


def rotary(frames, frequencies):
    inv = np.asarray(frequencies, dtype=np.float32)
    if inv.shape != (32,):
        raise ValueError('Expected 32 rotary inverse frequencies.')
    angles = np.arange(frames, dtype=np.float32)[:, None] * inv[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    return np.stack([np.cos(angles), np.sin(angles)]).reshape(2, frames, 1, 1, 64).astype(np.float32)


def compare_embeddings(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {'numerical_parity_passed': False, 'reason': 'shape mismatch or non-finite values',
                'reference_shape': list(reference.shape), 'candidate_shape': list(candidate.shape)}
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    cosine = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return {'numerical_parity_passed': bool(np.allclose(reference, candidate, atol=.03, rtol=.02)),
            'max_abs_error': float(delta.max()), 'mean_abs_error': float(delta.mean()),
            'mean_frame_cosine_similarity': float(cosine.mean()),
            'atol': .03, 'rtol': .02}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songmap-root', type=Path, required=True)
    parser.add_argument('--config')
    parser.add_argument('--seconds', type=int, choices=[30,120], default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Run Core ML inference on macOS.')
    import coremltools as ct
    import librosa
    cfg = config(args.config)
    if cfg['features']['long_window_seconds'] != 120:
        raise ValueError('This encoder check requires the app-compatible 30/120 profile.')
    root = args.songmap_root.expanduser().resolve() / 'SongMap/Resources/MLModels/SongFormer'
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Choose a new report path.')
    app, inventory = package_inventory(root)
    if any(not a['complete_for_size_measurement'] for a in inventory if a['role'] != 'head'):
        raise ValueError('Missing encoder assets or Git LFS pointers; fetch actual local weights.')
    if app['sample_rate'] != 24000 or app['encoder_layer'] != 10 or app['encoder_window_seconds'] != 120:
        raise ValueError('App encoder profile differs from cache configuration.')
    if app['stream_order'] != ['musicfm_30s','muq_30s','musicfm_long','muq_long']:
        raise ValueError('App stream order differs from cache configuration.')
    data, provenance = manifest(cfg)
    row = next(r for r in data['records'] if r['approved'] and r['split'] == 'val')
    features, offset, signature = next(iter_features(cfg, row))
    if offset != 0 or signature != provenance:
        raise ValueError('Expected first cache chunk with matching provenance.')
    # Load full audio before slicing: this matches cache resampling at window edges.
    waveform, _ = librosa.load(row['audio_path'], sr=24000, mono=True)
    waveform = waveform[:args.seconds * 24000]
    if len(waveform) <= 2048:
        raise ValueError('Audio too short.')
    def load(role):
        return ct.models.MLModel(str(root / app['models'][role]), compute_units=ct.ComputeUnit.CPU_ONLY)
    def first_input(model):
        return sorted(f.name for f in model.get_spec().description.input)[0]
    def first_array(output):
        return np.asarray(output[sorted(output)[0]])
    model = load('log_mel')
    padded = np.pad(waveform, (1024,1024), mode='reflect')[None].astype(np.float32)
    mel = first_array(model.predict({first_input(model): padded}))
    del model
    results = []
    for name, conv_role, encoder_role, short_column, long_column in [
        ('musicfm','musicfm_conv','musicfm',0,2048), ('muq','muq_conv','muq',1024,3072)]:
        print('Checking encoder:', name, '| seconds:', len(waveform)/24000, flush=True)
        conv = load(conv_role)
        convolved = first_array(conv.predict({first_input(conv): mel}))
        del conv
        if convolved.ndim != 3 or convolved.shape[0] != 1 or convolved.shape[-1] != 1024:
            raise ValueError(f'Unexpected convolution shape: {convolved.shape}')
        encoder = load(encoder_role)
        encoded = first_array(encoder.predict({'conv_in': convolved,
            'rotary': rotary(convolved.shape[1], app['rotary_inv_freq'][name])}))
        del encoder
        gc.collect()
        if encoded.ndim != 3 or encoded.shape[0] != 1 or encoded.shape[-1] != 1024:
            raise ValueError(f'Unexpected embedding shape: {encoded.shape}')
        column = short_column if args.seconds == 30 else long_column
        expected_frames = min(int(len(waveform)/24000*25), len(features))
        reference = features[:expected_frames, column:column+1024]
        count = min(len(reference), encoded.shape[1])
        if count < 1 or abs(encoded.shape[1] - len(reference)) > 4:
            raise ValueError(f'Encoder/cache frame counts diverge: {encoded.shape[1]} vs {len(reference)}')
        result = compare_embeddings(reference[:count], encoded[0,:count])
        result.update(encoder=name, frames_compared=count, cache_frames=len(reference),
                      coreml_frames=encoded.shape[1], cache_stream_column=column)
        results.append(result)
    write(output, {'track_id': row['track_id'], 'audio_seconds': len(waveform)/24000,
        'profile_seconds': args.seconds, 'plan_id': data['plan_id'], 'provenance_id': provenance,
        'results': results, 'ready_for_app_integration': False,
        'scope': 'Existing app encoders vs corresponding first short/long cached window only; no compression or full validation.',
        'note': 'Numerical mismatch is diagnostic, not permission to replace encoder weights. Includes app log-mel, reflect padding and rotary inputs.'})
    print('Report:', output)
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
