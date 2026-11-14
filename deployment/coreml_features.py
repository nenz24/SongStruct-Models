"""Core ML encoder feature extraction into a separate, resumable diagnostic cache."""
from pathlib import Path
import time
import platform
import numpy as np
from deployment.audit import package_inventory
from deployment.verify_encoders import rotary
from finetuning.common import fingerprint, read, sha_file


ROLES = ('log_mel','musicfm_conv','musicfm','muq_conv','muq')


def validate_musicfm_proof(proof):
    if proof.get('compute_precision') != 'float32':
        raise ValueError('MusicFM candidate must retain FLOAT32 compute export.')
    scheme = proof.get('compression_scheme')
    if scheme:
        if (scheme not in ('int8','palette6') or not proof.get('source_exact_window_passed')
                or not proof.get('window_finite_passed') or not proof.get('source_full_validation')):
            raise ValueError('Compressed MusicFM candidate needs verified source and finite-window checks.')
    elif not proof.get('exact_window_passed'):
        raise ValueError('MusicFM FP32 export must pass exact-window checks first.')
    if not proof.get('musicfm_weights_sha256') or not proof.get('musicfm_stats_sha256'):
        raise ValueError('MusicFM source weights/stats metadata required.')


class EncoderStageError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(f"Core ML encoder failure at {details.get('component')}: {details.get('reason')}")


class EncoderWindowError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(f"Encoder {details.get('component', 'unknown')} failed at "
                         f"{details['start_sample']/details['sample_rate']:.6f}s for "
                         f"{(details['end_sample']-details['start_sample'])/details['sample_rate']:.6f}s: {details.get('reason')}")


def array_summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'shape': list(values.shape), 'dtype': str(values.dtype),
            'nan_count': int(np.isnan(values).sum()), 'inf_count': int(np.isinf(values).sum()),
            'finite_min': float(finite.min()) if finite.size else None,
            'finite_max': float(finite.max()) if finite.size else None}


def native_input(values):
    result = np.array(values, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(result).all():
        raise ValueError('NaN/Inf in Core ML encoder input.')
    return result


def extract_chunks(waveform, encode_pair, settings):
    """Same stream/window ordering as frozen PyTorch feature extraction."""
    rate = settings['sample_rate']
    def streams(audio, seconds):
        parts = []
        for i in range(0,len(audio),seconds*rate):
            window = audio[i:i+seconds*rate]
            if len(window) <= 2048:
                continue
            try:
                parts.append(encode_pair(window))
            except (EncoderStageError, ValueError) as exc:
                details = dict(getattr(exc,'details',{'reason':str(exc)}))
                details.update(start_sample=offset+i, end_sample=offset+i+len(window),
                               sample_rate=rate, stream_window_seconds=seconds,
                               head_offset_samples=offset, audio_summary=array_summary(window))
                raise EncoderWindowError(details) from exc
        if not parts:
            raise ValueError('No valid encoder windows.')
        return [np.concatenate([p[k] for p in parts], axis=0) for k in (0,1)]
    for offset in range(0,len(waveform),settings['head_window_seconds']*rate):
        chunk = waveform[offset:offset+settings['head_window_seconds']*rate]
        if len(chunk) <= 2048:
            break
        short = streams(chunk,30)
        long = streams(chunk,settings['long_window_seconds'])
        arrays = short + long
        lengths = [len(a) for a in arrays]
        if max(lengths)-min(lengths) > 4:
            raise ValueError(f'Core ML streams misaligned: {lengths}')
        count = min(lengths)//3*3
        if not 3 <= count <= 10500 or any(a.ndim != 2 or a.shape[1] != 1024 for a in arrays):
            raise ValueError(f'Invalid fused feature dimensions: {[a.shape for a in arrays]}')
        features = np.concatenate([a[:count] for a in arrays], axis=-1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('NaN/Inf in Core ML features.')
        yield offset/rate, features


class CoreMLEncoders:
    def __init__(self, songmap_root, settings, musicfm_export_report=None):
        import coremltools as ct
        self.ct = ct
        self.root = Path(songmap_root).expanduser().resolve() / 'SongMap/Resources/MLModels/SongFormer'
        self.app, inventory = package_inventory(self.root)
        if any(not a['complete_for_size_measurement'] for a in inventory if a['role'] in ROLES):
            raise ValueError('Actual encoder packages required; missing assets/LFS pointers found.')
        if (settings['long_window_seconds'] != 120 or self.app['encoder_window_seconds'] != 120
                or self.app['sample_rate'] != settings['sample_rate'] or self.app['encoder_layer'] != 10
                or self.app['head_max_seconds'] != settings['head_window_seconds']
                or self.app['stream_order'] != ['musicfm_30s','muq_30s','musicfm_long','muq_long']):
            raise ValueError('Core ML app and training feature profiles differ.')
        import librosa
        self.package_paths = {role:(self.root/self.app['models'][role]).resolve() for role in ROLES}
        self.musicfm_override = None
        if musicfm_export_report:
            proof_path = Path(musicfm_export_report).expanduser().resolve()
            proof = read(proof_path)
            validate_musicfm_proof(proof)
            package = Path(proof['package']).expanduser().resolve()
            if not package.is_dir():
                raise ValueError('MusicFM FP32 candidate package missing.')
            self.package_paths['musicfm'] = package
            self.musicfm_override = {'report_sha256':sha_file(proof_path),
                'musicfm_weights_sha256':proof['musicfm_weights_sha256'],
                'musicfm_stats_sha256':proof['musicfm_stats_sha256'],
                'rotary_mode':'original_app_values_unchanged'}
            if proof.get('compression_scheme'):
                self.musicfm_override['compression_scheme'] = proof['compression_scheme']
                self.musicfm_override['candidate_validation_pending'] = True
        packages = {}
        for role in ROLES:
            path = self.package_paths[role]
            packages[role] = fingerprint([(str(p.relative_to(path)),sha_file(p))
                                         for p in sorted(path.rglob('*')) if p.is_file()])
        self.provenance = dict(packages=packages, app_manifest_sha256=sha_file(self.root/'songformer_coreml_manifest.json'),
                               coremltools=ct.__version__, librosa=librosa.__version__,
                               compute_units='CPU_ONLY', platform=platform.platform(), feature_settings=settings)
        if self.musicfm_override:
            self.provenance['musicfm_override'] = self.musicfm_override
        self.id = fingerprint(self.provenance)
        self.models = {}
        self.timings = {role: {'calls':0, 'prediction_seconds':0.0} for role in ROLES}

    def load(self, roles=ROLES):
        for role in roles:
            print('Loading Core ML encoder component:',role,flush=True)
            self.models[role] = self.ct.models.MLModel(str(self.package_paths[role]),
                compute_units=self.ct.ComputeUnit.CPU_ONLY)
            if role == 'musicfm' and self.musicfm_override:
                meta = self.models[role].user_defined_metadata
                if (meta.get('precision') != 'fp32' or meta.get('hidden_state_index') != '10'
                        or meta.get('musicfm_weights_sha256') != self.musicfm_override['musicfm_weights_sha256']
                        or meta.get('musicfm_stats_sha256') != self.musicfm_override['musicfm_stats_sha256']
                        or meta.get('compression_scheme') != self.musicfm_override.get('compression_scheme')):
                    raise ValueError('MusicFM candidate metadata does not match export proof.')

    def predict(self, role, values):
        input_stats = {name:array_summary(array) for name,array in values.items()}
        t = time.perf_counter()
        try:
            output = self.models[role].predict({name:native_input(array) for name,array in values.items()})
        except Exception as exc:
            raise EncoderStageError({'component':role, 'reason':'prediction_error',
                'error_type':type(exc).__name__, 'error':str(exc)[:2000], 'inputs':input_stats}) from exc
        self.timings[role]['calls'] += 1
        self.timings[role]['prediction_seconds'] += time.perf_counter()-t
        arrays = [np.asarray(output[key]) for key in sorted(output) if isinstance(output[key],np.ndarray)]
        if len(arrays) != 1:
            raise EncoderStageError({'component':role,'reason':'unexpected_output_count',
                                     'array_count':len(arrays),'inputs':input_stats})
        if not np.isfinite(arrays[0]).all():
            raise EncoderStageError({'component':role,'reason':'non_finite_output',
                                     'inputs':input_stats,'output':array_summary(arrays[0])})
        return arrays[0]

    def first_input(self, role):
        return sorted(f.name for f in self.models[role].get_spec().description.input)[0]

    def encode_pair(self, audio):
        print(f'    encoder window {len(audio)/24000:.2f}s',flush=True)
        padded = np.pad(audio,(1024,1024),mode='reflect')[None]
        mel = self.predict('log_mel',{self.first_input('log_mel'):padded})
        result = []
        for name in ('musicfm','muq'):
            convolved = self.predict(name+'_conv',{self.first_input(name+'_conv'):mel})
            if convolved.ndim != 3 or convolved.shape[0] != 1 or convolved.shape[-1] != 1024:
                raise ValueError(f'Unexpected {name} conv shape: {convolved.shape}')
            values = self.predict(name,{'conv_in':convolved,
                'rotary':rotary(convolved.shape[1],self.app['rotary_inv_freq'][name])})
            if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != 1024:
                raise ValueError(f'Unexpected {name} embedding shape: {values.shape}')
            result.append(values[0])
        return result

    def encode_musicfm(self, audio):
        print(f'    MusicFM-only encoder window {len(audio)/24000:.2f}s',flush=True)
        padded = np.pad(audio,(1024,1024),mode='reflect')[None]
        mel = self.predict('log_mel',{self.first_input('log_mel'):padded})
        convolved = self.predict('musicfm_conv',{self.first_input('musicfm_conv'):mel})
        if convolved.ndim != 3 or convolved.shape[0] != 1 or convolved.shape[-1] != 1024:
            raise ValueError(f'Unexpected MusicFM conv shape: {convolved.shape}')
        values = self.predict('musicfm',{'conv_in':convolved,
            'rotary':rotary(convolved.shape[1],self.app['rotary_inv_freq']['musicfm'])})
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != 1024:
            raise ValueError(f'Unexpected MusicFM embedding shape: {values.shape}')
        return values[0]
