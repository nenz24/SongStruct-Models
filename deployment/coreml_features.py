"""Core ML encoder feature extraction into a separate, resumable diagnostic cache."""
from pathlib import Path
import time
import platform
import numpy as np
from deployment.audit import package_inventory
from deployment.verify_encoders import rotary
from finetuning.common import fingerprint, sha_file


ROLES = ('log_mel','musicfm_conv','musicfm','muq_conv','muq')


def native_input(values):
    result = np.array(values, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(result).all():
        raise ValueError('NaN/Inf in Core ML encoder input.')
    return result


def extract_chunks(waveform, encode_pair, settings):
    """Same stream/window ordering as frozen PyTorch feature extraction."""
    rate = settings['sample_rate']
    def streams(audio, seconds):
        parts = [encode_pair(audio[i:i+seconds*rate]) for i in range(0,len(audio),seconds*rate)
                 if len(audio[i:i+seconds*rate]) > 2048]
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
    def __init__(self, songmap_root, settings):
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
        packages = {}
        for role in ROLES:
            path = self.root / self.app['models'][role]
            packages[role] = fingerprint([(str(p.relative_to(path)),sha_file(p))
                                         for p in sorted(path.rglob('*')) if p.is_file()])
        self.provenance = dict(packages=packages, app_manifest_sha256=sha_file(self.root/'songformer_coreml_manifest.json'),
                               coremltools=ct.__version__, librosa=librosa.__version__,
                               compute_units='CPU_ONLY', platform=platform.platform(), feature_settings=settings)
        self.id = fingerprint(self.provenance)
        self.models = {}
        self.timings = {role: {'calls':0, 'prediction_seconds':0.0} for role in ROLES}

    def load(self):
        for role in ROLES:
            print('Loading Core ML encoder component:',role,flush=True)
            self.models[role] = self.ct.models.MLModel(str(self.root/self.app['models'][role]),
                compute_units=self.ct.ComputeUnit.CPU_ONLY)

    def predict(self, role, values):
        t = time.perf_counter()
        output = self.models[role].predict({name:native_input(array) for name,array in values.items()})
        self.timings[role]['calls'] += 1
        self.timings[role]['prediction_seconds'] += time.perf_counter()-t
        arrays = [np.asarray(output[key]) for key in sorted(output) if isinstance(output[key],np.ndarray)]
        if len(arrays) != 1:
            raise ValueError(f'Expected one array from {role}; found {len(arrays)}.')
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
