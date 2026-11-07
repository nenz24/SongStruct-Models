import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
LABELS = ('intro', 'verse', 'chorus', 'bridge', 'inst', 'outro', 'silence', 'pre-chorus')
CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 26)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    tmp.replace(path)


def config(path=None):
    cfg = read(path or BASE / 'finetuning/config.json')
    for key in ('upstream_dir', 'labeled_dir', 'reviewed_dir', 'work_dir', 'checkpoint'):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((BASE / value).resolve() if not value.is_absolute() else value.resolve())
    f = cfg['features']
    expected = {'sample_rate': 24000, 'encoder_layer': 10, 'short_window_seconds': 30,
                'head_window_seconds': 420, 'encoder_fps': 25, 'stride': 3, 'storage_dtype': 'float32'}
    if any(f.get(k) != v for k, v in expected.items()) or f['long_window_seconds'] not in (120, 420):
        raise ValueError('Feature config must use the supported SongMapApp 30/120 or upstream 30/420 profile, float32.')
    return cfg


def device(name='auto'):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA tidak tersedia.')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS tidak tersedia.')
    return torch.device(name)


def upstream(cfg):
    root = Path(cfg['upstream_dir'])
    folder = root / 'src/SongFormer'
    if not (folder / 'models/SongFormer.py').is_file():
        raise ValueError('Source SongFormer belum ada. Ikuti SONGFORMER_SETUP.md.')
    sys.path[:0] = [str(folder), str(root / 'src/third_party')]
    os.chdir(folder)
    import numpy as np
    import scipy
    scipy.inf = np.inf
    return folder


def revision(cfg):
    return subprocess.check_output(['git', '-C', cfg['upstream_dir'], 'rev-parse', 'HEAD'], text=True).strip()


def load_head(cfg, target, checkpoint=None):
    upstream(cfg)
    import torch
    from ema_pytorch import EMA
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from models.SongFormer import Model
    hp = OmegaConf.load('configs/SongFormer.yaml')
    model = Model(hp)
    checkpoint = Path(checkpoint or cfg['checkpoint']).resolve()
    if checkpoint.suffix == '.safetensors':
        state = load_file(str(checkpoint), device='cpu')
        if any(k.startswith('ema_model.') for k in state):
            ema = EMA(model, include_online_model=False)
            ema.load_state_dict(state, strict=True)
            state = ema.ema_model.state_dict()
    else:
        # Only use trusted locally produced checkpoints here.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state, strict=True)
    return model.to(target), hp


def feature_path(cfg, record):
    return Path(cfg['work_dir']) / 'features' / record['id']


def iter_features(cfg, record):
    import numpy as np
    root = feature_path(cfg, record)
    meta = read(root / 'metadata.json')
    if meta['audio_sha256'] != record['audio_sha256'] or meta['features'] != cfg['features']:
        raise ValueError(f"Cache usang untuk {record['track_id']}; gunakan work_dir/cache baru.")
    for item in meta['chunks']:
        values = np.load(root / item['file'], mmap_mode='r', allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 4096 or values.shape[0] % 3:
            raise ValueError('Feature shape tidak valid.')
        yield values, item['offset'], meta['provenance_id']
