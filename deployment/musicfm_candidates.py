"""Replace only MusicFM streams, reusing validated, unchanged MuQ features."""
from itertools import zip_longest
import numpy as np
from deployment.coreml_features import EncoderStageError, EncoderWindowError, array_summary
from finetuning.common import fingerprint


def validate_muq_provenance(reused, candidate):
    identity=reused['id']
    if fingerprint({k:v for k,v in reused.items() if k!='id'})!=identity:
        raise ValueError('Reused baseline provenance identity invalid.')
    for key in ('coremltools','librosa','platform','compute_units','feature_settings','app_manifest_sha256'):
        if reused[key]!=candidate[key]:
            raise ValueError(f'Reused MuQ environment/profile differs: {key}')
    for role in ('log_mel','musicfm_conv','muq_conv','muq'):
        if reused['packages'][role]!=candidate['packages'][role]:
            raise ValueError(f'Unchanged frontend/MuQ package differs: {role}')
    return identity


def musicfm_chunks(waveform, encode, settings):
    rate=settings['sample_rate']
    def stream(audio, seconds, offset):
        parts=[]
        for start in range(0,len(audio),seconds*rate):
            window=audio[start:start+seconds*rate]
            if len(window)<=2048:
                continue
            try:
                parts.append(encode(window))
            except (EncoderStageError,ValueError) as exc:
                details=dict(getattr(exc,'details',{'reason':str(exc)}))
                details.update(start_sample=offset+start,end_sample=offset+start+len(window),
                               sample_rate=rate,stream_window_seconds=seconds,
                               head_offset_samples=offset,audio_summary=array_summary(window))
                raise EncoderWindowError(details) from exc
        if not parts:
            raise ValueError('No MusicFM windows.')
        return np.concatenate(parts,axis=0)
    step=settings['head_window_seconds']*rate
    for offset in range(0,len(waveform),step):
        audio=waveform[offset:offset+step]
        if len(audio)<=2048:
            break
        short=stream(audio,30,offset)
        long=stream(audio,settings['long_window_seconds'],offset)
        if abs(len(short)-len(long))>4:
            raise ValueError('MusicFM short/long frame counts diverged.')
        yield offset/rate,short,long


def reuse_muq_chunks(waveform, encode, settings, baseline_chunks):
    for produced,baseline in zip_longest(musicfm_chunks(waveform,encode,settings),baseline_chunks):
        if produced is None or baseline is None:
            raise ValueError('MusicFM and baseline cache chunk inventories differ.')
        offset,short,long=produced
        reference,reference_offset,_=baseline
        if offset!=reference_offset or reference.ndim!=2 or reference.shape[1]!=4096:
            raise ValueError('Baseline MuQ chunk offsets/shape differ.')
        count=len(reference)
        if any(a.ndim!=2 or a.shape[1]!=1024 or not count<=len(a)<=count+4 for a in (short,long)):
            raise ValueError('Candidate MusicFM frames cannot align to unchanged MuQ cache.')
        values=np.array(reference,dtype=np.float32,copy=True)
        values[:,:1024]=short[:count]
        values[:,2048:3072]=long[:count]
        if not np.isfinite(values).all():
            raise ValueError('Non-finite candidate MusicFM/reused MuQ features.')
        yield offset,values
