#!/usr/bin/env python3
"""Synthetic Python reference for the actual Swift pooling/timing/correction implementation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_section_context import pool_audio, geometry, LABELS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    out=p.parse_args().output_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    sections=[dict(start=.02,end=.10),dict(start=.10,end=.22),dict(start=.22,end=.24)]
    values=((np.arange(6*4096).reshape(6,4096)%97-48)/50).astype(np.float32)
    chunks=[]
    for index in range(2):
        name=f'chunk-{index}.f32';values[index*3:(index+1)*3].astype('<f4').tofile(out/name)
        chunks.append(dict(offset=index*.12,frames=3,file=name))
    means,coverage=pool_audio([(values[:3],0.,'fixture'),(values[3:],.12,'fixture')],sections,'fixture')
    means.astype('<f4').tofile(out/'means.f32');geometry(sections,.24).astype('<f4').tofile(out/'timing.f32')
    frame_p=[[.02,.03,.65,.15,.05,.04,.03,.03],[.02,.04,.18,.60,.04,.04,.04,.04]]
    frames=[dict(time=i*.12,duration=.12,probabilities=p) for i,p in enumerate(frame_p)]
    pooled=[]
    for s in sections:
        weights=[max(0,min(f['time']+f['duration'],s['end'])-max(f['time'],s['start'])) for f in frames]
        pooled.append([sum(w*f['probabilities'][j] for w,f in zip(weights,frames))/sum(weights) for j in range(8)])
    residual=torch.zeros(3,8);residual[0,3]=2;residual[1,2]=2
    logits=torch.tensor(pooled,dtype=torch.float32).clamp_min(1e-8).log()+residual
    fixture=dict(duration=.24,sections=sections,chunks=chunks,frames=frames,expectedCoverage=coverage.tolist(),
        residual=residual.flatten().tolist(),expectedPooled=pooled,expectedProbabilities=logits.softmax(-1).tolist(),
        expectedLabels=[LABELS[i] for i in logits.argmax(-1).tolist()])
    (out/'fixture.json').write_text(json.dumps(fixture,allow_nan=False))


if __name__=='__main__':main()
