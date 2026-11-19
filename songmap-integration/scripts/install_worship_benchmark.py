#!/usr/bin/env python3
"""Stage verified benchmark resources locally for the Debug-only worship integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def install(root,benchmark):
    root,benchmark=Path(root).resolve(),Path(benchmark).expanduser().resolve()
    assets=benchmark/'Assets'
    proof=json.loads((assets/'benchmark_manifest.json').read_text())
    if not proof.get('hybrid') or proof.get('class_ids')!=[0,1,2,3,4,5,6,26]:
        raise ValueError('Use the prepared hybrid palette6 pipeline project.')
    app_manifest=root/'SongMap/Resources/MLModels/SongFormer/songformer_coreml_manifest.json'
    manifest=json.loads(app_manifest.read_text())
    if manifest['sample_rate']!=24000 or manifest['head']['dataset_id']!=5:
        raise ValueError('Unexpected app profile.')
    if manifest['rotary_inv_freq']!=proof['rotary_inv_freq']:raise ValueError('Rotary frequencies differ.')
    roles=['log_mel','musicfm_conv','musicfm','muq_conv','muq','head']
    static=['musicfm_750','musicfm_3000','muq_750','muq_3000']
    for name in roles+static:
        folder=assets/(name+'.mlmodelc')
        if not folder.is_dir() or not any(p.is_file() for p in folder.rglob('*')):raise ValueError('Compiled resource missing: '+name)
    destination=root/'SongMap/Resources/MLModels/SongFormerWorship'
    if destination.exists():raise ValueError('Worship resources already exist; choose a fresh checkout or move that folder aside.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Stage first; do not leave a half-installed manifest if copying fails.
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix='.worship-stage-') as temp:
        staged=Path(temp)/'SongFormerWorship';staged.mkdir()
        for name in roles+static:
            shutil.copytree(assets/(name+'.mlmodelc'),staged/('Worship_'+name+'.mlmodelc'))
        manifest['models']={role:'Worship_'+role+'.mlmodelc' for role in roles}
        manifest['worship_static']={name:'Worship_'+name+'.mlmodelc' for name in static}
        manifest['head']['allowed_label_ids']=[0,1,2,3,4,5,6,26]
        manifest['head']['id_to_label']={str(k):v for k,v in zip(proof['class_ids'],proof['labels'])}
        manifest['installation_provenance']={'benchmark_manifest_sha256':hashlib.sha256((assets/'benchmark_manifest.json').read_bytes()).hexdigest(),
            'source_validation_sha256':proof['source_validation_sha256'],'package_digests':proof['package_digests']}
        (staged/'songformer_worship_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staged.rename(destination)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    benchmark=a.benchmark_dir
    if benchmark is None:
        candidates=list((root.parent/'Section-Annotations/outputs.local').glob('pipeline-hybrid-iphone-*/Assets/benchmark_manifest.json'))
        if not candidates:p.error('No sibling Section-Annotations hybrid benchmark found; supply --benchmark-dir.')
        benchmark=max(candidates,key=lambda p:p.stat().st_mtime).parent.parent
    print('Benchmark:',benchmark)
    print('Installed locally:',install(root,benchmark))
