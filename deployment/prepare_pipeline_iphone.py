"""Prepare full CPU-only PCM-to-section iPhone benchmark; preserves true tail lengths."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import numpy as np
from deployment.compress_muq import package_digest
from deployment.coreml_features import CoreMLEncoders
from deployment.iphone_project import project_text
from finetuning.common import config, read, write, sha_file, LABELS
from finetuning.engine import manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songmap-root',type=Path,required=True)
    p.add_argument('--muq-validation',type=Path,required=True)
    p.add_argument('--musicfm-compression-report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--track-id',help='Validation track; default is first available')
    p.add_argument('--config')
    p.add_argument('--team-id')
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run on macOS with Xcode.')
    import librosa
    cfg=config(a.config);data,provenance=manifest(cfg)
    validation_path=a.muq_validation.expanduser().resolve();v=read(validation_path)
    if not v.get('full_validation') or v.get('muq_scheme')!='palette6' or v.get('head_scheme')!='palette6':
        raise ValueError('Full MuQ palette6 validation required.')
    if v['plan_id']!=data['plan_id'] or v['provenance_id']!=provenance:raise ValueError('Dataset changed.')
    muq_proof=Path(v['muq_export_report']);mp=read(muq_proof)
    if sha_file(muq_proof)!=v['muq_export_report_sha256'] or package_digest(mp['package'])!=mp['package_digest']:
        raise ValueError('MuQ package/report changed.')
    combination_path=Path(v['baseline_combination']);combo=read(combination_path)
    if sha_file(combination_path)!=v['baseline_combination_sha256']:raise ValueError('Combination changed.')
    pipeline_path=Path(combo['encoder_pipeline_report']);pipeline=read(pipeline_path)
    if sha_file(pipeline_path)!=combo['encoder_pipeline_report_sha256']:raise ValueError('Encoder report changed.')
    fm_path=a.musicfm_compression_report.expanduser().resolve()
    if sha_file(fm_path)!=pipeline['musicfm_override']['report_sha256']:
        raise ValueError('MusicFM proof does not match the validated combination.')
    enc=CoreMLEncoders(a.songmap_root,cfg['features'],musicfm_export_report=fm_path)
    baseline_meta=read(Path(pipeline['diagnostic_cache'])/'encoder_provenance.json')
    for role in ('log_mel','musicfm_conv','muq_conv','musicfm'):
        if enc.provenance['packages'][role]!=baseline_meta['packages'][role]:raise ValueError('Package changed: '+role)
    if enc.provenance['app_manifest_sha256']!=mp['app_manifest_sha256']:raise ValueError('App manifest changed.')
    rows=[r for r in data['records'] if r['approved'] and r['split']=='val']
    row=next((r for r in rows if r['track_id']==a.track_id),None) if a.track_id else rows[0]
    if row is None:raise ValueError('Track not found in validation.')
    sections=[s for s in v['sections'] if s['track_id']==row['track_id']]
    if not sections or [(s['start'],s['end'],s['label_manual']) for s in sections]!=[(s['start'],s['end'],s['label']) for s in row['sections']]:
        raise ValueError('Section targets changed.')
    if sha_file(row['audio_path'])!=row['audio_sha256']:raise ValueError('Audio changed.')
    out=a.output_dir.expanduser().resolve()
    if out.exists():raise ValueError('Choose a new output directory.')
    assets=out/'Assets';assets.mkdir(parents=True)
    wav,_=librosa.load(row['audio_path'],sr=24000,mono=True)
    if not np.isfinite(wav).all():raise ValueError('Audio contains NaN/Inf.')
    np.asarray(wav,dtype='<f4').tofile(assets/'audio.f32')
    paths=dict(enc.package_paths);paths['muq']=Path(mp['package']);paths['head']=Path(combo['head_package'])
    roles=('log_mel','musicfm_conv','musicfm','muq_conv','muq','head')
    for role in roles:
        print('Compiling:',role,flush=True)
        subprocess.run(['xcrun','coremlcompiler','compile',str(paths[role]),str(assets)],check=True)
        compiled=assets/(paths[role].stem+'.mlmodelc')
        destination=assets/(role+'.mlmodelc')
        if compiled!=destination:compiled.rename(destination)
    import coremltools as ct
    h=ct.models.MLModel(str(paths['head']),skip_model_load=True)
    if h.user_defined_metadata.get('compression_scheme')!='palette6' or h.user_defined_metadata.get('checkpoint_sha256')!=v['checkpoint_sha256']:
        raise ValueError('Head metadata mismatch.')
    write(assets/'benchmark_manifest.json',{'track_id':row['track_id'],'sample_rate':24000,
        'audio_samples':len(wav),'head_window_seconds':420,'long_window_seconds':120,
        'rotary_inv_freq':enc.app['rotary_inv_freq'],'labels':list(LABELS),
        'class_ids':[0,1,2,3,4,5,6,26], 'sections':sections,
        'source_validation_sha256':sha_file(validation_path),'audio_pcm_sha256':sha_file(assets/'audio.f32'),
        'package_digests':{r:package_digest(paths[r]) for r in roles},
        'scope':'Resampled mono PCM -> app frontend -> palette6 encoders -> palette6 head -> fixed supplied Apple sections; no Apple boundary detection or audio decoding on phone.',
        'shape_policy':'Original variable lengths, CPU_ONLY; no zero padding or GPU dynamic-shape calls.'})
    template=Path(__file__).parent/'iphone_pipeline'
    for name in ('BenchmarkApp.swift','BenchmarkRunner.swift'):shutil.copyfile(template/name,out/name)
    project=out/'Benchmark.xcodeproj';project.mkdir()
    (project/'project.pbxproj').write_text(project_text('com.danendra.SongPipelineBenchmark',a.team_id,
        resource_names=[r+'.mlmodelc' for r in roles]+['audio.f32','benchmark_manifest.json']))
    print('Open:',project)


if __name__=='__main__':main()
