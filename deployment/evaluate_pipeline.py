"""Measure encoder conversion impact with the same verified FP32 head and Apple-section targets."""
import argparse
from pathlib import Path
import platform
import shutil
import tempfile
import time
import numpy as np
from deployment.compress_coreml import validate_source_report
from deployment.coreml_features import CoreMLEncoders, EncoderWindowError, extract_chunks
from deployment.verify_coreml import CoreMLAdapter
from deployment.musicfm_candidates import reuse_muq_chunks, validate_muq_provenance
from finetuning.common import config, fingerprint, read, sha_file, write
from finetuning.engine import evaluate, manifest


def cached_features(cache, row, candidate_id):
    folder = cache / row['id']
    meta = read(folder/'metadata.json')
    if meta['candidate_encoder_id'] != candidate_id or meta['audio_sha256'] != row['audio_sha256']:
        raise ValueError('Diagnostic feature cache stale; choose a new --cache-dir.')
    if not meta.get('chunks'):
        raise ValueError('Diagnostic cache metadata has no chunks.')
    for chunk in meta['chunks']:
        path = (folder/chunk['file']).resolve()
        path.relative_to(folder.resolve())
        values = np.load(path,mmap_mode='r',allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 4096 or not 3 <= len(values) <= 10500 or len(values)%3:
            raise ValueError('Invalid diagnostic feature cache shape.')
        yield values, chunk['offset'], candidate_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songmap-root',type=Path,required=True)
    parser.add_argument('--fp32-validation-report',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=Path('outputs.local/coreml-encoder-cache'))
    parser.add_argument('--limit',type=int,default=1,help='Validation tracks; 0=all')
    parser.add_argument('--reuse-muq-cache',type=Path,help='Reuse MuQ streams from a baseline Core ML diagnostic cache after source checks')
    parser.add_argument('--musicfm-export-report',type=Path,help='Optional verified FP32 MusicFM export; keeps original app frontend and rotary values')
    parser.add_argument('--config')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Core ML runtime requires macOS.')
    if args.limit < 0:
        raise ValueError('limit must be >=0')
    import coremltools as ct
    import librosa
    import torch
    torch.set_num_threads(2)
    cfg = config(args.config)
    cache, output, proof_path = [p.expanduser().resolve() for p in (args.cache_dir,args.output,args.fp32_validation_report)]
    if output.exists():
        raise ValueError('Choose a new report path.')
    proof = read(proof_path)
    package = Path(proof['package']).expanduser().resolve()
    validate_source_report(proof,package,proof.get('checkpoint_sha256'))
    data, provenance = manifest(cfg)
    all_rows = [r for r in data['records'] if r['approved'] and r['split']=='val']
    annotations_id = fingerprint([(r['track_id'],r['sections']) for r in all_rows])
    if (proof['plan_id'] != data['plan_id'] or proof['provenance_id'] != provenance
            or proof['annotations_id'] != annotations_id):
        raise ValueError('Validation labels/split/cache changed since FP32 verification.')
    rows = all_rows[:args.limit] if args.limit else all_rows
    if not rows:
        raise ValueError('No eligible validation tracks.')
    cache.mkdir(parents=True,exist_ok=True)
    if args.musicfm_export_report:
        override=read(args.musicfm_export_report.expanduser().resolve())
        original=read(Path(cfg['work_dir'])/'encoder_provenance.json')
        if (override.get('musicfm_weights_sha256') != original['musicfm_sha256']
                or override.get('musicfm_stats_sha256') != original['stats_sha256']):
            raise ValueError('MusicFM export weights/stats differ from frozen training provenance.')
    encoders = CoreMLEncoders(args.songmap_root,cfg['features'],musicfm_export_report=args.musicfm_export_report)
    reuse_cache = args.reuse_muq_cache.expanduser().resolve() if args.reuse_muq_cache else None
    reuse_id = None
    if reuse_cache:
        if reuse_cache == cache:
            raise ValueError('Candidate cache must differ from reused baseline cache.')
        reused = read(reuse_cache/'encoder_provenance.json')
        reuse_id = validate_muq_provenance(reused,encoders.provenance)
        for row in rows:
            for values,_,_ in cached_features(reuse_cache,row,reuse_id):
                pass
        encoders.provenance['reused_muq_encoder_id'] = reuse_id
        encoders.id = fingerprint(encoders.provenance)
    provenance_path = cache/'encoder_provenance.json'
    if provenance_path.exists() and read(provenance_path)['id'] != encoders.id:
        raise ValueError('Encoder packages/settings changed; use a new --cache-dir.')
    write(provenance_path,{'id':encoders.id,**encoders.provenance})
    pending = []
    for row in rows:
        if (cache/row['id']/'metadata.json').exists():
            # Check complete files and identity before deciding to reuse the cache.
            for values,_,_ in cached_features(cache,row,encoders.id):
                pass
        else:
            pending.append(row)
    required = sum(r['duration'] for r in pending)*25*4096*4
    if shutil.disk_usage(cache).free < required+1024**3:
        raise ValueError(f'Diagnostic features require about {required/1024**3:.1f} GiB plus 1 GiB free space.')
    started = time.perf_counter()
    if pending:
        encoders.load(roles=('log_mel','musicfm_conv','musicfm') if reuse_cache else ('log_mel','musicfm_conv','musicfm','muq_conv','muq'))
    for index,row in enumerate(pending,1):
        print(f'Core ML features [{index}/{len(pending)}]: {row["track_id"]}',flush=True)
        if sha_file(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Audio changed since prepare.')
        waveform,_ = librosa.load(row['audio_path'],sr=24000,mono=True)
        folder = cache/row['id']
        folder.mkdir(parents=True,exist_ok=True)
        chunks=[]
        chunks_iterator=(reuse_muq_chunks(waveform,encoders.encode_musicfm,cfg['features'],
                         cached_features(reuse_cache,row,reuse_id)) if reuse_cache else
                         extract_chunks(waveform,encoders.encode_pair,cfg['features']))
        try:
            for offset,values in chunks_iterator:
                name=f'chunk_{int(offset):06d}.npy'
                with tempfile.NamedTemporaryFile(dir=folder,delete=False) as handle:
                    temporary=Path(handle.name)
                    np.save(handle,values,allow_pickle=False)
                temporary.replace(folder/name)
                chunks.append({'file':name,'offset':offset,'frames':len(values)})
                print(f'  {offset:.1f}s: {len(values)} frames',flush=True)
        except EncoderWindowError as exc:
            failure = {**exc.details, 'track_id':row['track_id'], 'audio_sha256':row['audio_sha256'],
                       'candidate_encoder_id':encoders.id, 'cache_folder':str(folder),
                       'cache_preserved':True, 'non_finite_values_replaced':False,
                       'model_package_paths':{role:str(path) for role,path in encoders.package_paths.items()}}
            failure_path = cache/'last_encoder_failure.json'
            write(failure_path,failure)
            print('ENCODER FAILURE:',row['track_id'],str(exc),flush=True)
            print('Diagnostic report:',failure_path,flush=True)
            raise RuntimeError(f'Encoder failed. Share {failure_path}. Completed cache tracks were preserved.') from exc
        if not chunks:
            raise ValueError('No encoder chunks produced.')
        write(folder/'metadata.json',{'audio_sha256':row['audio_sha256'],
            'candidate_encoder_id':encoders.id,'chunks':chunks})
    # Keep encoders out of the head evaluation working set.
    encoders.models.clear()
    import gc
    gc.collect()
    model=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    if (model.user_defined_metadata.get('checkpoint_sha256') != proof['checkpoint_sha256']
            or model.user_defined_metadata.get('precision') != 'fp32'
            or model.user_defined_metadata.get('compression_scheme')):
        raise ValueError('Use the verified, uncompressed FP32 head.')
    adapter=CoreMLAdapter(model)
    baseline_details,converted_details=[],[]
    print('Evaluating cached reference features...',flush=True)
    baseline=evaluate(adapter,rows,cfg,torch.device('cpu'),provenance,details=baseline_details)
    print('Evaluating Core ML encoder features...',flush=True)
    converted=evaluate(adapter,rows,cfg,torch.device('cpu'),encoders.id,details=converted_details,
        feature_reader=lambda _cfg,row:cached_features(cache,row,encoders.id))
    changed=[]
    for a,b in zip(baseline_details,converted_details):
        if (a['track_id'],a['start'],a['end']) != (b['track_id'],b['start'],b['end']):
            raise ValueError('Section inventories differ.')
        if a['prediction'] != b['prediction']:
            changed.append({'track_id':a['track_id'],'start':a['start'],'end':a['end'],
                'label_manual':a['label_manual'],'reference_prediction':a['prediction'],
                'coreml_encoder_prediction':b['prediction'],
                'reference_correct':a['correct'],'coreml_correct':b['correct']})
    agreement=1-len(changed)/len(baseline_details)
    report={'tracks':len(rows),'full_validation':len(rows)==len(all_rows),
        'plan_id':data['plan_id'],'provenance_id':provenance,'candidate_encoder_id':encoders.id,
        'annotations_id':fingerprint([(r['track_id'],r['sections']) for r in rows]),
        'checkpoint_sha256':proof['checkpoint_sha256'],'fp32_head':str(package),
        'reference_cache_with_fp32_head':baseline,'coreml_encoders_with_fp32_head':converted,
        'macro_f1_delta':converted['macro_f1_present']-baseline['macro_f1_present'],
        'section_prediction_agreement':agreement,'changed_sections':changed,
        'encoder_prediction_timings_this_run':encoders.timings,
        'elapsed_seconds':time.perf_counter()-started,'newly_encoded_tracks':len(pending),
        'diagnostic_cache':str(cache),'ready_for_app_integration':False,
        'musicfm_override':encoders.musicfm_override,
        'reused_muq_cache':str(reuse_cache) if reuse_cache else None,
        'reused_muq_encoder_id':reuse_id,
        'scope':'CPU Core ML encoders from audio decoded/resampled by librosa, fixed Apple boundaries and FP32 head. Not Swift audio preprocessing or iPhone timing.'}
    write(output,report)
    print(f'Reference F1={baseline["macro_f1_present"]:.6f}; Core ML encoders F1={converted["macro_f1_present"]:.6f}; delta={report["macro_f1_delta"]:+.6f}')
    print(f'Section agreement={agreement:.4%}; changed={len(changed)}; report={output}')


if __name__=='__main__':
    main()
