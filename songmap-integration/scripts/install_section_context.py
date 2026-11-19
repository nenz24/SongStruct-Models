#!/usr/bin/env python3
"""Compile/stage the fully parity-checked context model locally; preserve existing worship resources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

LABELS = ['intro','verse','chorus','bridge','inst','outro','silence','pre-chorus']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def file_hashes(folder):
    paths=sorted(Path(folder).rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('Model resources must not contain symlinks.')
    return {str(p.relative_to(folder)):sha_file(p) for p in paths if p.is_file()}
def package_digest(folder): return fingerprint(list(file_hashes(folder).items()))
def find_report(paths, expected):
    return next((p.resolve() for p in paths if sha_file(p)==expected),None)
def compile_package(package, output):
    subprocess.run(['xcrun','coremlcompiler','compile',str(package),str(output)],check=True)


def install(root, parity_report, export_dir=None, songstruct_root=None, palette_report=None, compiler=compile_package):
    root=Path(root).expanduser().resolve(); parity_report=Path(parity_report).expanduser().resolve()
    parity=read(parity_report)
    if (parity.get('protocol')!='section-context-coreml-fp32-v1' or parity.get('split')!='val'
            or not parity.get('full_validation') or not parity.get('passed') or parity.get('status')!='numerically_consistent'
            or parity.get('precision')!='fp32' or parity.get('selected_epoch')!=5
            or parity.get('section_agreement')!=1 or parity.get('macro_f1_delta')!=0
            or len(parity.get('checks',[]))!=parity.get('tracks') or not parity.get('checks')
            or any(not c.get('passed') or any(not c.get(k,{}).get('passed') for k in ('residual','final_logits','probabilities'))
                   for c in parity['checks'])):
        raise ValueError('Full successful FP32 context parity is required before installation.')
    export_dir=Path(export_dir).expanduser().resolve() if export_dir else parity_report.parent.parent/'export'
    export=read(export_dir/'export_report.json')
    for key in ('protocol','context_checkpoint_sha256','palette_validation_sha256','palette_selection_identity','plan_id',
                'provenance_id','selected_epoch','torch','samples','package_digest'):
        if export[key]!=parity[key]: raise ValueError('Parity/export identity differs: '+key)
    package=Path(export['package']).expanduser().resolve()
    contract=export['input_contract']
    if (not export.get('native_synthetic_parity_passed') or not export.get('attention_parity_passed')
            or not export.get('trace_parity_passed') or export.get('precision')!='fp32'
            or export.get('declared_tensor_dtypes',{}).get('FLOAT16',0)!=0
            or not 1<=export['max_sections']<=512 or contract['dtype']!='float32'
            or contract['inputs']!={'section_audio':['N',4096],'section_timing':['N',6]}
            or contract['output']!={'context_residual':['N',8]} or contract['label_order']!=LABELS
            or not contract['all_original_sections_required'] or package_digest(package)!=export['package_digest']):
        raise ValueError('Validated context package/input contract changed.')
    source=find_report((root/'outputs.local').glob('context-palette6-validation-*/validation-all.json'),export['palette_validation_sha256'])
    if source is None: raise ValueError('Original completed palette6 context report is missing.')
    inputs=read(source.parent/'selection.json')
    if (inputs['identity']!=fingerprint({k:v for k,v in inputs.items() if k!='identity'})
            or inputs['identity']!=parity['palette_selection_identity']):
        raise ValueError('Palette6 context source identity changed.')
    songstruct=Path(songstruct_root).expanduser().resolve() if songstruct_root else root.parent/'Section-Annotations'
    raw=Path(palette_report).expanduser().resolve() if palette_report else find_report(
        (songstruct/'outputs.local').glob('muq-compression-*/palette6-all-validation.json'),inputs['palette6_report_sha256'])
    if raw is None or sha_file(raw)!=inputs['palette6_report_sha256']: raise ValueError('Palette6 source report missing/changed; supply --palette6-report.')
    raw_report=read(raw)
    if raw_report.get('head_scheme')!='palette6' or raw_report.get('muq_scheme')!='palette6' or not raw_report.get('full_validation'):
        raise ValueError('Expected fully validated palette6 source pipeline.')
    base=root/'SongMap/Resources/MLModels/SongFormerWorship'
    installed=read(base/'songformer_worship_manifest.json')
    if (installed.get('installation_provenance',{}).get('source_validation_sha256')!=sha_file(raw)
            or installed['sample_rate']!=24000 or installed['encoder_window_seconds']!=120
            or installed['head_max_seconds']!=420 or installed['head']['dataset_id']!=5
            or installed['head']['allowed_label_ids']!=[0,1,2,3,4,5,6,26]):
        raise ValueError('Installed worship resources differ from the palette6 context source. Install the matching benchmark first.')
    if (set(installed['models'])!={'log_mel','musicfm_conv','musicfm','muq_conv','muq','head'}
            or set(installed.get('worship_static',{}))!={'musicfm_750','musicfm_3000','muq_750','muq_3000'}):
        raise ValueError('The complete dynamic/fixed worship resource profile is required.')
    for filename in list(installed['models'].values())+list(installed.get('worship_static',{}).values()):
        model=base/filename
        if not model.is_dir() or not any(p.is_file() for p in model.rglob('*')):
            raise ValueError('Existing compiled worship resource is missing: '+filename)
    for key in ('plan_id','provenance_id'):
        if raw_report[key]!=parity[key]: raise ValueError('Source dataset profile differs: '+key)
    receipt=dict(schemaVersion=1,model='Worship_SectionContext.mlmodelc',precision='fp32',
        exportProtocol=parity['protocol'],selectedEpoch=5,checkpointSHA256=parity['context_checkpoint_sha256'],
        baseCheckpointSHA256=raw_report['checkpoint_sha256'],baseValidationSHA256=sha_file(raw),
        labelOrder=LABELS,maxSections=export['max_sections'],featureWidth=4096,sampleRate=24000,encoderFPS=25,
        shortWindowSeconds=30,longWindowSeconds=120,headWindowSeconds=420,
        packageDigest=export['package_digest'],parityReportSHA256=sha_file(parity_report),
        planID=parity['plan_id'],provenanceID=parity['provenance_id'])
    destination=base.parent/'SongFormerContext'; name='songformer_section_context_manifest.json'
    if destination.exists():
        previous=read(destination/name)
        if ({k:v for k,v in previous.items() if k!='compiledFiles'}!=receipt
                or previous.get('compiledFiles')!=file_hashes(destination/receipt['model'])):
            raise ValueError('Existing context installation differs/corrupt; move it aside before installing another package.')
        return destination
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix='.context-stage-') as directory:
        stage=Path(directory); compiler(package,stage)
        compiled=stage/(package.stem+'.mlmodelc')
        if not compiled.is_dir() or not file_hashes(compiled): raise ValueError('Compiler did not produce a compiled context model.')
        staged=stage/'SongFormerContext';staged.mkdir()
        shutil.move(str(compiled),staged/receipt['model'])
        receipt['compiledFiles']=file_hashes(staged/receipt['model'])
        (staged/name).write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        staged.rename(destination)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-report',type=Path)
    parser.add_argument('--export-dir',type=Path)
    parser.add_argument('--songstruct-root',type=Path)
    parser.add_argument('--palette6-report',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=args.parity_report
    if report is None:
        reports=list((root/'outputs.local').glob('context-coreml-fp32-*/parity/validation-all.json'))
        if not reports:parser.error('No full context Core ML parity report found; run section_context_coreml_fp32.ipynb first.')
        report=max(reports,key=lambda p:p.stat().st_mtime)
    print('Validated parity report:',report)
    print('Installed locally:',install(root,report,args.export_dir,args.songstruct_root,args.palette6_report))
    print('Debug launch argument: --songformer-worship-context (CPU_ONLY)')


if __name__=='__main__':main()
