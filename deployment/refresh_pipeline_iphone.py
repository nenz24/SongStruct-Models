"""Refresh benchmark Swift sources without recompiling or copying large Core ML assets."""
import argparse
from pathlib import Path
import shutil
from finetuning.common import read


def refresh(project):
    project=Path(project).expanduser().resolve()
    if not (project/'Benchmark.xcodeproj/project.pbxproj').is_file():
        raise ValueError('Pipeline benchmark Xcode project missing.')
    manifest=read(project/'Assets/benchmark_manifest.json')
    if (manifest.get('shape_policy') not in ('Original variable lengths, CPU_ONLY; no zero padding or GPU dynamic-shape calls.',
            'Static ALL for full 30/120s encoder windows; CPU_ONLY for exact-length tails, frontend and head.')
            or not manifest.get('sections') or manifest.get('class_ids')!=[0,1,2,3,4,5,6,26]):
        raise ValueError('This is not the supported full pipeline benchmark project.')
    source=Path(__file__).parent/'iphone_pipeline'
    for name in ('BenchmarkApp.swift','BenchmarkRunner.swift'):
        shutil.copyfile(source/name,project/name)
    return project/'Benchmark.xcodeproj'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--project-dir',type=Path)
    g.add_argument('--latest',action='store_true',help='Latest prepared outputs.local/pipeline-iphone-* project')
    args=p.parse_args()
    project=args.project_dir
    if args.latest:
        candidates=[p.parent for p in Path('outputs.local').glob('pipeline*-iphone-*/Benchmark.xcodeproj')
                    if (p.parent/'Assets/benchmark_manifest.json').is_file()]
        if not candidates:raise ValueError('No prepared pipeline benchmark found; supply --project-dir.')
        project=max(candidates,key=lambda p:(p/'Assets/benchmark_manifest.json').stat().st_mtime)
    print('Updated Swift only:',refresh(project))
    print('Stop the previous Xcode run, build/run again, then press Run full pipeline on iPhone.')


if __name__=='__main__':main()
