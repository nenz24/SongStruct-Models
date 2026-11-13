"""Evaluate pretrained or fine-tuned label models on frozen validation/test groups."""
import argparse
from pathlib import Path
from .common import config, device, load_head, write
from .engine import evaluate, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', type=Path, help='Default: pretrained baseline')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sections-output', type=Path, help='Optional JSON with every eligible section prediction for manual diagnosis')
    args = parser.parse_args()
    cfg = config(args.config)
    data, provenance = manifest(cfg)
    rows = [r for r in data['records'] if r['approved'] and r['split'] == args.split]
    if not rows:
        raise ValueError('Tidak ada approved audio pada split ini.')
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Output laporan sudah ada; pilih nama baru.')
    section_output = args.sections_output.expanduser().resolve() if args.sections_output else None
    if section_output and (section_output == output or section_output.exists()):
        raise ValueError('Section output must be a new path distinct from metrics output.')
    target = device(args.device)
    if checkpoint and checkpoint.suffix == '.pt':
        import torch
        meta = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if meta.get('plan_id') != data['plan_id'] or meta.get('provenance_id') != provenance or meta.get('config', {}).get('features') != cfg['features']:
            raise ValueError('Checkpoint tidak sesuai split/cache eksperimen ini.')
    model, _ = load_head(cfg, target, checkpoint)
    details = [] if section_output else None
    metrics = evaluate(model, rows, cfg, target, provenance, details=details)
    write(output, {'split': args.split, 'tracks': len(rows), 'plan_id': data['plan_id'],
                   'provenance_id': provenance, 'checkpoint': str(checkpoint or cfg['checkpoint']), **metrics})
    if section_output:
        write(section_output, {'split': args.split, 'plan_id': data['plan_id'],
                               'provenance_id': provenance, 'checkpoint': str(checkpoint or cfg['checkpoint']),
                               'index_basis': 'eligible targets; match original annotations by track_id/start/end',
                               'sections': details})
        print(f'Prediksi per section: {section_output}')
    print(f'Hasil: {output}\nSection acc={metrics["section_accuracy"]:.4f}; macro F1={metrics["macro_f1_present"]:.4f}')


if __name__ == '__main__':
    main()
