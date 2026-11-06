#!/usr/bin/env python3
"""Restore approved cloud corrections onto original local annotation files."""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LABELED_PATH = BASE / 'outputs.local' / 'labeled'
OUTPUT_PATH = BASE / 'outputs.local' / 'reviewed'
LABELS = {'intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'inst', 'outro', 'silence'}


def apply_corrections(original, reviewed):
    if reviewed.get('review_status') != 'approved':
        raise ValueError('Anotasi belum approved.')
    source_sections, corrections = original['sections'], reviewed['sections']
    if len(source_sections) != len(corrections):
        raise ValueError('Jumlah section berbeda dari sumber lokal.')
    if abs(float(original['duration']) - float(reviewed['duration'])) > 0.001:
        raise ValueError('Durasi berbeda dari sumber lokal.')
    result = {**original, 'sections': [], 'review_status': 'approved', 'boundary_locked': True,
              'annotator_id': reviewed.get('annotator_id'),
              'annotation_updated': reviewed.get('annotation_updated'),
              'annotation_version': reviewed.get('annotation_version')}
    for src, correction in zip(source_sections, corrections):
        if (src['start'] != correction['start'] or src['end'] != correction['end']):
            raise ValueError('Boundary cloud berbeda dari sumber lokal; hasil ditolak.')
        initial = src.get('label_initial', src.get('label'))
        if initial != correction.get('label_initial'):
            raise ValueError('Prediksi awal berbeda dari sumber lokal; hasil ditolak.')
        label = correction.get('label_manual')
        if label not in LABELS:
            raise ValueError('Label manual tidak valid atau belum diperiksa.')
        result['sections'].append({**src, 'label_initial': initial, 'label_manual': label,
                                   'label': label, 'note': correction.get('note', ''),
                                   'review_status': 'approved'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path, help='songstruct-approved.json yang diunduh dari aplikasi')
    args = parser.parse_args()
    payload = json.loads(args.export.expanduser().read_text(encoding='utf-8'))
    if not isinstance(payload.get('tracks'), list):
        parser.error('Format ekspor tidak valid: tracks harus berupa list.')
    sources = {}
    for file in LABELED_PATH.rglob('*.json'):
        original = json.loads(file.read_text(encoding='utf-8'))
        key = original.get('track_id')
        if key in sources:
            parser.error(f'Track ID lokal duplikat: {key}')
        sources[key] = (file, original)
    written, skipped, failed = 0, 0, 0
    for reviewed in payload['tracks']:
        key = reviewed.get('track_id')
        if reviewed.get('review_status') != 'approved':
            skipped += 1
            continue
        try:
            if key not in sources:
                raise ValueError('Track ID tidak ditemukan pada outputs.local/labeled.')
            file, original = sources[key]
            result = apply_corrections(original, reviewed)
            dest = OUTPUT_PATH / file.relative_to(LABELED_PATH)
            encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('x', encoding='utf-8') as handle:
                handle.write(encoded + '\n')
            written += 1
            print(f'OK: {key}', flush=True)
        except (ValueError, KeyError, OSError) as exc:
            failed += 1
            print(f'ERROR: {key}: {exc}', flush=True)
    print(f'{written} disimpan, {skipped} belum approved dilewati, {failed} gagal.')
    print(f'Hasil: {OUTPUT_PATH}. File sumber tidak diubah; output lama tidak ditimpa.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
