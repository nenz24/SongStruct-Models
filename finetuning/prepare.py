"""Validate annotations and freeze group-level train/validation/test splits."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import unicodedata
from .common import BASE, LABELS, config, fingerprint, read, sha_file, write
from .audit import allowed, audit


def group_suggestion(track_id):
    path = Path(track_id)
    name = path.parent.name if path.parent.name else re.sub(r'^\d+\s*[-_]\s*', '', path.stem)
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', name).casefold()).strip()


def validate_review(original, corrected, decisions=None, adjustments=None):
    if corrected.get('review_status') != 'approved':
        return None
    source, edited = original['sections'], corrected['sections']
    if len(source) != len(edited) or abs(float(original['duration']) - float(corrected['duration'])) > 0.001:
        raise ValueError('Approved annotation does not match source duration/sections.')
    duration = float(original['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid duration: {duration!r}')
    previous, result = 0.0, []
    for index, (a, b) in enumerate(zip(source, edited), 1):
        start, end = a['start'], a['end']
        # Only the final endpoint may exceed declared duration by at most 20 ms.
        # Keep source equality checks on raw times; clip only the training target.
        tail_limit = 0.020 + 1e-9 if index == len(source) else 0.0
        target_end = min(end, duration)
        if not (math.isfinite(start + end) and previous - 1e-6 <= start < target_end
                and end <= duration + tail_limit):
            raise ValueError(
                f'Section {index}: invalid time range; start={start!r}, end={end!r}, '
                f'previous_end={previous!r}, duration={duration!r}. '
                f'Overlap={max(0.0, previous - start):.9f}s; '
                f'end_overrun={max(0.0, end - duration):.9f}s.')
        if (start, end) != (b['start'], b['end']) or a.get('label_initial', a.get('label')) != b.get('label_initial'):
            raise ValueError('Boundary/prediksi awal berubah dari source.')
        if b.get('label_manual') not in LABELS:
            raise ValueError('Every approved section must have a valid manual label.')
        if end > duration and adjustments is not None:
            adjustments.append({'track_id': corrected.get('track_id', ''),
                                'section_index': index, 'original_end': end,
                                'training_end': target_end, 'trimmed_seconds': end - target_end})
        if allowed(corrected.get('track_id', ''), index, b, decisions or {}):
            result.append({'start': start, 'end': target_end, 'label': b['label_manual']})
        previous = end
    return result


def allocate(groups, ratios, seed):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError('split_ratios must be three positive values summing to 1.')
    ordered = sorted(set(groups), key=lambda g: fingerprint([seed, g]))
    if len(ordered) < 3:
        raise ValueError('Butuh minimal 3 kelompok lagu berbeda untuk split yang terpisah.')
    n = len(ordered)
    val = max(1, round(n * ratios[1]))
    test = max(1, round(n * ratios[2]))
    train = n - val - test
    if train < 1:
        raise ValueError('Rasio tidak menyisakan kelompok train.')
    return {g: 'train' if i < train else 'val' if i < train + val else 'test' for i, g in enumerate(ordered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--export', type=Path, help='Optional songstruct-approved.json from the web app')
    parser.add_argument('--confirm-groups', action='store_true', help='Confirm you reviewed and corrected group_map.csv')
    args = parser.parse_args()
    cfg = config(args.config)
    from check_dataset import DATASET_PATH
    root = DATASET_PATH.expanduser().resolve()
    work = Path(cfg['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    originals = {}
    for path in sorted(Path(cfg['labeled_dir']).rglob('*.json')):
        doc = read(path)
        track_id = doc.get('track_id')
        if not isinstance(track_id, str) or not track_id or track_id in originals:
            raise ValueError(f'Invalid/duplicate track_id: {track_id}')
        if doc.get('label_source') != 'ASLP-lab/SongFormer' or not doc.get('sections'):
            raise ValueError(f'Not a labeled SongFormer source: {path}')
        audio = (root / track_id).resolve()
        try:
            audio.relative_to(root)
        except ValueError:
            raise ValueError('Track path escapes dataset root.') from None
        if not audio.is_file():
            raise ValueError(f'Audio tidak ada: {audio}')
        originals[track_id] = (doc, audio, path)
    if not originals:
        raise ValueError('Belum ada JSON SongFormer di labeled_dir.')
    mapping = work / 'group_map.csv'
    group_map = {}
    if mapping.exists():
        with mapping.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['track_id'] in group_map or not row['group_id'].strip():
                    raise ValueError('Duplicate/empty group mapping.')
                group_map[row['track_id']] = row['group_id'].strip()
    missing = sorted(set(originals) - set(group_map))
    for key in missing:
        group_map[key] = group_suggestion(key)
    if missing or not mapping.exists():
        with mapping.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['track_id', 'group_id'])
            writer.writerows(sorted(group_map.items()))
    reviewed = {}
    documents = read(args.export.expanduser())['tracks'] if args.export else [read(p) for p in Path(cfg['reviewed_dir']).rglob('*.json')]
    decisions, audit_stats = audit(documents, work)
    validation_errors, time_adjustments, seen = [], [], set()
    for doc in documents:
        key = doc.get('track_id')
        if doc.get('review_status') != 'approved':
            continue
        if key not in originals or key in seen:
            raise ValueError(f'Unknown/duplicate approved track: {key}')
        seen.add(key)
        try:
            reviewed[key] = validate_review(originals[key][0], doc, decisions, time_adjustments)
        except (ValueError, TypeError, KeyError) as exc:
            validation_errors.append({'track_id': key, 'error': str(exc)})
    report = work / 'validation_errors.json'
    write(report, validation_errors)
    adjustment_report = work / 'time_adjustments.json'
    write(adjustment_report, time_adjustments)
    print(f'Penyesuaian akhir section <=20ms: {len(time_adjustments)}. Laporan: {adjustment_report}')
    if validation_errors:
        for failure in validation_errors:
            print(f"INVALID: {failure['track_id']} — {failure['error']}")
        raise ValueError(f'{len(validation_errors)} approved tracks failed validation. '
                         f'See {report}. No new manifest or split plan was written.')
    records, hashes = [], {}
    for key, (doc, audio, source) in originals.items():
        digest = sha_file(audio)
        group = group_map[key]
        if digest in hashes and hashes[digest] != group:
            raise ValueError(f'Identical audio bytes in different groups: {key}. Fix group_map.csv.')
        hashes[digest] = group
        records.append({'id': fingerprint(key)[:24], 'track_id': key, 'group_id': group,
                        'audio_path': str(audio), 'audio_sha256': digest,
                        'duration': doc['duration'], 'review_approved': key in reviewed,
                        'approved': bool(reviewed.get(key)),
                        'excluded_sections': len(doc['sections']) - len(reviewed[key]) if key in reviewed else 0,
                        'sections': reviewed.get(key, []), 'source_json': str(source),
                        'annotation_sha256': fingerprint(reviewed.get(key, []))})
    plan_path = work / 'split_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        if plan['groups'] != group_map or plan['seed'] != cfg['seed'] or plan['ratios'] != cfg['split_ratios']:
            raise ValueError('Frozen split differs from inventory/group map/config. Use a new work_dir for a new experiment.')
        if missing:
            raise ValueError('New tracks cannot be added silently to frozen split.')
    elif args.confirm_groups:
        split = allocate(group_map.values(), cfg['split_ratios'], cfg['seed'])
        plan = {'groups': group_map, 'group_splits': split, 'seed': cfg['seed'],
                'ratios': cfg['split_ratios'], 'groups_confirmed': True}
        plan['id'] = fingerprint(plan)
        write(plan_path, plan)
    else:
        plan = None
    for record in records:
        record['split'] = plan['group_splits'][record['group_id']] if plan else None
    manifest = {'schema_version': 1, 'plan_id': plan['id'] if plan else None,
                'groups_confirmed': bool(plan), 'records': records}
    write(work / 'manifest.json', manifest)
    counts = {s: sum(r['approved'] and r['split'] == s for r in records) for s in ('train', 'val', 'test')}
    gb = sum(float(r['duration']) for r in records) * 25 * 4096 * 4 / 1024 ** 3
    print(f"Inventory: {len(records)} audio; approved: {len(reviewed)}; approved split: {counts}")
    print(f"Section layak: {audit_stats['eligible_sections']}; ditahan/dikecualikan: {audit_stats['excluded_or_pending_sections']}.")
    print(f'Perkiraan cache float32: {gb:.1f} GiB (belum termasuk checkpoint).')
    print(f'Group map: {mapping}')
    if not plan:
        print('Review group_map.csv: gabungkan versi live/studio/cover/alias dari lagu yang sama. Lalu rerun --confirm-groups.')
    print('Tidak menjalankan training. Anotasi pending tidak menjadi target training.')


if __name__ == '__main__':
    main()
