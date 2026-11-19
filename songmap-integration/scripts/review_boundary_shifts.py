#!/usr/bin/env python3
"""Review local bar-boundary shifts from completed context app reports; never edit maps."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

LABELS = ['intro', 'verse', 'chorus', 'bridge', 'inst', 'outro', 'silence', 'pre-chorus']
PARAMETERS = dict(anchor_probability=.55, anchor_margin=.15, coverage=.98,
                  mean_log_gain=.10, minimum_shift_bars=.25, minimum_section_bars=1)
FIELDS = ['source_sha256', 'title', 'boundary_index', 'left_label', 'right_label',
          'original_seconds', 'candidate_seconds', 'status', 'reference_seconds', 'review_note']


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(report):
    if report.get('state') != 'completed' or not report.get('contextEvidence'):
        raise ValueError('A completed app report with contextEvidence is required.')
    evidence = report['structureEvidence']; context = report['contextEvidence']
    if evidence['labelOrder'] != LABELS or context['labelOrder'] != LABELS:
        raise ValueError('Expected the original eight-class order.')
    if context['selectedEpoch'] != 5 or context['precision'] != 'fp32':
        raise ValueError('Expected the frozen epoch-5 FP32 context model.')
    sections = evidence['assembledSections']; source = evidence['appleBoundaries']
    refinement = report.get('boundaryRefinementEvidence')
    if refinement:
        originals = [{k:s[k] for k in ('start','end','label')} for s in context['sections']]
        if (refinement.get('protocolVersion') != 'context-boundary-shift-review-v1'
                or refinement.get('originalSections') != originals or refinement.get('refinedSections') != sections):
            raise ValueError('Debug refinement source/final section identity changed.')
        # Always assess original context inputs, never cumulatively refine the already shifted output.
        sections = originals
    if not sections or len(sections) != len(context['sections']) or len(source) != len(sections):
        raise ValueError('All original sections and corrected labels are required.')
    previous = 0.
    for i, (section, corrected, apple) in enumerate(zip(sections, context['sections'], source)):
        start, end = section['start'], section['end']
        if (not all(finite(t) for t in (start, end)) or not previous - 1e-6 <= start < end
                or section['label'] not in LABELS or corrected['label'] != section['label']
                or corrected['start'] != start or corrected['end'] != end
                or apple['start'] != start or not finite(apple['end'])
                or abs(apple['end'] - end) > (.020000001 if i == len(sections)-1 else 1e-6)):
            raise ValueError('Original/context/assembled section identity or order changed.')
        previous = end
    frames = evidence['frames']; previous = -math.inf
    if not frames:
        raise ValueError('Saved base frame probabilities are required.')
    for frame in frames:
        time, duration, p = frame['time'], frame['duration'], frame['probabilities']
        if (not finite(time) or not finite(duration) or time < 0 or duration <= 0 or time < previous - 1e-6
                or len(p) != 8 or any(not finite(x) or not 0 <= x <= 1 for x in p) or abs(sum(p)-1) > 1e-4):
            raise ValueError('Non-finite, overlapping or malformed frame probabilities.')
        previous = time + duration
    bpm, meter, offset = evidence['bpm'], evidence['beatsPerBar'], evidence['downbeatOffset']
    if (not finite(bpm) or bpm <= 0 or not isinstance(meter, int) or isinstance(meter, bool)
            or not 1 <= meter <= 16 or not finite(offset) or offset < 0):
        raise ValueError('Invalid app bar grid.')
    bar = 60 / bpm * meter
    if not .25 <= bar <= 30:
        raise ValueError('Unsupported app bar duration.')
    return sections, frames, bar, offset


def overlap(frame, start, end):
    return max(0., min(end, frame['time'] + frame['duration']) - max(start, frame['time']))


def anchor(frames, start, end, label, bar):
    if end - start < bar * .5:
        return dict(passed=False, reason='short_anchor')
    sums = [0.] * 8; covered = 0.
    for frame in frames:
        weight = overlap(frame, start, end); covered += weight
        for j, p in enumerate(frame['probabilities']):
            sums[j] += weight * p
    p = [value / covered for value in sums] if covered else [0.] * 8
    k = LABELS.index(label); margin = p[k] - max(p[:k] + p[k+1:])
    return dict(start=start, end=end, coverage=covered/(end-start), probability=p[k], margin=margin,
                passed=covered/(end-start) >= PARAMETERS['coverage']
                and p[k] >= PARAMETERS['anchor_probability'] and margin >= PARAMETERS['anchor_margin'])


def propose(report):
    sections, frames, bar, offset = validate(report)
    rows = []
    for index, (left, right) in enumerate(zip(sections, sections[1:])):
        if left['label'] == right['label']:
            continue
        old = left['end']
        row = dict(boundary_index=index+1, left_label=left['label'], right_label=right['label'],
                   original_seconds=old, candidate_seconds=old, status='unchanged', candidate_checks=[])
        rows.append(row)
        if abs(old-right['start']) > 1e-6 or 'silence' in (left['label'], right['label']):
            row['status'] = 'abstain_gap_or_silence'; continue
        closest = offset + math.floor((old-offset)/bar + .5) * bar
        low, high = closest-bar, closest+bar
        row['left_anchor'] = anchor(frames, max(left['start'], low-bar), low, left['label'], bar)
        row['right_anchor'] = anchor(frames, high, min(right['end'], high+bar), right['label'], bar)
        if not row['left_anchor']['passed'] or not row['right_anchor']['passed']:
            row['status'] = 'abstain_weak_anchor'; continue
        li, ri = LABELS.index(left['label']), LABELS.index(right['label'])
        for point in (low, closest, high):
            shift = point-old
            if (abs(shift) < bar*PARAMETERS['minimum_shift_bars']
                    or point-left['start'] < bar*PARAMETERS['minimum_section_bars']
                    or right['end']-point < bar*PARAMETERS['minimum_section_bars']):
                continue
            new_class, old_class = (li, ri) if shift > 0 else (ri, li)
            gain = covered = 0.
            for frame in frames:
                weight = overlap(frame, min(old, point), max(old, point)); covered += weight
                p = frame['probabilities']
                gain += weight*(math.log(max(p[new_class], 1e-8))-math.log(max(p[old_class], 1e-8)))
            mean_gain = gain/covered if covered else 0.
            row['candidate_checks'].append(dict(seconds=point, shift_seconds=shift,
                coverage=covered/abs(shift), log_gain_seconds=gain, mean_log_gain=mean_gain,
                passed=covered/abs(shift) >= PARAMETERS['coverage'] and mean_gain >= PARAMETERS['mean_log_gain']))
        passed = [c for c in row['candidate_checks'] if c['passed']]
        if passed:
            best = max(passed, key=lambda c: (c['log_gain_seconds'], -abs(c['shift_seconds'])))
            row.update(candidate_seconds=best['seconds'], status='candidate_requires_review',
                       log_gain_seconds=best['log_gain_seconds'], mean_log_gain=best['mean_log_gain'])
        else:
            row['status'] = 'abstain_weak_gain_or_short_section'
    simulated = copy.deepcopy(sections)
    for row in sorted((r for r in rows if r['status'] == 'candidate_requires_review'),
                      key=lambda r: (-r['log_gain_seconds'], r['boundary_index'])):
        i = row['boundary_index']-1; point = row['candidate_seconds']
        if (point-simulated[i]['start'] < bar*PARAMETERS['minimum_section_bars']
                or simulated[i+1]['end']-point < bar*PARAMETERS['minimum_section_bars']):
            row.update(status='abstain_neighbor_conflict', candidate_seconds=row['original_seconds'])
            continue
        simulated[i]['end'] = point; simulated[i+1]['start'] = point
    result = dict(title=report['title'], source_analysis_id=report.get('id'), bar_seconds=bar,
        context_checkpoint_sha256=report['contextEvidence']['checkpointSHA256'], boundaries=rows,
        simulated_sections=simulated, candidate_count=sum(r['status']=='candidate_requires_review' for r in rows))
    native = report.get('boundaryRefinementEvidence')
    if native:
        expected = [(r['boundary_index'],r['candidate_seconds'],r['status']=='candidate_requires_review') for r in rows]
        actual = [(r['boundaryIndex'],r['candidateSeconds'],r['applied']) for r in native['decisions']]
        same_rows = len(actual)==len(expected) and all(a[0]==b[0] and abs(a[1]-b[1])<=1e-6 and a[2]==b[2] for a,b in zip(actual,expected))
        same_sections = len(simulated)==len(native['refinedSections']) and all(a['label']==b['label'] and
            abs(a['start']-b['start'])<=1e-6 and abs(a['end']-b['end'])<=1e-6 for a,b in zip(simulated,native['refinedSections']))
        same_parameters = native['parameters']==dict(anchorProbability=PARAMETERS['anchor_probability'],
            anchorMargin=PARAMETERS['anchor_margin'],coverage=PARAMETERS['coverage'],meanLogGain=PARAMETERS['mean_log_gain'],
            minimumShiftBars=PARAMETERS['minimum_shift_bars'],minimumSectionBars=PARAMETERS['minimum_section_bars'])
        if not (same_rows and same_sections and same_parameters and native['appliedCount']==result['candidate_count']):
            raise ValueError('Native Debug boundary result does not reproduce the fixed Python rule.')
        result['native_runtime_comparison'] = dict(passed=True, applied_count=native['appliedCount'],
            scope='Recomputed from original context boundaries; not a second cumulative shift or independent accuracy measurement.')
    return result


def reference_metrics(tracks, path):
    expected = {(t['source_sha256'], str(r['boundary_index'])): (t, r)
                for t in tracks for r in t['boundaries']}
    before = []; after = []; seen = set(); touched = set()
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        for item in csv.DictReader(handle):
            key = (item['source_sha256'], item['boundary_index'])
            if key not in expected or key in seen:
                raise ValueError('Unknown, stale or duplicate manual reference boundary.')
            seen.add(key); track, row = expected[key]
            for field in ('title', 'left_label', 'right_label', 'status'):
                if item[field] != str(track['title'] if field=='title' else row[field]):
                    raise ValueError('Reference row identity changed: '+field)
            for field in ('original_seconds', 'candidate_seconds'):
                if float(item[field]) != row[field]:
                    raise ValueError('Reference proposal changed: '+field)
            if not item['reference_seconds'].strip():
                continue
            value = float(item['reference_seconds'])
            i = row['boundary_index']-1; sections = track['simulated_sections']
            if not finite(value) or not sections[i]['start'] < value < sections[i+1]['end']:
                raise ValueError('Manual reference must be inside the neighboring sections.')
            before.append(abs(row['original_seconds']-value)); after.append(abs(row['candidate_seconds']-value))
            touched.add(track['source_sha256'])
    if not before:
        raise ValueError('Reference CSV has no manually filled reference_seconds.')
    def summary(values):
        return dict(mae_seconds=sum(values)/len(values), max_error_seconds=max(values),
                    within_0_5_seconds=sum(v<=.5 for v in values)/len(values),
                    within_1_second=sum(v<=1 for v in values)/len(values))
    return dict(reviewed_boundaries=len(before), reviewed_reports=len(touched), baseline=summary(before),
        candidate=summary(after), improved=sum(a<b-1e-9 for a,b in zip(after,before)),
        worsened=sum(a>b+1e-9 for a,b in zip(after,before)),
        note='Selected manual spot checks only, not an independent dataset-wide boundary accuracy estimate.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, action='append', default=[])
    p.add_argument('--reports-dir', type=Path)
    p.add_argument('--reference-csv', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    sources = list(args.report)
    if args.reports_dir:
        sources += sorted(args.reports_dir.expanduser().glob('analysis-*.json'))
    if not sources:
        p.error('Supply --report PATH or --reports-dir DIRECTORY containing analysis-*.json.')
    tracks = []; skipped = []; seen = set()
    for path in sources:
        path = path.expanduser().resolve(); raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            continue
        seen.add(sha); report = json.loads(raw)
        if report.get('state') != 'completed' or not report.get('contextEvidence'):
            skipped.append(dict(path=str(path), reason='No completed context evidence')); continue
        track = propose(report); track.update(source_path=str(path), source_sha256=sha); tracks.append(track)
    if not tracks:
        raise ValueError('No completed context app reports found. Export a new context-enabled analysis JSON.')
    result = dict(protocol='context-boundary-shift-review-v1', parameters=PARAMETERS,
        reports=len(tracks), unique_titles=len({t['title'] for t in tracks}),
        candidates=sum(t['candidate_count'] for t in tracks), tracks=tracks, skipped_reports=skipped,
        maps_changed=False, manual_labels_changed=False, encoder_executed=False, training_started=False,
        boundary_accuracy_validated=False, ready_for_app_integration=False,
        limitations=['Experimental review defaults; no threshold fitting or tuning is performed.',
            'Only adjacent different-label boundaries move; labels and section count remain fixed.',
            'Uniform app bar clock may drift from observed beats. No independent beat correction is made.',
            'A proposed shift is evidence for review, not calibrated confidence or a new instrumental label.',
            'No context/model rerun on changed geometry. No new missing boundaries or large outro shifts are inserted.'])
    if args.reference_csv:
        result['manual_spot_check'] = reference_metrics(tracks, args.reference_csv.expanduser())
        result['reference_csv_sha256'] = hashlib.sha256(args.reference_csv.expanduser().read_bytes()).hexdigest()
    out = args.output_dir.expanduser(); out.mkdir(parents=True, exist_ok=False)
    (out/'review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (out/'boundary_review.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for track in tracks:
            for row in track['boundaries']:
                writer.writerow({**{k:row[k] for k in FIELDS if k in row}, 'source_sha256':track['source_sha256'], 'title':track['title']})
    for track in tracks:
        for row in track['boundaries']:
            if row['status']=='candidate_requires_review':
                print(f'{track["title"]}: {row["left_label"]} -> {row["right_label"]}: '
                      f'{row["original_seconds"]:.3f}s -> {row["candidate_seconds"]:.3f}s', flush=True)
    print(f'{len(tracks)} reports, {result["unique_titles"]} titles, {result["candidates"]} candidates. Saved: {out/"review.json"}')


if __name__ == '__main__':
    main()
