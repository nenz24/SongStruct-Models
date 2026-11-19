#!/usr/bin/env python3
"""Compare in-app MusicUnderstanding boundary audits with the dataset's Mac export.

Input: analysis diagnostics JSON files from a Debug run with `--musicunderstanding-boundary-audit`
(Library → ⋯ → Analysis reports → Share report), and the dataset boundary exports
(Section-Annotations/outputs.local/apple/**.json, written by export_apple_boundaries.py on the Mac).

For every configuration the report gives the section count, one-to-one boundary hit rates within
0.5 s and 3 s of the Mac export (precision/recall/F), the mean absolute shift of matched boundaries,
and run-to-run agreement between the two identical exporter-style passes. A verdict per song says
which factor explains the difference: the app's call (fixable in the app), run-to-run variation, or
the device (iOS vs macOS model), which the app cannot change.
"""
import argparse
import json
from pathlib import Path

TOLERANCES = (0.5, 3.0)
APP = 'app-combined-default-timing'
EXPORTER = 'structure-only-precise-timing'
REPEAT = 'structure-only-precise-timing-repeat'
DEFAULT_TIMING = 'structure-only-default-timing'


def boundaries(sections, duration, edge=0.05):
    """Interior section boundaries: every start/end except the song edges, de-duplicated within `edge`."""
    points = sorted(float(t) for s in sections for t in (s['start'], s['end']))
    result = []
    for t in points:
        if edge < t < duration - edge and (not result or t - result[-1] > edge):
            result.append(t)
    return result


def match(reference, estimate, tolerance):
    """One-to-one matching by increasing distance (as mir_eval hit rate); returns pairs."""
    pairs = sorted((abs(r - e), i, j) for i, r in enumerate(reference) for j, e in enumerate(estimate)
                   if abs(r - e) <= tolerance)
    used_r, used_e, matched = set(), set(), []
    for distance, i, j in pairs:
        if i not in used_r and j not in used_e:
            used_r.add(i); used_e.add(j); matched.append((reference[i], estimate[j]))
    return matched


def scores(reference, estimate):
    result = {'reference_boundaries': len(reference), 'estimated_boundaries': len(estimate)}
    for tolerance in TOLERANCES:
        matched = match(reference, estimate, tolerance)
        precision = len(matched) / len(estimate) if estimate else float(not reference)
        recall = len(matched) / len(reference) if reference else float(not estimate)
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        result[f'hit_{tolerance:g}s'] = {'precision': precision, 'recall': recall, 'f': f,
                                         'mean_abs_shift': (sum(abs(r - e) for r, e in matched) / len(matched)) if matched else None}
    return result


def identical(a, b, tolerance=0.05):
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))


def verdict(runs):
    """Which factor explains the app/Mac difference for one song."""
    app, exporter, repeat = runs.get(APP), runs.get(EXPORTER), runs.get(REPEAT)
    if exporter is None:
        return 'incomplete: exporter-style run missing'
    if repeat is not None and not identical(exporter['boundaries'], repeat['boundaries']):
        return 'run-to-run variation: identical exporter-style calls differ on this device'
    exporter_f = exporter['vs_mac']['hit_0.5s']['f']
    app_f = app['vs_mac']['hit_0.5s']['f'] if app else None
    if exporter_f >= 0.99:
        if app_f is not None and app_f < 0.99:
            return 'app call: exporter-style call reproduces the Mac export; the app call does not (fixable in the app)'
        return 'consistent: app and exporter-style calls reproduce the Mac export'
    if app_f is not None and exporter_f > app_f + 1e-9:
        return 'partly app call, partly device: exporter-style is closer to Mac but still differs'
    return 'device: even the exporter-style call differs from the Mac export (iOS vs macOS model; not fixable by the call)'


def missing_audit_hint(report):
    """Why a report has no audit: old build, launch argument missing, or an interrupted audit."""
    events = report.get('events', [])
    names = [e.get('name') for e in events]
    stages = [e['name'] for e in events if e.get('kind') == 'stage']
    if 'music_understanding_boundary_audit' in names:
        return 'the audit stage started but did not finish (analysis interrupted?); re-run the analysis'
    muq = 'songformer.muq.start' in names
    return (f"started {report.get('startedAt')}; stages: {', '.join(stages) or 'none'}; MuQ path: {'yes' if muq else 'no'}. "
            'No audit stage: the build did not include the audit or --musicunderstanding-boundary-audit was not set '
            'for this run. Rebuild after git pull, check the Run arguments of the scheme you launch, then analyse again.')


def load_mac_exports(folder):
    exports = {}
    for path in Path(folder).expanduser().rglob('*.json'):
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get('boundary_source') == 'apple_musicunderstanding.sections':
            exports[Path(document['audio_path']).stem] = document
    return exports


def compare(reports, exports):
    results = []
    for path in reports:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        audit = report.get('boundaryAuditEvidence')
        title = report.get('title')
        if not audit:
            results.append({'title': title, 'report': str(path), 'verdict': 'no boundary audit (run with --musicunderstanding-boundary-audit)',
                            'hint': missing_audit_hint(report)})
            continue
        mac = exports.get(title)
        if mac is None:
            results.append({'title': title, 'report': str(path), 'verdict': 'no Mac export with this audio file name'})
            continue
        duration = float(mac['duration'])
        reference = boundaries(mac['sections'], duration)
        runs = {}
        for run in audit['runs']:
            if run.get('error'):
                runs[run['configuration']] = {'error': run['error']}
                continue
            estimate = boundaries(run['sections'], duration)
            runs[run['configuration']] = {'sections': len(run['sections']), 'seconds': run['seconds'],
                                          'boundaries': estimate, 'vs_mac': scores(reference, estimate)}
        ok = {k: v for k, v in runs.items() if 'error' not in v}
        results.append({'title': title, 'report': str(path), 'mac_sections': len(mac['sections']),
                        'mac_os': mac.get('os_version'), 'runs': runs, 'verdict': verdict(ok)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path, help='Analysis diagnostics JSON files (or folders)')
    parser.add_argument('--mac-exports', type=Path, default=Path(__file__).resolve().parents[2] / 'Section-Annotations/outputs.local/apple')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    files = [f for p in args.reports for f in (sorted(p.glob('analysis-*.json')) if p.is_dir() else [p])]
    exports = load_mac_exports(args.mac_exports)
    if not exports:
        parser.error(f'No Mac boundary exports found under {args.mac_exports}')
    results = compare(files, exports)
    for item in results:
        print(f"\n== {item['title']}: {item['verdict']}")
        if item.get('hint'):
            print('  ' + item['hint'])
        for name, run in item.get('runs', {}).items():
            if 'error' in run:
                print(f'  {name:<38} error: {run["error"]}')
                continue
            h05, h3 = run['vs_mac']['hit_0.5s'], run['vs_mac']['hit_3s']
            shift = h3['mean_abs_shift']
            print(f"  {name:<38} sections {run['sections']:>3} (Mac {item['mac_sections']})  "
                  f"F@0.5s {h05['f']:.2f}  F@3s {h3['f']:.2f}  mean shift {'-' if shift is None else f'{shift:.2f}s'}  "
                  f"{run['seconds']:.1f}s")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print('\nReport:', args.output)


if __name__ == '__main__':
    main()
