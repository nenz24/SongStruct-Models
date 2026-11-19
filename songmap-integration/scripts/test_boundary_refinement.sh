#!/bin/bash
set -eu
boundary_root="$(cd "$(dirname "$0")/.." && pwd)"
boundary_python="${BOUNDARY_REFINEMENT_PYTHON:-python3}"
boundary_temp="$(mktemp -d "${TMPDIR:-/tmp}/songformer-boundary.XXXXXX")"
trap 'rm -rf "$boundary_temp"' EXIT
"$boundary_python" "$boundary_root/scripts/generate_boundary_shift_fixture.py" --output "$boundary_temp/reference.json"
xcrun swiftc -O -swift-version 5 -default-isolation MainActor \
    "$boundary_root/SongMap/Data/SongAnalysis/Structure/SongFormerBoundaryRefiner.swift" \
    "$boundary_root/scripts/tests/SongFormerBoundaryRefinerChecks.swift" \
    -o "$boundary_temp/checks"
"$boundary_temp/checks" "$boundary_temp/reference.json"
