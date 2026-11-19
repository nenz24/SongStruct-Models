#!/bin/bash
set -eu
context_root="$(cd "$(dirname "$0")/.." && pwd)"
context_python="${SECTION_CONTEXT_PYTHON:-$context_root/../Section-Annotations/.venv-songformer311/bin/python}"
context_temp="$(mktemp -d "${TMPDIR:-/tmp}/songformer-context.XXXXXX")"
trap 'rm -rf "$context_temp"' EXIT
"$context_python" "$context_root/scripts/generate_context_math_fixture.py" --output-dir "$context_temp/fixture"
xcrun swiftc -O -swift-version 5 -default-isolation MainActor \
    "$context_root/SongMap/Data/SongAnalysis/Structure/SongFormerContextMath.swift" \
    "$context_root/SongMap/Data/SongAnalysis/Structure/SongFormerBoundaryRefiner.swift" \
    "$context_root/SongMap/Data/SongAnalysis/Structure/SongFormerSectionContext.swift" \
    "$context_root/SongMap/Data/SongAnalysis/Structure/SongFormerTensorBridge.swift" \
    "$context_root/SongMap/Data/SongAnalysis/SongAnalysisDiagnostics.swift" \
    "$context_root/scripts/tests/SongFormerSectionContextChecks.swift" \
    -o "$context_temp/checks"
"$context_temp/checks" "$context_temp/fixture"
