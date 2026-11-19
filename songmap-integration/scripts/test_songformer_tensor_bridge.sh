#!/bin/bash
set -eu
bridge_root="$(cd "$(dirname "$0")/.." && pwd)"
bridge_temp="$(mktemp -d "${TMPDIR:-/tmp}/songformer-bridge.XXXXXX")"
trap 'rm -rf "$bridge_temp"' EXIT
xcrun swiftc -O \
    "$bridge_root/SongMap/Data/SongAnalysis/Structure/SongFormerTensorBridge.swift" \
    "$bridge_root/scripts/tests/SongFormerTensorBridgeChecks.swift" \
    -o "$bridge_temp/checks"
"$bridge_temp/checks"
