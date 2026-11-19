#!/bin/bash
# Typecheck the MuQ-only app pipeline against the real context math, tensor bridge, diagnostics,
# model locator (asset packs) and progress sources for iOS 27 (the app's deployment target).
# FrameLabel/LabeledSection come from minimal copies of their declarations so
# the large analyzer/labeler files and their dependencies are not needed. The full app build in Xcode
# remains the definitive compile.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
temp="$(mktemp -d "${TMPDIR:-/tmp}/songformer-muq.XXXXXX")"
trap 'rm -rf "$temp"' EXIT
cat > "$temp/Stubs.swift" <<'SWIFT'
import Foundation
actor SongFormerAnalyzer {
    nonisolated struct FrameLabel: Sendable, Equatable {
        let time: Double
        let label: String
        let probabilities: [Double]?
        let duration: Double
        init(time: Double, label: String, probabilities: [Double]? = nil, duration: Double = 0.12) {
            self.time = time; self.label = label; self.probabilities = probabilities; self.duration = duration
        }
    }
}
nonisolated enum SectionLabeler {
    nonisolated struct LabeledSection: Sendable, Equatable { var label: String; var start: Double; var end: Double }
}
SWIFT
sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
structure="$root/SongMap/Data/SongAnalysis/Structure"
xcrun swiftc -typecheck -swift-version 5 -default-isolation MainActor -target arm64-apple-ios27.0 -sdk "$sdk" \
    "$temp/Stubs.swift" \
    "$structure/SongFormerMuQPipeline.swift" \
    "$structure/SongFormerContextMath.swift" \
    "$structure/SongFormerTensorBridge.swift" \
    "$structure/SongFormerBoundaryRefiner.swift" \
    "$root/SongMap/Data/SongAnalysis/SongAnalysisDiagnostics.swift" \
    "$root/SongMap/Data/SongAnalysis/ModelLocator.swift" \
    "$root/SongMap/Data/SongAnalysis/ModelStore.swift" \
    "$root/SongMap/Data/SongAnalysis/AnalysisProgress.swift"
echo "MuQ-only pipeline typecheck passed (iOS SDK)."
