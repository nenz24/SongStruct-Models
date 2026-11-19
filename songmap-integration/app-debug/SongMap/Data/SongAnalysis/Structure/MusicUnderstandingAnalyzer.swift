//
//  MusicUnderstandingAnalyzer.swift
//  SongMap
//

import AVFoundation
import CoreMedia
import Foundation
import MusicUnderstanding

/// What Apple's MusicUnderstanding hears in a song. Times are in seconds.
nonisolated struct MusicUnderstandingResult: Sendable {
    /// Section boundaries only; SongFormer names them.
    let sections: [ClosedRange<Double>]
    let beats: [Double]
    let downbeats: [Double]
    let tempoBPM: Double?
    /// Sorted by start; each region lasts until the next.
    let keys: [KeyRegion]
    /// Where someone is singing: empty for an instrumental, nil when vocal activity couldn't be analysed.
    let vocalRanges: [ClosedRange<Double>]?
    /// Mean activity score (0...1) for the instruments exposed by MusicUnderstanding.
    let instrumentActivity: [String: Float]
    /// The same pass's activity ranges with its own rhythm, as instrument detection needs them.
    var mixActivity: InstrumentActivityAnalysis?
}

/// Just where instruments play, for instrument detection. Times are in seconds.
nonisolated struct InstrumentActivityAnalysis: Sendable, Codable {
    /// nil when instrument activity couldn't be analysed.
    let ranges: MusicActivityRanges?
    /// Empty unless rhythm was requested.
    let beats: [Double]
    let tempoBPM: Double?
}

/// MusicUnderstanding's four activity classes as plain time ranges (seconds, sorted by start).
nonisolated struct MusicActivityRanges: Sendable, Equatable, Codable {
    let vocal: [ClosedRange<Double>]
    let drums: [ClosedRange<Double>]
    let bass: [ClosedRange<Double>]
    /// Everything that isn't vocal, drums or bass (guitars, keys, strings, synths, ...).
    let other: [ClosedRange<Double>]
}

actor MusicUnderstandingAnalyzer {
    func analyze(fileURL: URL) async throws -> MusicUnderstandingResult {
        let result: MusicUnderstandingSession.SessionResult
        do {
            result = try await Self.run(fileURL, for: [.structure, .rhythm, .key, .instrumentActivity])
        } catch {
            // Vocal activity only improves lyrics; don't let it take the rest of the analysis down.
            try Task.checkCancellation()
            result = try await Self.run(fileURL, for: [.structure, .rhythm, .key])
        }

        let rhythm = result.rhythm
        return MusicUnderstandingResult(
            sections: (result.structure?.sections ?? []).compactMap(Self.seconds),
            beats: rhythm?.beats.compactMap(Self.seconds) ?? [],
            downbeats: rhythm?.bars.compactMap(Self.seconds) ?? [],
            tempoBPM: rhythm?.beatsPerMinute.map(Double.init),
            keys: Self.keyRegions(result.key),
            vocalRanges: result.instrumentActivity.map { ($0.ranges[.vocal] ?? []).compactMap(Self.seconds) },
            instrumentActivity: Self.activityScores(result.instrumentActivity),
            mixActivity: result.instrumentActivity.map {
                InstrumentActivityAnalysis(
                    ranges: Self.activityRanges($0),
                    beats: rhythm?.beats.compactMap(Self.seconds) ?? [],
                    tempoBPM: rhythm?.beatsPerMinute.map(Double.init)
                )
            }
        )
    }

    /// Structure only, as the dataset exporter (SongStruct-Models apple/ExportBoundaries.swift) runs it:
    /// `[.structure]` alone and, when `preciseTiming`, AVURLAssetPreferPreciseDurationAndTimingKey. Debug audit only.
    func structureSections(fileURL: URL, preciseTiming: Bool) async throws -> [ClosedRange<Double>] {
        let asset = preciseTiming
            ? AVURLAsset(url: fileURL, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
            : AVURLAsset(url: fileURL)
        let session = try await MusicUnderstandingSession(asset: asset)
        let result = try await withTaskCancellationHandler {
            try await session.analyze(for: [.structure])
        } onCancel: {
            Task { await session.cancel() }
        }
        return (result.structure?.sections ?? []).compactMap(Self.seconds)
    }

    /// Instrument activity, plus beats and tempo when `includeRhythm`. Skips structure and key, so it's
    /// much quicker than `analyze`; a separated stem only needs the activity.
    func instrumentActivity(fileURL: URL, includeRhythm: Bool) async throws -> InstrumentActivityAnalysis {
        let result = try await Self.run(fileURL, for: includeRhythm ? [.rhythm, .instrumentActivity] : [.instrumentActivity])
        return InstrumentActivityAnalysis(
            ranges: result.instrumentActivity.map(Self.activityRanges),
            beats: result.rhythm?.beats.compactMap(Self.seconds) ?? [],
            tempoBPM: result.rhythm?.beatsPerMinute.map(Double.init)
        )
    }

    private static func activityRanges(_ result: InstrumentActivityResult) -> MusicActivityRanges {
        func ranges(_ instrument: InstrumentActivityResult.Instrument) -> [ClosedRange<Double>] {
            (result.ranges[instrument] ?? []).compactMap(seconds).sorted { $0.lowerBound < $1.lowerBound }
        }
        return MusicActivityRanges(vocal: ranges(.vocal), drums: ranges(.drum), bass: ranges(.bass), other: ranges(.other))
    }

    private static func run(
        _ fileURL: URL,
        for types: Set<AnalysisType>
    ) async throws -> MusicUnderstandingSession.SessionResult {
        let session = try await MusicUnderstandingSession(asset: AVURLAsset(url: fileURL))
        return try await withTaskCancellationHandler {
            try await session.analyze(for: types)
        } onCancel: {
            Task { await session.cancel() }
        }
    }

    /// Key changes in time order, with repeats of the same key merged.
    private static func keyRegions(_ result: KeyResult?) -> [KeyRegion] {
        let ranges = (result?.ranges ?? []).sorted { CMTimeGetSeconds($0.range.start) < CMTimeGetSeconds($1.range.start) }
        var regions: [KeyRegion] = []
        for range in ranges {
            let start = CMTimeGetSeconds(range.range.start)
            guard start.isFinite else { continue }
            let region = KeyRegion(
                start: max(0, start),
                tonic: pitchClass(of: range.value.tonic),
                isMinor: range.value.mode == .minor
            )
            if let last = regions.last, last.tonic == region.tonic, last.isMinor == region.isMinor { continue }
            regions.append(region)
        }
        return regions
    }

    private static func seconds(_ time: CMTime) -> Double? {
        let value = CMTimeGetSeconds(time)
        return value.isFinite ? value : nil
    }

    private static func seconds(_ range: CMTimeRange) -> ClosedRange<Double>? {
        guard let start = seconds(range.start), let end = seconds(range.end), end > start else { return nil }
        return start...end
    }

    private static func activityScores(
        _ result: InstrumentActivityResult?
    ) -> [String: Float] {
        guard let result else { return [:] }
        return result.activity.reduce(into: [:]) { scores, entry in
            let values = entry.value.map(\.value)
            guard !values.isEmpty else { return }
            scores[entry.key.rawValue] = values.reduce(0, +) / Float(values.count)
        }
    }

    private static func pitchClass(of tonic: KeyResult.Tonic) -> Int {
        switch tonic {
        case .c: return 0
        case .cSharp, .dFlat: return 1
        case .d: return 2
        case .dSharp, .eFlat: return 3
        case .e: return 4
        case .f: return 5
        case .fSharp, .gFlat: return 6
        case .g: return 7
        case .gSharp, .aFlat: return 8
        case .a: return 9
        case .aSharp, .bFlat: return 10
        case .b: return 11
        }
    }
}
