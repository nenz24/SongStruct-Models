//
//  OnDeviceSongAnalysisProvider.swift
//  SongMap
//

import Foundation
import OSLog

enum SongAnalysisError: LocalizedError {
    case audioUnavailable

    var errorDescription: String? {
        switch self {
        case .audioUnavailable: return "The audio file for this song isn't on this device yet."
        }
    }
}

/// Analyzes a song entirely on device, one stage after another so only one stage's models are in memory:
///  1. MusicUnderstanding (Apple): beats, bars, tempo, key, section boundaries and where the vocals are
///     Validated Beat This! FP32 resources, when installed, replace only the beat/bar timestamps.
///  2. SongFormer: what each section is (intro, verse, chorus, ...)
///  3. BTC: chords
///  4. Whisper / SpeechTranscriber: the sung language and the lyrics
/// Sections and rhythm are required; chords and lyrics are best effort, so a failure there still
/// gives a SongMap, just without them.
final class OnDeviceSongAnalysisProvider: SongAnalysisProvider {
    private let documentsURL: URL
    private let logger = Logger(subsystem: "com.salamtoleransi.SongMap", category: "SongAnalysis")

    init(documentsURL: URL = URL.documentsDirectory) {
        self.documentsURL = documentsURL
    }

    /// The structure and lyrics models can ship in Apple-hosted asset packs instead of the app.
    var needsModelDownload: Bool {
        !ModelLocator.packsToPrepare(for: ModelResource.songAnalysis).isEmpty
    }

    func prepareModels(progress: @escaping @Sendable (Double) -> Void) async throws {
        try await ModelLocator.prepare(ModelResource.songAnalysis, progress: progress)
    }

    #if DEBUG
    /// Debug only and recording only: re-runs MusicUnderstanding the way the dataset exporter did, twice,
    /// plus structure-only with default timing, so app/Mac boundary differences can be attributed.
    private static func auditBoundaries(audioURL: URL, appSections: [ClosedRange<Double>], appSeconds: Double,
                                        diagnostics: SongAnalysisDiagnostics) async {
        typealias Audit = SongAnalysisDiagnostics.BoundaryAuditEvidence
        func intervals(_ ranges: [ClosedRange<Double>]) -> [SongAnalysisDiagnostics.StructureEvidence.Interval] {
            ranges.map { SongAnalysisDiagnostics.StructureEvidence.Interval(start: $0.lowerBound, end: $0.upperBound) }
        }
        let analyzer = MusicUnderstandingAnalyzer()
        var runs = [Audit.Run(configuration: "app-combined-default-timing", seconds: appSeconds, sections: intervals(appSections), error: nil)]
        let variants: [(String, Bool)] = [("structure-only-precise-timing", true), ("structure-only-precise-timing-repeat", true),
                                          ("structure-only-default-timing", false)]
        for (name, precise) in variants {
            if Task.isCancelled { break }
            let start = ProcessInfo.processInfo.systemUptime
            do {
                let sections = try await analyzer.structureSections(fileURL: audioURL, preciseTiming: precise)
                runs.append(Audit.Run(configuration: name, seconds: ProcessInfo.processInfo.systemUptime - start,
                                      sections: intervals(sections), error: nil))
            } catch {
                runs.append(Audit.Run(configuration: name, seconds: ProcessInfo.processInfo.systemUptime - start,
                                      sections: [], error: error.localizedDescription))
            }
        }
        diagnostics.boundaryAudit(Audit(runs: runs))
    }
    #endif

    func analyze(song: Song, progress: @escaping @Sendable (Double) -> Void) async throws -> DetectedSongAnalysis {
        let diagnostics = SongAnalysisDiagnostics(songID:song.id.uuidString,title:song.title ?? "Untitled")
        do {
        let audioURL = documentsURL.appending(path: song.audioPath)
        guard !song.audioPath.isEmpty, FileManager.default.fileExists(atPath: audioURL.path()) else {
            throw SongAnalysisError.audioUnavailable
        }
        // The generation queue downloads the models first (showing it on the song); this is a
        // no-op then, and covers any other caller. Without it, Beat This! would quietly be skipped.
        try await prepareModels { _ in }
        let metadataStart = diagnostics.begin("audio_metadata")
        let duration = try AudioDecoder.duration(of: audioURL)
        diagnostics.end("audio_metadata",since:metadataStart)
        diagnostics.event("audio.duration_seconds",detail:String(duration))

        // Rough share of the total time each stage takes (measured on the simulator; Whisper is faster
        // on a device, where it runs on the GPU).
        let rhythmProgress = ProgressSpan(start: 0, length: 0.08, report: progress)
        let sectionProgress = ProgressSpan(start: 0.08, length: 0.47, report: progress)
        let chordProgress = ProgressSpan(start: 0.55, length: 0.05, report: progress)
        let lyricsProgress = ProgressSpan(start: 0.6, length: 0.4, report: progress)

        rhythmProgress(0)
        let musicStart = diagnostics.begin("music_understanding")
        var music = try await MusicUnderstandingAnalyzer().analyze(fileURL: audioURL)
        let musicSeconds = ProcessInfo.processInfo.systemUptime - musicStart
        diagnostics.end("music_understanding",since:musicStart)
        #if DEBUG
        if SongAnalysisDiagnostics.launchArgument("musicunderstanding-boundary-audit") {
            let auditStart = diagnostics.begin("music_understanding_boundary_audit")
            await Self.auditBoundaries(audioURL: audioURL, appSections: music.sections, appSeconds: musicSeconds, diagnostics: diagnostics)
            diagnostics.end("music_understanding_boundary_audit",since:auditStart)
        }
        #endif
        if let mixActivity = music.mixActivity {
            // Instrument detection on the SongMap screen starts from this pass instead of repeating it.
            do {
                try InstrumentAnalysisStore.saveMixActivity(mixActivity, forAudioAt: audioURL)
            } catch {
                logger.error("Couldn't save instrument activity: \(error.localizedDescription, privacy: .public)")
            }
        }
        rhythmProgress(0.4)
        // Decoded once; every stage below derives the rate and channels its model needs.
        let decodeStart = diagnostics.begin("audio_decode_native")
        var audio = try AudioDecoder.loadNative(url: audioURL)
        diagnostics.end("audio_decode_native",since:decodeStart)
        let rhythmID = song.id.uuidString
        let title = song.title ?? "Untitled"
        let reportedTempo = music.tempoBPM.map { String($0) } ?? "unavailable"
        let beatPreview = String(describing: Array(music.beats.prefix(8)))
        let barPreview = String(describing: Array(music.downbeats.prefix(8)))
        logger.notice("Rhythm raw [\(rhythmID, privacy: .public)] \(title, privacy: .public), duration \(duration, privacy: .public)s: MusicUnderstanding tempo \(reportedTempo, privacy: .public) BPM; \(music.beats.count, privacy: .public) beats, \(music.downbeats.count, privacy: .public) bars; first beat times \(beatPreview, privacy: .public); first bar times \(barPreview, privacy: .public)")

        var rhythmDetector = "MusicUnderstanding"
        var primaryRhythm: BeatGrid.Resolution? = nil
        let beatStart = diagnostics.begin("beat_this_optional")
        if BeatThisAnalyzer.isAvailable {
            do {
                let result = try await BeatThisAnalyzer.analyze(audio: audio) { rhythmProgress(0.4 + $0 * 0.6) }
                let learned = BeatGrid.fromBeatThis(beats: result.beats, downbeats: result.downbeats)
                let tempo = learned.grid.bpm
                primaryRhythm = learned
                music = MusicUnderstandingResult(sections: music.sections, beats: result.beats,
                    downbeats: result.downbeats, tempoBPM: tempo, keys: music.keys,
                    vocalRanges: music.vocalRanges, instrumentActivity: music.instrumentActivity)
                rhythmDetector = "BeatThisW8FP32"
                let footprint = result.pointSampledPeakFootprintMiB.map { String(format: "%.1f", $0) } ?? "unavailable"
                let loadTime = result.modelLoadSeconds.map { String(format: "%.3f", $0) } ?? "unavailable"
                logger.notice("Rhythm BeatThis runtime [\(rhythmID, privacy: .public)]: detector=\(rhythmDetector, privacy: .public); storedWeightBits=8; compute=FP32; default=true; load=\(loadTime, privacy: .public)s; pointSampledPeakFootprintMiB=\(footprint, privacy: .public); continuousPeak=false; processWide=true")
                logger.notice("Rhythm BeatThis [\(rhythmID, privacy: .public)] \(title, privacy: .public): CPU_ONLY; \(result.frameCount, privacy: .public) frames, \(result.beats.count, privacy: .public) beats, \(result.downbeats.count, privacy: .public) downbeats; elapsed=\(result.elapsedSeconds, privacy: .public)s; first beats=\(String(describing: Array(result.beats.prefix(8))), privacy: .public); first bars=\(String(describing: Array(result.downbeats.prefix(8))), privacy: .public)")
            } catch {
                try Task.checkCancellation()
                diagnostics.event("beat_this_fallback",detail:error.localizedDescription)
                logger.error("Rhythm BeatThis fallback [\(rhythmID, privacy: .public)]: \(error.localizedDescription, privacy: .public)")
            }
        } else {
            logger.error("Rhythm BeatThis unavailable: default W8 model/metadata/frontend resources are incomplete")
        }
        diagnostics.end("beat_this_optional",since:beatStart)
        rhythmProgress(1)

        let songformerStart = diagnostics.begin("songformer")
        let structure = try await SongFormerAnalyzer().analyze(audio: audio, boundaries: music.sections, duration: duration,
                                                              progress: sectionProgress, diagnostics: diagnostics)
        let frameLabels = structure.frameLabels
        diagnostics.end("songformer",since:songformerStart)
        let chordStart = diagnostics.begin("chords")
        let chordStage = try await detectChords(in: audio, keys: music.keys, downbeats: music.downbeats,
                                               primaryBeats: primaryRhythm == nil ? nil : music.beats, progress: chordProgress, diagnostics:diagnostics)
        diagnostics.end("chords",since:chordStart)
        let chords = chordStage?.chords
        let audioRhythm = chordStage?.rhythm
        // The independent tracker is used only on the fallback path. A successful
        // Beat This! result supplies both the grid and chord-segmentation beats.
        let trackerBeats = primaryRhythm == nil ? (audioRhythm?.beats ?? chords?.beats ?? []) : []
        let independentTempo: String
        if trackerBeats.count >= 8 {
            let independent = BeatGrid.fit(beats: trackerBeats, downbeats: [], tempoBPM: nil)
            independentTempo = String(independent.bpm)
        } else {
            independentTempo = "unavailable"
        }
        let rhythm = SongAnalysisAssembler.rhythm(music: music, chords: chords, audioRhythm: audioRhythm,
                                                  primaryRhythm: primaryRhythm)
        let grid = rhythm.grid
        let support = primaryRhythm != nil ? "BeatThis primary; legacy tracker bypassed" : audioRhythm?.evidence.map {
            "correlation=\($0.correlation), half-time=\($0.halfTimeCorrelation), windows=\($0.supportingWindows)/\($0.analyzedWindows)"
        } ?? "unavailable"
        let meterSupport: String
        if rhythm.source == .beatThis {
            meterSupport = rhythm.selectionDetails
        } else if grid.beatUnit == 8, let evidence = rhythm.meterEvidence {
            meterSupport = evidence.supportDescription
        } else {
            meterSupport = "bar timing; no compatible confident compound evidence"
        }
        let meterStatus = rhythm.meterNeedsReview ? "needsReview" : "estimated"
        let meterCandidates = rhythm.meterDiagnostics.isEmpty ? "audio meter analysis unavailable" : rhythm.meterDiagnostics
        if !rhythm.selectionDetails.isEmpty {
            logger.notice("Rhythm timing validation [\(rhythmID, privacy: .public)] \(title, privacy: .public): \(rhythm.selectionDetails, privacy: .public)")
        }
        logger.notice("Rhythm grid anchor [\(rhythmID, privacy: .public)] \(title, privacy: .public): downbeatOffset=\(grid.downbeatOffset, privacy: .public)s; barDuration=\(grid.barDuration, privacy: .public)s; firstObservedBar=\(music.downbeats.first ?? -1, privacy: .public)s")
        let detectorTempo = music.tempoBPM.map { String($0) } ?? "unavailable"
        logger.notice("Rhythm selected [\(rhythmID, privacy: .public)] \(title, privacy: .public): detector=\(rhythmDetector, privacy: .public), tempo \(detectorTempo, privacy: .public) BPM; independent audio tracker \(independentTempo, privacy: .public) BPM (\(trackerBeats.count, privacy: .public) beats); SongMap grid \(grid.bpm, privacy: .public) BPM per 1/\(grid.beatUnit, privacy: .public), meter \(grid.beatsPerBar, privacy: .public)/\(grid.beatUnit, privacy: .public); source=\(rhythm.source.rawValue, privacy: .public); support=\(support, privacy: .public); meterSupport=\(meterSupport, privacy: .public); meterStatus=\(meterStatus, privacy: .public)")
        logger.notice("Rhythm meter candidates [\(rhythmID, privacy: .public)] \(title, privacy: .public): \(meterCandidates, privacy: .public)")
        // Lyrics separation wants 44.1 kHz stereo. Keep only that from here on, so the native decode
        // is released before UVR, the stage that needs the most memory.
        let lyricsInputStart = diagnostics.begin("lyrics_audio_conversion")
        audio = DecodedAudio(
            channels: try AudioDecoder.convert(audio, sampleRate: MDXSeparator.sampleRate, maxChannels: 2),
            sampleRate: MDXSeparator.sampleRate
        )
        diagnostics.end("lyrics_audio_conversion",since:lyricsInputStart)
        let lyricsStart = diagnostics.begin("lyrics")
        let lyrics = try await recognizeLyrics(
            in: audioURL,
            audio: audio,
            vocalRanges: music.vocalRanges,
            duration: duration,
            progress: lyricsProgress, diagnostics:diagnostics
        )

        diagnostics.end("lyrics",since:lyricsStart)
        var finalSections = structure.contextSections
        #if DEBUG
        if structure.boundaryRefinementEnabled {
            let refinementStart = diagnostics.begin("section_boundary_refinement")
            guard let originalSections = structure.contextSections else {
                throw SongFormerBoundaryRefiner.RefinementError.invalid("The context labels are required.")
            }
            let refinementFrames = try frameLabels.map { frame -> SongFormerBoundaryRefiner.Frame in
                guard let p = frame.probabilities else {
                    throw SongFormerBoundaryRefiner.RefinementError.invalid("Missing base frame probabilities.")
                }
                return .init(time: frame.time, duration: frame.duration, probabilities: p)
            }
            let refinement = try SongFormerBoundaryRefiner.refine(
                sections: originalSections.map { .init(start: $0.start, end: $0.end, label: $0.label) },
                frames: refinementFrames,
                grid: .init(bpm: grid.bpm, beatsPerBar: grid.beatsPerBar, downbeatOffset: min(grid.downbeatOffset, duration)))
            finalSections = refinement.refinedSections.map { .init(label: $0.label, start: $0.start, end: $0.end) }
            diagnostics.boundaryRefinement(refinement)
            diagnostics.end("section_boundary_refinement",since:refinementStart)
            NSLog("WORSHIP BOUNDARY_REFINEMENT_DONE debug-opt-in changed=%ld", refinement.appliedCount)
        }
        #endif
        let assemblyStart = diagnostics.begin("assemble_and_section_pooling")
        let analysis = SongAnalysisAssembler(
            duration: duration,
            music: music,
            frameLabels: frameLabels,
            chords: chords,
            audioRhythm: audioRhythm,
            lyrics: lyrics
        ).makeAnalysis(resolvedRhythm: rhythm, sectionsOverride: finalSections)
        diagnostics.end("assemble_and_section_pooling",since:assemblyStart)
        diagnostics.structure(SongAnalysisDiagnostics.StructureEvidence(
            labelOrder:["intro","verse","chorus","bridge","inst","outro","silence","pre-chorus"],
            appleBoundaries:music.sections.map { .init(start:$0.lowerBound,end:$0.upperBound) },
            frames:frameLabels.compactMap { frame -> SongAnalysisDiagnostics.StructureEvidence.Frame? in
                guard let probabilities = frame.probabilities else { return nil }
                return .init(time:frame.time,duration:frame.duration,probabilities:probabilities)
            },
            assembledSections:analysis.sections.map { .init(label:$0.label,start:$0.startTime,end:$0.endTime) },
            bpm:analysis.bpm,beatsPerBar:analysis.timeSignature.beatsPerBar,downbeatOffset:analysis.downbeatOffset))
        diagnostics.finish(state:"completed")
        return analysis
        } catch {
            diagnostics.finish(state:error is CancellationError ? "cancelled" : "failed",error:error.localizedDescription)
            throw error
        }
    }

    private func detectChords(in audio: DecodedAudio, keys: [KeyRegion], downbeats: [Double], primaryBeats: [Double]?,
                              progress: ProgressSpan, diagnostics: SongAnalysisDiagnostics) async throws -> ChordEngine.Analysis? {
        do {
            let result = try await ChordEngine.analyze(audio: audio, keys: keys, options: ChordEngineOptions(), progress: progress,
                                                      downbeats: downbeats, primaryBeats: primaryBeats)
            if let failure = result.chordFailure {
                diagnostics.event("chord_partial_failure",detail:String(describing:failure),persist:true)
                logger.error("Chord detection failed (audio rhythm retained): \(failure, privacy: .public)")
            }
            return result
        } catch {
            try Task.checkCancellation()
            diagnostics.event("chord_failure",detail:error.localizedDescription,persist:true)
            logger.error("Chord detection failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    /// Instrumentals (MusicUnderstanding found no vocals) skip transcription altogether.
    private func recognizeLyrics(
        in audioURL: URL,
        audio: DecodedAudio,
        vocalRanges: [ClosedRange<Double>]?,
        duration: TimeInterval,
        progress: ProgressSpan, diagnostics: SongAnalysisDiagnostics
    ) async throws -> LyricsRecognizer.Result? {
        if let vocalRanges, vocalRanges.isEmpty {
            progress(1)
            return nil
        }
        do {
            return try await LyricsRecognizer().recognize(
                fileURL: audioURL,
                audio: audio,
                vocalRanges: vocalRanges ?? [],
                duration: duration,
                progress: progress
            )
        } catch {
            try Task.checkCancellation()
            diagnostics.event("lyrics_failure",detail:error.localizedDescription,persist:true)
            logger.error("Lyrics recognition failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }
}
