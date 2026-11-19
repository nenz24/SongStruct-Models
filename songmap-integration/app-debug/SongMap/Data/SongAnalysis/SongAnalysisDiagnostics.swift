import Foundation

/// Local timing and structure predictions: no waveform, lyrics, or encoder embeddings. Shared across analysis actors.
nonisolated final class SongAnalysisDiagnostics: @unchecked Sendable {
    nonisolated struct Event: Codable, Sendable {
        let name: String
        let kind: String
        let seconds: Double?
        let detail: String?
    }
    nonisolated struct StructureEvidence: Codable, Sendable {
        nonisolated struct Interval: Codable, Sendable { let start: Double; let end: Double }
        nonisolated struct Frame: Codable, Sendable { let time: Double; let duration: Double; let probabilities: [Double] }
        nonisolated struct Section: Codable, Sendable { let label: String; let start: Double; let end: Double }
        let labelOrder: [String]
        let appleBoundaries: [Interval]
        let frames: [Frame]
        let assembledSections: [Section]
        let bpm: Double
        let beatsPerBar: Int
        let downbeatOffset: Double
        let note = "Frame values are base SongFormer probabilities over eight allowed classes. Context-corrected assembled labels, when enabled, are described separately in contextEvidence. Display bars belong to the section containing the bar midpoint. No manual reference labels are inferred."
    }
    nonisolated struct ContextEvidence: Codable, Sendable {
        nonisolated struct Section: Codable, Sendable {
            let start: Double; let end: Double
            let baselineLabel: String; let label: String
            let baseProbabilities: [Double]; let probabilities: [Double]
            let audioCoverage: Double; let probabilityCoverage: Double
        }
        let checkpointSHA256: String
        let baseCheckpointSHA256: String
        let audioSHA256: String
        let timingSHA256: String
        let lowAudioCoverageSections: [Int]
        let sections: [Section]
        let selectedEpoch = 5
        let precision = "fp32"
        let computeUnits = "CPU_ONLY"
        let labelOrder = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence", "pre-chorus"]
        let note = "All original Apple sections provide audio context. These are the original context-input ranges; optional later Debug boundary shifts are recorded in boundaryRefinementEvidence. Before/after labels are predictions, not manual ground truth. Hashes describe little-endian Float32 means/timing, not exported embeddings."
    }
    /// Debug `--musicunderstanding-boundary-audit`: the same file analysed by MusicUnderstanding in several
    /// configurations. Recording only; the analysis always uses the app's own pass.
    nonisolated struct BoundaryAuditEvidence: Codable, Sendable {
        nonisolated struct Run: Codable, Sendable {
            let configuration: String
            let seconds: Double
            let sections: [StructureEvidence.Interval]
            let error: String?
        }
        let runs: [Run]
        var note = "app-combined-default-timing is the pass used by the analysis ([.structure, .rhythm, .key, .instrumentActivity], default AVURLAsset). structure-only-precise-timing matches the dataset exporter; its repeat checks run-to-run consistency; structure-only-default-timing isolates the timing option."
    }
    nonisolated struct Report: Codable, Sendable {
        var schemaVersion = 1
        let id: String
        let songID: String
        let title: String
        let startedAt: String
        let os: String
        var state = "running"
        var activeStage: String?
        var totalSeconds: Double = 0
        var events: [Event] = []
        var error: String?
        var structureEvidence: StructureEvidence?
        var contextEvidence: ContextEvidence?
        var boundaryRefinementEvidence: SongFormerBoundaryRefiner.Result?
        var boundaryAuditEvidence: BoundaryAuditEvidence?
        let note = "Parent and child timings overlap; do not sum all events. Running state after termination is a partial checkpoint, not a crash diagnosis. Timing includes diagnostic overhead."
    }
    /// Debug launch flags typed in Xcode: macOS Smart Dashes can turn "--name" into "—name" or "–name",
    /// and a single "-" is a common slip. Matches `name` (without dashes) after any leading dashes.
    static func launchArgument(_ name: String, in arguments: [String] = ProcessInfo.processInfo.arguments) -> Bool {
        arguments.contains { $0.drop(while: { $0 == "-" || $0 == "\u{2013}" || $0 == "\u{2014}" }) == Substring(name) }
    }
    static var directory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AnalysisDiagnostics", isDirectory: true)
    }
    private let lock = NSLock()
    private let started = ProcessInfo.processInfo.systemUptime
    private let url: URL
    private var report: Report

    init(songID: String, title: String) {
        let id = UUID().uuidString
        report = Report(id:id, songID:songID, title:title,
            startedAt:ISO8601DateFormatter().string(from:Date()),
            os:ProcessInfo.processInfo.operatingSystemVersionString)
        url = Self.directory.appendingPathComponent("analysis-\(id).json")
        lock.lock(); defer { lock.unlock() }; saveLocked()
    }
    func begin(_ stage: String) -> Double {
        lock.lock(); defer { lock.unlock() }
        report.activeStage = stage
        report.events.append(Event(name:stage,kind:"begin",seconds:nil,detail:nil))
        saveLocked()
        return ProcessInfo.processInfo.systemUptime
    }
    func end(_ stage: String, since start: Double) {
        lock.lock(); defer { lock.unlock() }
        report.events.append(Event(name:stage,kind:"stage",seconds:ProcessInfo.processInfo.systemUptime-start,detail:nil))
        report.activeStage = nil; saveLocked()
    }
    /// Fine-grained measurements stay in memory until the next stage checkpoint.
    func metric(_ name: String, since start: Double, detail: String? = nil) {
        lock.lock(); defer { lock.unlock() }
        report.events.append(Event(name:name,kind:"metric",seconds:ProcessInfo.processInfo.systemUptime-start,detail:detail))
    }
    func event(_ name: String, detail: String, persist: Bool = false) {
        lock.lock(); defer { lock.unlock() }
        report.events.append(Event(name:name,kind:"event",seconds:nil,detail:detail))
        if persist { saveLocked() }
    }
    func structure(_ evidence: StructureEvidence) {
        lock.lock(); defer { lock.unlock() }
        report.structureEvidence = evidence; saveLocked()
    }
    func context(_ evidence: ContextEvidence) {
        lock.lock(); defer { lock.unlock() }
        report.contextEvidence = evidence; saveLocked()
    }
    func boundaryRefinement(_ evidence: SongFormerBoundaryRefiner.Result) {
        lock.lock(); defer { lock.unlock() }
        report.boundaryRefinementEvidence = evidence; saveLocked()
    }
    func boundaryAudit(_ evidence: BoundaryAuditEvidence) {
        lock.lock(); defer { lock.unlock() }
        report.boundaryAuditEvidence = evidence; saveLocked()
    }
    func finish(state: String, error: String? = nil) {
        lock.lock(); defer { lock.unlock() }
        report.state = state; report.error = error
        if state == "completed" { report.activeStage = nil }
        saveLocked()
    }
    /// Reports are written in Debug only, where Library → ⋯ → Analysis reports shows them. Release keeps
    /// nothing on disk.
    private func saveLocked() {
        report.totalSeconds = ProcessInfo.processInfo.systemUptime-started
        #if DEBUG
        do {
            try FileManager.default.createDirectory(at:Self.directory,withIntermediateDirectories:true)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
            try encoder.encode(report).write(to:url,options:.atomic)
        } catch { NSLog("ANALYSIS_DIAGNOSTIC_SAVE_FAILED %@",error.localizedDescription) }
        #endif
    }
}
