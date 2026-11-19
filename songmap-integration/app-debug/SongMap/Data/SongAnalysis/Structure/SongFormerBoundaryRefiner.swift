import Foundation

/// Fixed post-context experiment. No audio, model inference, relabeling, or new sections.
nonisolated enum SongFormerBoundaryRefiner {
    static let labels = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence", "pre-chorus"]
    static let protocolVersion = "context-boundary-shift-review-v1"
    nonisolated struct Section: Codable, Sendable, Equatable { let start: Double; var end: Double; let label: String }
    nonisolated struct Frame: Codable, Sendable { let time: Double; let duration: Double; let probabilities: [Double] }
    nonisolated struct Grid: Codable, Sendable { let bpm: Double; let beatsPerBar: Int; let downbeatOffset: Double }
    nonisolated struct Parameters: Codable, Sendable, Equatable {
        let anchorProbability: Double; let anchorMargin: Double; let coverage: Double
        let meanLogGain: Double; let minimumShiftBars: Double; let minimumSectionBars: Double
    }
    static let parameters = Parameters(anchorProbability: 0.55, anchorMargin: 0.15, coverage: 0.98,
        meanLogGain: 0.10, minimumShiftBars: 0.25, minimumSectionBars: 1)
    nonisolated struct Anchor: Codable, Sendable {
        let passed: Bool
        var reason: String? = nil
        var start: Double? = nil; var end: Double? = nil; var coverage: Double? = nil
        var probability: Double? = nil; var margin: Double? = nil
    }
    nonisolated struct Candidate: Codable, Sendable {
        let seconds: Double; let shiftSeconds: Double; let coverage: Double
        let logGainSeconds: Double; let meanLogGain: Double; let passed: Bool
    }
    nonisolated struct Decision: Codable, Sendable {
        let boundaryIndex: Int; let leftLabel: String; let rightLabel: String; let originalSeconds: Double
        var candidateSeconds: Double; var status: String; var candidateChecks: [Candidate]
        var leftAnchor: Anchor? = nil; var rightAnchor: Anchor? = nil
        var logGainSeconds: Double? = nil; var meanLogGain: Double? = nil
        var applied = false
    }
    nonisolated struct Result: Codable, Sendable {
        let protocolVersion: String; let parameters: Parameters; let barSeconds: Double
        let originalSections: [Section]; let refinedSections: [Section]; let decisions: [Decision]; let appliedCount: Int
    }
    nonisolated enum RefinementError: LocalizedError {
        case invalid(String)
        var errorDescription: String? { if case .invalid(let detail) = self { return "Boundary refinement: " + detail }; return nil }
    }

    static func refine(sections: [Section], frames: [Frame], grid: Grid) throws -> Result {
        guard !sections.isEmpty, !frames.isEmpty, grid.bpm.isFinite, grid.bpm > 0,
              (1...16).contains(grid.beatsPerBar), grid.downbeatOffset.isFinite, grid.downbeatOffset >= 0 else {
            throw RefinementError.invalid("Missing sections/frames or invalid grid.")
        }
        let bar = 60 / grid.bpm * Double(grid.beatsPerBar)
        guard bar.isFinite, (0.25...30).contains(bar) else { throw RefinementError.invalid("Unsupported bar duration.") }
        var previous = 0.0
        for section in sections {
            guard section.start.isFinite, section.end.isFinite, section.start >= previous - 1e-6,
                  section.start < section.end, labels.contains(section.label) else {
                throw RefinementError.invalid("Invalid/overlapping context sections.")
            }
            previous = section.end
        }
        previous = -.infinity
        for frame in frames {
            let p = frame.probabilities
            guard frame.time.isFinite, frame.duration.isFinite, frame.time >= 0, frame.duration > 0,
                  frame.time >= previous - 1e-6, p.count == 8,
                  p.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }), abs(p.reduce(0, +) - 1) <= 1e-4 else {
                throw RefinementError.invalid("Invalid/overlapping frame probabilities.")
            }
            previous = frame.time + frame.duration
        }
        var rows: [Decision] = []
        for index in 0..<max(0, sections.count - 1) {
            try Task.checkCancellation()
            let left = sections[index]; let right = sections[index + 1]; let old = left.end
            if left.label == right.label { continue }
            var row = Decision(boundaryIndex: index + 1, leftLabel: left.label, rightLabel: right.label,
                originalSeconds: old, candidateSeconds: old, status: "unchanged", candidateChecks: [])
            if abs(old - right.start) > 1e-6 || left.label == "silence" || right.label == "silence" {
                row.status = "abstain_gap_or_silence"; rows.append(row); continue
            }
            let closest = grid.downbeatOffset + floor((old - grid.downbeatOffset) / bar + 0.5) * bar
            guard closest.isFinite else { throw RefinementError.invalid("Non-finite bar candidate.") }
            let low = closest - bar; let high = closest + bar
            row.leftAnchor = anchor(frames, start: max(left.start, low - bar), end: low, label: left.label, bar: bar)
            row.rightAnchor = anchor(frames, start: high, end: min(right.end, high + bar), label: right.label, bar: bar)
            guard row.leftAnchor?.passed == true, row.rightAnchor?.passed == true else {
                row.status = "abstain_weak_anchor"; rows.append(row); continue
            }
            let li = labels.firstIndex(of: left.label)!; let ri = labels.firstIndex(of: right.label)!
            for point in [low, closest, high] {
                let shift = point - old
                if abs(shift) < bar * parameters.minimumShiftBars
                    || point - left.start < bar * parameters.minimumSectionBars
                    || right.end - point < bar * parameters.minimumSectionBars { continue }
                let newClass = shift > 0 ? li : ri; let oldClass = shift > 0 ? ri : li
                var gain = 0.0; var covered = 0.0
                for frame in frames {
                    let weight = overlap(frame, start: min(old, point), end: max(old, point))
                    covered += weight
                    gain += weight * (log(max(frame.probabilities[newClass], 1e-8)) - log(max(frame.probabilities[oldClass], 1e-8)))
                }
                let mean = covered > 0 ? gain / covered : 0
                row.candidateChecks.append(Candidate(seconds: point, shiftSeconds: shift, coverage: covered / abs(shift),
                    logGainSeconds: gain, meanLogGain: mean,
                    passed: covered / abs(shift) >= parameters.coverage && mean >= parameters.meanLogGain))
            }
            // First candidate wins an exact tie, matching Python's stable max.
            var best: Candidate?
            for candidate in row.candidateChecks where candidate.passed {
                if let current = best {
                    if candidate.logGainSeconds > current.logGainSeconds
                        || (candidate.logGainSeconds == current.logGainSeconds && abs(candidate.shiftSeconds) < abs(current.shiftSeconds)) {
                        best = candidate
                    }
                } else { best = candidate }
            }
            if let best {
                row.candidateSeconds = best.seconds; row.status = "candidate_requires_review"
                row.logGainSeconds = best.logGainSeconds; row.meanLogGain = best.meanLogGain
            } else { row.status = "abstain_weak_gain_or_short_section" }
            rows.append(row)
        }
        var refined = sections
        let order = rows.indices.filter { rows[$0].status == "candidate_requires_review" }.sorted {
            let a = rows[$0]; let b = rows[$1]
            if a.logGainSeconds! == b.logGainSeconds! { return a.boundaryIndex < b.boundaryIndex }
            return a.logGainSeconds! > b.logGainSeconds!
        }
        for position in order {
            let row = rows[position]; let index = row.boundaryIndex - 1; let point = row.candidateSeconds
            if point - refined[index].start < bar * parameters.minimumSectionBars
                || refined[index + 1].end - point < bar * parameters.minimumSectionBars {
                rows[position].status = "abstain_neighbor_conflict"; rows[position].candidateSeconds = row.originalSeconds
                continue
            }
            refined[index].end = point
            refined[index + 1] = Section(start: point, end: refined[index + 1].end, label: refined[index + 1].label)
            rows[position].applied = true; rows[position].status = "applied_debug"
        }
        return Result(protocolVersion: protocolVersion, parameters: parameters, barSeconds: bar,
            originalSections: sections, refinedSections: refined, decisions: rows, appliedCount: rows.filter(\.applied).count)
    }

    private static func overlap(_ frame: Frame, start: Double, end: Double) -> Double {
        max(0, min(end, frame.time + frame.duration) - max(start, frame.time))
    }

    private static func anchor(_ frames: [Frame], start: Double, end: Double, label: String, bar: Double) -> Anchor {
        guard end - start >= bar * 0.5 else { return Anchor(passed: false, reason: "short_anchor") }
        var sums = [Double](repeating: 0, count: 8); var covered = 0.0
        for frame in frames {
            let weight = overlap(frame, start: start, end: end); covered += weight
            for j in 0..<8 { sums[j] += weight * frame.probabilities[j] }
        }
        let p = covered > 0 ? sums.map { $0 / covered } : sums
        let k = labels.firstIndex(of: label)!
        let margin = p[k] - p.indices.filter { $0 != k }.map { p[$0] }.max()!
        return Anchor(passed: covered / (end - start) >= parameters.coverage
            && p[k] >= parameters.anchorProbability && margin >= parameters.anchorMargin,
            start: start, end: end, coverage: covered / (end - start), probability: p[k], margin: margin)
    }
}
