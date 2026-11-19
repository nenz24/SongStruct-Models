//
//  SectionLabeler.swift
//  SongMap
//

import Foundation

/// Turns MusicUnderstanding's section boundaries and SongFormer's frame labels into the sections of a chart.
nonisolated enum SectionLabeler {
    nonisolated struct LabeledSection: Sendable, Equatable {
        var label: String
        var start: Double
        var end: Double

        var length: Double { end - start }
    }

    /// Label used when SongFormer gave nothing to go on; shown as a custom section.
    static let fallbackLabel = "Section"
    private static let silenceLabel = "silence"

    /// - Each section gets the label most of its frames have.
    /// - Sections cover the whole song without gaps, and boundaries sit on barlines,
    ///   so every bar of the SongMap belongs to exactly one section.
    /// - Silence and sections shorter than a bar join a neighbour; neighbours with the same label merge.
    static func sections(
        boundaries: [ClosedRange<Double>],
        frameLabels: [SongFormerAnalyzer.FrameLabel],
        grid: BeatGrid,
        duration: Double
    ) -> [LabeledSection] {
        guard duration > 0 else { return [] }
        if !boundaries.isEmpty, frameLabels.contains(where: { $0.probabilities != nil }) {
            return pooled(boundaries: boundaries, frames: frameLabels)
        }
        let ranges = boundaries.isEmpty ? labelRuns(frameLabels) : boundaries
        var sections = ranges
            .sorted { $0.lowerBound < $1.lowerBound }
            .map { LabeledSection(label: majorityLabel(in: $0, frameLabels: frameLabels), start: $0.lowerBound, end: $0.upperBound) }
        sections = contiguous(sections, duration: duration)
        if sections.isEmpty {
            sections = [LabeledSection(label: fallbackLabel, start: 0, end: duration)]
        }
        sections = snappedToBarlines(sections, grid: grid, duration: duration)
        sections = folding(sections) { $0.label == silenceLabel }
        sections = folding(sections) { $0.length < grid.barDuration }
        return mergingRepeats(sections)
    }

    /// Match training: allowed-class softmax first, then overlap-duration probability pooling.
    /// Keep Apple's boundaries, silence and repeated labels in this experimental path.
    private static func pooled(boundaries: [ClosedRange<Double>], frames: [SongFormerAnalyzer.FrameLabel]) -> [LabeledSection] {
        let labels = ["intro","verse","chorus","bridge","inst","outro","silence","pre-chorus"]
        return boundaries.sorted { $0.lowerBound < $1.lowerBound }.map { range in
            var sums = Array(repeating: 0.0, count: labels.count)
            var coverage = 0.0
            for frame in frames {
                guard let probabilities = frame.probabilities, probabilities.count == labels.count else { continue }
                let weight = max(0, min(frame.time+frame.duration,range.upperBound)-max(frame.time,range.lowerBound))
                guard weight > 0 else { continue }
                for j in labels.indices { sums[j] += probabilities[j]*weight }
                coverage += weight
            }
            var best = 0
            for j in 1..<labels.count where sums[j] > sums[best] { best = j }
            return LabeledSection(label: coverage > 0 ? labels[best] : fallbackLabel,
                start: range.lowerBound,end:range.upperBound)
        }
    }

    // MARK: Labels

    /// The most common frame label inside `range`; ties go to the label heard closest to its middle.
    private static func majorityLabel(in range: ClosedRange<Double>, frameLabels: [SongFormerAnalyzer.FrameLabel]) -> String {
        let inside = frameLabels.filter { $0.time >= range.lowerBound && $0.time < range.upperBound }
        let middle = (range.lowerBound + range.upperBound) / 2
        let counts = Dictionary(grouping: inside, by: \.label).mapValues(\.count)
        func distanceToMiddle(_ label: String) -> Double {
            inside.filter { $0.label == label }.map { abs($0.time - middle) }.min() ?? .greatestFiniteMagnitude
        }
        let best = counts.keys.max {
            (counts[$0, default: 0], -distanceToMiddle($0)) < (counts[$1, default: 0], -distanceToMiddle($1))
        }
        return best
            ?? frameLabels.min { abs($0.time - middle) < abs($1.time - middle) }?.label
            ?? fallbackLabel
    }

    /// Stretches of consecutive frames with the same label; used when there are no boundaries.
    private static func labelRuns(_ frameLabels: [SongFormerAnalyzer.FrameLabel]) -> [ClosedRange<Double>] {
        guard let first = frameLabels.first, let last = frameLabels.last else { return [] }
        var ranges: [ClosedRange<Double>] = []
        var runStart = first.time
        var runLabel = first.label
        for frame in frameLabels.dropFirst() where frame.label != runLabel {
            ranges.append(runStart...frame.time)
            runStart = frame.time
            runLabel = frame.label
        }
        let frameStep = frameLabels.count > 1 ? frameLabels[1].time - frameLabels[0].time : 0
        ranges.append(runStart...(last.time + frameStep))
        return ranges
    }

    // MARK: Boundaries

    /// The first section starts at 0, each one ends where the next starts, and the last ends with the song.
    private static func contiguous(_ sections: [LabeledSection], duration: Double) -> [LabeledSection] {
        var result = sections.filter { $0.start < duration }
        guard !result.isEmpty else { return [] }
        result[0].start = 0
        for index in result.indices.dropLast() {
            result[index].end = result[index + 1].start
        }
        result[result.count - 1].end = duration
        return result.filter { $0.end > $0.start }
    }

    /// Moves every inner boundary to the nearest barline. Sections that end up empty disappear.
    private static func snappedToBarlines(_ sections: [LabeledSection], grid: BeatGrid, duration: Double) -> [LabeledSection] {
        var result = sections
        for index in result.indices.dropFirst() {
            let barline = min(max(grid.nearestBarline(to: result[index].start), result[index - 1].start), duration)
            result[index].start = barline
            result[index - 1].end = barline
        }
        return result.filter { $0.end > $0.start }
    }

    /// Removes the sections matching `shouldFold` and gives their time to the section before them,
    /// or to the one after them at the start of the song.
    private static func folding(_ sections: [LabeledSection], where shouldFold: (LabeledSection) -> Bool) -> [LabeledSection] {
        var result: [LabeledSection] = []
        var leadingStart: Double?
        for section in sections {
            if !shouldFold(section) {
                var kept = section
                if result.isEmpty, let leadingStart { kept.start = leadingStart }
                result.append(kept)
            } else if result.isEmpty {
                leadingStart = leadingStart ?? section.start
            } else {
                result[result.count - 1].end = section.end
            }
        }
        // Everything matched (e.g. a silent file): keep one section spanning the song.
        if result.isEmpty, let first = sections.first, let last = sections.last {
            return [LabeledSection(label: first.label, start: first.start, end: last.end)]
        }
        return result
    }

    private static func mergingRepeats(_ sections: [LabeledSection]) -> [LabeledSection] {
        var result: [LabeledSection] = []
        for section in sections {
            if let last = result.last, last.label == section.label {
                result[result.count - 1].end = section.end
            } else {
                result.append(section)
            }
        }
        return result
    }
}
