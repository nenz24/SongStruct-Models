import Foundation

@main struct SongFormerSectionContextChecks {
    nonisolated struct Fixture: Decodable {
        nonisolated struct Chunk: Decodable { let offset: Double; let frames: Int; let file: String }
        nonisolated struct Frame: Decodable { let time: Double; let duration: Double; let probabilities: [Double] }
        let duration: Double
        let sections: [SongFormerContextMath.Interval]
        let chunks: [Chunk]
        let frames: [Frame]
        let expectedCoverage: [Double]
        let residual: [Float]
        let expectedPooled: [[Double]]
        let expectedProbabilities: [[Double]]
        let expectedLabels: [String]
    }
    /// -O builds drop precondition messages; print the failing check before exiting.
    static func fail(_ message: String, file: StaticString = #file, line: UInt = #line) -> Never {
        FileHandle.standardError.write(Data("CHECK FAILED (\(file):\(line)): \(message)\n".utf8)); exit(1)
    }
    static func check(_ condition: @autoclosure () -> Bool, _ message: @autoclosure () -> String = "condition", file: StaticString = #file, line: UInt = #line) {
        if !condition() { fail(message(), file: file, line: line) }
    }
    static func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        check(data.count % 4 == 0)
        return data.withUnsafeBytes { raw in
            stride(from: 0, to: data.count, by: 4).map { Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian) }
        }
    }
    static func close(_ a: [Double], _ b: [Double], tolerance: Double = 1e-6) {
        check(a.count == b.count)
        check(zip(a, b).allSatisfy { pair in pair.0.isFinite && pair.1.isFinite && abs(pair.0 - pair.1) <= tolerance }, "Python/Swift values differ")
    }
    static func main() throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
        let accumulator = try SongFormerContextMath.Accumulator(intervals: fixture.sections, duration: fixture.duration)
        for chunk in fixture.chunks {
            let values = try floats(root.appendingPathComponent(chunk.file))
            try values.withUnsafeBufferPointer { try accumulator.append($0, frames: chunk.frames, offset: chunk.offset) }
        }
        let features = try accumulator.features()
        close(features.audio.map(Double.init), try floats(root.appendingPathComponent("means.f32")).map(Double.init))
        close(features.timing.map(Double.init), try floats(root.appendingPathComponent("timing.f32")).map(Double.init))
        close(accumulator.coverage, fixture.expectedCoverage, tolerance: 1e-9)
        let frames = fixture.frames.map { SongFormerContextMath.Frame(time: $0.time, duration: $0.duration, probabilities: $0.probabilities) }
        let pools = try SongFormerContextMath.pool(accumulator.intervals, frames: frames)
        for (pool, expected) in zip(pools, fixture.expectedPooled) { close(pool.probabilities, expected, tolerance: 1e-9) }
        let predictions = try SongFormerContextMath.corrected(probabilities: pools.map(\.probabilities), residual: fixture.residual)
        check(predictions.map(\.label) == fixture.expectedLabels)
        for (prediction, expected) in zip(predictions, fixture.expectedProbabilities) { close(prediction.probabilities, expected) }
        // MuQ-only pipeline: MuQ30 + MuQ120 columns (2048 wide). Pooling is linear, so the 2048-wide
        // accumulator must equal the MuQ columns of the 4096-wide Python means.
        let muq = try SongFormerContextMath.Accumulator(intervals: fixture.sections, duration: fixture.duration, width: 2048)
        let muqColumns = Array(1024..<2048) + Array(3072..<4096)
        for chunk in fixture.chunks {
            let values = try floats(root.appendingPathComponent(chunk.file))
            var sliced = [Float](); sliced.reserveCapacity(chunk.frames * 2048)
            for frame in 0..<chunk.frames { for column in muqColumns { sliced.append(values[frame * 4096 + column]) } }
            try sliced.withUnsafeBufferPointer { try muq.append($0, frames: chunk.frames, offset: chunk.offset) }
        }
        let muqFeatures = try muq.features(); let fullMeans = features.audio
        check(muqFeatures.audio.count == fixture.sections.count * 2048)
        close(muqFeatures.audio.map(Double.init),
              (0..<fixture.sections.count).flatMap { s in muqColumns.map { Double(fullMeans[s * 4096 + $0]) } })
        close(muq.coverage, accumulator.coverage, tolerance: 1e-12)
        do { _ = try SongFormerContextMath.Accumulator(intervals: fixture.sections, duration: fixture.duration, width: 1000); fail("Width accepted") }
        catch SongFormerContextMath.ContextError.invalid { }
        let tie = try SongFormerContextMath.corrected(probabilities: [Array(repeating: 0.125, count: 8)], residual: Array(repeating: 0, count: 8))
        check(tie[0].label == "intro")

        // Reproduce the short final section that failed the original percentage-only gate.
        let tail = try SongFormerContextMath.Accumulator(intervals: [.init(start: 261.902, end: 263.723)], duration: 263.723)
        let ones = [Float](repeating: 1, count: 6591 * 4096)
        try ones.withUnsafeBufferPointer { try tail.append($0, frames: 6591, offset: 0) }
        let tailFeatures = try tail.features()
        check(tailFeatures.lowCoverage == [0]); check(tailFeatures.audio.allSatisfy { abs($0 - 1) < 1e-6 })
        let gap = try SongFormerContextMath.Accumulator(intervals: [.init(start: 0, end: 0.36)], duration: 0.36)
        let block = [Float](repeating: 1, count: 3 * 4096)
        try block.withUnsafeBufferPointer { try gap.append($0, frames: 3, offset: 0) }
        try block.withUnsafeBufferPointer { try gap.append($0, frames: 3, offset: 0.24) }
        do { _ = try gap.features(); fail("Unexplained gap accepted") }
        catch SongFormerContextMath.ContextError.invalid { }
        do {
            _ = try SongFormerContextMath.normalize([.init(start: 0, end: 1), .init(start: 0.5, end: 2)], duration: 2, maximum: 512)
            fail("Overlap accepted")
        } catch SongFormerContextMath.ContextError.invalid { }
        print("Section context checks passed: Python/Swift means, timing, probability pooling, FP32 correction, tie order, bounded tails, and invalid gaps. Runtime sources compiled; no iPhone inference performed.")
    }
}
