import Foundation

@main struct SongFormerBoundaryRefinerChecks {
    nonisolated struct Fixture: Decodable {
        let name: String
        let sections: [SongFormerBoundaryRefiner.Section]
        let frames: [SongFormerBoundaryRefiner.Frame]
        let grid: SongFormerBoundaryRefiner.Grid
        let expected: SongFormerBoundaryRefiner.Result
    }
    static func compare(_ actual: Any, _ expected: Any, path: String) {
        if let a = actual as? [String: Any], let b = expected as? [String: Any] {
            precondition(Set(a.keys) == Set(b.keys), "Different keys: \(path)")
            for key in b.keys { compare(a[key]!, b[key]!, path: path + "." + key) }
        } else if let a = actual as? [Any], let b = expected as? [Any] {
            precondition(a.count == b.count, "Different lengths: \(path)")
            for i in b.indices { compare(a[i], b[i], path: path + "[\(i)]") }
        } else if let a = actual as? NSNumber, let b = expected as? NSNumber {
            let x = a.doubleValue; let y = b.doubleValue
            precondition(x.isFinite && y.isFinite && abs(x - y) <= 1e-9 + 1e-9 * abs(y), "Numeric mismatch: \(path): \(x), \(y)")
        } else if let a = actual as? String, let b = expected as? String {
            precondition(a == b, "Different strings: \(path): \(a), \(b)")
        } else { preconditionFailure("Unsupported/mismatched JSON types: \(path)") }
    }
    static func main() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let cases = try JSONDecoder().decode([Fixture].self, from: data)
        for fixture in cases {
            let result = try SongFormerBoundaryRefiner.refine(sections: fixture.sections, frames: fixture.frames, grid: fixture.grid)
            precondition(result.originalSections == fixture.sections)
            precondition(result.refinedSections.map(\.label) == fixture.sections.map(\.label))
            let encoder = JSONEncoder()
            compare(try JSONSerialization.jsonObject(with: encoder.encode(result)),
                    try JSONSerialization.jsonObject(with: encoder.encode(fixture.expected)), path: fixture.name)
            print("BOUNDARY CHECK OK: \(fixture.name); shifted=\(result.appliedCount)")
        }
        guard let first = cases.first else { preconditionFailure("No reference cases") }
        do {
            _ = try SongFormerBoundaryRefiner.refine(sections: first.sections, frames: first.frames,
                grid: .init(bpm: 0, beatsPerBar: 4, downbeatOffset: 0))
            preconditionFailure("Invalid grid accepted")
        } catch SongFormerBoundaryRefiner.RefinementError.invalid { }
        do {
            _ = try SongFormerBoundaryRefiner.refine(sections: first.sections,
                frames: [first.frames[0], first.frames[0]], grid: first.grid)
            preconditionFailure("Overlapping frames accepted")
        } catch SongFormerBoundaryRefiner.RefinementError.invalid { }
        print("Boundary Swift/Python parity passed for \(cases.count) cases, including the real app report. No models or iPhone inference were run.")
    }
}
