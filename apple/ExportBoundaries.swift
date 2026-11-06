import Foundation
import AVFoundation
import CoreMedia
import MusicUnderstanding

struct Track: Decodable {
    let id: String
    let audio_path: String
    let output_path: String
}

@main
struct ExportBoundaries {
    static func ranges(_ values: [CMTimeRange]) throws -> [[String: Double]] {
        try values.map { range in
            let start = CMTimeGetSeconds(range.start)
            let end = CMTimeGetSeconds(range.end)
            guard start.isFinite, end.isFinite, start >= 0, end > start else {
                throw NSError(domain: "ExportBoundaries", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid structure time range"])
            }
            return ["start": start, "end": end]
        }
    }

    static func main() async {
        guard CommandLine.arguments.count == 2 else {
            print("Usage: export-boundaries manifest.json")
            exit(2)
        }
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
            let tracks = try JSONDecoder().decode([Track].self, from: data)
            var failures = 0
            for (index, track) in tracks.enumerated() {
                print("[\(index + 1)/\(tracks.count)] \(track.id)")
                do {
                    let output = URL(fileURLWithPath: track.output_path)
                    guard !FileManager.default.fileExists(atPath: output.path) else {
                        print("SKIP: output already exists")
                        continue
                    }
                    let asset = AVURLAsset(url: URL(fileURLWithPath: track.audio_path),
                        options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
                    let duration = CMTimeGetSeconds(try await asset.load(.duration))
                    guard duration.isFinite, duration > 0 else {
                        throw NSError(domain: "ExportBoundaries", code: 2,
                            userInfo: [NSLocalizedDescriptionKey: "Invalid audio duration"])
                    }
                    let session = try await MusicUnderstandingSession(asset: asset)
                    let result = try await session.analyze(for: [.structure])
                    guard let structure = result.structure, !structure.sections.isEmpty else {
                        throw NSError(domain: "ExportBoundaries", code: 3,
                            userInfo: [NSLocalizedDescriptionKey: "No sections returned"])
                    }
                    let sections = try ranges(structure.sections)
                    let raw = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result))
                    let document: [String: Any] = [
                        "schema_version": 1, "track_id": track.id,
                        "audio_path": track.audio_path, "duration": duration,
                        "boundary_source": "apple_musicunderstanding.sections",
                        "sections": sections,
                        "segments": try ranges(structure.segments),
                        "phrases": try ranges(structure.phrases),
                        "raw_result": raw,
                        "review_status": "pending",
                        "exported_at": ISO8601DateFormatter().string(from: Date()),
                        "os_version": ProcessInfo.processInfo.operatingSystemVersionString
                    ]
                    let encoded = try JSONSerialization.data(withJSONObject: document,
                                                             options: [.prettyPrinted, .sortedKeys])
                    try FileManager.default.createDirectory(at: output.deletingLastPathComponent(),
                                                             withIntermediateDirectories: true)
                    try encoded.write(to: output, options: [.withoutOverwriting])
                    print("OK: \(sections.count) sections → \(output.path)")
                } catch {
                    failures += 1
                    print("ERROR: \(track.id): \(error)")
                }
            }
            if failures > 0 { exit(1) }
        } catch {
            print("ERROR: \(error)")
            exit(1)
        }
    }
}
