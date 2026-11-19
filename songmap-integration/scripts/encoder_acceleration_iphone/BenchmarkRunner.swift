import CoreML
import CryptoKit
import Foundation
import Darwin

actor BenchmarkRunner {
    struct Manifest: Decodable {
        struct Fixture: Decodable {
            let name: String
            let frames: Int
            let resource: String
            let package_bytes: Int
            let compiled_bytes: Int
            let resource_digest: String
            let files: [String: String]
            let hashes: [String: String]
        }
        let graphProtocol: String
        enum CodingKeys: String, CodingKey {
            case graphProtocol = "protocol"
            case encoder, compression_scheme, source_state_sha256, export_report_sha256, fixtures
        }
        let encoder: String
        let compression_scheme: String
        let source_state_sha256: String
        let export_report_sha256: String
        let fixtures: [Fixture]
    }
    nonisolated static var lastStageURL: URL {
        URL.documentsDirectory.appendingPathComponent("encoder-graph-last-stage.json")
    }
    private func footprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let code = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return code == KERN_SUCCESS ? info.phys_footprint : nil
    }
    private func floats(_ fixture: Manifest.Fixture, _ key: String) throws -> [Float] {
        guard let name = fixture.files[key], let expected = fixture.hashes[key],
              let url = Bundle.main.url(forResource: name, withExtension: "f32") else { throw error("Missing fixture: \(key)") }
        let data = try Data(contentsOf: url)
        let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard actual == expected, data.count % 4 == 0 else { throw error("Fixture checksum/layout differs: \(key)") }
        let values = data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
        guard values.allSatisfy({ $0.isFinite }) else { throw error("Non-finite fixture") }
        return values
    }
    private func error(_ message: String) -> NSError {
        NSError(domain: "EncoderGraphProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    func run(fixtureName: String, mode: String) throws -> URL {
        guard let manifestURL = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json") else { throw error("Missing manifest") }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.graphProtocol == "encoder-rank3-explicit-glu-v1",
              let fixture = manifest.fixtures.first(where: { $0.name == fixtureName }), [750, 3000].contains(fixture.frames) else {
            throw error("Unsupported graph protocol/profile")
        }
        let units: MLComputeUnits
        switch mode {
        case "cpuOnly": units = .cpuOnly
        case "cpuAndNeuralEngine": units = .cpuAndNeuralEngine
        case "cpuAndGPU": units = .cpuAndGPU
        default: throw error("Unknown compute mode")
        }
        let id = UUID().uuidString
        var report: [String: Any] = ["protocol": manifest.graphProtocol, "id": id, "encoder": manifest.encoder,
            "fixture": fixture.name, "frames": fixture.frames, "compression_scheme": manifest.compression_scheme,
            "requested_compute_units": mode, "source_state_sha256": manifest.source_state_sha256,
            "export_report_sha256": manifest.export_report_sha256, "compiled_resource_digest": fixture.resource_digest,
            "compiled_resource_bytes": fixture.compiled_bytes, "package_bytes": fixture.package_bytes,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "thermal_before": ProcessInfo.processInfo.thermalState.rawValue, "ready_for_app_integration": false,
            "scope": "Static encoder candidate only on fixed real inputs. No frontend, head, boundaries or whole-song timing.",
            "note": "Requested units do not prove GPU/ANE placement. Compare same profile after fresh launch; first calls include runtime setup. Memory snapshots are not peak memory. GPU fatal assertions cannot be caught here."]
        #if targetEnvironment(simulator)
        report["simulator"] = true
        #else
        report["simulator"] = false
        #endif
        var system = utsname(); uname(&system)
        let capacity = MemoryLayout.size(ofValue: system.machine)
        report["hardware_identifier"] = withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: capacity) { String(cString: $0) }
        }
        func checkpoint(_ stage: String) throws {
            report["stage"] = stage
            report["state"] = "running"
            if let bytes = footprint() { report["last_process_footprint_bytes"] = bytes }
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: Self.lastStageURL, options: .atomic)
            NSLog("ENCODER_GRAPH %@ %@ %@", fixture.name, mode, stage)
        }
        try checkpoint("reading_inputs")
        let conv = try floats(fixture, "conv_in"), cos = try floats(fixture, "rotary_cos"), sin = try floats(fixture, "rotary_sin")
        let reference = try floats(fixture, "reference")
        guard conv.count == fixture.frames * 1024, reference.count == conv.count,
              cos.count == fixture.frames * 64, sin.count == cos.count else { throw error("Fixture dimensions differ") }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "conv_in": MLMultiArray(MLShapedArray<Float>(scalars: conv, shape: [1, fixture.frames, 1024])),
            "rotary_cos": MLMultiArray(MLShapedArray<Float>(scalars: cos, shape: [fixture.frames, 64])),
            "rotary_sin": MLMultiArray(MLShapedArray<Float>(scalars: sin, shape: [fixture.frames, 64]))
        ])
        guard let url = Bundle.main.url(forResource: fixture.resource, withExtension: "mlmodelc") else { throw error("Missing compiled model") }
        let configuration = MLModelConfiguration(); configuration.computeUnits = units
        try checkpoint("load_start")
        let loadStart = ProcessInfo.processInfo.systemUptime
        let model = try MLModel(contentsOf: url, configuration: configuration)
        report["model_load_seconds"] = ProcessInfo.processInfo.systemUptime - loadStart
        if let bytes = footprint() { report["process_footprint_after_load_bytes"] = bytes }
        try checkpoint("load_done")
        var calls: [[String: Any]] = []
        for call in 1...4 {
            try checkpoint("predict_start_\(call)")
            let result: [String: Any] = try autoreleasepool {
                let start = ProcessInfo.processInfo.systemUptime
                let output = try model.prediction(from: inputs)
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                guard let array = output.featureValue(for: "embedding")?.multiArrayValue,
                      array.shape.map({ $0.intValue }) == [1, fixture.frames, 1024] else { throw error("Unexpected embedding output") }
                let values = MLShapedArray<Float>(converting: array).scalars
                var invalid = 0, maximum = 0.0, sum = 0.0, passed = true
                for (actual, expected) in zip(values, reference) {
                    guard actual.isFinite else { invalid += 1; passed = false; continue }
                    let difference = abs(Double(actual) - Double(expected))
                    maximum = max(maximum, difference); sum += difference
                    if difference > 0.001 + 0.001 * abs(Double(expected)) { passed = false }
                }
                return ["call": call, "seconds": elapsed, "invalid_elements": invalid,
                    "max_abs_error": maximum, "mean_abs_error_over_all_elements": sum / Double(values.count),
                    "numerical_parity_passed": passed, "atol": 0.001, "rtol": 0.001]
            }
            calls.append(result); report["calls"] = calls
            try checkpoint("predict_done_\(call)")
        }
        let warm = calls.dropFirst().compactMap { $0["seconds"] as? Double }.sorted()
        report["warm_median_seconds"] = warm[1]
        report["first_prediction_seconds"] = calls[0]["seconds"]
        report["all_predictions_finite"] = calls.allSatisfy { ($0["invalid_elements"] as? Int) == 0 }
        report["all_calls_numerically_consistent"] = calls.allSatisfy { ($0["numerical_parity_passed"] as? Bool) == true }
        report["thermal_after"] = ProcessInfo.processInfo.thermalState.rawValue
        if let bytes = footprint() { report["process_footprint_after_predictions_bytes"] = bytes }
        report["state"] = "completed"; report["stage"] = "completed"
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        let resultURL = URL.documentsDirectory.appendingPathComponent("encoder-graph-\(id).json")
        try data.write(to: resultURL, options: .atomic); try data.write(to: Self.lastStageURL, options: .atomic)
        return resultURL
    }
}
