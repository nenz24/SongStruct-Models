import CoreML
import Foundation
import Darwin

actor BenchmarkRunner {
    struct Manifest: Decodable {
        struct Candidate: Decodable { let name: String; let resource: String; let package_bytes: Int }
        let frames: Int
        let feature_width: Int
        let reference_label: String
        let features_sha256: String
        let checkpoint_sha256: String
        let models: [Candidate]
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

    private func floats(_ name: String) throws -> [Float] {
        guard let url = Bundle.main.url(forResource: name, withExtension: "f32") else {
            throw NSError(domain: "Benchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing \(name).f32"])
        }
        let data = try Data(contentsOf: url)
        guard data.count % 4 == 0 else { throw NSError(domain: "Benchmark", code: 2) }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }

    func run(mode: String) throws -> String {
        guard let manifestURL = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json") else {
            throw NSError(domain: "Benchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing benchmark manifest"])
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let values = try floats("features")
        let reference = try floats("reference_logits")
        guard manifest.feature_width == 4096, manifest.frames % 3 == 0,
              values.count == manifest.frames * 4096, reference.count == manifest.frames / 3 * 128 else {
            throw NSError(domain: "Benchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Invalid fixture dimensions"])
        }
        let input = MLMultiArray(MLShapedArray<Float>(scalars: values, shape: [1, manifest.frames, 4096]))
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_embeddings": input])
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        var records: [[String: Any]] = []
        let config = MLModelConfiguration()
        switch mode {
        case "cpuOnly": config.computeUnits = .cpuOnly
        case "cpuAndGPU": config.computeUnits = .cpuAndGPU
        case "cpuAndNeuralEngine": config.computeUnits = .cpuAndNeuralEngine
        default: config.computeUnits = .all
        }
        for candidate in manifest.models {
            let record: [String: Any] = try autoreleasepool {
                guard let url = Bundle.main.url(forResource: candidate.resource, withExtension: "mlmodelc") else {
                    throw NSError(domain: "Benchmark", code: 5, userInfo: [NSLocalizedDescriptionKey: "Missing compiled \(candidate.resource)"])
                }
                let before = footprint()
                let start = ProcessInfo.processInfo.systemUptime
                let model = try MLModel(contentsOf: url, configuration: config)
                let load = ProcessInfo.processInfo.systemUptime - start
                let afterLoad = footprint()
                func predict() throws -> (MLFeatureProvider, Double) {
                    let t = ProcessInfo.processInfo.systemUptime
                    let output = try model.prediction(from: provider)
                    return (output, ProcessInfo.processInfo.systemUptime - t)
                }
                let (first, firstSeconds) = try predict()
                guard let logits = first.featureValue(for: "function_logits")?.multiArrayValue else {
                    throw NSError(domain: "Benchmark", code: 6, userInfo: [NSLocalizedDescriptionKey: "Missing function_logits"])
                }
                let output = MLShapedArray<Float>(converting: logits).scalars
                guard output.count == reference.count, output.allSatisfy({ $0.isFinite }), reference.allSatisfy({ $0.isFinite }) else { throw NSError(domain: "Benchmark", code: 7) }
                let ids = [0, 1, 2, 3, 4, 5, 6, 26]
                var maxError = 0.0
                var matches = 0
                var maxProbabilityError = 0.0
                for frame in 0..<(manifest.frames / 3) {
                    let a = ids.map { Double(reference[frame * 128 + $0]) }
                    let b = ids.map { Double(output[frame * 128 + $0]) }
                    let topA = a.indices.max { a[$0] < a[$1] }!
                    let topB = b.indices.max { b[$0] < b[$1] }!
                    if topA == topB { matches += 1 }
                    let ea = a.map { exp($0 - a.max()!) }
                    let eb = b.map { exp($0 - b.max()!) }
                    let sa = ea.reduce(0, +), sb = eb.reduce(0, +)
                    for index in ids.indices {
                        maxError = max(maxError, abs(a[index] - b[index]))
                        maxProbabilityError = max(maxProbabilityError, abs(ea[index]/sa - eb[index]/sb))
                    }
                }
                var warm: [Double] = []
                for _ in 0..<5 { warm.append(try autoreleasepool { try predict().1 }) }
                let afterPrediction = footprint()
                let sorted = warm.sorted()
                var result: [String: Any] = ["candidate": candidate.name,
                    "source_package_bytes": candidate.package_bytes,
                    "model_load_seconds": load, "first_prediction_seconds": firstSeconds,
                    "warm_prediction_seconds": warm, "warm_median_seconds": sorted[sorted.count / 2],
                    "audio_seconds_represented": Double(manifest.frames) / 25,
                    "allowed_frame_agreement_with_mac_fp32": Double(matches) / Double(manifest.frames / 3),
                    "allowed_max_logit_error": maxError, "allowed_max_probability_error": maxProbabilityError]
                if let before { result["process_footprint_before_load_bytes"] = before }
                if let afterLoad { result["process_footprint_after_load_bytes"] = afterLoad }
                if let afterPrediction { result["process_footprint_after_predictions_bytes"] = afterPrediction }
                return result
            }
            records.append(record)
        }
        var system = utsname()
        uname(&system)
        let capacity = MemoryLayout.size(ofValue: system.machine)
        let hardware = withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: capacity) { String(cString: $0) }
        }
        #if targetEnvironment(simulator)
        let simulator = true
        #else
        let simulator = false
        #endif
        let report: [String: Any] = ["hardware_identifier": hardware,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "simulator": simulator, "requested_compute_units": mode,
            "thermal_state_before": thermalBefore,
            "thermal_state_after": ProcessInfo.processInfo.thermalState.rawValue,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "reference": manifest.reference_label,
            "fixture_features_sha256": manifest.features_sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "scope": "Head only, identical real cached features; no encoder or Apple boundary timing",
            "memory_note": "Process footprint snapshots, not peak. Includes fixture/app/system allocations; candidate order can affect allocator state.",
            "timing_note": "Compiled-model load and first prediction are not guaranteed cold-OS-cache timings. Requested compute units do not prove ANE placement.",
            "results": records]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
