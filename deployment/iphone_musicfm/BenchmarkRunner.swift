import CoreML
import Foundation
import Darwin

actor BenchmarkRunner {
    struct Manifest: Decodable {
        struct Candidate: Decodable {
            let name: String
            let resource: String
            let package_bytes: Int
            let compiled_bytes: Int
        }
        struct Fixture: Decodable {
            let name: String
            let frames: Int
            let conv_resource: String
            let rotary_resource: String
            let reference_resource: String
            let conv_sha256: String
            let rotary_sha256: String
        }
        let models: [Candidate]
        let fixtures: [Fixture]
        let musicfm_weights_sha256: String
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
            throw NSError(domain: "MusicFMBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing \(name).f32"])
        }
        let data = try Data(contentsOf: url)
        guard data.count % 4 == 0 else { throw NSError(domain: "MusicFMBenchmark", code: 2) }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }

    func run(mode: String, candidateName: String, fixtureName: String) throws -> String {
        guard let manifestURL = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json") else {
            throw NSError(domain: "MusicFMBenchmark", code: 3)
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard let candidate = manifest.models.first(where: { $0.name == candidateName }),
              let fixture = manifest.fixtures.first(where: { $0.name == fixtureName }) else {
            throw NSError(domain: "MusicFMBenchmark", code: 4, userInfo: [NSLocalizedDescriptionKey: "Candidate or fixture not bundled"])
        }
        let conv = try floats(fixture.conv_resource)
        let rotary = try floats(fixture.rotary_resource)
        let reference = try floats(fixture.reference_resource)
        guard conv.count == fixture.frames * 1024, rotary.count == 2 * fixture.frames * 64,
              reference.count == conv.count, conv.allSatisfy({ $0.isFinite }),
              rotary.allSatisfy({ $0.isFinite }), reference.allSatisfy({ $0.isFinite }) else {
            throw NSError(domain: "MusicFMBenchmark", code: 5, userInfo: [NSLocalizedDescriptionKey: "Invalid fixture"])
        }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "conv_in": MLMultiArray(MLShapedArray<Float>(scalars: conv, shape: [1, fixture.frames, 1024])),
            "rotary": MLMultiArray(MLShapedArray<Float>(scalars: rotary, shape: [2, fixture.frames, 1, 1, 64]))
        ])
        guard let url = Bundle.main.url(forResource: candidate.resource, withExtension: "mlmodelc") else {
            throw NSError(domain: "MusicFMBenchmark", code: 6)
        }
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        let beforeLoad = footprint()
        let config = MLModelConfiguration()
        switch mode {
        case "cpuOnly": config.computeUnits = .cpuOnly
        case "cpuAndGPU": config.computeUnits = .cpuAndGPU
        case "cpuAndNeuralEngine": config.computeUnits = .cpuAndNeuralEngine
        default: config.computeUnits = .all
        }
        let loadStart = ProcessInfo.processInfo.systemUptime
        let model = try MLModel(contentsOf: url, configuration: config)
        let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStart
        let afterLoad = footprint()
        func prediction() throws -> ([Float], Double) {
            let start = ProcessInfo.processInfo.systemUptime
            let output = try model.prediction(from: inputs)
            let elapsed = ProcessInfo.processInfo.systemUptime - start
            guard let array = output.featureValue(for: "embedding")?.multiArrayValue,
                  array.shape.map({ $0.intValue }) == [1, fixture.frames, 1024] else {
                throw NSError(domain: "MusicFMBenchmark", code: 7, userInfo: [NSLocalizedDescriptionKey: "Unexpected embedding output"])
            }
            return (MLShapedArray<Float>(converting: array).scalars, elapsed)
        }
        let (output, firstSeconds) = try prediction()
        var badFrames: [Int] = []
        var nanCount = 0, infCount = 0, finiteElements = 0, validCosFrames = 0
        var maxError = 0.0, sumError = 0.0, sumCosine = 0.0
        for frame in 0..<fixture.frames {
            var valid = true
            var dot = 0.0, normA = 0.0, normB = 0.0
            for channel in 0..<1024 {
                let index = frame * 1024 + channel
                let b = output[index]
                if b.isNaN { nanCount += 1 }
                if b.isInfinite { infCount += 1 }
                if !b.isFinite { valid = false; continue }
                let a = Double(reference[index]), value = Double(b)
                let error = abs(a - value)
                maxError = max(maxError, error)
                sumError += error
                finiteElements += 1
                dot += a * value
                normA += a * a
                normB += value * value
            }
            if valid {
                sumCosine += dot / max(sqrt(normA * normB), 1e-12)
                validCosFrames += 1
            } else { badFrames.append(frame) }
        }
        var warm: [Double] = []
        var warmInvalidCounts: [Int] = []
        for _ in 0..<3 {
            let item: (Double, Int) = try autoreleasepool {
                let (values, seconds) = try prediction()
                return (seconds, values.filter { !$0.isFinite }.count)
            }
            warm.append(item.0)
            warmInvalidCounts.append(item.1)
        }
        let afterPredictions = footprint()
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
        var report: [String: Any] = [
            "candidate": candidateName, "fixture": fixtureName,
            "frames": fixture.frames, "audio_seconds_represented_approximately": Double(fixture.frames)/25,
            "hardware_identifier": hardware, "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "simulator": simulator, "requested_compute_units": mode,
            "thermal_before": thermalBefore, "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "model_load_seconds": loadSeconds, "first_prediction_seconds": firstSeconds,
            "warm_prediction_seconds": warm, "warm_median_seconds": warm.sorted()[warm.count/2],
            "first_prediction_nan_count": nanCount, "first_prediction_inf_count": infCount,
            "first_prediction_bad_frame_indices": badFrames,
            "warm_prediction_invalid_counts": warmInvalidCounts,
            "all_predictions_finite": badFrames.isEmpty && warmInvalidCounts.allSatisfy({ $0 == 0 }),
            "source_package_bytes": candidate.package_bytes, "compiled_resource_bytes": candidate.compiled_bytes,
            "conv_sha256": fixture.conv_sha256, "rotary_sha256": fixture.rotary_sha256,
            "musicfm_weights_sha256": manifest.musicfm_weights_sha256,
            "finite_elements_compared": finiteElements,
            "reference": "Mac Core ML repaired FP32 MusicFM on identical conv/rotary inputs",
            "scope": "MusicFM L10 encoder only; no frontend, MuQ, label head, Apple boundaries or full song timing",
            "memory_note": "Process footprint snapshots, not peak memory. Restart app between candidates for cleaner comparisons.",
            "timing_note": "Compiled resources. First prediction/load are not guaranteed cold OS cache. Requested compute units do not prove ANE placement. Numerical comparisons use first output; all warm calls are checked for invalid values.",
            "ready_for_app_integration": false
        ]
        if finiteElements > 0 {
            report["finite_subset_max_abs_error"] = maxError
            report["finite_subset_mean_abs_error"] = sumError / Double(finiteElements)
        }
        if validCosFrames > 0 { report["fully_finite_frame_mean_cosine"] = sumCosine / Double(validCosFrames) }
        if let beforeLoad { report["process_footprint_before_load_bytes"] = beforeLoad }
        if let afterLoad { report["process_footprint_after_load_bytes"] = afterLoad }
        if let afterPredictions { report["process_footprint_after_predictions_bytes"] = afterPredictions }
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
