import Foundation
import CoreML
import CryptoKit
import Darwin

actor BenchmarkRunner {
    struct Target: Decodable {
        let index: Int
        let label: String
        let prediction: String
        let probabilities: [Double]
        let base_probabilities: [Double]
    }
    struct Track: Decodable {
        let id: String
        let title: String
        let duration: Double
        let samples: Int
        let pcm: String
        let pcm_sha256: String
        let means: String
        let means_sha256: String
        let intervals: [SongFormerContextMath.Interval]
        let targets: [Target]
    }
    struct Manifest: Decodable {
        let version: String
        let quality_sha256: String
        let probe_source_sha256: String
        let static_graph_policy: String
        let frontend_route_probe: Bool?
        let frontend_gpu_policy: String?
        let graph_repair_sha256: String?
        let mac_preflight_policy: String?
        let mac_feature_parity_passed: Bool?
        let canonical_encoder_policy: String?
        let canonical_encoder_schedule: String?
        let tensor_transfer_policy: String?
        let cpu_control_receipt_sha256: String?
        let canonical_encoder_functions: [String: [String: String]]?
        let canonical_encoder_reports: [String: String]?
        let compiled_encoder_bytes: Int?
        /// nil: MusicFM + MuQ (4096 features). ["muq"]: MuQ-only label pipeline (2048 features).
        let encoder_roles: [String]?
        let graph_states: [String: String]
        let base_checkpoint: String
        let context_checkpoint: String
        let rotary: [String: [Float]]
        let files: [String: [String: String]]
        let tracks: [Track]
    }
    static var documents: URL { URL.documentsDirectory }
    static var checkpoint: URL { documents.appendingPathComponent("pipeline-last-stage.json") }
    static var batchURL: URL { documents.appendingPathComponent("pipeline-batch-latest.json") }
    private var verified = Set<String>()
    private var busy = false
    private func failure(_ text: String) -> NSError { NSError(domain: "EncoderPipelineProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
    private func footprint() -> UInt64? {
        var value = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &value) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? value.phys_footprint : nil
    }
    private func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var h = SHA256()
        while let data = try handle.read(upToCount: 4*1024*1024), !data.isEmpty { h.update(data: data) }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private func readFloats(_ name: String, expected: String) throws -> [Float] {
        guard let url = Bundle.main.url(forResource: name, withExtension: "f32"), try hash(url) == expected else { throw failure("Fixture checksum differs: \(name)") }
        let data = try Data(contentsOf: url)
        guard data.count % 4 == 0 else { throw failure("Invalid Float32 fixture") }
        let values = data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self))) }
        }
        guard values.allSatisfy({ $0.isFinite }) else { throw failure("Non-finite fixture") }
        return values
    }
    private nonisolated static func roles(_ m: Manifest) -> [String] { m.encoder_roles ?? ["musicfm", "muq"] }
    private nonisolated static func streamNames(_ roles: [String]) -> [String] { [30, 120].flatMap { s in roles.map { "\($0)_\(s)" } } }
    private func input(_ values: [Float], _ shape: [Int]) -> MLMultiArray { MLMultiArray(MLShapedArray<Float>(scalars: values, shape: shape)) }
    private func values(_ array: MLMultiArray) throws -> [Float] { MLShapedArray<Float>(try SongFormerTensorBridge.float32(array)).scalars }
    private nonisolated static func featureDiagnostics(actualValues: [Float], reference: [Float], sourceIndices: [Int], intervals: [SongFormerContextMath.Interval],
                                                       streamNames: [String] = ["musicfm_30", "muq_30", "musicfm_120", "muq_120"]) throws -> [[String: Any]] {
        let width = streamNames.count*1024
        guard !streamNames.isEmpty, !sourceIndices.isEmpty, actualValues.count == sourceIndices.count*width,
            reference.count == intervals.count*width, sourceIndices.allSatisfy({ intervals.indices.contains($0) }),
            Set(sourceIndices).count == sourceIndices.count else { throw NSError(domain: "EncoderPipelineProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid feature diagnostic dimensions"]) }
        var streamDiagnostics = [[String: Any]]()
        for stream in streamNames.indices {
            var maximum = 0.0, total = 0.0, failed = 0, worstSection = 0, worstFeature = 0
            var perSection = [[String: Any]]()
            for (local, section) in sourceIndices.enumerated() {
                var sectionMax = 0.0, sectionSum = 0.0, sectionFailed = 0
                for feature in 0..<1024 {
                    let actualIndex = local*width+stream*1024+feature
                    let referenceIndex = section*width+stream*1024+feature
                    let actual = Double(actualValues[actualIndex]), expected = Double(reference[referenceIndex])
                    let error = abs(actual-expected)
                    total += error; sectionSum += error; sectionMax = max(sectionMax,error)
                    if error > 1e-4+1e-4*abs(expected) { failed += 1; sectionFailed += 1 }
                    if error > maximum { maximum = error; worstSection = local; worstFeature = feature }
                }
                perSection.append(["index": section, "start": intervals[section].start, "end": intervals[section].end,
                    "max_abs_error": sectionMax, "mean_abs_error": sectionSum/1024, "failed_values": sectionFailed])
            }
            let index = worstSection*width+stream*1024+worstFeature
            let sourceSection = sourceIndices[worstSection]
            let referenceIndex = sourceSection*width+stream*1024+worstFeature
            streamDiagnostics.append(["stream": streamNames[stream], "max_abs_error": maximum,
                "mean_abs_error": total/Double(sourceIndices.count*1024), "failed_values": failed,
                "worst_section_index": sourceSection, "worst_feature_index": worstFeature,
                "worst_reference": Double(reference[referenceIndex]), "worst_actual": Double(actualValues[index]),
                "sections": perSection])
        }
        return streamDiagnostics
    }
    private func manifest() throws -> Manifest {
        guard let url = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json") else { throw failure("Missing manifest") }
        let result = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        guard result.version == "encoder-pipeline-phone-v2", result.tracks.count >= 1 else { throw failure("Unsupported manifest") }
        guard result.encoder_roles == nil || (result.encoder_roles == ["muq"] && result.canonical_encoder_policy != nil) else { throw failure("Unsupported encoder roles") }
        if result.canonical_encoder_policy != nil {
            let functions = ["750": "short_30", "3000": "long_120", "partial": "partial"]
            let roles = Self.roles(result)
            guard result.canonical_encoder_policy == "long-profile-shared-palette-v1",
                result.canonical_encoder_schedule == "one-role-at-a-time-v1",
                result.tensor_transfer_policy == ProbeOwnedTensorCopy.policy,
                result.canonical_encoder_functions == Dictionary(uniqueKeysWithValues: roles.map { ($0, functions) }),
                result.canonical_encoder_reports?.count == roles.count,
                result.compiled_encoder_bytes != nil,
                result.frontend_route_probe == false, result.frontend_gpu_policy == "disabled",
                result.static_graph_policy == "split-before-conv-v2",
                Set(result.files.keys) == Set(["log_mel", "head", "context"] + roles + roles.map { $0 + "_conv" })
            else { throw failure("Invalid canonical encoder function/resource policy") }
            if result.encoder_roles == ["muq"] {
                // MuQ-only admission: Mac functional gates (labels, base/final probabilities) on the same models.
                guard result.mac_preflight_policy == "muq-functional-v1" else { throw failure("MuQ-only probe requires its Mac functional preflight") }
            } else if result.mac_preflight_policy == "owned-buffer-diagnostic" {
                guard result.cpu_control_receipt_sha256 == "066006ed14b4850c4bce05a5445b5a53ff5bc962ce30ec1ad85082a6b4a00770" else { throw failure("Missing reviewed CPU control for owned-buffer diagnostic") }
            } else {
                guard result.mac_preflight_policy == "strict-feature", result.mac_feature_parity_passed == true else { throw failure("Strict canonical preflight did not pass") }
            }
        }
        return result
    }
    func gpuAvailable() throws -> Bool { try manifest().static_graph_policy == "split-before-conv-v2" }
    func frontendGPUAvailable() throws -> Bool {
        let m = try manifest()
        return m.frontend_route_probe == true && m.frontend_gpu_policy == "full-30s-only-v1" && m.static_graph_policy == "split-before-conv-v2" && m.graph_repair_sha256 != nil
    }
    private func manifestHash() throws -> String {
        guard let url = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json") else { throw failure("Missing manifest") }
        return try hash(url)
    }
    func tracks() throws -> [String] { try manifest().tracks.map(\.title) }
    func run(trackIndex: Int, mode: String, frontendMode: String = "cpuOnly", progress: @escaping @Sendable (String) -> Void) throws -> URL {
        guard !busy else { throw failure("Another benchmark is running") }
        busy = true; defer { busy = false }
        return try autoreleasepool { try execute(trackIndex: trackIndex, mode: mode, frontendMode: frontendMode, progress: progress) }
    }
    private struct BatchRun: Sendable {
        let pass: Int
        let trackIndex: Int
        let mode: String
        let frontendMode: String
    }
    func batch(mode: String, frontendMode: String = "cpuOnly", progress: @escaping @Sendable (String) -> Void) async throws -> URL {
        let m = try manifest()
        let schedule = (1...3).flatMap { pass in m.tracks.indices.map { BatchRun(pass: pass, trackIndex: $0, mode: mode, frontendMode: frontendMode) } }
        return try await performBatch(schedule: schedule, kind: "same-route-three-pass", progress: progress)
    }
    func compareFrontends(progress: @escaping @Sendable (String) -> Void) async throws -> URL {
        guard try frontendGPUAvailable() else { throw failure("Frontend route comparison is not enabled by validated preparation") }
        let m = try manifest()
        let schedule = m.tracks.indices.flatMap { index in
            ["cpuOnly", "gpuShort30", "gpuShort30", "cpuOnly"].enumerated().map { position, mode in
                BatchRun(pass: position/2+1, trackIndex: index, mode: "cpuAndGPU", frontendMode: mode)
            }
        }
        return try await performBatch(schedule: schedule, kind: "frontend-ABBA", progress: progress)
    }
    private func performBatch(schedule: [BatchRun], kind: String, progress: @escaping @Sendable (String) -> Void) async throws -> URL {
        guard !busy else { throw failure("Another benchmark is running") }
        busy = true; defer { busy = false }
        let m = try manifest(); let digest = try manifestHash()
        var entries = [[String: Any]](); let id = UUID().uuidString; var errorText: String?; var activeRun = 0
        let encoderModes = Set(schedule.map(\.mode)); let frontendModes = Set(schedule.map(\.frontendMode))
        func persist(_ state: String, active: Int) throws {
            var object: [String: Any] = ["batch_id": id, "state": state, "active_run": active, "batch_kind": kind,
                "quality_sha256": m.quality_sha256, "probe_source_sha256": m.probe_source_sha256, "static_graph_policy": m.static_graph_policy,
                "manifest_sha256": digest, "graph_repair_sha256": m.graph_repair_sha256 ?? "none",
                "mode": encoderModes.count == 1 ? encoderModes.first! : "mixed",
                "frontend_mode": frontendModes.count == 1 ? frontendModes.first! : "mixed",
                "frontend_gpu_policy": m.frontend_gpu_policy ?? "disabled",
                "requested_runs": schedule.count, "completed_runs": entries.count, "runs": entries,
                "note": "Same-process scheduled runs with 2-second idle gaps. Frontend ABBA uses CPU, GPU-short-30s, GPU-short-30s, CPU per song; long/partial frontends stay CPU and encoder modes stay fixed. OS cache is not guaranteed cold. No automatic app approval."]
            if let errorText { object["error"] = errorText }
            object["mac_preflight_policy"] = m.mac_preflight_policy ?? "legacy_strict"
            if let parity = m.mac_feature_parity_passed { object["mac_feature_parity_passed"] = parity }
            if let policy = m.canonical_encoder_policy {
                object["canonical_encoder_policy"] = policy
                object["canonical_encoder_schedule"] = m.canonical_encoder_schedule
                object["tensor_transfer_policy"] = m.tensor_transfer_policy
                object["cpu_control_receipt_sha256"] = m.cpu_control_receipt_sha256
                object["canonical_encoder_reports"] = m.canonical_encoder_reports
                object["compiled_encoder_bytes"] = m.compiled_encoder_bytes
            }
            object["encoder_roles"] = Self.roles(m)
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: Self.batchURL, options: .atomic)
        }
        try persist("running", active: 1)
        do {
            for request in schedule {
                try Task.checkCancellation()
                let number = entries.count+1; activeRun = number; try persist("running", active: number)
                let url = try autoreleasepool {
                    try execute(trackIndex: request.trackIndex, mode: request.mode, frontendMode: request.frontendMode) {
                        progress("Run \(number)/\(schedule.count), frontend \(request.frontendMode): " + $0)
                    }
                }
                let data = try Data(contentsOf: url)
                let report = try JSONSerialization.jsonObject(with: data)
                entries.append(["pass": request.pass, "track_index": request.trackIndex, "frontend_mode": request.frontendMode,
                    "report": report, "report_file": url.lastPathComponent])
                try persist("running", active: number)
                guard (report as? [String: Any])?["label_parity_passed"] as? Bool == true else {
                    throw failure("Run completed but label/probability parity failed. Batch stopped; share batch and last-stage JSON.")
                }
                try await Task.sleep(nanoseconds: 2_000_000_000)
            }
            try persist("completed", active: 0)
        } catch {
            errorText = String(describing: error); try persist("failed", active: activeRun); throw error
        }
        return Self.batchURL
    }
    private func execute(trackIndex: Int, mode: String, frontendMode: String, progress: @escaping @Sendable (String) -> Void) throws -> URL {
        let m = try manifest()
        guard m.tracks.indices.contains(trackIndex), ["cpuOnly", "cpuAndGPU"].contains(mode) else { throw failure("Invalid selection") }
        guard mode == "cpuOnly" || m.static_graph_policy == "split-before-conv-v2" else { throw failure("GPU route for the old graph is blocked after a fatal MPS assertion. Prepare the repaired candidate.") }
        guard ["cpuOnly", "gpuShort30"].contains(frontendMode) else { throw failure("Invalid frontend mode") }
        guard frontendMode == "cpuOnly" || (m.frontend_route_probe == true && m.frontend_gpu_policy == "full-30s-only-v1" && m.static_graph_policy == "split-before-conv-v2" && m.graph_repair_sha256 != nil) else { throw failure("Frontend GPU route requires full repaired-graph validation and isolated probe preparation") }
        let track = m.tracks[trackIndex]; let id = UUID().uuidString
        var report: [String: Any] = ["version": m.version, "run_id": id, "track_id": track.id, "title": track.title,
            "manifest_sha256": try manifestHash(), "graph_repair_sha256": m.graph_repair_sha256 ?? "none",
            "requested_convolution_units": frontendMode == "gpuShort30" ? "mixed_by_window" : "cpuOnly",
            "convolution_mode": frontendMode, "frontend_gpu_policy": m.frontend_gpu_policy ?? "disabled", "log_mel_units": "cpuOnly",
            "requested_full_window_units": mode, "partial_window_units": "cpuOnly", "quality_sha256": m.quality_sha256,
            "probe_source_sha256": m.probe_source_sha256, "static_graph_policy": m.static_graph_policy,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "process_identifier": ProcessInfo.processInfo.processIdentifier,
            "bundle_identifier": Bundle.main.bundleIdentifier ?? "unknown",
            "thermal_before": ProcessInfo.processInfo.thermalState.rawValue, "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "ready_for_app_integration": false,
            "scope": "Bundled 24-kHz mono PCM -> actual frontends -> static full / legacy CPU partial encoders -> palette6 head -> frozen native context. Supplied fixed Apple boundaries; no MP3 decoding, MusicUnderstanding, beats, chords or lyrics."]
        report["mac_preflight_policy"] = m.mac_preflight_policy ?? "legacy_strict"
        if let parity = m.mac_feature_parity_passed { report["mac_feature_parity_passed"] = parity }
        if let policy = m.canonical_encoder_policy {
            report["canonical_encoder_policy"] = policy
            report["canonical_encoder_schedule"] = m.canonical_encoder_schedule
            report["tensor_transfer_policy"] = m.tensor_transfer_policy
            report["cpu_control_receipt_sha256"] = m.cpu_control_receipt_sha256
            report["canonical_encoder_reports"] = m.canonical_encoder_reports
            report["compiled_encoder_bytes"] = m.compiled_encoder_bytes
            report["scope"] = "Bundled PCM -> CPU frontends -> canonical multi-function encoders (full windows selected CPU or CPU+GPU; partial CPU) -> palette6 head -> frozen native context. Fixed Apple boundaries; no MP3 decoding, MusicUnderstanding, beats, chords or lyrics."
        }
        #if targetEnvironment(simulator)
        report["simulator"] = true
        #else
        report["simulator"] = false
        #endif
        var system = utsname(); uname(&system)
        let capacity = MemoryLayout.size(ofValue: system.machine)
        report["hardware_identifier"] = withUnsafePointer(to: &system.machine) { p in p.withMemoryRebound(to: CChar.self, capacity: capacity) { String(cString: $0) } }
        var memory = [[String: Any]](); var loadTimes = [[String: Any]](); var predictions = [[String: Any]](); var routes = [[String: Any]]()
        var completedChunks = [[String: Any]](); var inlineDiagnosticsSeconds = 0.0
        let checkpointStarted = ProcessInfo.processInfo.systemUptime
        func emit(_ stage: String) throws {
            progress(track.title + ": " + stage); NSLog("PIPELINE_GRAPH %@ %@", mode, stage)
            report["last_stage"] = stage; report["state"] = "running"
            report["checkpoint_written_at_unix_seconds"] = Date().timeIntervalSince1970
            report["checkpoint_elapsed_seconds"] = ProcessInfo.processInfo.systemUptime-checkpointStarted
            report["model_load_events"] = loadTimes; report["prediction_events"] = predictions; report["routes"] = routes
            report["completed_head_chunks"] = completedChunks
            if let bytes = footprint() { memory.append(["stage": stage, "bytes": bytes]); report["memory_snapshots"] = memory }
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: Self.checkpoint, options: .atomic)
        }
        do {
        try emit("integrity_start")
        if m.tensor_transfer_policy == ProbeOwnedTensorCopy.policy {
            try ProbeOwnedTensorCopy.selfTest()
            report["native_owned_tensor_self_test_passed"] = true
        }
        let preflightStart = ProcessInfo.processInfo.systemUptime
        for (resource, files) in m.files where !verified.contains(resource) {
            guard let folder = Bundle.main.url(forResource: resource, withExtension: "mlmodelc") else { throw failure("Missing \(resource)") }
            for (path, expected) in files {
                guard !path.hasPrefix("/"), !path.split(separator: "/").contains(".."), try hash(folder.appendingPathComponent(path)) == expected else { throw failure("Model file checksum differs: \(resource)/\(path)") }
            }
            verified.insert(resource)
        }
        let audio = try readFloats(track.pcm, expected: track.pcm_sha256)
        let referenceMeans = try readFloats(track.means, expected: track.means_sha256)
        let roles = Self.roles(m), streamNames = Self.streamNames(roles), featureWidth = streamNames.count*1024
        report["encoder_roles"] = roles; report["feature_width"] = featureWidth
        if m.encoder_roles == ["muq"] {
            report["scope"] = "Bundled PCM -> CPU log-mel + MuQ frontends -> canonical MuQ multi-function bank only (MusicFM not bundled or loaded) -> 2048-input palette6 head -> 2048-input native context. Fixed Apple boundaries; no MP3 decoding, MusicUnderstanding, beats, chords or lyrics."
        }
        guard audio.count == track.samples, referenceMeans.count == track.intervals.count*featureWidth else { throw failure("Fixture dimensions differ") }
        report["integrity_and_fixture_read_seconds"] = ProcessInfo.processInfo.systemUptime-preflightStart
        try emit("pipeline_start")
        let started = ProcessInfo.processInfo.systemUptime
        let accumulator = try SongFormerContextMath.Accumulator(intervals: track.intervals, duration: track.duration, width: featureWidth)
        var modelUnits = [String: String](); var modelFunctions = [String: String](); var models = [String: (String, MLModel)]()
        func load(_ slot: String, _ resource: String, gpu: Bool, key: String? = nil, functionName: String? = nil) throws -> MLModel {
            let identity = (key ?? resource) + (functionName.map { ":function=" + $0 } ?? "")
            if let existing = models[slot], existing.0 == identity { return existing.1 }
            autoreleasepool { _ = models.removeValue(forKey: slot) }
            report.removeValue(forKey: "active_prediction")
            report["active_model"] = ["slot": slot, "resource": resource, "function_name": functionName ?? "main", "units": gpu ? "cpuAndGPU" : "cpuOnly"]
            try emit("load_start \(resource) \(gpu ? "cpuAndGPU" : "cpuOnly")")
            guard let url = Bundle.main.url(forResource: resource, withExtension: "mlmodelc") else { throw failure("Missing model \(resource)") }
            let configuration = MLModelConfiguration(); configuration.computeUnits = gpu ? .cpuAndGPU : .cpuOnly
            if let functionName { configuration.functionName = functionName }
            let begin = ProcessInfo.processInfo.systemUptime; let model = try MLModel(contentsOf: url, configuration: configuration)
            let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
            if let functionName {
                guard m.canonical_encoder_policy == "long-profile-shared-palette-v1", resource == slot,
                    ["musicfm", "muq"].contains(slot),
                    m.canonical_encoder_functions?[slot]?.values.contains(functionName) == true
                else { throw failure("Unapproved canonical encoder function") }
            } else if resource.hasSuffix("_750") || resource.hasSuffix("_3000") {
                if m.static_graph_policy == "split-before-conv-v2" {
                    guard metadata["graph_repair_policy"] == m.static_graph_policy else { throw failure("Missing split-convolution repair metadata") }
                }
                guard metadata["graph_protocol"] == "encoder-rank3-explicit-glu-v1", metadata["source_state_sha256"] == m.graph_states[slot], metadata["compression_scheme"] == "palette6" else { throw failure("Static graph metadata differs") }
            } else if slot == "head" {
                guard metadata["checkpoint_sha256"] == m.base_checkpoint, metadata["compression_scheme"] == "palette6" else { throw failure("Head metadata differs") }
            } else if slot == "context" {
                let lineage = m.encoder_roles == ["muq"] ? metadata["variant"] == "muq" && metadata["input_width"] == "2048" : metadata["selected_epoch"] == "5"
                guard metadata["context_checkpoint_sha256"] == m.context_checkpoint, lineage, metadata["precision"] == "fp32" else { throw failure("Context metadata differs") }
            }
            loadTimes.append(["resource": resource, "function_name": functionName ?? "main", "units": gpu ? "cpuAndGPU" : "cpuOnly", "seconds": ProcessInfo.processInfo.systemUptime-begin])
            modelUnits[slot] = gpu ? "cpuAndGPU" : "cpuOnly"
            modelFunctions[slot] = functionName ?? "main"
            models[slot] = (identity, model); try emit("load_done \(resource)"); return model
        }
        func predict(_ model: MLModel, role: String, inputs: [String: MLMultiArray]) throws -> MLMultiArray {
            report["active_prediction"] = ["role": role, "resource_key": models[role]?.0 ?? role, "requested_units": modelUnits[role] ?? "unknown",
                "function_name": modelFunctions[role] ?? "main",
                "inputs": inputs.mapValues { ["shape": $0.shape.map(\.intValue), "strides": $0.strides.map(\.intValue), "dtype": $0.dataType.rawValue] as [String: Any] }] as [String: Any]
            try emit("predict_start \(role)")
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            let begin = ProcessInfo.processInfo.systemUptime
            var copySeconds = 0.0
            let array: MLMultiArray = try autoreleasepool {
                let output = try model.prediction(from: provider)
                guard output.featureNames.count == 1, let key = output.featureNames.first,
                    let source = output.featureValue(for: key)?.multiArrayValue else { throw failure("Invalid model output: \(role)") }
                if m.tensor_transfer_policy == ProbeOwnedTensorCopy.policy {
                    let startedCopy = ProcessInfo.processInfo.systemUptime
                    // Complete the independent copy while the producer provider is still retained.
                    let owned = try withExtendedLifetime(output) { try ProbeOwnedTensorCopy.float32(source) }
                    copySeconds = ProcessInfo.processInfo.systemUptime-startedCopy
                    return owned
                }
                return source
            }
            let elapsed = ProcessInfo.processInfo.systemUptime-begin
            predictions.append(["component": role, "resource_key": models[role]?.0 ?? role, "function_name": modelFunctions[role] ?? "main", "requested_units": modelUnits[role] ?? "unknown", "seconds": elapsed, "owned_copy_seconds": copySeconds, "shape": array.shape.map(\.intValue)])
            try emit("predict_done \(role)"); return array
        }
        func frontend(_ role: String, tensor: MLMultiArray, shortFullWindow: Bool = false) throws -> MLMultiArray {
            let shape = tensor.shape.map(\.intValue)
            let useGPU = frontendMode == "gpuShort30" && shortFullWindow && shape == [1,128,3000] && ["musicfm_conv", "muq_conv"].contains(role)
            let identity = role + ":" + (useGPU ? "cpuAndGPU" : "cpuOnly") + ":" + shape.map { String($0) }.joined(separator: "x")
            let model = try load(role, role, gpu: useGPU, key: identity)
            guard model.modelDescription.inputDescriptionsByName.count == 1, let key = model.modelDescription.inputDescriptionsByName.keys.first else { throw failure("Unexpected frontend contract") }
            return try SongFormerTensorBridge.float32(predict(model, role: role, inputs: [key: tensor]))
        }
        func pair(_ start: Int, _ end: Int, seconds: Int, roles: [String] = ["musicfm", "muq"]) throws -> [[Float]] {
            try autoreleasepool {
                let n = end-start; guard n > 2048 else { throw failure("Encoder window too short") }
                var padded = [Float](); padded.reserveCapacity(n+2048)
                for i in -1024..<(n+1024) {
                    let index = i < 0 ? -i : (i >= n ? 2*n-2-i : i); padded.append(audio[start+index])
                }
                report["active_window"] = ["start_sample": start, "samples": n, "stream_seconds": seconds]
                try emit("window \(Double(start)/24000)s length=\(Double(n)/24000)s stream=\(seconds)s")
                let mel = try frontend("log_mel", tensor: input(padded, [1,padded.count])); var result = [[Float]]()
                for role in roles {
                    let values: [Float] = try autoreleasepool {
                        let conv = try frontend(role+"_conv", tensor: mel, shortFullWindow: seconds == 30 && n == 30*24000); let shape = conv.shape.map(\.intValue)
                        guard shape.count == 3, shape[0] == 1, shape[2] == 1024, let frequencies = m.rotary[role], frequencies.count == 32 else { throw failure("Unexpected convolution/rotary contract") }
                        let count = shape[1]; let full = n == seconds*24000 && count == seconds*25
                        if m.canonical_encoder_policy != nil && n == seconds*24000 && !full { throw failure("Canonical full-window frame count differs") }
                        var cosines = [Float](repeating: 0, count: count*64), sines = cosines
                        for t in 0..<count { for c in 0..<32 {
                            let angle = Float(t)*frequencies[c]; let cosine = cosf(angle), sine = sinf(angle)
                            cosines[t*64+c] = cosine; cosines[t*64+c+32] = cosine
                            sines[t*64+c] = sine; sines[t*64+c+32] = sine
                        } }
                        var resource = full ? "\(role)_\(count)" : role
                        var functionName: String? = nil
                        if m.canonical_encoder_policy != nil {
                            guard let selected = m.canonical_encoder_functions?[role]?[full ? String(count) : "partial"] else { throw failure("Missing canonical function for this window") }
                            resource = role; functionName = selected
                        }
                        let model = try load(role, resource, gpu: full && mode == "cpuAndGPU", key: resource+":\(count)", functionName: functionName)
                        let inputs: [String: MLMultiArray] = full
                            ? ["conv_in": conv, "rotary_cos": input(cosines,[count,64]), "rotary_sin": input(sines,[count,64])]
                            : ["conv_in": conv, "rotary": input(cosines+sines,[2,count,1,1,64])]
                        let output = try predict(model, role: role, inputs: inputs)
                        guard output.shape.map(\.intValue) == shape else { throw failure("Encoder output shape differs") }
                        routes.append(["role": role, "start": Double(start)/24000, "seconds": Double(n)/24000, "stream": seconds,
                            "frames": count, "function_name": functionName ?? "main",
                            "route": m.canonical_encoder_policy != nil ? (full ? "canonical_full" : "canonical_partial_cpu") : (full ? "new_static" : "legacy_partial_cpu")])
                        return try self.values(output)
                    }
                    result.append(values)
                }
                return result
            }
        }
        var frames = [SongFormerContextMath.Frame]()
        for headStart in stride(from: 0, to: audio.count, by: 420*24000) {
            let headEnd = min(audio.count, headStart+420*24000); if headEnd-headStart <= 2048 { break }
            try autoreleasepool {
                var streams = [[Float]]()
                if m.canonical_encoder_schedule == "one-role-at-a-time-v1" {
                    var ordered = [[Float]](repeating: [], count: 2*roles.count)
                    var windowFrames = [String: Int]()
                    for (roleIndex, role) in roles.enumerated() {
                        try emit("role_phase_start \(role)")
                        for (streamIndex, seconds) in [30,120].enumerated() {
                            var values = [Float]()
                            for start in stride(from: headStart, to: headEnd, by: seconds*24000) {
                                let end = min(headEnd,start+seconds*24000); if end-start <= 2048 { continue }
                                let encoded = try pair(start,end,seconds: seconds,roles: [role])
                                guard encoded.count == 1 else { throw failure("Serial encoder role contract differs") }
                                let key = "\(seconds):\(start)", count = encoded[0].count/1024
                                guard encoded[0].count % 1024 == 0 else { throw failure("Serial encoder width differs") }
                                if roleIndex == 0 { windowFrames[key] = count }
                                else if windowFrames[key] != count { throw failure("Serial per-window frame counts differ; refusing shifted alignment") }
                                values.append(contentsOf: encoded[0])
                            }
                            ordered[streamIndex*roles.count+roleIndex] = values
                        }
                        autoreleasepool { models.removeAll() }
                        try emit("role_phase_released \(role)")
                    }
                    streams = ordered
                } else {
                guard roles == ["musicfm", "muq"] else { throw failure("Single-encoder probes require the serial schedule") }
                for seconds in [30,120] {
                    var fm = [Float](), muq = [Float]()
                    for start in stride(from: headStart, to: headEnd, by: seconds*24000) {
                        let end = min(headEnd,start+seconds*24000); if end-start <= 2048 { continue }
                        let encoded = try pair(start,end,seconds: seconds); let count = min(encoded[0].count,encoded[1].count)/1024*1024
                        fm.append(contentsOf: encoded[0].prefix(count)); muq.append(contentsOf: encoded[1].prefix(count))
                    }
                    streams.append(fm); streams.append(muq)
                }
                }
                let count = (streams.map { $0.count/1024 }.min() ?? 0)/3*3; guard count > 0 else { throw failure("No fused frames") }
                autoreleasepool { models.removeAll() }; try emit("encoders_released_before_fusion")
                let embeddings = try SongFormerTensorBridge.fuse(streams: streams, frames: count); streams.removeAll()
                let offset = Double(headStart)/24000, effectiveEnd = offset+Double(count)/25
                let completeIndices = accumulator.intervals.indices.filter {
                    accumulator.intervals[$0].start >= offset && accumulator.intervals[$0].end <= effectiveEnd
                }
                var chunkDiagnostics = [[String: Any]]()
                var appendError: Error?
                embeddings.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
                    do {
                        guard strides[1] == featureWidth, strides[2] == 1 else { throw failure("Unexpected fused layout") }
                        try accumulator.append(UnsafeBufferPointer(buffer), frames: count, offset: Double(headStart)/24000)
                        if !completeIndices.isEmpty && m.canonical_encoder_policy != nil {
                            let begin = ProcessInfo.processInfo.systemUptime
                            let diagnostic = try SongFormerContextMath.Accumulator(intervals: completeIndices.map { accumulator.intervals[$0] }, duration: track.duration, width: featureWidth)
                            try diagnostic.append(UnsafeBufferPointer(buffer), frames: count, offset: offset)
                            let complete = try diagnostic.features()
                            chunkDiagnostics = try Self.featureDiagnostics(actualValues: complete.audio, reference: referenceMeans,
                                sourceIndices: completeIndices, intervals: accumulator.intervals, streamNames: streamNames)
                            inlineDiagnosticsSeconds += ProcessInfo.processInfo.systemUptime-begin
                        }
                    } catch { appendError = error }
                }
                if let appendError { throw appendError }
                report["current_chunk_feature_diagnostics"] = ["start": offset, "effective_end": effectiveEnd,
                    "source_indices": completeIndices, "streams": chunkDiagnostics,
                    "note": "Only intervals fully contained in this completed feature chunk; no final context or global parity verdict."]
                let head = try load("head", "head", gpu: false)
                let output = try predict(head, role: "head", inputs: ["input_embeddings": embeddings])
                guard output.shape.map(\.intValue) == [1,count/3,128] else { throw failure("Invalid head shape") }
                let logits = try values(output); autoreleasepool { models.removeAll() }
                let ids = [0,1,2,3,4,5,6,26]
                for i in 0..<(count/3) {
                    let selected = ids.map { Double(logits[i*128+$0]) }; let top = selected.max()!
                    let exponentials = selected.map { exp($0-top) }; let total = exponentials.reduce(0,+)
                    frames.append(.init(time: Double(headStart)/24000+Double(i)*0.12, duration: 0.12, probabilities: exponentials.map { $0/total }))
                }
                completedChunks.append(["start": offset, "effective_end": effectiveEnd, "frames": count,
                    "source_indices": completeIndices, "pooled_feature_diagnostics": chunkDiagnostics,
                    "head_prediction_complete": true])
                report.removeValue(forKey: "current_chunk_feature_diagnostics")
                report["inline_feature_diagnostics_seconds"] = inlineDiagnosticsSeconds
                try emit("head_chunk_done \(Double(headStart)/24000)s")
            }
        }
        let prepared = try accumulator.features(); let ranges = accumulator.intervals
        guard !ranges.isEmpty, prepared.audio.count == ranges.count*featureWidth, prepared.audio.count == referenceMeans.count else { throw failure("Pooled reference feature dimensions differ") }
        let pools = try SongFormerContextMath.pool(ranges, frames: frames)
        let context = try load("context", "context", gpu: false)
        let output = try predict(context, role: "context", inputs: ["section_audio": input(prepared.audio,[ranges.count,featureWidth]), "section_timing": input(prepared.timing,[ranges.count,6])])
        guard output.shape.map(\.intValue) == [ranges.count,8] else { throw failure("Invalid context output") }
        let corrected = try SongFormerContextMath.corrected(probabilities: pools.map(\.probabilities), residual: values(output))
        autoreleasepool { models.removeAll() }
        var sections = [[String: Any]](); var agreement = 0, correct = 0; var maxProbabilityError = 0.0, maxPriorError = 0.0
        for target in track.targets {
            guard ranges.indices.contains(target.index), target.probabilities.count == 8, target.base_probabilities.count == 8 else { throw failure("Invalid target mapping") }
            let prediction = corrected[target.index]; let range = ranges[target.index]
            agreement += prediction.label == target.prediction ? 1 : 0; correct += prediction.label == target.label ? 1 : 0
            var sectionProbabilityError = 0.0, sectionPriorError = 0.0
            for j in 0..<8 {
                maxProbabilityError = max(maxProbabilityError,abs(prediction.probabilities[j]-target.probabilities[j]))
                maxPriorError = max(maxPriorError,abs(pools[target.index].probabilities[j]-target.base_probabilities[j]))
                sectionProbabilityError = max(sectionProbabilityError,abs(prediction.probabilities[j]-target.probabilities[j]))
                sectionPriorError = max(sectionPriorError,abs(pools[target.index].probabilities[j]-target.base_probabilities[j]))
            }
            sections.append(["start": range.start, "end": range.end, "manual": target.label, "mac_prediction": target.prediction,
                "prediction": prediction.label, "probabilities": prediction.probabilities,
                "index": target.index, "mac_probabilities": target.probabilities,
                "base_probabilities": pools[target.index].probabilities, "mac_base_probabilities": target.base_probabilities,
                "max_probability_error": sectionProbabilityError, "max_base_probability_error": sectionPriorError,
                "audio_coverage": accumulator.coverage[target.index]/(range.end-range.start),
                "probability_coverage": pools[target.index].coveredSeconds/(range.end-range.start)])
        }
        var maxMeanError = 0.0
        for (actual, expected) in zip(prepared.audio, referenceMeans) { maxMeanError = max(maxMeanError,abs(Double(actual)-Double(expected))) }
        report["pcm_pipeline_seconds"] = ProcessInfo.processInfo.systemUptime-started
        report["audio_seconds"] = Double(audio.count)/24000; report["section_agreement_with_mac"] = Double(agreement)/Double(track.targets.count)
        report["section_accuracy"] = Double(correct)/Double(track.targets.count); report["max_section_probability_error"] = maxProbabilityError
        report["max_base_probability_error"] = maxPriorError; report["max_pooled_audio_error"] = maxMeanError
        report["sections"] = sections; report["model_load_events"] = loadTimes; report["prediction_events"] = predictions
        report["routes"] = routes; report["low_audio_coverage_sections"] = prepared.lowCoverage
        report["thermal_after"] = ProcessInfo.processInfo.thermalState.rawValue
        report["all_predictions_finite"] = true
        report["label_parity_passed"] = agreement == track.targets.count && maxProbabilityError <= 1e-4 && maxPriorError <= 1e-4 && prepared.lowCoverage.isEmpty
        report["parity_checks"] = ["labels_identical": agreement == track.targets.count,
            "base_probability_within_1e_4": maxPriorError <= 1e-4,
            "context_probability_within_1e_4": maxProbabilityError <= 1e-4,
            "audio_coverage_passed": prepared.lowCoverage.isEmpty]
        // Reporting only, after the pipeline timer: locate drift without changing inference or admission.
        let diagnosticsStarted = ProcessInfo.processInfo.systemUptime
        let streamDiagnostics = try Self.featureDiagnostics(actualValues: prepared.audio, reference: referenceMeans,
            sourceIndices: Array(ranges.indices), intervals: ranges, streamNames: streamNames)
        report["pooled_feature_diagnostics"] = streamDiagnostics
        report["feature_diagnostics_note"] = "All source intervals, four ordered encoder streams vs the SAME canonical Mac means. Elementwise atol=rtol=1e-4 is diagnostic only; existing final-label/base/context probability gates are unchanged. No isolated component or exact window cause is established by pooled errors alone."
        report["feature_diagnostics_seconds"] = ProcessInfo.processInfo.systemUptime-diagnosticsStarted
        report["memory_note"] = "Process footprint sampled at stage checkpoints, not true peak memory. Timings include progress/checkpoint and inline feature-diagnostic overhead (reported separately); integrity/fixture reads and final feature diagnostics are separate. Requested units do not prove GPU placement."
        try emit("pipeline_done")
        report["state"] = "completed"
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys])
        let url = Self.documents.appendingPathComponent("pipeline-graph-\(id).json")
        try data.write(to: url, options: .atomic); try data.write(to: Self.checkpoint, options: .atomic)
        return url
        } catch {
            report["state"] = "failed"; report["error"] = String(describing: error)
            report["caught_swift_error"] = true
            report["error_domain"] = (error as NSError).domain; report["error_code"] = (error as NSError).code
            if let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys]) {
                try? data.write(to: Self.checkpoint, options: .atomic)
            }
            throw error
        }
    }
}
