import Foundation
import CoreML
import Darwin

actor BenchmarkRunner {
    struct Section: Decodable {
        let start: Double
        let end: Double
        let label_manual: String
        let prediction: String
        let probabilities: [String: Double]
    }
    struct Manifest: Decodable {
        let track_id: String
        let sample_rate: Int
        let audio_samples: Int
        let head_window_seconds: Int
        let long_window_seconds: Int
        let rotary_inv_freq: [String: [Float]]
        let labels: [String]
        let class_ids: [Int]
        let sections: [Section]
        let source_validation_sha256: String
        let scope: String
        let shape_policy: String
        let hybrid: Bool?
        let static_resources: [String: String]?
    }
    func failure(_ message: String) -> NSError {
        NSError(domain: "PipelineBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    func footprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let code = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return code == KERN_SUCCESS ? info.phys_footprint : nil
    }
    func floats(_ a: MLMultiArray) throws -> [Float] {
        // Match Python native_input: convert frontend FP16/FP32 outputs explicitly.
        // The converting initializer also copies strided arrays into canonical order.
        let result = MLShapedArray<Float>(converting: a).scalars
        guard result.allSatisfy({ $0.isFinite }) else { throw failure("NaN/Inf output") }
        return result
    }
    func input(_ values: [Float], _ shape: [Int]) -> MLMultiArray {
        MLMultiArray(MLShapedArray<Float>(scalars: values, shape: shape))
    }
    func run(progress: @escaping @Sendable (String) -> Void) throws -> String {
        // Ensure autoreleased Core ML objects from a completed run are drained before the next run.
        return try autoreleasepool { try execute(progress: progress) }
    }
    private func execute(progress: @escaping @Sendable (String) -> Void) throws -> String {
        let checkpointURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pipeline-last-stage.json")
        let runID = UUID().uuidString
        func emit(_ message: String) {
            NSLog("PIPELINE %@", message)
            progress(message)
            if message.hasPrefix("START") || message.hasPrefix("WINDOW") || message.hasPrefix("FUSE") ||
                message.hasPrefix("MEMORY") || message.hasPrefix("HEAD_CHUNK_DONE") || message.hasPrefix("DONE") {
                var checkpoint: [String: Any] = ["run_id":runID,"last_stage":message,
                    "recorded_at":ISO8601DateFormatter().string(from: Date()),
                    "state":message.hasPrefix("DONE") ? "completed" : "running",
                    "note":"Last recorded stage only; this file does not determine the OS termination reason."]
                if let bytes = footprint() { checkpoint["process_footprint_bytes"] = bytes }
                do {
                    let data = try JSONSerialization.data(withJSONObject: checkpoint, options: [.prettyPrinted,.sortedKeys])
                    try data.write(to: checkpointURL, options: .atomic)
                } catch { NSLog("PIPELINE checkpoint write failed: %@", error.localizedDescription) }
            }
        }
        emit("START v3 staged memory: reading manifest and PCM")
        let bundle = Bundle.main
        guard let url = bundle.url(forResource: "benchmark_manifest", withExtension: "json"),
              let audioURL = bundle.url(forResource: "audio", withExtension: "f32") else { throw failure("Missing resources") }
        let m = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        let hybrid = m.hybrid ?? false
        let data = try Data(contentsOf: audioURL)
        guard data.count == m.audio_samples * 4 else { throw failure("PCM size mismatch") }
        let audio: [Float] = data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
        guard audio.allSatisfy({ $0.isFinite }), m.sample_rate == 24000,
              m.labels.count == 8, m.class_ids == [0,1,2,3,4,5,6,26] else { throw failure("Invalid PCM/profile") }
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        var snapshots: [UInt64] = []
        if let b = footprint() { snapshots.append(b) }
        let loadStart = ProcessInfo.processInfo.systemUptime
        let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
        var models: [String: MLModel] = [:]
        var activeEncoderResource: [String: String] = [:]
        var encoderLoadSeconds = 0.0
        var stagedLoadSeconds = 0.0
        var memoryEvents = [[String: Any]]()
        var encoderLoadEvents = [[String: Any]]()
        for role in ["log_mel","musicfm_conv","musicfm","muq_conv","muq","head"] {
            if hybrid && (role == "musicfm" || role == "muq" || role == "head") { continue }
            guard let path = bundle.url(forResource: role, withExtension: "mlmodelc") else { throw failure("Missing \(role)") }
            emit("LOAD_START: \(role)")
            models[role] = try MLModel(contentsOf: path, configuration: config)
            emit("LOAD_DONE: \(role)")
        }
        let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStart
        if let b = footprint() { snapshots.append(b) }
        func memoryEvent(_ stage: String) {
            if let bytes = footprint() {
                snapshots.append(bytes)
                memoryEvents.append(["stage":stage,"bytes":bytes])
                emit("MEMORY: \(stage), \(bytes) bytes")
            }
        }
        func loadCPU(_ role: String) throws {
            if models[role] != nil { return }
            guard let path = bundle.url(forResource: role, withExtension: "mlmodelc") else { throw failure("Missing \(role)") }
            emit("STAGED_LOAD_START: \(role), cpuOnly")
            let started = ProcessInfo.processInfo.systemUptime
            models[role] = try MLModel(contentsOf: path, configuration: config)
            stagedLoadSeconds += ProcessInfo.processInfo.systemUptime - started
            emit("STAGED_LOAD_DONE: \(role)")
        }
        var timings: [String: Double] = [:], calls: [String: Int] = [:]
        func predict(_ role: String, _ inputs: [String: MLMultiArray]) throws -> MLMultiArray {
            let started = ProcessInfo.processInfo.systemUptime
            emit("PREDICT_START: \(role), call \((calls[role] ?? 0) + 1)")
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            let result = try autoreleasepool { try models[role]!.prediction(from: provider) }
            timings[role, default: 0] += ProcessInfo.processInfo.systemUptime - started
            calls[role, default: 0] += 1
            emit("PREDICT_DONE: \(role), \(String(format: "%.3f", ProcessInfo.processInfo.systemUptime - started))s")
            guard let key = result.featureNames.sorted().first,
                  let output = result.featureValue(for: key)?.multiArrayValue else { throw failure("Missing output \(role)") }
            emit("OUTPUT: \(role), shape=\(output.shape), dtype=\(output.dataType.rawValue), strides=\(output.strides)")
            return output
        }
        func first(_ role: String) -> String { models[role]!.modelDescription.inputDescriptionsByName.keys.sorted().first! }
        func selectEncoder(_ role: String, _ frames: Int, _ samples: Int, _ seconds: Int) throws -> String {
            if !hybrid { return "cpuOnly_dynamic" }
            let expected = seconds == 30 ? 750 : 3000
            let useStatic = (seconds == 30 || seconds == 120) && samples == seconds*24000 && frames == expected
            let key = "\(role)_\(frames)"
            let resource: String
            if useStatic {
                guard let bundled = m.static_resources?[key] else { throw failure("Missing static resource \(key)") }
                resource = bundled
            } else { resource = role }
            if activeEncoderResource[role] != resource {
                // Keep one active encoder model per role instead of all shapes at once.
                models.removeValue(forKey: role)
                guard let path = bundle.url(forResource: resource, withExtension: "mlmodelc") else { throw failure("Missing \(resource)") }
                let route = useStatic ? "all_static" : "cpuOnly_dynamic_tail"
                emit("ENCODER_LOAD_START: \(resource), \(route)")
                let started = ProcessInfo.processInfo.systemUptime
                let cfg = MLModelConfiguration(); cfg.computeUnits = useStatic ? .all : .cpuOnly
                models[role] = try MLModel(contentsOf: path, configuration: cfg)
                let elapsed = ProcessInfo.processInfo.systemUptime - started
                encoderLoadSeconds += elapsed
                encoderLoadEvents.append(["role":role,"resource":resource,"route":route,"seconds":elapsed])
                activeEncoderResource[role] = resource
                emit("ENCODER_LOAD_DONE: \(resource), \(elapsed)s")
                if let b = footprint() { snapshots.append(b) }
            }
            return useStatic ? "all_static" : "cpuOnly_dynamic_tail"
        }
        var windows: [[String: Any]] = []
        func pair(_ audio: [Float], _ offset: Int, _ seconds: Int) throws -> [[Float]] {
            return try autoreleasepool {
            emit("WINDOW: start=\(Double(offset)/24000)s, length=\(Double(audio.count)/24000)s, stream=\(seconds)s")
            let n = audio.count
            var padded = [Float](); padded.reserveCapacity(n + 2048)
            for i in -1024..<(n+1024) {
                let index = i < 0 ? -i : (i >= n ? 2*n-2-i : i)
                padded.append(audio[index])
            }
            let melOutput = try predict("log_mel", [first("log_mel"): input(padded,[1,n+2048])])
            let mel = input(try floats(melOutput), melOutput.shape.map { $0.intValue })
            var results = [[Float]]()
            for role in ["musicfm","muq"] {
                let convOutput = try predict(role+"_conv", [first(role+"_conv"): mel])
                let shape = convOutput.shape.map { $0.intValue }
                guard shape.count == 3, shape[0] == 1, shape[2] == 1024 else { throw failure("Bad \(role) conv shape: \(shape)") }
                emit("READ_CONV: \(role), converting frontend output to Float32")
                let conv = input(try floats(convOutput), shape)
                emit("READ_CONV_DONE: \(role)")
                let frames = shape[1]
                guard let frequencies = m.rotary_inv_freq[role], frequencies.count == 32 else { throw failure("Invalid rotary") }
                var rotary = [Float](repeating: 0, count: 2*frames*64)
                for t in 0..<frames {
                    for c in 0..<64 {
                        let angle = Float(t) * frequencies[c % 32]
                        rotary[t*64+c] = cos(angle)
                        rotary[frames*64+t*64+c] = sin(angle)
                    }
                }
                let route = try selectEncoder(role, frames, n, seconds)
                emit("ROUTE: \(role), \(frames) frames, \(route)")
                let output = try predict(role, ["conv_in": conv,"rotary": input(rotary,[2,frames,1,1,64])])
                guard output.shape.map({$0.intValue}) == shape else { throw failure("Encoder shape mismatch") }
                results.append(try floats(output))
                windows.append(["encoder":role,"start_sample":offset,"samples":n,"stream_seconds":seconds,"frames":frames,"route":route])
            }
            return results
            }
        }
        var sums = Array(repeating: Array(repeating: 0.0, count: 8), count: m.sections.count)
        var covered = Array(repeating: 0.0, count: m.sections.count)
        let begin = ProcessInfo.processInfo.systemUptime
        let headStep = m.head_window_seconds * m.sample_rate
        for headOffset in stride(from: 0, to: audio.count, by: headStep) {
            let end = min(audio.count, headOffset+headStep)
            if end-headOffset <= 2048 { break }
            if hybrid {
                for role in ["log_mel","musicfm_conv","muq_conv"] { try loadCPU(role) }
            }
            var streams = [[Float]]()
            for seconds in [30,m.long_window_seconds] {
                var fm = [Float](), muq = [Float]()
                for start in stride(from: headOffset, to: end, by: seconds*m.sample_rate) {
                    let stop = min(end,start+seconds*m.sample_rate)
                    if stop-start <= 2048 { continue }
                    let encoded = try pair(Array(audio[start..<stop]),start,seconds)
                    fm.append(contentsOf: encoded[0]);muq.append(contentsOf: encoded[1])
                }
                streams.append(fm);streams.append(muq)
            }
            let lengths = streams.map { $0.count/1024 }
            guard let minimum = lengths.min(), let maximum = lengths.max(), maximum-minimum <= 4 else { throw failure("Stream lengths diverged") }
            let frames = minimum/3*3
            guard frames >= 3, frames <= 10500 else { throw failure("Invalid fused frame count") }
            emit("FUSE: \(frames) frames; then head and section pooling")
            var fused = [Float](repeating: 0,count: frames*4096)
            for t in 0..<frames {
                for stream in 0..<4 {
                    fused.replaceSubrange((t*4096+stream*1024)..<(t*4096+(stream+1)*1024),with: streams[stream][(t*1024)..<((t+1)*1024)])
                }
            }
            streams.removeAll(keepingCapacity: false)
            guard fused.allSatisfy({ $0.isFinite }) else { throw failure("NaN/Inf in fused head input") }
            let headInput = input(fused,[1,frames,4096])
            fused.removeAll(keepingCapacity: false)
            memoryEvent("before_head_release")
            if hybrid {
                autoreleasepool {
                    for role in ["musicfm","muq","log_mel","musicfm_conv","muq_conv"] {
                        models.removeValue(forKey: role)
                    }
                    activeEncoderResource.removeAll()
                }
                memoryEvent("after_encoder_release")
                try loadCPU("head")
            }
            memoryEvent("before_head_prediction")
            let result = try predict("head",["input_embeddings":headInput])
            guard result.shape.map({$0.intValue}) == [1,frames/3,128] else { throw failure("Invalid head output shape") }
            let logits = try floats(result)
            if hybrid {
                autoreleasepool { _ = models.removeValue(forKey: "head") }
                memoryEvent("after_head_release")
            }
            for t in 0..<(frames/3) {
                let selected = m.class_ids.map { Double(logits[t*128+$0]) }
                let maximum = selected.max()!
                let expValues = selected.map { exp($0-maximum) }
                let total = expValues.reduce(0,+)
                let time = Double(headOffset)/Double(m.sample_rate) + Double(t)*0.12
                for (i,s) in m.sections.enumerated() {
                    let overlap = max(0,min(time+0.12,s.end)-max(time,s.start))
                    if overlap > 0 {
                        for j in 0..<8 { sums[i][j] += expValues[j]/total*overlap }
                        covered[i] += overlap
                    }
                }
            }
            if let b = footprint() { snapshots.append(b) }
            emit("HEAD_CHUNK_DONE: offset=\(headOffset), frames=\(frames)")
        }
        let inferenceSeconds = ProcessInfo.processInfo.systemUptime - begin
        var sections = [[String: Any]](), agreement = 0, correct = 0, lowCoverage = 0
        var maxProbabilityError = 0.0
        for (i,s) in m.sections.enumerated() {
            guard covered[i] > 0 else { throw failure("Section has no coverage") }
            let probs = sums[i].map { $0/covered[i] }
            let pred = probs.indices.max(by: { probs[$0] < probs[$1] })!
            let label = m.labels[pred]
            agreement += label == s.prediction ? 1 : 0
            correct += label == s.label_manual ? 1 : 0
            let coverage = covered[i]/(s.end-s.start)
            lowCoverage += coverage < 0.98 ? 1 : 0
            for j in 0..<8 { maxProbabilityError = max(maxProbabilityError,abs(probs[j]-(s.probabilities[m.labels[j]] ?? 0))) }
            sections.append(["start":s.start,"end":s.end,"manual":s.label_manual,"mac_prediction":s.prediction,
                             "prediction":label,"coverage":coverage,"probabilities":probs])
        }
        var system = utsname();uname(&system)
        let machine = withUnsafePointer(to: &system.machine) { ptr in ptr.withMemoryRebound(to: CChar.self,capacity: 1) { String(cString:$0) } }
        let report: [String: Any] = ["track_id":m.track_id,"hardware_identifier":machine,
            "os":ProcessInfo.processInfo.operatingSystemVersionString,"requested_compute_units":hybrid ? "hybrid_all_static_cpu_tail" : "cpuOnly",
            "encoder_model_load_seconds_during_pipeline":encoderLoadSeconds,
            "staged_cpu_model_load_seconds_during_pipeline":stagedLoadSeconds,"memory_events":memoryEvents,
            "diagnostic_note":"Includes stage checkpoint writes; compare speed again after stability is established.","encoder_load_events":encoderLoadEvents,
            "total_load_and_pcm_pipeline_seconds":loadSeconds+inferenceSeconds,
            "model_load_seconds":loadSeconds,"pcm_pipeline_seconds":inferenceSeconds,
            "audio_seconds":Double(audio.count)/Double(m.sample_rate),"prediction_seconds_by_component":timings,
            "calls_by_component":calls,"sections":sections,"section_accuracy":Double(correct)/Double(sections.count),
            "section_agreement_with_mac":Double(agreement)/Double(sections.count),"max_section_probability_error":maxProbabilityError,
            "low_coverage_sections":lowCoverage,"encoder_windows":windows,"all_predictions_finite":true,
            "process_footprint_snapshots_bytes":snapshots,"memory_note":"Snapshots, not peak memory; Process keeps frontend/head and one active model per encoder role; snapshots are not peak memory.",
            "thermal_before":thermalBefore,"thermal_after":ProcessInfo.processInfo.thermalState.rawValue,"low_power_mode":lowPower,
            "source_validation_sha256":m.source_validation_sha256,"scope":m.scope,"shape_policy":m.shape_policy,
            "ready_for_app_integration":false]
        emit("DONE: preparing JSON report")
        return String(decoding:try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]),as:UTF8.self)
    }
}
