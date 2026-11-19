import CoreML
import CryptoKit
import Foundation

/// MuQ-only worship section labels (no MusicFM), CPU only.
///
/// 24 kHz mono PCM → log-mel → MuQ convolution → canonical multi-function MuQ bank (`short_30`,
/// `long_120`, `partial`) on every 30 s and 120 s window → [MuQ30, MuQ120] 2048-wide features →
/// palette6 function head per 420 s chunk → 2048-wide section context over all Apple sections.
/// Same order, windows, inputs and gates as the iPhone probe that matched the Mac reference
/// (docs/WORSHIP_INTEGRATION.md); the resources come from `scripts/install_muq_pipeline.py`.
nonisolated final class SongFormerMuQPipeline {
    nonisolated struct Result: Sendable {
        let frames: [SongFormerAnalyzer.FrameLabel]
        let sections: [SectionLabeler.LabeledSection]
    }
    nonisolated enum PipelineError: LocalizedError {
        case invalid(String)
        var errorDescription: String? { if case .invalid(let message) = self { return "MuQ SongFormer: " + message }; return nil }
    }
    nonisolated struct Specification: Decodable {
        nonisolated struct Models: Decodable {
            let log_mel: String
            let muq_conv: String
            let muq: String
            let head: String
            let context: String
        }
        let schemaVersion: Int
        let pipeline: String
        let models: Models
        let canonicalFunctions: [String: String]
        let rotaryInverseFrequencies: [Float]
        let headCheckpointSHA256: String
        let contextCheckpointSHA256: String
        let labelOrder: [String]
        let allowedLabelIDs: [Int]
        let maxSections: Int
        let featureWidth: Int
        let sampleRate: Int
        let shortWindowSeconds: Int
        let longWindowSeconds: Int
        let headWindowSeconds: Int
    }

    static let manifestName = "songformer_muq_manifest"
    static var isInstalled: Bool { Bundle.main.url(forResource: manifestName, withExtension: "json") != nil }

    private static let reflectPadding = 1024
    private static let embeddingWidth = 1024
    private static let classCount = 128
    private let spec: Specification
    private let diagnostics: SongAnalysisDiagnostics?
    /// One live instance per slot; a different function/shape releases the previous one first.
    private var models: [String: (identity: String, model: MLModel)] = [:]

    init(diagnostics: SongAnalysisDiagnostics?) throws {
        guard let url = Bundle.main.url(forResource: Self.manifestName, withExtension: "json") else {
            throw PipelineError.invalid("Install the MuQ-only resources first (scripts/install_muq_pipeline.py).")
        }
        let decoded = try JSONDecoder().decode(Specification.self, from: Data(contentsOf: url))
        guard decoded.schemaVersion == 1, decoded.pipeline == "muq-only-v1",
              decoded.canonicalFunctions == ["750": "short_30", "3000": "long_120", "partial": "partial"],
              decoded.rotaryInverseFrequencies.count == 32, decoded.labelOrder == SongFormerContextMath.labels,
              decoded.allowedLabelIDs == [0, 1, 2, 3, 4, 5, 6, 26], decoded.featureWidth == 2048,
              (1...512).contains(decoded.maxSections), decoded.sampleRate == 24000,
              decoded.shortWindowSeconds == 30, decoded.longWindowSeconds == 120, decoded.headWindowSeconds == 420 else {
            throw PipelineError.invalid("Installed resources do not match the validated MuQ-only profile.")
        }
        spec = decoded
        self.diagnostics = diagnostics
    }

    func analyze(samples: [Float], boundaries: [SongFormerContextMath.Interval], duration: Double,
                 progress: ProgressSpan) throws -> Result {
        defer { releaseModels() }
        let started = ProcessInfo.processInfo.systemUptime
        let rate = spec.sampleRate, width = spec.featureWidth
        let headSamples = spec.headWindowSeconds * rate
        let streamSeconds = [spec.shortWindowSeconds, spec.longWindowSeconds]
        let accumulator = try SongFormerContextMath.Accumulator(intervals: boundaries, duration: duration,
                                                                maximum: spec.maxSections, width: width)
        let headStarts = Array(stride(from: 0, to: samples.count, by: headSamples))
        let totalWindows = max(1, headStarts.reduce(0) { total, headStart in
            let length = min(samples.count, headStart + headSamples) - headStart
            return total + streamSeconds.reduce(0) { $0 + (length + $1 * rate - 1) / ($1 * rate) }
        })
        var completedWindows = 0
        var frames: [SongFormerAnalyzer.FrameLabel] = []
        var contextFrames: [SongFormerContextMath.Frame] = []
        diagnostics?.event("songformer.muq.start", detail: "muq-only-v1 CPU_ONLY sections=\(boundaries.count)", persist: true)
        for headStart in headStarts {
            let headEnd = min(samples.count, headStart + headSamples)
            if headEnd - headStart <= 2 * Self.reflectPadding { break }
            var streams: [[Float]] = []
            for seconds in streamSeconds {
                var values: [Float] = []
                for start in stride(from: headStart, to: headEnd, by: seconds * rate) {
                    defer { completedWindows += 1; progress(0.95 * Double(completedWindows) / Double(totalWindows)) }
                    let end = min(headEnd, start + seconds * rate)
                    if end - start <= 2 * Self.reflectPadding { continue }
                    let encoded = try autoreleasepool { try encode(samples, from: start, to: end, seconds: seconds) }
                    guard encoded.count % Self.embeddingWidth == 0 else { throw PipelineError.invalid("Encoder width differs.") }
                    values.append(contentsOf: encoded)
                }
                streams.append(values)
            }
            let count = (streams.map { $0.count / Self.embeddingWidth }.min() ?? 0) / 3 * 3
            guard count > 0 else { throw PipelineError.invalid("No fused frames.") }
            releaseModels()  // encoders and frontends go before the head is loaded
            let embeddings = try SongFormerTensorBridge.fuse(streams: streams, frames: count)
            streams.removeAll()
            let offset = Double(headStart) / Double(rate)
            var appendError: Error?
            embeddings.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
                do {
                    guard strides[1] == width, strides[2] == 1 else { throw PipelineError.invalid("Unexpected fused layout.") }
                    try accumulator.append(UnsafeBufferPointer(buffer), frames: count, offset: offset)
                } catch { appendError = error }
            }
            if let appendError { throw appendError }
            let head = try load("head", resource: spec.models.head, identity: "head")
            let output = try predict(head, ["input_embeddings": embeddings], stage: "head")
            guard output.shape.map(\.intValue) == [1, count / 3, Self.classCount] else { throw PipelineError.invalid("Invalid head shape.") }
            let logits = try values(output)
            releaseModels()
            for index in 0..<(count / 3) {
                let selected = spec.allowedLabelIDs.map { Double(logits[index * Self.classCount + $0]) }
                guard selected.allSatisfy({ $0.isFinite }), let top = selected.max() else { throw PipelineError.invalid("Invalid head logits.") }
                let exponentials = selected.map { exp($0 - top) }
                let total = exponentials.reduce(0, +)
                let probabilities = exponentials.map { $0 / total }
                // Strict > keeps the first class on ties, matching Python argmax.
                var best = 0
                for j in 1..<probabilities.count where probabilities[j] > probabilities[best] { best = j }
                let time = offset + Double(index) * 0.12
                frames.append(.init(time: time, label: spec.labelOrder[best], probabilities: probabilities))
                contextFrames.append(.init(time: time, duration: 0.12, probabilities: probabilities))
            }
        }
        guard !frames.isEmpty else { throw PipelineError.invalid("Audio is too short for SongFormer.") }
        let prepared = try accumulator.features()
        let ranges = accumulator.intervals
        let pools = try SongFormerContextMath.pool(ranges, frames: contextFrames)
        let context = try load("context", resource: spec.models.context, identity: "context")
        let residual = try predict(context, ["section_audio": input(prepared.audio, [ranges.count, width]),
                                             "section_timing": input(prepared.timing, [ranges.count, 6])], stage: "context")
        guard residual.shape.map(\.intValue) == [ranges.count, 8] else { throw PipelineError.invalid("Invalid context residual.") }
        let predictions = try SongFormerContextMath.corrected(probabilities: pools.map(\.probabilities), residual: values(residual))
        releaseModels()
        diagnostics?.context(SongAnalysisDiagnostics.ContextEvidence(
            checkpointSHA256: spec.contextCheckpointSHA256, baseCheckpointSHA256: spec.headCheckpointSHA256,
            audioSHA256: Self.digest(prepared.audio), timingSHA256: Self.digest(prepared.timing),
            lowAudioCoverageSections: prepared.lowCoverage.map { $0 + 1 },
            sections: ranges.indices.map { i in
                SongAnalysisDiagnostics.ContextEvidence.Section(start: ranges[i].start, end: ranges[i].end, baselineLabel: predictions[i].baselineLabel,
                      label: predictions[i].label, baseProbabilities: pools[i].probabilities,
                      probabilities: predictions[i].probabilities,
                      audioCoverage: accumulator.coverage[i] / (ranges[i].end - ranges[i].start),
                      probabilityCoverage: pools[i].coveredSeconds / (ranges[i].end - ranges[i].start))
            }))
        diagnostics?.metric("songformer.muq.total", since: started, detail: "frames=\(frames.count) sections=\(ranges.count)")
        progress(1)
        let sections = zip(ranges, predictions).map { SectionLabeler.LabeledSection(label: $0.1.label, start: $0.0.start, end: $0.0.end) }
        return Result(frames: frames, sections: sections)
    }

    // MARK: - Encoder windows

    /// One 30 s or 120 s window → frames × 1024 MuQ layer-10 values (owned).
    private func encode(_ samples: [Float], from start: Int, to end: Int, seconds: Int) throws -> [Float] {
        try Task.checkCancellation()
        let n = end - start
        let padding = Self.reflectPadding
        var padded: [Float] = []
        padded.reserveCapacity(n + 2 * padding)
        for i in -padding..<(n + padding) {
            let index = i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i)
            padded.append(samples[start + index])
        }
        let melModel = try load("log_mel", resource: spec.models.log_mel, identity: "log_mel:\(padded.count)")
        let mel = try predict(melModel, [try inputName(of: melModel): input(padded, [1, padded.count])], stage: "log_mel")
        let melShape = mel.shape.map { String($0.intValue) }.joined(separator: "x")
        let convModel = try load("muq_conv", resource: spec.models.muq_conv, identity: "muq_conv:" + melShape)
        let conv = try predict(convModel, [try inputName(of: convModel): mel], stage: "muq_conv")
        let shape = conv.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, shape[2] == Self.embeddingWidth else { throw PipelineError.invalid("Invalid MuQ frontend shape \(shape).") }
        let count = shape[1]
        let fullLength = n == seconds * spec.sampleRate
        let full = fullLength && count == seconds * 25
        if fullLength && !full { throw PipelineError.invalid("Full-window frame count differs.") }
        guard let function = spec.canonicalFunctions[full ? String(count) : "partial"] else {
            throw PipelineError.invalid("No canonical MuQ function for \(count) frames.")
        }
        var cosines = [Float](repeating: 0, count: count * 64), sines = cosines
        let frequencies = spec.rotaryInverseFrequencies
        for t in 0..<count {
            for c in 0..<32 {
                let angle = Float(t) * frequencies[c]
                let cosine = cosf(angle), sine = sinf(angle)
                cosines[t * 64 + c] = cosine; cosines[t * 64 + c + 32] = cosine
                sines[t * 64 + c] = sine; sines[t * 64 + c + 32] = sine
            }
        }
        let bank = try load("muq", resource: spec.models.muq, identity: "muq:\(function):\(count)", functionName: function)
        let inputs: [String: MLMultiArray] = full
            ? ["conv_in": conv, "rotary_cos": input(cosines, [count, 64]), "rotary_sin": input(sines, [count, 64])]
            : ["conv_in": conv, "rotary": input(cosines + sines, [2, count, 1, 1, 64])]
        let output = try predict(bank, inputs, stage: "muq_" + function)
        guard output.shape.map(\.intValue) == shape else { throw PipelineError.invalid("MuQ output shape differs.") }
        return try values(output)
    }

    // MARK: - Models

    private func load(_ slot: String, resource: String, identity: String, functionName: String? = nil) throws -> MLModel {
        if let cached = models[slot], cached.identity == identity { return cached.model }
        autoreleasepool { _ = models.removeValue(forKey: slot) }
        let name = (resource as NSString).deletingPathExtension
        // In the app in Debug; in the "sections" asset pack in Release (prepared before analysis).
        guard let url = ModelLocator.modelURL(named: name) else { throw ModelStoreError.missing(name) }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        if let functionName { configuration.functionName = functionName }
        let start = ProcessInfo.processInfo.systemUptime
        let model = try MLModel(contentsOf: url, configuration: configuration)
        diagnostics?.metric("songformer.muq.model_load", since: start, detail: name + (functionName.map { ":" + $0 } ?? ""))
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        switch slot {
        case "head":
            guard metadata["checkpoint_sha256"] == spec.headCheckpointSHA256, metadata["compression_scheme"] == "palette6",
                  metadata["input_width"] == String(spec.featureWidth) else { throw PipelineError.invalid("Head metadata differs from the installed manifest.") }
        case "context":
            guard metadata["context_checkpoint_sha256"] == spec.contextCheckpointSHA256, metadata["variant"] == "muq",
                  metadata["input_width"] == String(spec.featureWidth), metadata["precision"] == "fp32" else {
                throw PipelineError.invalid("Context metadata differs from the installed manifest.")
            }
        default:
            break
        }
        models[slot] = (identity, model)
        return model
    }

    private func releaseModels() { autoreleasepool { models.removeAll() } }

    /// Single-output prediction; the output is copied into an owned Float32 array while its provider is alive.
    private func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], stage: String) throws -> MLMultiArray {
        try Task.checkCancellation()
        let start = ProcessInfo.processInfo.systemUptime
        defer { diagnostics?.metric("songformer.muq.predict." + stage, since: start) }
        return try autoreleasepool {
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
            guard output.featureNames.count == 1, let key = output.featureNames.first,
                  let array = output.featureValue(for: key)?.multiArrayValue else {
                throw PipelineError.invalid("Invalid \(stage) output.")
            }
            return try withExtendedLifetime(output) {
                MLMultiArray(MLShapedArray<Float>(scalars: try values(array), shape: array.shape.map(\.intValue)))
            }
        }
    }

    private func inputName(of model: MLModel) throws -> String {
        let names = model.modelDescription.inputDescriptionsByName.keys
        guard names.count == 1, let name = names.first else { throw PipelineError.invalid("Unexpected frontend input contract.") }
        return name
    }

    private func input(_ values: [Float], _ shape: [Int]) -> MLMultiArray {
        MLMultiArray(MLShapedArray<Float>(scalars: values, shape: shape))
    }

    private func values(_ array: MLMultiArray) throws -> [Float] {
        MLShapedArray<Float>(try SongFormerTensorBridge.float32(array)).scalars
    }

    private static func digest(_ values: [Float]) -> String {
        values.withUnsafeBufferPointer { buffer in
            let data = Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
    }
}
