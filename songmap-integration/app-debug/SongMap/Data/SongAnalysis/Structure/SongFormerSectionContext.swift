import CoreML
import CryptoKit
import Foundation

/// One analysis owns this instance; encoders/head are released before its small CPU model is loaded.
nonisolated final class SongFormerSectionContext {
    nonisolated struct Result: Sendable {
        let intervals: [SongFormerContextMath.Interval]
        let predictions: [SongFormerContextMath.Prediction]
    }
    nonisolated private struct Specification: Decodable {
        let schemaVersion: Int
        let model: String
        let precision: String
        let exportProtocol: String
        let selectedEpoch: Int
        let checkpointSHA256: String
        let baseCheckpointSHA256: String
        let baseValidationSHA256: String
        let labelOrder: [String]
        let maxSections: Int
        let featureWidth: Int
        let sampleRate: Int
        let encoderFPS: Int
        let shortWindowSeconds: Int
        let longWindowSeconds: Int
        let headWindowSeconds: Int
    }
    private let spec: Specification
    private let modelURL: URL
    private let accumulator: SongFormerContextMath.Accumulator
    private let diagnostics: SongAnalysisDiagnostics?
    private var baseHeadValidated = false

    init(boundaries: [SongFormerContextMath.Interval], duration: Double, baseValidationSHA256: String?,
         diagnostics: SongAnalysisDiagnostics?, resources: URL? = nil) throws {
        let name = "songformer_section_context_manifest"
        let receipt = resources?.appendingPathComponent(name + ".json") ?? Bundle.main.url(forResource: name, withExtension: "json")
        guard let receipt else { throw SongFormerContextMath.ContextError.invalid("Install the validated section-context resources first.") }
        let decoded = try JSONDecoder().decode(Specification.self, from: Data(contentsOf: receipt))
        guard decoded.schemaVersion == 1, decoded.model == "Worship_SectionContext.mlmodelc", decoded.precision == "fp32",
              decoded.exportProtocol == "section-context-coreml-fp32-v1", decoded.selectedEpoch == 5,
              decoded.labelOrder == SongFormerContextMath.labels, (1...512).contains(decoded.maxSections),
              decoded.featureWidth == 4096, decoded.sampleRate == 24000, decoded.encoderFPS == 25,
              decoded.shortWindowSeconds == 30, decoded.longWindowSeconds == 120, decoded.headWindowSeconds == 420,
              decoded.baseValidationSHA256 == baseValidationSHA256 else {
            throw SongFormerContextMath.ContextError.invalid("Context receipt does not match the installed palette6 profile/source validation.")
        }
        let model = resources?.appendingPathComponent(decoded.model)
            ?? Bundle.main.url(forResource: "Worship_SectionContext", withExtension: "mlmodelc")
        guard let model, FileManager.default.fileExists(atPath: model.path) else {
            throw SongFormerContextMath.ContextError.invalid("Compiled context model is missing from the app bundle.")
        }
        spec = decoded; modelURL = model; self.diagnostics = diagnostics
        accumulator = try SongFormerContextMath.Accumulator(intervals: boundaries, duration: duration, maximum: decoded.maxSections)
    }

    func validateBaseHead(_ model: MLModel) throws {
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard metadata["checkpoint_sha256"] == spec.baseCheckpointSHA256, metadata["compression_scheme"] == "palette6" else {
            throw SongFormerContextMath.ContextError.invalid("Installed SongFormer head differs from the validated palette6 context inputs.")
        }
        baseHeadValidated = true
    }

    func append(embeddings: MLMultiArray, frames: Int, offset: Double) throws {
        let started = ProcessInfo.processInfo.systemUptime
        defer { diagnostics?.metric("songformer.context.audio_pooling", since: started, detail: "frames=\(frames)") }
        guard embeddings.shape.map(\.intValue) == [1, frames, 4096] else {
            throw SongFormerContextMath.ContextError.invalid("Fused embeddings do not follow the four-stream contract.")
        }
        // Normal fusion already has this layout. Canonicalize an unusual strided allocation only when needed.
        let canonical: MLMultiArray
        if embeddings.dataType == .float32, embeddings.strides[1].intValue == 4096, embeddings.strides[2].intValue == 1 {
            canonical = embeddings
        } else { canonical = try SongFormerTensorBridge.float32(embeddings) }
        guard canonical.dataType == .float32, canonical.strides[1].intValue == 4096, canonical.strides[2].intValue == 1 else {
            throw SongFormerContextMath.ContextError.invalid("Unsupported canonical fused layout.")
        }
        var failure: Error?
        canonical.withUnsafeMutableBufferPointer(ofType: Float.self) { values, _ in
            do { try accumulator.append(UnsafeBufferPointer(values), frames: frames, offset: offset) }
            catch { failure = error }
        }
        if let failure { throw failure }
    }

    func corrected(frames: [SongFormerContextMath.Frame]) throws -> Result {
        guard baseHeadValidated else { throw SongFormerContextMath.ContextError.invalid("Base head was not validated.") }
        try Task.checkCancellation()
        let prepared = try accumulator.features()
        let ranges = accumulator.intervals
        let pools = try SongFormerContextMath.pool(ranges, frames: frames)
        let audio = MLMultiArray(MLShapedArray<Float>(scalars: prepared.audio, shape: [ranges.count, 4096]))
        let timing = MLMultiArray(MLShapedArray<Float>(scalars: prepared.timing, shape: [ranges.count, 6]))
        let audioHash = Self.digest(prepared.audio); let timingHash = Self.digest(prepared.timing)
        diagnostics?.event("songformer.context.model_load_start", detail: "epoch=5 precision=fp32 CPU_ONLY sections=\(ranges.count)", persist: true)
        let loadStart = ProcessInfo.processInfo.systemUptime
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard metadata["context_checkpoint_sha256"] == spec.checkpointSHA256,
              metadata["precision"] == "fp32", metadata["export_protocol"] == spec.exportProtocol,
              metadata["selected_epoch"] == "5", metadata["max_sections"] == String(spec.maxSections),
              metadata["label_order"] == SongFormerContextMath.labels.joined(separator: ","),
              Set(model.modelDescription.inputDescriptionsByName.keys) == Set(["section_audio", "section_timing"]),
              Set(model.modelDescription.outputDescriptionsByName.keys) == Set(["context_residual"]) else {
            throw SongFormerContextMath.ContextError.invalid("Context model metadata/input contract differs from its receipt.")
        }
        diagnostics?.metric("songformer.context.model_load", since: loadStart)
        diagnostics?.event("songformer.context.predict_start", detail: "CPU_ONLY N=\(ranges.count) audio_sha256=\(audioHash) timing_sha256=\(timingHash)", persist: true)
        let predictStart = ProcessInfo.processInfo.systemUptime
        let output = try autoreleasepool {
            try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["section_audio": audio, "section_timing": timing]))
        }
        diagnostics?.metric("songformer.context.predict", since: predictStart)
        guard let array = output.featureValue(for: "context_residual")?.multiArrayValue,
              array.dataType == .float32, array.shape.map(\.intValue) == [ranges.count, 8] else {
            throw SongFormerContextMath.ContextError.invalid("Invalid context residual output.")
        }
        let residual = MLShapedArray<Float>(array).scalars
        let postStart = ProcessInfo.processInfo.systemUptime
        let predictions = try SongFormerContextMath.corrected(probabilities: pools.map(\.probabilities), residual: residual)
        diagnostics?.metric("songformer.context.postprocess", since: postStart)
        let evidence = ranges.indices.map { i in
            SongAnalysisDiagnostics.ContextEvidence.Section(start: ranges[i].start, end: ranges[i].end,
                baselineLabel: predictions[i].baselineLabel, label: predictions[i].label,
                baseProbabilities: pools[i].probabilities, probabilities: predictions[i].probabilities,
                audioCoverage: accumulator.coverage[i] / (ranges[i].end - ranges[i].start),
                probabilityCoverage: pools[i].coveredSeconds / (ranges[i].end - ranges[i].start))
        }
        diagnostics?.context(SongAnalysisDiagnostics.ContextEvidence(checkpointSHA256: spec.checkpointSHA256,
            baseCheckpointSHA256: spec.baseCheckpointSHA256, audioSHA256: audioHash, timingSHA256: timingHash,
            lowAudioCoverageSections: prepared.lowCoverage.map { $0 + 1 }, sections: evidence))
        let changed = predictions.filter { $0.baselineLabel != $0.label }.count
        NSLog("WORSHIP CONTEXT_DONE epoch=5 CPU_ONLY sections=%ld changed=%ld", ranges.count, changed)
        return Result(intervals: ranges, predictions: predictions)
    }

    private static func digest(_ values: [Float]) -> String {
        values.withUnsafeBufferPointer { buffer in
            let data = Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
    }
}
