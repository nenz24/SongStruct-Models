//
//  SongFormerAnalyzer.swift
//  SongMap
//

import CoreML
import Foundation

/// Song-function labels (intro, verse, chorus, ...) for every ~0.12 s of a song, from SongFormer run with
/// Core ML: waveform → log-mel → MuQ and MusicFM encoders on 30 s and 120 s windows → function head.
actor SongFormerAnalyzer {
    nonisolated struct FrameLabel: Sendable, Equatable {
        let time: Double
        let label: String
        let probabilities: [Double]?
        let duration: Double

        init(time: Double, label: String, probabilities: [Double]? = nil, duration: Double = 0.12) {
            self.time = time; self.label = label
            self.probabilities = probabilities; self.duration = duration
        }
    }
    nonisolated struct Analysis: Sendable {
        let frameLabels: [FrameLabel]
        let contextSections: [SectionLabeler.LabeledSection]?
        let boundaryRefinementEnabled: Bool
    }

    nonisolated enum AnalyzerError: LocalizedError {
        case missingResource(String)
        case invalidModel(String)
        case inference(String)

        var errorDescription: String? {
            switch self {
            case .missingResource(let name): return "Missing SongFormer resource: \(name)"
            case .invalidModel(let message): return "Invalid SongFormer model: \(message)"
            case .inference(let message): return "SongFormer inference failed: \(message)"
            }
        }
    }

    private static let shortWindowSeconds = 30.0
    private static let reflectPadding = 1024
    /// Width of one MuQ or MusicFM frame. The head reads four of them side by side.
    private static let embeddingWidth = 1024
    private static let classCount = 128

    private let manifest: Manifest
    /// The validated MuQ-only pipeline (no MusicFM, 2048-wide head and context, CPU only): always in
    /// Release, and in Debug unless a launch option picks another path (e.g. `--songformer-legacy`).
    private let muqOnly: Bool
    private let worship: Bool
    private let hybrid: Bool
    private let contextEnabled: Bool
    private let boundaryRefinementEnabled: Bool
    private var diagnostics: SongAnalysisDiagnostics?
    private var frameRate: Double { manifest.head_fps ?? 25.0 / 3.0 }

    init() throws {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        let preferences = UserDefaults.standard
        if args.contains("--songformer-worship-boundary-refinement"), args.contains("--songformer-worship-no-boundary-refinement") {
            throw AnalyzerError.invalidModel("Enable only one boundary refinement launch option.")
        }
        if args.contains("--songformer-legacy") {
            preferences.set(false,forKey:"songformer.debug.worship")
            preferences.set(false,forKey:"songformer.debug.hybrid")
            preferences.set(false,forKey:"songformer.debug.context")
            preferences.set(false,forKey:"songformer.debug.boundaryRefinement")
        } else if args.contains("--songformer-worship-boundary-refinement") {
            preferences.set(true,forKey:"songformer.debug.worship")
            preferences.set(true,forKey:"songformer.debug.context")
            preferences.set(true,forKey:"songformer.debug.boundaryRefinement")
        } else if args.contains("--songformer-worship-no-boundary-refinement") {
            preferences.set(true,forKey:"songformer.debug.worship")
            preferences.set(true,forKey:"songformer.debug.context")
            preferences.set(false,forKey:"songformer.debug.boundaryRefinement")
        } else if args.contains("--songformer-worship-context") {
            preferences.set(true,forKey:"songformer.debug.worship")
            preferences.set(true,forKey:"songformer.debug.context")
            preferences.set(false,forKey:"songformer.debug.boundaryRefinement")
        } else if args.contains("--songformer-worship-no-context") {
            preferences.set(true,forKey:"songformer.debug.worship")
            preferences.set(false,forKey:"songformer.debug.context")
            preferences.set(false,forKey:"songformer.debug.boundaryRefinement")
        } else if args.contains("--songformer-worship") {
            preferences.set(true,forKey:"songformer.debug.worship")
            preferences.set(false,forKey:"songformer.debug.hybrid")
        }
        let muqRefinement = SongAnalysisDiagnostics.launchArgument("songformer-muq-boundary-refinement", in: args)
        if SongAnalysisDiagnostics.launchArgument("songformer-muq", in: args) || muqRefinement {
            preferences.set(true,forKey:"songformer.debug.muq")
            preferences.set(muqRefinement,forKey:"songformer.debug.boundaryRefinement")
        } else if args.contains(where: { $0.hasPrefix("--songformer-") }) {
            // Any other explicit SongFormer launch choice leaves the MuQ-only path.
            preferences.set(false,forKey:"songformer.debug.muq")
        }
        // Persist only explicit Debug choices so Home Screen runs use the same experiment.
        // Release ignores these preferences. The known fatal MusicFM shape is still blocked.
        // Without a choice Debug runs what Release runs: the MuQ-only pipeline.
        let useMuQ = preferences.object(forKey:"songformer.debug.muq") as? Bool ?? true
        muqOnly = useMuQ
        let useWorship = !useMuQ && preferences.bool(forKey:"songformer.debug.worship")
        worship = useWorship
        contextEnabled = useWorship && preferences.bool(forKey:"songformer.debug.context")
        boundaryRefinementEnabled = (useMuQ || (useWorship && preferences.bool(forKey:"songformer.debug.context")))
            && preferences.bool(forKey:"songformer.debug.boundaryRefinement")
        // Both MusicFM and MuQ have produced fatal MPS compiler assertions in the app.
        // Ignore and clear previously persisted opt-ins; try/catch cannot recover these crashes.
        preferences.set(false,forKey:"songformer.debug.hybrid")
        hybrid = false
        #else
        // Release: the MuQ-only pipeline (6-bit head + section context over Apple's boundaries, CPU only).
        muqOnly = true; worship = false; hybrid = false; contextEnabled = false; boundaryRefinementEnabled = false
        #endif
        let manifestName = worship ? "songformer_worship_manifest" : "songformer_coreml_manifest"
        guard let url = Bundle.main.url(forResource: manifestName, withExtension: "json") else {
            throw AnalyzerError.missingResource(manifestName+".json")
        }
        manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        if worship {
            guard manifest.sample_rate == 24000, manifest.encoder_window_seconds == 120,
                  manifest.head_max_seconds == 420, manifest.head.dataset_id == 5,
                  manifest.head.allowed_label_ids == [0,1,2,3,4,5,6,26] else {
                throw AnalyzerError.invalidModel("Worship model contract differs from validated profile.")
            }
        }
        guard manifest.rotary_inv_freq.muq.count == 32, manifest.rotary_inv_freq.musicfm.count == 32 else {
            throw AnalyzerError.invalidModel("rotary_inv_freq is missing or invalid.")
        }
    }

    func frameLabels(fileURL: URL, progress: ProgressSpan) throws -> [FrameLabel] {
        guard !contextEnabled && !muqOnly else { throw AnalyzerError.invalidModel("Section context requires the analyze API with Apple boundaries.") }
        return try frameLabels(progress: progress) { try AudioDecoder.loadMono(url: fileURL, sampleRate: $0) }
    }

    /// `audio`: the song as decoded once for the whole analysis.
    func frameLabels(audio: DecodedAudio, progress: ProgressSpan, diagnostics: SongAnalysisDiagnostics? = nil) throws -> [FrameLabel] {
        guard !contextEnabled && !muqOnly else { throw AnalyzerError.invalidModel("Section context requires the analyze API with Apple boundaries.") }
        self.diagnostics = diagnostics
        defer { self.diagnostics = nil }
        return try frameLabels(progress: progress) { try AudioDecoder.convert(audio, sampleRate: $0, maxChannels: 1)[0] }
    }

    /// Retain means while fused frames are alive; return only labels/probabilities after model inference.
    func analyze(audio: DecodedAudio, boundaries: [ClosedRange<Double>], duration: Double,
                 progress: ProgressSpan, diagnostics: SongAnalysisDiagnostics?) throws -> Analysis {
        self.diagnostics = diagnostics
        defer { self.diagnostics = nil }
        do {
            if muqOnly {
                let pipeline = try SongFormerMuQPipeline(diagnostics: diagnostics)
                let decodeStart = ProcessInfo.processInfo.systemUptime
                let samples = try AudioDecoder.convert(audio, sampleRate: 24000, maxChannels: 1)[0]
                diagnostics?.metric("songformer.decode_resample", since: decodeStart, detail: "24 kHz mono")
                let result = try pipeline.analyze(samples: samples,
                                                  boundaries: boundaries.map { .init(start: $0.lowerBound, end: $0.upperBound) },
                                                  duration: duration, progress: progress)
                return Analysis(frameLabels: result.frames, contextSections: result.sections,
                                boundaryRefinementEnabled: boundaryRefinementEnabled)
            }
            let context: SongFormerSectionContext?
            if contextEnabled {
                context = try SongFormerSectionContext(
                    boundaries: boundaries.map { .init(start: $0.lowerBound, end: $0.upperBound) }, duration: duration,
                    baseValidationSHA256: manifest.installation_provenance?.source_validation_sha256, diagnostics: diagnostics)
            } else { context = nil }
            let baseProgress = context == nil ? progress : progress.span(from: 0, to: 0.97)
            let frames = try frameLabels(progress: baseProgress, context: context) {
                try AudioDecoder.convert(audio, sampleRate: $0, maxChannels: 1)[0]
            }
            guard let context else { return Analysis(frameLabels: frames, contextSections: nil, boundaryRefinementEnabled: false) }
            let probabilityFrames = try frames.map { frame -> SongFormerContextMath.Frame in
                guard let probabilities = frame.probabilities else { throw AnalyzerError.inference("Context requires allowed-class probabilities.") }
                return .init(time: frame.time, duration: frame.duration, probabilities: probabilities)
            }
            let corrected = try context.corrected(frames: probabilityFrames)
            let sections = zip(corrected.intervals, corrected.predictions).map {
                SectionLabeler.LabeledSection(label: $0.1.label, start: $0.0.start, end: $0.0.end)
            }
            progress(1)
            return Analysis(frameLabels: frames, contextSections: sections, boundaryRefinementEnabled: boundaryRefinementEnabled)
        } catch {
            diagnostics?.event("songformer.analysis.failed", detail: error.localizedDescription, persist: true)
            throw error
        }
    }

    /// `monoSamples` returns the song as mono at the given sample rate.
    private func frameLabels(progress: ProgressSpan, context: SongFormerSectionContext? = nil,
                             monoSamples: (Double) throws -> [Float]) throws -> [FrameLabel] {
        let stageStart = ProcessInfo.processInfo.systemUptime
        diagnostics?.event("songformer.tensor_bridge",detail:"v1: reuse contiguous Float32, vector Float16 conversion, bulk feature copies")
        diagnostics?.event("songformer.configuration",detail:"worship=\(worship), hybrid_opt_in=\(hybrid); all encoder GPU routing disabled after fatal MPS compiler assertions",persist:true)
        NSLog("WORSHIP ANALYSIS_START v10 CPU fixed shapes; section_context=%@; boundary_refinement=%@",
              context == nil ? "off" : "epoch5-fp32", boundaryRefinementEnabled ? "debug-opt-in" : "off")
        defer { NSLog("WORSHIP ANALYSIS_END elapsed=%.3fs", ProcessInfo.processInfo.systemUptime-stageStart) }
        return try autoreleasepool {
        let models = try loadModels()
        progress(0.05)
        let sampleRate = Double(manifest.sample_rate)
        let resampleStart = ProcessInfo.processInfo.systemUptime
        let samples = try monoSamples(sampleRate)
        diagnostics?.metric("songformer.decode_resample",since:resampleStart,detail:"24 kHz mono")
        progress(0.08)

        // The log-mel/encoder models take at most `encoder_window_seconds` (120 s), so the encoders run
        // per window and their frames are concatenated. The head takes up to `head_max_seconds` (420 s)
        // at once, so long-range labels such as bridge aren't decided from an isolated 120 s context.
        let encoderSamples = Int(Double(manifest.encoder_window_seconds) * sampleRate)
        let headSamples = Int(Double(manifest.head_max_seconds) * sampleRate)
        let shortSamples = Int(Self.shortWindowSeconds * sampleRate)

        let headWindows = Self.windows(count: samples.count, size: headSamples)
        let totalSteps = headWindows.reduce(0) { total, window in
            total + Self.windows(count: window.count, size: encoderSamples).count
                + Self.windows(count: window.count, size: shortSamples).count + 1
        }
        var completedSteps = 0
        let encoding = progress.span(from: 0.08, to: 1)
        let advance = {
            completedSteps += 1
            encoding(Double(completedSteps) / Double(max(totalSteps, 1)))
        }

        var functionLogits: [[Float]] = []
        var worshipFrames: [FrameLabel] = []
        for headWindow in headWindows {
            let headAudio = samples[headWindow]
            guard headAudio.count > Self.reflectPadding * 2 else { break }

            var short = try encode(headAudio, windowSize: shortSamples, models: models, advance: advance)
            var long = try encode(headAudio, windowSize: encoderSamples, models: models, advance: advance)
            // The head reads frames in groups of three (25 fps in, 8.33 fps out).
            let frames = min(long.frames, short.frames) / 3 * 3
            guard frames > 0 else {
                advance()
                continue
            }

            let fusionStart = ProcessInfo.processInfo.systemUptime
            let embeddings = try Self.headEmbeddings(short: short, long: long, frames: frames)
            diagnostics?.metric("songformer.feature_fusion",since:fusionStart,detail:"frames=\(frames)")
            short = EncodedStreams(); long = EncodedStreams()
            if worship { models.releaseEncoders() }
            try context?.append(embeddings: embeddings, frames: frames, offset: Double(headWindow.lowerBound) / sampleRate)
            let logits = try autoreleasepool {
                defer { if worship { models.releaseHead() } }
                let head = try models.head()
                try context?.validateBaseHead(head)
                return try functionLogitRows(embeddings: embeddings, frames: frames, head: head)
            }
            if worship {
                if context != nil, logits.count != frames / 3 { throw AnalyzerError.inference("Context base head frame count differs from the validated stride.") }
                let softmaxStart = ProcessInfo.processInfo.systemUptime
                for (index, row) in logits.enumerated() {
                    let ids = manifest.head.allowed_label_ids
                    guard row.count == 128, row.allSatisfy({ $0.isFinite }) else { throw AnalyzerError.inference("Invalid worship logits.") }
                    let selected = ids.map { Double(row[$0]) }
                    let top = selected.max()!
                    let values = selected.map { exp($0-top) }; let total = values.reduce(0,+)
                    let probabilities = values.map { $0/total }
                    // Strict > preserves the first class on ties, matching Python argmax.
                    var best = 0
                    for j in 1..<probabilities.count where probabilities[j] > probabilities[best] { best = j }
                    worshipFrames.append(FrameLabel(time: Double(headWindow.lowerBound)/sampleRate + Double(index)*0.12,
                        label: label(for: ids[best]), probabilities: probabilities))
                }
                diagnostics?.metric("songformer.allowed_softmax_frames",since:softmaxStart)
                advance(); continue
            }
            let startFrame = Int(Double(headWindow.lowerBound) / sampleRate * frameRate)
            let endFrame = startFrame + logits.count
            if functionLogits.count < endFrame {
                functionLogits += Array(
                    repeating: Array(repeating: 0, count: Self.classCount),
                    count: endFrame - functionLogits.count
                )
            }
            functionLogits.replaceSubrange(startFrame..<endFrame, with: logits)
            advance()
        }

        if worship { return worshipFrames }
        return functionLogits.enumerated().map { index, logits in
            FrameLabel(time: Double(index) / frameRate, label: label(for: bestLabel(in: logits)))
        }
        }
    }

    // MARK: - Loading

    nonisolated private final class LoadedModels {
        private var frontends: [String: MLModel] = [:]
        private let names: Manifest.Models
        private let staticNames: [String: String]
        private let hybrid: Bool
        private let diagnostics: SongAnalysisDiagnostics?
        private var active: [String: (String, MLModel)] = [:]
        private var loadedHead: MLModel?
        init(names: Manifest.Models, staticNames: [String: String], hybrid: Bool, diagnostics: SongAnalysisDiagnostics?) throws {
            self.names = names; self.staticNames = staticNames; self.hybrid = hybrid; self.diagnostics = diagnostics

        }
        func frontend(_ role: String) throws -> MLModel {
            if let model = frontends[role] { return model }
            let filename: String
            switch role {
            case "log_mel": filename = names.log_mel
            case "musicfm_conv": filename = names.musicfm_conv
            case "muq_conv": filename = names.muq_conv
            default: throw AnalyzerError.invalidModel("Unknown frontend \(role)")
            }
            let model = try Self.load(filename, units: .cpuOnly, diagnostics:diagnostics)
            frontends[role] = model; return model
        }
        private static func load(_ filename: String, units: MLComputeUnits, diagnostics: SongAnalysisDiagnostics?) throws -> MLModel {
            let name = (filename as NSString).deletingPathExtension
            guard let url = ModelLocator.modelURL(named: name) else {
                throw AnalyzerError.missingResource(name)
            }
            let config = MLModelConfiguration(); config.computeUnits = units
            let start = ProcessInfo.processInfo.systemUptime
            defer { diagnostics?.metric("songformer.model_load",since:start,detail:name) }
            NSLog("WORSHIP LOAD_START %@", name)
            do {
                let model = try MLModel(contentsOf: url, configuration: config)
                NSLog("WORSHIP LOAD_DONE %@ elapsed=%.3fs", name, ProcessInfo.processInfo.systemUptime-start)
                return model
            } catch {
                throw AnalyzerError.inference("Loading \(name): \(error.localizedDescription)")
            }
        }
        func encoder(_ role: String, frames: Int, fullWindow: Bool, forceDynamicCPU: Bool = false) throws -> MLModel {
            let staticName = staticNames["\(role)_\(frames)"]
            // Fixed shape and compute units are independent: use the verified fixed graphs on CPU.
            let useStatic = !forceDynamicCPU && fullWindow && (frames == 750 || frames == 3000) && staticName != nil
            let filename = useStatic ? staticName! : (role == "muq" ? names.muq : names.musicfm)
            // Dynamic CPU instances are never reused across different tensor lengths.
            let key = filename + ":cpu:frames=\(frames)"
            if let cached = active[role], cached.0 == key { return cached.1 }
            autoreleasepool { _ = active.removeValue(forKey: role) }
            let model = try Self.load(filename, units: .cpuOnly, diagnostics:diagnostics)
            active[role] = (key,model)
            let route = useStatic ? "cpuOnly_static" : "cpuOnly_dynamic"
            diagnostics?.event("songformer.route",detail:"\(role) frames=\(frames) route=\(route)")
            NSLog("WORSHIP encoder %@ frames=%ld route %@",role,frames,route)
            return model
        }
        func head() throws -> MLModel {
            if let loadedHead { return loadedHead }
            let model = try Self.load(names.head, units: .cpuOnly, diagnostics:diagnostics); loadedHead = model; return model
        }
        func releaseEncoders(reason: String = "before head") {
            autoreleasepool { active.removeAll(); frontends.removeAll() }
            NSLog("WORSHIP RELEASE encoder and frontend model references %@", reason)
        }
        func releaseHead() { autoreleasepool { loadedHead = nil } }
    }

    private func loadModels() throws -> LoadedModels {
        try LoadedModels(names: manifest.models, staticNames: manifest.worship_static ?? [:], hybrid: hybrid, diagnostics:diagnostics)
    }

    // MARK: - Encoders

    /// MusicFM and MuQ embeddings of consecutive windows, concatenated: `frames` rows of 1024 values each.
    nonisolated private struct EncodedStreams {
        var musicFM: [Float] = []
        var muq: [Float] = []
        var frames = 0
    }

    private func encode(
        _ audio: ArraySlice<Float>,
        windowSize: Int,
        models: LoadedModels,
        advance: () -> Void
    ) throws -> EncodedStreams {
        var streams = EncodedStreams()
        for window in Self.windows(count: audio.count, size: windowSize) {
            defer { advance() }
            try Task.checkCancellation()
            let chunk = audio[(audio.startIndex + window.lowerBound)..<(audio.startIndex + window.upperBound)]
            // A sliver at the very end is too short for the encoders; the other stream drops it too.
            guard chunk.count > Self.reflectPadding * 2 else { continue }
            let windowDetail = "start=\(Double(chunk.startIndex) / Double(manifest.sample_rate))s length=\(Double(chunk.count) / Double(manifest.sample_rate))s stream=\(Double(windowSize) / Double(manifest.sample_rate))s"
            diagnostics?.event("songformer.encoder_window",detail:windowDetail,persist:true)
            NSLog("WORSHIP ENCODER_WINDOW %@",windowDetail)

            try autoreleasepool {
            let mel = try logMel(Self.reflectPad(chunk, padding: Self.reflectPadding), model: models.frontend("log_mel"))
            let frequencies = manifest.rotary_inv_freq
            // Keep only owned Swift feature values after each encoder. Drain its temporary Core ML
            // output providers/tensors before entering the other encoder in this same audio window.
            let musicFM = try autoreleasepool {
                try encoder(mel, conv: models.frontend("musicfm_conv"), models: models, role: "musicfm", fullWindow: chunk.count == windowSize,
                            inverseFrequencies: frequencies.musicfm, name: "MusicFM")
            }
            let muq = try autoreleasepool {
                try encoder(mel, conv: models.frontend("muq_conv"), models: models, role: "muq", fullWindow: chunk.count == windowSize,
                            inverseFrequencies: frequencies.muq, name: "MuQ")
            }
            let frames = min(musicFM.count, muq.count) / Self.embeddingWidth
            streams.musicFM += musicFM.prefix(frames * Self.embeddingWidth)
            streams.muq += muq.prefix(frames * Self.embeddingWidth)
            streams.frames += frames
            }
        }
        return streams
    }

    private func logMel(_ waveform: [Float], model: MLModel) throws -> MLMultiArray {
        let input = MLMultiArray(MLShapedArray<Float>(scalars: waveform, shape: [1, waveform.count]))
        let outputs = try Self.predict(model, inputs: MLDictionaryFeatureProvider(dictionary: [Self.firstInputName(of: model): input]), stage: "log_mel", diagnostics:diagnostics)
        guard let mel = Self.firstMultiArray(in: outputs) else {
            throw AnalyzerError.inference("Log-mel model did not return an array.")
        }
        return try Self.floatInput(mel, diagnostics:diagnostics)
    }

    /// Convolution front end, then the conformer with rotary position embeddings.
    /// Returns frames × 1024 values, row-major.
    private func encoder(
        _ mel: MLMultiArray,
        conv: MLModel,
        models: LoadedModels,
        role: String,
        fullWindow: Bool,
        inverseFrequencies: [Float],
        name: String
    ) throws -> [Float] {
        let convOutputs = try Self.predict(conv, inputs: MLDictionaryFeatureProvider(dictionary: [Self.firstInputName(of: conv): mel]), stage: role+"_conv", diagnostics:diagnostics)
        guard let convolved = Self.firstMultiArray(in: convOutputs) else {
            throw AnalyzerError.inference("\(name) convolution output is missing.")
        }
        let shape = convolved.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, shape[2] == Self.embeddingWidth else {
            throw AnalyzerError.inference("\(name) convolution output has invalid shape \(shape).")
        }

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "conv_in": Self.floatInput(convolved, diagnostics:diagnostics),
            "rotary": Self.rotaryEmbedding(frameCount: shape[1], inverseFrequencies: inverseFrequencies)
        ])
        func run(_ model: MLModel) throws -> [Float] {
            let outputs = try Self.predict(model, inputs: inputs, stage: role, diagnostics:diagnostics)
            guard let output = Self.firstMultiArray(in: outputs), output.shape.map(\.intValue) == shape else {
                throw AnalyzerError.inference("\(name) embedding shape mismatch.")
            }
            let readStart = ProcessInfo.processInfo.systemUptime
            let values = MLShapedArray<Float>(converting: output).scalars
            diagnostics?.metric("songformer.embedding_read",since:readStart,detail:role)
            guard values.allSatisfy({ $0.isFinite }) else { throw AnalyzerError.inference("NaN/Inf in \(name).") }
            return values
        }
        do {
            return try autoreleasepool {
                try run(models.encoder(role, frames: shape[1], fullWindow: fullWindow))
            }
        } catch {
            try Task.checkCancellation()
            guard worship && fullWindow && (shape[1] == 750 || shape[1] == 3000) else { throw error }
            diagnostics?.event("songformer.cpu_retry",detail:"\(role): \(error.localizedDescription)",persist:true)
            NSLog("WORSHIP CPU retry %@: %@",role,error.localizedDescription)
            // The old retry retained the other encoder and frontends. Release every cached reference
            // after the failed call's pool has drained; retain the exact prepared inputs and features.
            models.releaseEncoders(reason: "before fresh CPU retry for \(role)")
            let layout = inputs.featureNames.sorted().map { key in
                guard let array = inputs.featureValue(for: key)?.multiArrayValue else { return key + "=non-array" }
                return "\(key):shape=\(array.shape),strides=\(array.strides),dtype=\(array.dataType.rawValue)"
            }.joined(separator:"; ")
            diagnostics?.event("songformer.cpu_retry_after_release",detail:"\(role); \(layout)",persist:true)
            return try autoreleasepool {
                try run(models.encoder(role, frames: shape[1], fullWindow: false, forceDynamicCPU: true))
            }
        }
    }

    private static func floatInput(_ array: MLMultiArray, diagnostics: SongAnalysisDiagnostics?) throws -> MLMultiArray {
        let start = ProcessInfo.processInfo.systemUptime
        defer { diagnostics?.metric("songformer.tensor_conversion",since:start,detail:"shape=\(array.shape)") }
        return try SongFormerTensorBridge.float32(array)
    }

    /// cos and sin planes of shape [2, frames, 1, 1, 64].
    private static func rotaryEmbedding(frameCount: Int, inverseFrequencies: [Float]) -> MLMultiArray {
        let halfDimension = inverseFrequencies.count
        let dimension = halfDimension * 2
        let planeSize = frameCount * dimension
        var values = [Float](repeating: 0, count: 2 * planeSize)
        for frame in 0..<frameCount {
            let base = frame * dimension
            for index in 0..<halfDimension {
                let angle = Float(frame) * inverseFrequencies[index]
                values[base + index] = cosf(angle)
                values[base + index + halfDimension] = cosf(angle)
                values[planeSize + base + index] = sinf(angle)
                values[planeSize + base + index + halfDimension] = sinf(angle)
            }
        }
        return MLMultiArray(MLShapedArray<Float>(scalars: values, shape: [2, frameCount, 1, 1, dimension]))
    }

    // MARK: - Head

    private func functionLogitRows(
        embeddings: MLMultiArray,
        frames: Int,
        head: MLModel
    ) throws -> [[Float]] {
        try Task.checkCancellation()
        let inputNames = head.modelDescription.inputDescriptionsByName.keys.sorted()
        var inputs: [String: MLMultiArray] = [:]
        if let name = inputNames.first(where: { $0.localizedCaseInsensitiveContains("embedding") }) ?? inputNames.first {
            inputs[name] = embeddings
        }
        if let name = inputNames.first(where: { $0.localizedCaseInsensitiveContains("dataset") }) {
            inputs[name] = MLMultiArray(MLShapedArray<Float>(scalars: [Float(manifest.head.dataset_id)], shape: [1]))
        }
        if let name = inputNames.first(where: { $0.localizedCaseInsensitiveContains("mask") }) {
            inputs[name] = MLMultiArray(MLShapedArray<Float>(scalars: labelMask(), shape: [1, 1, Self.classCount]))
        }

        let outputs = try Self.predict(head, inputs: MLDictionaryFeatureProvider(dictionary: inputs), stage: "head", diagnostics:diagnostics)
        let names = outputs.featureNames.sorted()
        guard let name = names.first(where: { $0.localizedCaseInsensitiveContains("function") }) ?? names.first,
              let logits = outputs.featureValue(for: name)?.multiArrayValue,
              let width = logits.shape.last?.intValue, width > 0 else {
            throw AnalyzerError.inference("The function head returned no logits.")
        }
        let scalars = MLShapedArray<Float>(converting: logits).scalars
        return stride(from: 0, to: scalars.count - width + 1, by: width).map { Array(scalars[$0..<($0 + width)]) }
    }

    /// [1, frames, 4096]: MusicFM 30 s, MuQ 30 s, MusicFM 120 s and MuQ 120 s side by side,
    /// the `stream_order` the head was exported with.
    private static func headEmbeddings(short: EncodedStreams, long: EncodedStreams, frames: Int) throws -> MLMultiArray {
        try SongFormerTensorBridge.fuse(streams:[short.musicFM,short.muq,long.musicFM,long.muq],frames:frames)
    }

    // MARK: - Labels

    private func bestLabel(in logits: [Float]) -> Int {
        let allowed = manifest.head.allowed_label_ids.filter { $0 < logits.count }
        return allowed.max { logits[$0] < logits[$1] } ?? 0
    }

    private func label(for id: Int) -> String {
        manifest.head.id_to_label[String(id)] ?? "label-\(id)"
    }

    /// 0 for the labels this dataset uses, 1 (masked out) for the rest.
    private func labelMask() -> [Float] {
        var mask = [Float](repeating: 1, count: Self.classCount)
        for id in manifest.head.allowed_label_ids where id < mask.count { mask[id] = 0 }
        return mask
    }

    // MARK: - Helpers

    /// Consecutive index ranges of at most `size` covering `0..<count`.
    private static func windows(count: Int, size: Int) -> [Range<Int>] {
        guard size > 0 else { return [0..<count] }
        return stride(from: 0, to: count, by: size).map { $0..<min($0 + size, count) }
    }

    /// The log-mel model expects the waveform reflect-padded at both ends, like torch.stft(center: true).
    private static func reflectPad(_ samples: ArraySlice<Float>, padding: Int) -> [Float] {
        let values = Array(samples)
        guard values.count > 1, padding > 0 else { return values }
        let left = (1...padding).map { values[min($0, values.count - 1)] }.reversed()
        let right = (0..<padding).map { values[max(0, values.count - 2 - $0)] }
        return Array(left) + values + right
    }

    private static func predict(_ model: MLModel, inputs: MLFeatureProvider, stage: String, diagnostics: SongAnalysisDiagnostics?) throws -> MLFeatureProvider {
        try Task.checkCancellation()
        let shapes = inputs.featureNames.sorted().map { name in
            "\(name)=\(inputs.featureValue(for: name)?.multiArrayValue?.shape ?? [])"
        }.joined(separator: ", ")
        let start = ProcessInfo.processInfo.systemUptime
        defer { diagnostics?.metric("songformer.predict."+stage,since:start,detail:shapes) }
        NSLog("WORSHIP PREDICT_START %@ %@",stage,shapes)
        do {
            let result = try autoreleasepool { try model.prediction(from: inputs) }
            NSLog("WORSHIP PREDICT_DONE %@ elapsed=%.3fs",stage,ProcessInfo.processInfo.systemUptime-start)
            return result
        } catch {
            let ns = error as NSError
            diagnostics?.event("songformer.prediction_failed",detail:"\(stage) \(shapes): \(ns.domain) \(ns.code) \(ns.localizedDescription)",persist:true)
            NSLog("WORSHIP FAILED %@ domain=%@ code=%ld %@",stage,ns.domain,ns.code,ns.localizedDescription)
            throw AnalyzerError.inference("\(stage) [\(shapes)]: \(ns.domain) \(ns.code) — \(ns.localizedDescription)")
        }
    }

    private static func firstInputName(of model: MLModel) -> String {
        model.modelDescription.inputDescriptionsByName.keys.sorted().first ?? "input"
    }

    private static func firstMultiArray(in outputs: MLFeatureProvider) -> MLMultiArray? {
        outputs.featureNames.sorted().lazy.compactMap { outputs.featureValue(for: $0)?.multiArrayValue }.first
    }

    // MARK: - Manifest

    /// songformer_coreml_manifest.json, written by the export script next to the models.
    nonisolated private struct Manifest: Decodable {
        nonisolated struct Models: Decodable {
            let log_mel: String
            let muq_conv: String
            let muq: String
            let musicfm_conv: String
            let musicfm: String
            let head: String
        }

        nonisolated struct RotaryFrequencies: Decodable {
            let muq: [Float]
            let musicfm: [Float]
        }

        nonisolated struct Head: Decodable {
            let dataset_id: Int
            let allowed_label_ids: [Int]
            let id_to_label: [String: String]
        }
        nonisolated struct Installation: Decodable { let source_validation_sha256: String }

        let sample_rate: Int
        let head_fps: Double?
        let encoder_window_seconds: Int
        let head_max_seconds: Int
        let models: Models
        let rotary_inv_freq: RotaryFrequencies
        let head: Head
        let worship_static: [String: String]?
        let installation_provenance: Installation?
    }
}
