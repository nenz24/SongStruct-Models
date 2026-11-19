import SwiftUI
import UIKit

@main
struct BenchmarkApp: App {
    var body: some Scene { WindowGroup { BenchmarkView() } }
}
struct BenchmarkView: View {
    @State private var titles: [String] = []
    @State private var track = 0
    @State private var mode = "cpuOnly"
    @State private var running = false
    @State private var gpuAvailable = false
    @State private var frontendAvailable = false
    @State private var frontendMode = "cpuOnly"
    @State private var progress = "Ready"
    @State private var reportURL: URL?
    @State private var checkpointURL: URL?
    @State private var message = "Use a physical iPhone. GPU candidate mode requires the split-convolution repair. Complete one selected-track run before the three-pass batch. No SongMap models are installed by this probe."
    private let runner = BenchmarkRunner()
    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Track", selection: $track) { ForEach(titles.indices, id: \.self) { Text(titles[$0]).tag($0) } }.disabled(running)
                if frontendAvailable {
                    Text("Full-window encoders: CPU + GPU (fixed)").font(.caption)
                    Picker("Convolution frontend", selection: $frontendMode) {
                        Text("CPU control").tag("cpuOnly")
                        Text("GPU 30s / CPU others").tag("gpuShort30")
                    }.disabled(running)
                } else {
                    Picker("Full-window encoder mode", selection: $mode) {
                        Text("CPU only").tag("cpuOnly")
                        if gpuAvailable { Text("CPU + GPU candidate").tag("cpuAndGPU") }
                    }.disabled(running)
                }
                Button(running ? "Running…" : "Run selected track") { start(batch: false) }.disabled(running || titles.isEmpty)
                if frontendAvailable {
                    Button("Compare frontend routes (AB / BA)") { start(batch: true, comparison: true) }.disabled(running || titles.isEmpty)
                } else {
                    Button("Run 3 passes over all tracks") { start(batch: true) }.disabled(running || titles.isEmpty)
                }
                Text(progress).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                if let reportURL { ShareLink("Share result JSON", item: reportURL) }
                if let checkpointURL { ShareLink("Share last recorded stage", item: checkpointURL) }
                ScrollView { Text(message).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }.padding().navigationTitle("Encoder pipeline probe")
                .task {
                    do {
                        titles = try await runner.tracks(); gpuAvailable = try await runner.gpuAvailable()
                        frontendAvailable = try await runner.frontendGPUAvailable()
                        if frontendAvailable {
                            mode = "cpuAndGPU"
                            message = "GPU frontend is restricted to full 30-second windows. Long and partial frontends, log-mel, partial encoders, head and context stay CPU. Full-window repaired encoders keep CPU+GPU. Check label_parity_passed before AB/BA."
                        }
                    } catch { message = error.localizedDescription }
                    if FileManager.default.fileExists(atPath: BenchmarkRunner.checkpoint.path()) { checkpointURL = BenchmarkRunner.checkpoint }
                    if FileManager.default.fileExists(atPath: BenchmarkRunner.batchURL.path()) { reportURL = BenchmarkRunner.batchURL }
                }
        }
    }
    private func start(batch: Bool, comparison: Bool = false) {
        running = true; reportURL = nil; progress = "Starting…"
        UIApplication.shared.isIdleTimerDisabled = true
        Task {
            defer { running = false; UIApplication.shared.isIdleTimerDisabled = false }
            let update: @Sendable (String) -> Void = { text in Task { @MainActor in progress = text } }
            do {
                let url: URL
                if comparison { url = try await runner.compareFrontends(progress: update) }
                else if batch { url = try await runner.batch(mode: mode, frontendMode: frontendMode, progress: update) }
                else { url = try await runner.run(trackIndex: track, mode: mode, frontendMode: frontendMode, progress: update) }
                reportURL = url; message = try String(contentsOf: url, encoding: .utf8); progress = "Completed"
                if let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
                   object["label_parity_passed"] as? Bool == false { progress = "Completed; label/probability parity failed" }
            } catch {
                message = "Stopped: \(error.localizedDescription)"; progress = "Stopped; share the last-stage report"
                if batch && FileManager.default.fileExists(atPath: BenchmarkRunner.batchURL.path()) { reportURL = BenchmarkRunner.batchURL }
            }
            if FileManager.default.fileExists(atPath: BenchmarkRunner.checkpoint.path()) { checkpointURL = BenchmarkRunner.checkpoint }
        }
    }
}
