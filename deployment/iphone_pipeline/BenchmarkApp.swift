import SwiftUI
import UIKit
@main struct BenchmarkApp: App {
    var body: some Scene { WindowGroup { BenchmarkView() } }
}
struct BenchmarkView: View {
    @State private var running = false
    @State private var progress = "Ready"
    @State private var text = "Full PCM pipeline (routing selected by benchmark manifest). Keep this app open while running. Models and real audio are bundled."
    @State private var reportURL: URL?
    @State private var diagnosticURL: URL?
    let runner = BenchmarkRunner()
    var body: some View {
        NavigationStack {
            VStack {
                Button(running ? "Running…" : "Run full pipeline") {
                    running = true; reportURL = nil; progress = "Starting…"
                    UIApplication.shared.isIdleTimerDisabled = true
                    Task {
                        defer { running = false; UIApplication.shared.isIdleTimerDisabled = false }
                        do {
                            text = try await runner.run { message in
                                Task { @MainActor in progress = message }
                            }
                            let url = FileManager.default.temporaryDirectory.appendingPathComponent("pipeline-benchmark-\(UUID().uuidString).json")
                            try text.write(to: url, atomically: true, encoding: .utf8)
                            reportURL = url
                        } catch {
                            let message = "FAILED: \(error.localizedDescription)"
                            NSLog("PIPELINE %@", message)
                            progress = message
                            text = String(describing: error)
                        }
                    }
                }.disabled(running)
                Text(progress).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                if let reportURL { ShareLink("Share JSON report", item: reportURL) }
                if let diagnosticURL { ShareLink("Share last recorded stage", item: diagnosticURL) }
                ScrollView { Text(text).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
            }.padding().navigationTitle("Song pipeline benchmark")
                .task {
                    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                        .appendingPathComponent("pipeline-last-stage.json")
                    if FileManager.default.fileExists(atPath: url.path) {
                        diagnosticURL = url
                        if let saved = try? String(contentsOf: url, encoding: .utf8) { text = "Last recorded stage (not a crash diagnosis):\n" + saved }
                    }
                }
        }
    }
}
