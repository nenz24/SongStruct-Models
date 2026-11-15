import SwiftUI
import UIKit
@main struct BenchmarkApp: App {
    var body: some Scene { WindowGroup { BenchmarkView() } }
}
struct BenchmarkView: View {
    @State private var running = false
    @State private var text = "CPU-only full PCM pipeline. Keep this app open while running. Models and real audio are bundled."
    @State private var reportURL: URL?
    let runner = BenchmarkRunner()
    var body: some View {
        NavigationStack {
            VStack {
                Button(running ? "Running…" : "Run full pipeline") {
                    running = true; reportURL = nil
                    UIApplication.shared.isIdleTimerDisabled = true
                    Task {
                        defer { running = false; UIApplication.shared.isIdleTimerDisabled = false }
                        do {
                            text = try await runner.run()
                            let url = FileManager.default.temporaryDirectory.appendingPathComponent("pipeline-benchmark-\(UUID().uuidString).json")
                            try text.write(to: url, atomically: true, encoding: .utf8)
                            reportURL = url
                        } catch { text = error.localizedDescription }
                    }
                }.disabled(running)
                if let reportURL { ShareLink("Share JSON report", item: reportURL) }
                ScrollView { Text(text).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
            }.padding().navigationTitle("Song pipeline CPU")
        }
    }
}
