import SwiftUI

@main
struct BenchmarkApp: App {
    var body: some Scene { WindowGroup { BenchmarkView() } }
}

struct BenchmarkView: View {
    @State private var mode = "all"
    @State private var running = false
    @State private var report = "Run on your physical iPhone. This measures the label head only."
    @State private var reportURL: URL?
    private let runner = BenchmarkRunner()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Compute units", selection: $mode) {
                    ForEach(["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"], id: \.self) { Text($0) }
                }.disabled(running)
                Button(running ? "Measuring…" : "Benchmark FP32 / INT8 / palette6") {
                    running = true
                    reportURL = nil
                    Task {
                        defer { running = false }
                        do {
                            report = try await runner.run(mode: mode)
                            let url = FileManager.default.temporaryDirectory.appendingPathComponent("head-benchmark-\(UUID().uuidString).json")
                            try report.write(to: url, atomically: true, encoding: .utf8)
                            reportURL = url
                        } catch { report = error.localizedDescription }
                    }
                }.disabled(running)
                if let reportURL { ShareLink("Share JSON report", item: reportURL) }
                ScrollView { Text(report).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }.padding().navigationTitle("SongFormer benchmark")
        }
    }
}
