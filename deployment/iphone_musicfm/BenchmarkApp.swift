import SwiftUI

@main
struct BenchmarkApp: App {
    var body: some Scene { WindowGroup { BenchmarkView() } }
}

struct BenchmarkView: View {
    @State private var mode = "cpuOnly"
    @State private var candidate = "int8"
    @State private var candidates = ["int8", "palette6"]
    @State private var fixture = "short"
    @State private var running = false
    @State private var report = "Fixed-shape benchmark: begin with cpuOnly and short, then failed_full. Test GPU modes separately. Use a physical iPhone."
    @State private var reportURL: URL?
    private let runner = BenchmarkRunner()
    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Candidate", selection: $candidate) { ForEach(candidates, id: \.self) { Text($0) } }.disabled(running)
                Picker("Fixture", selection: $fixture) {
                    Text("Short prefix (~30s)").tag("short")
                    Text("Pilihanku failed window (~119s)").tag("failed_full")
                }.disabled(running)
                Picker("Compute units", selection: $mode) {
                    ForEach(["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"], id: \.self) { Text($0) }
                }.disabled(running)
                Button(running ? "Measuring…" : "Run MusicFM benchmark") {
                    running = true
                    reportURL = nil
                    Task {
                        defer { running = false }
                        do {
                            report = try await runner.run(mode: mode, candidateName: candidate, fixtureName: fixture)
                            let url = FileManager.default.temporaryDirectory.appendingPathComponent("musicfm-benchmark-\(UUID().uuidString).json")
                            try report.write(to: url, atomically: true, encoding: .utf8)
                            reportURL = url
                        } catch { report = error.localizedDescription }
                    }
                }.disabled(running)
                if let reportURL { ShareLink("Share JSON report", item: reportURL) }
                ScrollView { Text(report).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }.padding().navigationTitle("MusicFM benchmark")
                .task {
                    if let url = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json"),
                       let data = try? Data(contentsOf: url),
                       let manifest = try? JSONDecoder().decode(BenchmarkRunner.Manifest.self, from: data) {
                        candidates = manifest.models.map(\.name)
                    }
                }
        }
    }
}
