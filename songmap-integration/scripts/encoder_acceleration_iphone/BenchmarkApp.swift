import SwiftUI

@main
struct BenchmarkApp: App {
    var body: some Scene { WindowGroup { BenchmarkView() } }
}

struct BenchmarkView: View {
    @State private var fixture = ""
    @State private var fixtures: [String] = []
    @State private var mode = "cpuOnly"
    @State private var started = false
    @State private var message = "Start with CPU only. Relaunch before changing profile or compute mode. This isolated graph candidate has not been approved for SongMap."
    @State private var reportURL: URL?
    private let runner = BenchmarkRunner()
    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Profile", selection: $fixture) {
                    ForEach(fixtures, id: \.self) { Text($0).tag($0) }
                }.disabled(started)
                Picker("Compute mode", selection: $mode) {
                    ForEach(["cpuOnly", "cpuAndNeuralEngine", "cpuAndGPU"], id: \.self) { Text($0).tag($0) }
                }.disabled(started)
                Button(started ? "Relaunch for another measurement" : "Run encoder benchmark") {
                    started = true
                    Task {
                        do {
                            reportURL = try await runner.run(fixtureName: fixture, mode: mode)
                            message = try String(contentsOf: reportURL!, encoding: .utf8)
                        } catch {
                            message = "Stopped: \(error.localizedDescription). The last-stage report is available below."
                            if FileManager.default.fileExists(atPath: BenchmarkRunner.lastStageURL.path()) {
                                reportURL = BenchmarkRunner.lastStageURL
                            }
                        }
                    }
                }.disabled(started || fixture.isEmpty)
                if let reportURL { ShareLink("Share JSON report", item: reportURL) }
                ScrollView {
                    Text(message).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }.padding().navigationTitle("Encoder graph probe")
                .task {
                    do {
                        guard let url = Bundle.main.url(forResource: "benchmark_manifest", withExtension: "json") else { return }
                        let manifest = try JSONDecoder().decode(BenchmarkRunner.Manifest.self, from: Data(contentsOf: url))
                        fixtures = manifest.fixtures.map(\.name); fixture = fixtures.first ?? ""
                        if FileManager.default.fileExists(atPath: BenchmarkRunner.lastStageURL.path()) {
                            reportURL = BenchmarkRunner.lastStageURL
                        }
                    } catch { message = error.localizedDescription }
                }
        }
    }
}
