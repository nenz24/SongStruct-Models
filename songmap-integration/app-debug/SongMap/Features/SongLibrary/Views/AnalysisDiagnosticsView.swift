import SwiftUI

/// Debug-only entry point from Library. Files remain available after leaving Xcode or restarting.
struct AnalysisDiagnosticsView: View {
    private struct Item: Identifiable {
        let url: URL
        let report: SongAnalysisDiagnostics.Report
        var id: String { report.id }
    }
    @State private var items: [Item] = []
    var body: some View {
        List {
            SwiftUI.Section {
                Text("Saved locally on this iPhone. Reports include song title, timing, routing and errors; no audio or lyrics.").font(.caption)
            }
            ForEach(items) { item in
                VStack(alignment:.leading,spacing:8) {
                    Text(item.report.title).font(.headline)
                    Text("\(item.report.state) · \(item.report.totalSeconds,specifier:"%.1f") s").font(.caption)
                    if let stage = item.report.activeStage { Text("Last stage: " + stage).font(.caption) }
                    if let error = item.report.error { Text(error).font(.caption).foregroundStyle(.red) }
                    ShareLink("Share report",item:item.url)
                }
            }
            if items.isEmpty { Text("No analysis reports yet. Analyze a song first.") }
        }
        .navigationTitle("Analysis reports")
        .toolbar { Button("Refresh") { reload() } }
        .onAppear { reload() }
    }
    private func reload() {
        let urls = (try? FileManager.default.contentsOfDirectory(at:SongAnalysisDiagnostics.directory,
            includingPropertiesForKeys:[.contentModificationDateKey])) ?? []
        items = urls.filter { $0.pathExtension == "json" }.compactMap { url in
            guard let data = try? Data(contentsOf:url),
                  let report = try? JSONDecoder().decode(SongAnalysisDiagnostics.Report.self,from:data) else { return nil }
            return Item(url:url,report:report)
        }.sorted { $0.report.startedAt > $1.report.startedAt }
    }
}
