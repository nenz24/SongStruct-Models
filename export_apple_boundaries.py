#!/usr/bin/env python3
"""Build and run the native Apple boundary exporter on a Mac."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from check_dataset import DATASET_PATH

# Path audio dibaca dari DATASET_PATH di check_dataset.py.
# Output dan build selalu relatif terhadap folder skrip, bukan folder terminal.
BASE = Path(__file__).resolve().parent
OUTPUT_PATH = BASE / "outputs.local" / "apple"
PILOT_LIMIT = 5  # 0 = seluruh dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=PILOT_LIMIT)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit harus >= 0")
    if platform.system() != "Darwin":
        parser.error("Exporter Apple harus dijalankan di macOS dengan Xcode dan MusicUnderstanding SDK.")
    root = DATASET_PATH.expanduser().resolve()
    if not root.is_dir():
        parser.error("Edit DATASET_PATH di check_dataset.py terlebih dahulu.")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp3")
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error("Tidak ada MP3 ditemukan.")
    # Preserve relative paths, including extension, to avoid same-stem collisions.
    tracks = [{"id": p.relative_to(root).as_posix(), "audio_path": str(p),
               "output_path": str(OUTPUT_PATH / (p.relative_to(root).as_posix() + ".json"))}
              for p in files]
    build = BASE / ".build"
    build.mkdir(exist_ok=True)
    binary = build / "export-boundaries"
    print("Memeriksa SDK dan mengompilasi exporter Swift...", flush=True)
    try:
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        subprocess.run(["xcrun", "--sdk", "macosx", "swiftc", "-parse-as-library",
                        "-sdk", sdk, str(BASE / "apple" / "ExportBoundaries.swift"),
                        "-o", str(binary)], check=True)
        print(f"Memproses {len(tracks)} lagu. Output: {OUTPUT_PATH}", flush=True)
        with tempfile.TemporaryDirectory(prefix="songstruct-") as tmp:
            manifest = Path(tmp) / "tracks.json"
            manifest.write_text(json.dumps(tracks, ensure_ascii=False), encoding="utf-8")
            subprocess.run([str(binary), str(manifest)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Ekspor gagal: {exc}", flush=True)
        print("Output yang sudah berhasil tetap disimpan; jalankan ulang untuk melewati file tersebut.")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
