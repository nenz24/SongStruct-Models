#!/usr/bin/env python3
"""Check a local audio dataset without third-party dependencies."""
import argparse
import json
from pathlib import Path
import sys

# Ubah ke path absolut folder audio Anda; boleh di luar repositori.
DATASET_PATH = Path("/path/absolut/Dataset-Automation/dataset_mp3")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Opsional: mengganti DATASET_PATH di file ini.")
    parser.add_argument("--manifest", type=Path, help="Write a JSON audio inventory to a NEW file.")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    print(f"Folder kerja: {Path.cwd()}", flush=True)
    print(f"Dataset: {root}", flush=True)
    if not root.is_dir():
        parser.error("Folder dataset tidak ditemukan. Edit DATASET_PATH di check_dataset.py ke path absolut folder audio Anda.")
    try:
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp3")
        records = [{"id": p.relative_to(root).as_posix(), "audio_path": str(p),
                    "relative_path": p.relative_to(root).as_posix(),
                    "size_bytes": p.stat().st_size} for p in files]
    except OSError as exc:
        parser.exit(1, f"Gagal membaca dataset: {exc}\n")
    print(f"Jumlah MP3: {len(files)}", flush=True)
    for item in records[:5]:
        print(item["audio_path"])
    if not files:
        parser.exit(1, "Tidak ada MP3 ditemukan. Periksa lokasi dataset.\n")
    if args.manifest:
        try:
            with args.manifest.expanduser().open("x", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "dataset_root": str(root), "tracks": records},
                          handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError as exc:
            parser.exit(1, f"Gagal menulis manifest (file lama tidak ditimpa): {exc}\n")
        print(f"Manifest: {args.manifest.expanduser().resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
