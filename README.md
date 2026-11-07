# SongStruct Models — anotasi worship

Tahap awal pipeline anotasi Apple Music Understanding → label SongFormer → koreksi manual → fine-tuning.

## Jalankan di terminal VSCode

Edit `DATASET_PATH` di bagian atas `check_dataset.py` ke path absolut folder audio Anda. Setelah itu cukup jalankan:

```bash
python3 check_dataset.py
```

Path contoh di skrip harus diganti sebelum digunakan. Argumen `--dataset` di bawah tetap tersedia sebagai override opsional.

Butuh Python 3.9 atau lebih baru. Skrip pemeriksaan ini tidak membutuhkan instalasi paket.

```bash
python3 check_dataset.py --dataset "/path/absolut/Dataset-Automation/dataset_mp3"
```

Path harus menunjuk audio yang tersedia di komputer tempat Python berjalan. Folder notebook atau skrip boleh berbeda dari folder dataset. Anda juga bisa menyeret folder dari Finder ke terminal untuk mendapatkan path.

Opsional, simpan inventaris audio ke file baru:

```bash
python3 check_dataset.py --dataset "/path/absolut/Dataset-Automation/dataset_mp3" --manifest tracks.local.json
```

Manifest berisi path lokal dan ukuran file, bukan hasil analisis audio. ID menggunakan path relatif agar nama file yang sama pada subfolder berbeda tetap berbeda. Gunakan audio yang sama pada Mac dan server; jangan mengubah panjang atau memangkas audio di salah satu sisi.

## Periksa kesiapan Mac

```bash
sw_vers -productVersion
xcodebuild -version
```

## Ekspor boundary Apple di Mac

Setelah mengisi DATASET_PATH di check_dataset.py, jalankan:

```bash
python3 export_apple_boundaries.py
```

Default memproses 5 MP3 pertama secara alfabetis. Exporter Python mengompilasi kode Swift menggunakan Xcode aktif, lalu menjalankan analisis .structure satu lagu per proses iterasi. Tidak ada paket pip tambahan.

Output: outputs.local/apple/<path-relatif-audio>.json. Setiap file menyimpan sections, segments, phrases dalam detik, hasil mentah Apple, durasi, dan status pending. Boundary tidak di-snap, digabung, atau dilebur. Label SongFormer belum ditambahkan. Jangan anggap boundary ini sebagai ground truth sebelum review manual.

File output yang sudah ada dilewati, sehingga bisa melanjutkan batch setelah kegagalan. Untuk mengulang sebuah lagu, pindahkan/hapus file outputnya terlebih dahulu. Jangan gunakan output lama setelah audio sumber diubah.

Setelah pilot diperiksa, seluruh dataset dapat diproses dengan:

```bash
python3 export_apple_boundaries.py --limit 0
```

Butuh macOS/SDK yang menyediakan MusicUnderstanding. Versi Xcode saja belum membuktikan framework berhasil dikompilasi atau dijalankan; jika ada error, kirim log kompilasi/runtime. Wrapper Python telah diuji sintaks/CLI di Linux; exporter Swift belum dapat diuji di Mac pada lingkungan pengembangan ini.

Skrip ini belum menjalankan SongFormer atau training.

## Tahap berikutnya

1. Ekspor boundary asli Apple `structure.sections` di Mac, tanpa merge/fold/snapping otomatis.
2. Jalankan SongFormer dengan konteks lagu dan simpan probabilitas label per frame.
3. Agregasikan probabilitas pada setiap rentang section Apple.
4. Koreksi label dan boundary manual, pertahankan hasil awal dan status review.
5. Fine-tune hanya anotasi approved; pisahkan versi lagu yang sama dalam satu kelompok split.
6. Evaluasi sebelum dan sesudah ekspor Core ML dan palettization 6-bit.

Jangan commit audio, checkpoint, path lokal, atau kredensial ke repositori publik.


## Aplikasi anotasi bersama

Aplikasi web cloud tersedia di [annotation_web/README.md](annotation_web/README.md). Versi GitHub + Cloudflare memakai kode akses berbeda per anotator; tidak membutuhkan akun ChatGPT untuk teman. Fitur: unggah folder JSON berlabel dan MP3, pemutaran per section, boundary terkunci, koreksi label, penugasan, simpan draf, kirim review, persetujuan, dan ekspor JSON bersama. Tidak ada audio/checkpoint/kredensial di repository.

Setelah deploy mengikuti panduan aplikasi dan mengekspor hasil approved, pulihkan ke format lokal:

```bash
python import_cloud_annotations.py /path/songstruct-approved.json
```

Skrip mencocokkan track ID dan boundary dengan JSON lokal, lalu menyimpan ke outputs.local/reviewed tanpa menimpa sumber atau output lama. Source aplikasi tersedia; deployment ke akun Cloudflare Anda dan pengujian browser tim masih perlu dilakukan.


## Pipeline fine-tuning worship

Lihat [finetuning/README.md](finetuning/README.md) untuk persiapan inventaris, audit kelompok lagu, split tetap, cache fitur MuQ/MusicFM, training label head, dan evaluasi holdout. Pipeline hanya memakai label_manual yang sudah approved, mempertahankan boundary Apple, dan tidak mengubah database anotasi.

Mulai dari persiapan ringan, tanpa memuat model atau menjalankan training:

```bash
python3 -m finetuning.prepare
```

Periksa outputs.local/finetuning/group_map.csv sebelum mengonfirmasi kelompok. Ekstraksi cache bisa ditunda dan membutuhkan ruang disk tambahan; pelabelan lama belum menyimpan fitur ini. Training model penuh belum dijalankan pada dataset pengguna. Pengujian CPU mencakup data/split, gradient dan freeze, loop training sintetis, serta checkpoint EMA round-trip.
