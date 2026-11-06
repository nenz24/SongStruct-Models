# SongStruct Models — anotasi worship

Tahap awal pipeline anotasi Apple Music Understanding → label SongFormer → koreksi manual → fine-tuning.

## Jalankan di terminal VSCode

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

Versi macOS dan SDK perlu diperiksa sebelum implementasi ekspor Music Understanding. Skrip ini tidak menjalankan framework Apple, SongFormer, atau training.

## Tahap berikutnya

1. Ekspor boundary asli Apple `structure.sections` di Mac, tanpa merge/fold/snapping otomatis.
2. Jalankan SongFormer dengan konteks lagu dan simpan probabilitas label per frame.
3. Agregasikan probabilitas pada setiap rentang section Apple.
4. Koreksi label dan boundary manual, pertahankan hasil awal dan status review.
5. Fine-tune hanya anotasi approved; pisahkan versi lagu yang sama dalam satu kelompok split.
6. Evaluasi sebelum dan sesudah ekspor Core ML dan palettization 6-bit.

Jangan commit audio, checkpoint, path lokal, atau kredensial ke repositori publik.
