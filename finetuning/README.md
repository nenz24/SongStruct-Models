# Fine-tuning SongFormer untuk label worship

Pipeline: anotasi approved → validasi terhadap boundary awal → split per identitas lagu → cache fitur MuQ/MusicFM → fine-tune label head → evaluasi → checkpoint. Boundary tetap ditentukan Apple. Default tidak melatih boundary head, MuQ, atau MusicFM. Head SongFormer tetap memiliki 128 output dengan dataset ID 5; loss dan evaluasi hanya menggunakan kelas 0,1,2,3,4,5,6,26.

## Yang bisa dikerjakan sambil menunggu anotasi

Dari root repo, gunakan environment SongFormer yang sudah berhasil menjalankan pelabelan:

```bash
source .venv-songformer311/bin/activate
python3 -m finetuning.prepare
```

Perintah ini membaca DATASET_PATH dari check_dataset.py, JSON awal dari outputs.local/labeled, dan koreksi approved dari outputs.local/reviewed jika tersedia. Tidak memuat model, mengunggah audio, atau memulai training. Menghasilkan outputs.local/finetuning/manifest.json dan group_map.csv, menghitung SHA256 audio untuk mendeteksi salinan identik, serta memperkirakan kebutuhan cache.

Buka **outputs.local/finetuning/group_map.csv**. Satu group_id harus mencakup semua versi dari komposisi yang sama: live, studio, cover, instrumental, judul alternatif, dan typo nama folder. Saran awal berdasarkan nama folder hanya titik awal; tidak membuktikan identitas musik. Hash mendeteksi file MP3 yang identik secara byte, bukan cover atau encoding ulang. Gabungkan alias seperti Way Maker / Way Maker (Original) secara manual jika memang lagu yang sama.

Setelah benar-benar diperiksa:

```bash
python3 -m finetuning.prepare --confirm-groups
```

Ini mengunci split group 80/10/10 ke split_plan.json dengan seed 42. Seluruh inventaris dipakai untuk merencanakan split meskipun belum approved. Setelahnya, persetujuan anotasi tambahan tidak memindahkan lagu antar split. Jika komposisi inventaris, kelompok, seed, atau rasio berubah, gunakan work_dir eksperimen baru; jangan mengedit frozen plan.

## Cache fitur sekali, sebelum atau sesudah anotasi selesai

Konfigurasi ada di finetuning/config.json. Semua path relatif terhadap root repo, bukan direktori model upstream. Default profil mengikuti SongMapApp: mono 24 kHz, layer 10, stream MusicFM30 + MuQ30 + MusicFM120 + MuQ120, blok head 420 detik, output 25/3 Hz. Profil upstream 30/420 dapat dipilih dengan mengubah long_window_seconds; gunakan work_dir berbeda untuk membandingkan profil.

Uji satu audio dahulu:

```bash
python3 -m finetuning.features --device mps --limit 1
```

Jika disk dan perangkat memadai:

```bash
python3 -m finetuning.features --device mps
```

Pada server NVIDIA, gunakan --device cuda. Pemilihan auto mencoba CUDA, lalu MPS, lalu CPU. Lakukan pilot karena dukungan backward MPS pada model penuh belum diuji.

**Ekstraksi pertama tetap mahal.** Run pelabelan lama tidak menyimpan embedding, sehingga fitur tidak bisa dipulihkan dari label saja. Cache ini membutuhkan satu kali ekstraksi tambahan. Setelah tersimpan, training tidak menghitung MuQ/MusicFM ulang setiap epoch. Simpan float32 untuk menjaga parity dengan preprocessing awal; lagu 4 menit kira-kira 94 MiB / 98 MB, sekitar 30 GiB untuk 323 lagu dengan rata-rata tersebut. Panjang lagu sebenarnya menentukan kebutuhan disk.

Metadata cache mencatat hash audio, konfigurasi fitur, revisi upstream, hash MusicFM dan stats, serta snapshot commit MuQ. Snapshot awal memakai cache Hugging Face lokal jika tersedia; setelah itu revisi dikunci. Cache selesai dilewati saat dijalankan ulang. File NPY dan metadata ditulis secara atomik. Jangan mencampur cache dari encoder berbeda. Audio di luar region section Apple tidak diberi label otomatis.

## Setelah hasil anotasi disetujui

Di web, Ekspor disetujui mengunduh songstruct-approved.json. Ada dua pilihan:

```bash
python3 -m finetuning.prepare --export "$HOME/Downloads/songstruct-approved.json"
```

Atau gunakan jalur lokal yang sudah ada:

```bash
python3 import_cloud_annotations.py "$HOME/Downloads/songstruct-approved.json"
python3 -m finetuning.prepare
```

Hanya review_status=approved dengan label_manual valid pada **semua section** yang menjadi target. label_initial/label prediksi tidak digunakan sebagai ground truth. Boundary, durasi, dan prediksi awal dibandingkan dengan JSON sumber; perubahan yang tidak cocok ditolak. Anotasi pending/submitted diabaikan. Jalankan prepare lagi setelah ekspor baru untuk snapshot training berikutnya. Jangan mengganti manifest saat training sedang berjalan.

## Audit catatan sebelum training

Status approved di web berarti anotasi diterima pengelola, bukan jaminan semua boundary benar. Sebelum mempersiapkan target, jalankan audit pada ekspor terbaru:

```bash
python3 -m finetuning.audit --export "$HOME/Downloads/songstruct-approved.json"
```

Tidak memerlukan audio/model. Hasil: outputs.local/finetuning/annotation_audit.json dan section_review.csv. CSV memuat seluruh section approved. Filter kolom issue_hint selain clean untuk melihat catatan; kategori boundary_or_timing dan worship_term hanya petunjuk, bukan diagnosis otomatis.

- Section tanpa catatan: default decision=include.
- Section bercatatan: default pending dan belum dipakai training.
- Ubah decision menjadi include hanya jika label dan boundary memang layak; exclude jika salah atau tidak ingin dipakai. pending juga menahan section.
- Jika boundary salah memengaruhi section sebelum/sesudahnya, tandai tetangganya exclude juga. Semua section tersedia di CSV untuk keperluan ini.
- Baris dengan boundary_issue=true atau training_eligible=false pada data sumber selalu dikecualikan, meskipun CSV berisi include.
- Jangan mengedit start/end, track_id, section_index, atau source_digest di CSV. CSV ini mengatur kelayakan training, bukan memperbaiki timestamp.
- Keputusan lama dipertahankan selama label manual, waktu, catatan, dan identitas section tidak berubah. Catatan/label berubah → keputusan lama tidak berlaku dan section bercatatan ditahan lagi. Ekspor parsial tidak menghapus keputusan untuk lagu yang tidak ada di ekspor tersebut.

Prepare menjalankan audit yang sama secara otomatis:

```bash
python3 -m finetuning.prepare --export "$HOME/Downloads/songstruct-approved.json"
```

Hanya section layak dari lagu approved yang dimasukkan ke manifest. Region yang dikecualikan tidak diisi dengan label tetangga; fungsi loss dan metrik melewatinya. Lagu yang semua section-nya ditahan tetap tercatat sebagai review_approved tetapi tidak menjadi sampel training/validation. Split group tetap tidak dipindahkan. JSON asli, catatan cloud, dan boundary tidak diubah.

## Latih label head

Butuh approved audio pada split train **dan** validation. Jika validation belum mendapat anotasi, tunggu atau prioritaskan review lagu validation; jangan memindahkan holdout untuk memaksa training.

```bash
python3 -m finetuning.train --device mps --run-name worship-v1
```

Default: function_head saja, AdamW LR 1e-4, accumulation 4 chunk, maksimal 20 epoch, clipping gradient 1, early stopping 4 epoch. MuQ/MusicFM tidak dimuat saat training; input berasal dari cache. Frozen backbone tetap eval agar dropout tidak mengubah representasinya. Objective merata-ratakan probabilitas frame pada rentang section dengan bobot overlap durasi, lalu menghitung negative log probability label manual. Bagian section yang memotong blok 420 detik dipelajari per chunk; evaluasi menggabungkan seluruh chunk sebelum menentukan label section.

Pilihan tahap berikutnya: training.trainable=label_backbone, LR lebih kecil misalnya 1e-5, dan run-name baru. Ini melatih projection/transformer/function head SongFormer, tetap tidak melatih MuQ/MusicFM atau parameter boundary head. Karena backbone berbagi, prediksi boundary model bisa berubah; pipeline tetap menggunakan boundary Apple. Bukan LoRA, QLoRA, atau quantization-aware training.

Setiap run baru dibuat terpisah dan gagal jika nama sudah dipakai. Tidak menimpa pretrained checkpoint, hasil pelabelan, atau database anotasi. Resume optimizer di tengah run belum diimplementasikan. latest.pt adalah snapshot bobot terbaru; best dipilih berdasarkan macro F1 kelas yang hadir pada validation. Jika tidak mengalahkan pretrained baseline, best tetap baseline.

Output di outputs.local/finetuning/runs/worship-v1:

- run.json: konfigurasi, identitas split/encoder/anotasi, dan hash checkpoint awal.
- baseline_val.json: hasil pretrained pada validation yang sama.
- history.jsonl: loss training dan metrik validation per epoch.
- best.pt, latest.pt: state model lengkap beserta metadata eksperimen.
- best.safetensors: EMA wrapper dengan bobot best, mengikuti format loader upstream dan skrip pelabelan repo ini. Ini snapshot bobot best, bukan EMA temporal dari seluruh riwayat optimizer.

## Evaluasi holdout

Gunakan test hanya untuk penilaian akhir; tidak dipakai optimizer, scheduler, atau early stopping. Bandingkan baseline dan checkpoint pada split yang sama:

```bash
python3 -m finetuning.evaluate --split test --device mps --output outputs.local/test-baseline.json
python3 -m finetuning.evaluate --split test --device mps --checkpoint outputs.local/finetuning/runs/worship-v1/best.pt --output outputs.local/test-worship-v1.json
```

Laporan mencakup section accuracy, duration-weighted accuracy, macro F1 kelas yang hadir, confusion matrix 8 kelas, support per kelas, dan section dengan coverage fitur kurang dari 98%. Tidak mengklaim perbaikan boundary karena boundary tidak dilatih. Evaluasi best.pt memeriksa identitas split/cache; safetensors sendiri tidak menyimpan metadata eksperimen, jadi gunakan .pt untuk evaluasi yang terikat eksperimen.

## Prediksi dengan model hasil fine-tuning

Skrip label_apple_sections.py sekarang mendukung checkpoint dan folder output terpisah:

```bash
python3 label_apple_sections.py --device mps --limit 1 --checkpoint outputs.local/finetuning/runs/worship-v1/best.safetensors --output-dir outputs.local/labeled-worship-v1
```

Jangan mengganti JSON awal yang menjadi acuan anotasi berjalan. Tetap memakai profil encoder 120 detik bawaan skrip pelabelan; jika meneliti profil 420 detik, samakan ENCODER_WINDOW_SECONDS secara eksplisit atau gunakan evaluator cache.

## iPhone dan 6-bit

Pipeline ini menghasilkan checkpoint label presisi penuh. Ekspor Core ML dan palettization 6-bit merupakan tahap berikutnya setelah akurasi worship terbukti. Belum ada jaminan akurasi, latensi, RAM, atau parity Core ML dari pipeline ini. Seluruh encoder juga berkontribusi pada ukuran aplikasi. Uji FP32 → Core ML → 6-bit pada holdout identik sebelum mengganti model aplikasi.

## Validasi yang sudah dilakukan

```bash
python3 -m unittest discover -s tests -p test_finetuning.py -v
```

Unit/integration smoke tests memakai anotasi serta fitur sintetis dan model kecil: menolak anotasi belum approved/boundary berubah, split reproducible, gradient menuju label manual, hanya label head berubah saat dibekukan, loop training/validation/early stopping berjalan, dan round-trip checkpoint EMA safetensors identik. Tidak mengunduh bobot MuQ/MusicFM atau menjalankan fine-tuning model penuh di lingkungan pengembang; model penuh, data worship nyata, serta akselerasi Mac masih perlu pilot pada perangkat Anda.
