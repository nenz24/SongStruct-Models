# Integrasi label section ke SongMap — dokumentasi percobaan

Folder ini mengarsipkan semua percobaan yang dilakukan untuk membawa model label section worship (hasil fine-tuning di repo ini) ke app SongMap di iPhone: skrip, notebook, tes, kode Debug app, dan catatan lengkapnya. Hasil akhirnya dipakai di app lewat PR SongMapApp [#63](https://github.com/nenz24/SongMapApp/pull/63); yang masuk ke app hanya model dan pipeline akhir, sedangkan semua bahan percobaan ada di sini.

- Catatan harian lengkap (bahasa Inggris, urut waktu): [`WORSHIP_INTEGRATION.md`](WORSHIP_INTEGRATION.md)
- Periode: 15–19 November 2026. Perangkat uji: iPhone 17 (iPhone18,1), iOS 27.0.1. Mac untuk training dan validasi.

## Tujuan

Label section lagu worship (intro, verse, chorus, bridge, inst, outro, silence, pre-chorus) yang berjalan **offline di iPhone**, dengan:

1. akurasi tetap stabil (tidak lebih buruk dari model produksi),
2. ukuran model kecil (bobot sekitar 6-bit),
3. analisis cepat dan stabil (tidak crash, hasil sama setiap kali).

## Hasil akhir yang dipakai di app

**SongFormer MuQ-only + konteks section, CPU saja.**

```
MP3 → 24 kHz mono
  → log-mel → frontend konvolusi MuQ
  → bank encoder MuQ (fungsi short_30 / long_120 / partial) di setiap jendela 30 dtk dan 120 dtk
  → gabung [MuQ 30 dtk, MuQ 120 dtk] = fitur 2048 per frame
  → head fungsi 6-bit (palette6) per potongan 420 dtk → label setiap 0,12 dtk (softmax 8 kelas)
  → probabilitas dirata-rata per section Apple MusicUnderstanding
  → model konteks section (melihat seluruh lagu) → satu label per section
```

| Model (compiled `.mlmodelc`) | Ukuran |
|---|---:|
| `SongFormerMuQ_LogMel` | 17,3 MB |
| `SongFormerMuQ_Conv` | 26,3 MB |
| `SongFormerMuQ_Encoder` (bank MuQ multi-fungsi, palette6) | 184,5 MB |
| `SongFormerMuQ_Head` (palette6, input 2048) | 16,8 MB |
| `SongFormerMuQ_Context` (FP32, input 2048) | 0,7 MB |
| **Total** | **≈ 246 MB** (sebelumnya MusicFM + MuQ ≈ 555 MB) |

Di Release, model diunduh lewat asset pack `sections` (Background Assets); di Debug ikut di dalam app.

### Angka utama

| Pengukuran | Produksi lama (MusicFM + MuQ + konteks) | MuQ-only + konteks |
|---|---|---|
| Validasi 32 lagu / 451 section, PyTorch | 0.7938 / 0.7755 (acc / macro F1) | 0.8027 / 0.7674 |
| **Test terkunci 34 lagu / 520 section** | 0.8000 / 0.6997 | **0.8135 / 0.7182** |
| Validasi, pipeline Core ML lengkap | 0.7916 / 0.7672 | 0.8004 / 0.7664 |
| iPhone, PCM → label, Alleluia (4,4 menit) | 46,1 / 45,5 / 51,2 dtk | 19,5 / 20,7 / 21,3 dtk |
| iPhone, PCM → label, Raise a Hallelujah (11 menit) | 113,8 / 124,8 / 147,5 dtk | 53,8 / 55,5 / 57,1 dtk |
| Ukuran bank encoder compiled | 369,0 MB (dua encoder) | 184,5 MB |

- Test: Δakurasi +0.0135 (bootstrap per lagu 95%: −0.0043…+0.0333), Δmacro F1 +0.0185 (−0.0070…+0.0436). Klaimnya **tidak lebih buruk** (non-inferior) dengan estimasi yang condong lebih baik, bukan peningkatan yang terbukti.
- Trade-off kelas: chorus, intro, pre-chorus, inst membaik; recall bridge turun (32/57 → 28/57) walau presisi bridge naik (0.64 → 0.70).
- iPhone (probe terisolasi, CPU): label identik dengan referensi Mac di semua run, error probabilitas ≤ 5e-6. Sekitar **2,2× lebih cepat**.
- Di app (Debug, iPhone): tahap SongFormer 31 dtk (4,4 menit), 37 dtk (5,6 menit), 74 dtk (11 menit). Label sama dengan referensi Mac 19/19 dan 23/24 section.

## Aturan keputusan

Semua keputusan memakai aturan yang ditetapkan **sebelum** hasil keluar:

- Sebuah varian lolos (*candidate*) bila akurasi dan macro F1 section masing-masing turun **paling banyak 0.01** dibanding kontrol, pada section yang sama persis.
- Bootstrap per lagu (95%), jumlah section yang hanya benar di satu sisi, dan daftar section yang berubah dilaporkan sebagai informasi ketidakpastian, bukan sebagai gerbang yang dilonggarkan.
- Split test hanya dipakai sekali per keputusan, setelah `selection.json` mengunci checkpoint, aturan, dan daftar lagu. Tidak ada tuning dari angka test.
- Batas section dan label manual tidak pernah diubah.

## Kronologi percobaan

### 1. Model worship ke app (Debug) dan pooling per section Apple
Notebook: `notebooks/manual_context_audit.ipynb`, catatan "Worship integration — Debug experiment".
Model worship palette6 (MusicFM + MuQ) dijalankan di app Debug, dan label per section dihitung dengan merata-rata probabilitas per section Apple (sesuai cara training), bukan voting mayoritas frame. Masalah yang ditemui: assertion GPU (MPS) yang membuat app crash → semua encoder dikunci ke CPU; jalur tensor bridge dioptimalkan; laporan waktu per tahap ditambahkan.

### 2. Bridge vs chorus: audit dan kalibrasi
Notebook: `train_bridge_audit.ipynb`, `bridge_calibration_experiment.ipynb`.
Audit train (251 lagu, 4.013 section): dari 531 bridge manual, 262 diprediksi chorus. Adapter biner kecil (ridge 0.1, dipilih dengan CV per grup lagu di train) di validasi: 0.7605 / 0.7238 → 0.7650 / 0.7285, bridge F1 0.378 → 0.409. Perbaikan kecil; **tidak dipakai**, diganti konteks section.

### 3. Model konteks section (akurasi dulu)
Notebook: `section_context_accuracy.ipynb`, `section_context_test.ipynb`.
Model residual kecil (304.776 parameter): rata-rata audio 4096 per section + fitur waktu + satu lapis attention antar section → koreksi 8 kelas di atas log-probabilitas SongFormer. Output mulai dari nol (awal = model lama persis). Epoch 5 terpilih: validasi 0.7938 / 0.7755, bridge F1 0.571 (33 section diperbaiki, 18 jadi salah). Test 34 lagu: 0.8000 / 0.6997 vs worship-v3 0.7750 / 0.6806; bridge benar 22/57 → 32/57. **Dipakai** (versi akhirnya dilatih ulang untuk MuQ-only).

### 4. Konteks dengan input palette6 dan ekspor Core ML
Notebook: `section_context_palette6_validation.ipynb`, `section_context_coreml_fp32.ipynb`, `section_context_app_install.ipynb`.
Dengan fitur dari encoder palette6: 450/451 label sama, 0.7916 / 0.7672. Ekspor Core ML FP32 (N section ≤ 512): 100% section sama, error logit maks 2.86e-6. Dipasang ke app Debug.

### 5. Percobaan batas section dan lirik (tidak dipakai)
Notebook: `boundary_shift_review.ipynb`, `boundary_validation_review.ipynb`, `boundary_refinement_debug.ipynb`, `lyric_structure_review.ipynb`, `lyric_label_experiment.ipynb`, `repetition_context_experiment.ipynb`.
- Geser batas lokal ke downbeat (Who You Say I Am, 281.3 → 283.4 dtk) dicoba sebagai opsi Debug; terdengar benar di satu lagu, tapi tidak ada data batas manual untuk menilainya secara umum. Kode: `app-debug/.../SongFormerBoundaryRefiner.swift`.
- Audit lirik (I Surrender) dan percobaan label dari pengulangan lirik/audio: hanya menambah satu section benar di validasi; **tidak dipakai**.

### 6. Kecepatan dan ukuran encoder (dua encoder)
Notebook: `encoder_acceleration_experiment.ipynb`, `encoder_glu_quality.ipynb`, `encoder_glu_repair.ipynb`, `frontend_routes_iphone.ipynb`, `encoder_pipeline_quality.ipynb`, `encoder_pipeline_iphone.ipynb`, `encoder_weight_packaging.ipynb`, `encoder_sharing_audit.ipynb`, `encoder_canonical_bank.ipynb`, `canonical_pipeline_*.ipynb`, `canonical_owned_diagnostic_iphone.ipynb`.
- Graf encoder diperbaiki (pemisahan GLU) setelah assertion GPU; label validasi tetap identik.
- **Bank encoder kanonik multi-fungsi**: satu file per encoder dengan fungsi `short_30`, `long_120`, `partial` yang berbagi bobot palette6. Ukuran compiled 1.009,3 MiB → 351,9 MiB (−65,1%), label validasi identik.
- CPU+GPU gagal paritas di lagu panjang (fitur jendela 120 dtk bergeser) → **hanya CPU**.
- Error alokasi mmap pada MuQ partial → model dimuat satu per satu (serial) dan output disalin ke buffer milik sendiri (owned Float32).

### 7. Ablasi encoder: apakah satu encoder cukup?
Notebook: `encoder_stream_ablation.ipynb`. Skrip: `scripts/encoder_stream_ablation.py`.
Proyeksi input head (4096→2048) dipotong kolomnya per stream lalu dilatih ulang singkat.
- Percobaan 1 (jendela training 30 dtk): **tidak ada yang lolos**. MusicFM saja 0.6918 / 0.6489; MuQ saja 0.7539 / 0.6973; 30-dtk saja 0.7384 / 0.6939 (kontrol 0.7605 / 0.7238).
- Percobaan 2, dideklarasikan sebelumnya (jendela 120 dtk, LR 1e-5): kontrol all4 0.7783 / 0.7438, **MuQ saja 0.7783 / 0.7428** → lolos.
- Catatan teknis: MPS kehabisan memori (20 GB) dengan potongan 420 dtk → training memakai jendela; LR 1e-4 tidak stabil → 1e-5 dengan akumulasi diskalakan.

### 8. MuQ-only sampai ke iPhone
Notebook: `variant_context.ipynb`, `variant_pipeline_test.ipynb`, `muq_coreml_export.ipynb`, `muq_encoder_quality.ipynb`, `muq_pipeline_iphone.ipynb`.
1. Konteks dilatih ulang per head: MuQ + konteks 0.8027 / 0.7674 vs all4 + konteks 0.7849 / 0.7687 dan produksi 0.7938 / 0.7755 → lolos kedua gerbang.
2. Test terkunci (tabel di atas) → lolos.
3. Ekspor Core ML: head palette6 16,8 MB (FP32 87,6 MB), konteks 0,7 MB. FP32: label identik (error 9.5e-7). Palette6: 0.8049 / 0.7692, 450/451 sama.
4. Fitur dari encoder MuQ Core ML (bukan cache PyTorch): 0.8004 / 0.7664, 449/451 sama → lolos.
5. Probe iPhone terisolasi (app "MuQ Only", 5 model): CPU lolos 7/7 run, ≈2,2× lebih cepat; CPU+GPU gagal paritas di lagu panjang.
6. Integrasi ke app Debug, lalu Release (asset pack `sections`).

### 9. Audit konsistensi batas MusicUnderstanding
Kode: `app-debug/.../OnDeviceSongAnalysisProvider.swift` (`auditBoundaries`), `app-debug/.../MusicUnderstandingAnalyzer.swift`. Skrip: `scripts/compare_boundary_audit.py`.
Di iPhone, MusicUnderstanding dipanggil 4 cara (cara app, cara exporter Mac dua kali, timing default). Hasil: **deterministik** (pemanggilan berulang identik) dan opsi timing tidak berpengaruh. Alleluia identik dengan export Mac (F 1.00). Raise a Hallelujah: 26 section di iPhone vs 28 di Mac (F@0,5 dtk 0.83, F@3 dtk 0.94); bedanya berasal dari analisis iOS vs macOS, bukan cara pemanggilan. Keputusan: batas iPhone dipakai apa adanya.

## Yang tidak dipakai dan alasannya

| Percobaan | Alasan |
|---|---|
| Kalibrasi bridge/chorus | Perbaikan kecil; digantikan konteks section |
| Geser batas ke downbeat | Belum ada batas manual untuk menilai; tetap opsi Debug |
| Label dari lirik / pengulangan | Hanya +1 section di validasi |
| CPU+GPU / Neural Engine | Gagal paritas atau assertion fatal; CPU sudah cukup cepat |
| MusicFM saja, jendela 30 dtk saja | Gagal aturan 0.01 |

## Yang masih terbuka

- **Bridge vs chorus** tetap kesalahan terbesar (recall bridge 28/57 di test).
- **Batas section**: butuh 20–30 lagu dengan batas hasil koreksi manual untuk menilai perbaikan batas.
- **Lirik** (Whisper) kini memakan sekitar 60% waktu analisis karena encoder Core ML Whisper (`ggml-small-encoder.mlmodelc`) tidak ada di app.

## Isi folder

| Folder | Isi |
|---|---|
| `notebooks/` | 33 notebook percobaan (tanpa output), dijalankan dari root SongMapApp |
| `scripts/` | Skrip Python per percobaan (ekspor, validasi, training, persiapan app probe iPhone) |
| `scripts/tests/` | Tes Python dan pemeriksaan Swift native untuk skrip di atas |
| `scripts/encoder_pipeline_iphone/`, `scripts/encoder_acceleration_iphone/` | Kode Swift app probe iPhone (BenchmarkRunner dan lainnya) |
| `app-debug/` | Snapshot kode app SongMap versi Debug: konteks 4096, boundary refiner, diagnostik dan tampilannya, audit batas, pipeline MuQ versi Debug, analyzer dengan opsi `--songformer-*` |
| `experiments/structure/` | Contoh input review (laporan analisis Who You Say I Am, sidecar probabilitas I Surrender) |
| `WORSHIP_INTEGRATION.md` | Catatan lengkap setiap tahap: perintah, gerbang, hasil, dan keputusan |

### Notebook dan skrip utama per tahap

| Tahap | Notebook | Skrip |
|---|---|---|
| Ablasi encoder | `encoder_stream_ablation.ipynb` | `encoder_stream_ablation.py` |
| Konteks per varian | `variant_context.ipynb` | `train_variant_context.py` |
| Test terkunci | `variant_pipeline_test.ipynb` | `evaluate_variant_test.py` |
| Ekspor Core ML MuQ | `muq_coreml_export.ipynb` | `export_muq_pipeline.py`, `context_coreml.py` |
| Fitur encoder Core ML | `muq_encoder_quality.ipynb` | `muq_encoder_quality.py` |
| Probe iPhone | `muq_pipeline_iphone.ipynb` | `prepare_muq_pipeline_iphone.py` |
| Pasang ke app | — | `install_muq_pipeline.py` |
| Audit batas | — | `compare_boundary_audit.py` |
| Konteks section (4096) | `section_context_*.ipynb` | `train_section_context.py`, `evaluate_frozen_context.py`, `evaluate_palette6_context.py` |
| Bank encoder kanonik | `encoder_canonical_bank.ipynb`, `canonical_pipeline_*.ipynb` | `canonical_encoder_weights.py`, `canonical_encoder_quality.py`, `prepare_canonical_pipeline_iphone.py` |

## Menjalankan ulang

File di sini adalah arsip sesuai keadaan saat dijalankan, di branch `worship-hybrid-integration` SongMapApp (commit `bfc0fa9`). Path di dalamnya relatif ke **root SongMapApp**, dan banyak skrip memanggil repo ini sebagai folder saudara (`../Section-Annotations`, environment `.venv-songformer311`). Untuk menjalankan ulang:

1. Checkout SongMapApp di commit `bfc0fa9` (semua file ini ada di sana di lokasi aslinya), atau salin `scripts/` dan `notebooks/*.ipynb` ke root SongMapApp.
2. Siapkan environment seperti di [`../SONGFORMER_SETUP.md`](../SONGFORMER_SETUP.md) dan [`../finetuning/README.md`](../finetuning/README.md). Tes konversi Core ML membutuhkan coremltools 8.x (torch 2.7, x-transformers 2.4.14).
3. Jalankan notebook dari root SongMapApp; hasil ditulis ke `outputs.local/` (tidak di-commit).

Data audio, cache fitur, dan checkpoint tidak disertakan (sesuai aturan repo ini). Dua file transkripsi lirik I Surrender (`i-surrender.json`, `i-surrender-review.json`) sengaja tidak diarsipkan karena berisi lirik berhak cipta; versi lengkapnya ada di commit `bfc0fa9` SongMapApp. Di `scripts/tests/test_lyric_structure.py`, dua potongan lirik pada data tes diganti teks netral; maksud tesnya tidak berubah.
