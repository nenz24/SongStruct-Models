# Label awal SongFormer pada section Apple

Boundary Apple tetap dikunci. Koreksi manual cukup mengisi label_manual dan review_status; label_initial dipertahankan. Field label awalnya sama dengan label_initial; saat ekspor training nanti gunakan label_manual jika tersedia. Jangan gunakan anotasi pending sebagai ground truth.

## Setup sekali di terminal VSCode (Mac)

Dari root repositori SongStruct-Models:

```bash
git clone --recursive https://github.com/ASLP-lab/SongFormer.git vendor/SongFormer
python3.11 -m venv .venv-songformer311
source .venv-songformer311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
```

Gunakan Python 3.11 dengan NumPy 1.26.4 dan SciPy 1.17.1. Versi SciPy lama gagal memuat binary PROPACK pada Mac pengguna (error __DATA/__thread_bss); SciPy baru menggunakan implementasi PROPACK C. Triton juga dihapus karena tidak tersedia pada macOS. Ini penyesuaian dependency, bukan perubahan checkpoint.

Jika sudah punya environment Python 3.10, jalankan deactivate dahulu, lalu buat environment baru .venv-songformer311. Environment lama tidak perlu dihapus. Jika python3.11 belum tersedia dan Anda menggunakan Homebrew: brew install python@3.11.

Uji sebelum memuat model:

```bash
python -m pip check
python -c "import numpy, scipy; import scipy.sparse.linalg; import sklearn; from muq import MuQ; print(numpy.__version__, scipy.__version__); print('Import berhasil')"
```

Kombinasi versi ini mengikuti persyaratan SciPy 1.17 (Python >=3.11, NumPy >=1.26.4); inferensi penuh SongFormer di Mac belum terverifikasi. Jika instalasi atau import gagal, kirim log sebelum melanjutkan. Jangan gunakan requirements upstream yang akan memasang kembali SciPy lama.

Checkpoint yang sudah ada di vendor/SongFormer/src/SongFormer/ckpts tetap digunakan. Tidak perlu mengunduhnya lagi.

Download checkpoint resmi:

```bash
(cd vendor/SongFormer/src/SongFormer && python utils/fetch_pretrained.py)
```

MuQ otomatis diunduh dari Hugging Face saat pemuatan pertama. Setup pertama membutuhkan internet dan penyimpanan untuk seluruh bobot. Audio diproses lokal, tidak diunggah.

## Jalankan

```bash
python label_apple_sections.py
```

Input otomatis dari outputs.local/apple; default 5 JSON pertama, output ke outputs.local/labeled dengan subfolder yang sama. Path audio diambil dari JSON hasil Apple. Tidak perlu menyalin audio ke repo. JSON dan audio harus tersedia pada komputer ini.

Default memakai CUDA jika tersedia, selain itu CPU. CPU bisa lambat dan menggunakan banyak RAM. MPS bersifat opsional dan belum diuji:

```bash
python label_apple_sections.py --device mps
```

Jika MPS menemui operasi tidak didukung, coba CPU. Jangan langsung memproses 323 lagu sebelum satu pilot berhasil.

```bash
python label_apple_sections.py --limit 1
python label_apple_sections.py --limit 0
```

## Output

Setiap section mempertahankan start/end Apple dan mendapat:
- label_initial dan label: prediksi awal.
- label_manual: null, diisi anotator.
- label_scores: rata-rata softmax berbobot durasi overlap frame pada section. Ini bukan probabilitas akurasi terkalibrasi.
- coverage_fraction dan needs_review: membantu menemukan prediksi ambigu/timeline tidak tercakup.
- review_status: pending.

Tidak ada merge section sama label, snapping, atau boundary SongFormer yang digunakan. Kelas: intro, verse, chorus, bridge, inst, outro, silence, pre-chorus.

Output yang sudah ada tidak ditimpa, agar koreksi manual aman. Untuk menghitung ulang, pindahkan output label lama dahulu; JSON Apple tidak diubah.

Encoder pendek 30 detik dan panjang 120 detik mengikuti SongMapApp; head memakai blok sampai 420 detik. Ini berbeda dari baseline upstream encoder panjang 420 detik. Ganti ENCODER_WINDOW_SECONDS di skrip ke 420 jika ingin menguji baseline resmi dengan kebutuhan memori lebih tinggi. Window yang digunakan dicatat pada output.

Skrip mengikuti model dan urutan embedding upstream. Sintaks Python dan agregasi dengan data sintetis sudah diuji; inferensi model penuh belum dijalankan di lingkungan pengembang karena audio pengguna dan environment model tidak tersedia. Ini pelabelan pretrained, belum fine-tuning atau kuantisasi 6-bit.
