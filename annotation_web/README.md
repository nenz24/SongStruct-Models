# SongStruct Annotation Studio

Aplikasi cloud untuk koreksi label section lagu worship bersama. Audio di Cloudflare R2; anotasi, penugasan, dan histori di D1. Boundary Apple terkunci; prediksi SongFormer dipertahankan terpisah dari koreksi manusia. Tidak menjalankan model AI di server.

## Mode GitHub + Cloudflare (tanpa akun ChatGPT untuk anotator)

Kode dapat disimpan di GitHub; GitHub Pages sendiri tidak menyediakan backend database dan penyimpanan privat yang dipakai aplikasi ini. Deploy Worker dari akun Cloudflare Anda. Teman membuka URL Worker di browser dan memasukkan kode masing-masing. Mereka tidak perlu clone repo atau install Python.

Butuh Node.js 22+ dan akun Cloudflare. Jalankan dari folder aplikasi ini, bukan root pipeline Python:

```bash
npm install --save-dev wrangler
npx wrangler login
cp wrangler.example.jsonc wrangler.jsonc
npx wrangler d1 create songstruct-annotations
```

Salin `database_id` yang dikembalikan perintah terakhir ke `wrangler.jsonc`. Nama resource dapat diganti jika sudah digunakan di akun Anda; samakan dengan config.

Aktifkan R2 pada akun Cloudflare jika belum aktif, kemudian:

```bash
npx wrangler r2 bucket create songstruct-audio
npx wrangler d1 migrations apply DB --remote
```

Buat kode admin acak, simpan di password manager, lalu masukkan saat perintah secret meminta input. Jangan commit kode admin:

```bash
openssl rand -hex 32
npx wrangler secret put ADMIN_ACCESS_CODE
npm run build
npx wrangler deploy
```

`wrangler deploy` menampilkan URL aplikasi yang bisa dibagikan. Halaman login dapat diakses publik, tetapi daftar lagu, audio, edit, dan ekspor memerlukan kode yang valid. `ADMIN_ACCESS_CODE` dipasang sebagai secret di akun Anda. Kode anotator disimpan hanya sebagai hash di database dan dikirim lewat Authorization header; tidak ditaruh di URL. Kode akses tersimpan di sessionStorage tab browser dan dibersihkan saat keluar/tab ditutup. Jangan berbagi kode admin dengan anotator.

### Siapkan 3 anotator

1. Buka URL Worker. Masuk dengan kode admin.
2. Di **Tim & hasil**, buat kode untuk Anotator 1, 2, dan 3 menggunakan nama masing-masing.
3. Salin kode saat ditampilkan; server tidak dapat menampilkan ulang kode aslinya.
4. Kirim URL dan kode masing-masing secara pribadi. Pengiriman ini Anda lakukan sendiri.
5. Teman mengambil lagu yang belum ditugaskan, atau pengelola menugaskan melalui dropdown pada lagu.
6. Kode dapat dicabut melalui **Cabut kode**. Koreksi lama tetap tersimpan untuk audit.

## Unggah dataset

Di **Unggah lagu**, pilih file atau folder JSON dari `outputs.local/labeled/`, dan file/folder audio dari `DATASET_PATH`. Browser hanya membaca file yang Anda pilih. Tidak mengunggah checkpoint atau folder lain.

Upload pertama coba satu lagu. MP3 dipasangkan melalui `track_id`/nama file; file tidak boleh diubah panjangnya sejak analisis. Jika ada nama sama, unggah pasangan yang ambigu secara terpisah. JSON lokal sampai 128 MiB dibaca dan diringkas otomatis: hasil mentah Apple/raw_result tidak dikirim. Payload anotasi ke server tetap maksimal 2 MB; audio maksimal 90 MB per lagu. Audio yang sudah tersedia dan anotasi yang sudah ada tidak ditimpa otomatis. Unggah gagal dapat diulang; lagu tanpa audio ditandai. Path absolut lokal dan hasil Apple mentah tidak dibagikan ke anggota.

Untuk mode kode, audio diunduh ke memori browser saat lagu dibuka (hingga 90 MB), kemudian dapat di-seek per section. Pada mode Sites, pemutaran menggunakan HTTP Range. Audio tidak dimasukkan ke repository GitHub.

## Cara anotasi

- Ambil tugas. Dengarkan section melalui tombol play pada baris atau timeline.
- Ubah **Label final** jika prediksi salah; klik **✓** jika prediksi benar. Label kosong berarti belum diperiksa.
- Catatan opsional untuk kasus ambigu. Boundary tidak dapat digeser, split, atau merge.
- Klik **Simpan draf** secara eksplisit. Indikator menunjukkan perubahan yang belum disimpan; tidak ada janji autosave.
- Setelah semua section diperiksa, **Kirim untuk review**. Anotator tidak bisa mengubah hasil terkirim sebelum pengelola membukanya kembali.
- Pengelola **Setujui anotasi** atau **Kembalikan**. Hanya approved masuk ekspor training.
- Dua editor dengan versi berbeda mendapat konflik 409, bukan overwrite diam-diam. Muat ulang setelah menyimpan salinan catatan Anda jika terjadi konflik.

## Ekspor

Pengelola dapat mengunduh `songstruct-approved.json` untuk data disetujui, atau `songstruct-all.json` sebagai cadangan semua status. Bentuk: `{schema_version, exported_at, count, tracks:[...]}`. Setiap section menyimpan `start`, `end`, `label_initial`, `label_manual`, `label`, `review_status`, dan catatan. Data approved masih perlu pemisahan train/validation/test menurut identitas lagu sebelum training.

Jalankan `python import_cloud_annotations.py /path/songstruct-approved.json` dari root repo Python untuk mencocokkan boundary dengan JSON lokal dan menyimpan file per lagu ke `outputs.local/reviewed/`. Skrip itu berada satu tingkat di atas folder aplikasi dalam SongStruct-Models.

## Biaya dan kapasitas

Mulai dari paket gratis jika kapasitas akun dan penggunaan memungkinkan; bukan jaminan bebas biaya. R2 perlu diaktifkan pada akun hosting. Cek ukuran seluruh audio dan billing akun sebelum upload besar. Harga Cloudflare langsung berbeda dari hosting Sites: [R2](https://developers.cloudflare.com/r2/pricing/) dan [Workers](https://developers.cloudflare.com/workers/platform/pricing/). Jangan aktifkan paket berbayar tanpa memahami billing.

## Preview Sites

Versi privat yang dibuat di percakapan menggunakan identitas ChatGPT bawaan Sites dan allowlist anggota. Mode itu mempertahankan autentikasi platform. Set `OWNER_EMAIL` ke email akun pemilik di runtime, jangan aktifkan `INVITE_AUTH`. Versi GitHub/Cloudflare di atas memakai `INVITE_AUTH=1`, tanpa ketergantungan ChatGPT.

## Pemeriksaan dan batas validasi

```bash
node tests/api.test.mjs
node --check ui/app.js
npm run build
```

Pengujian mencakup batas akses, penugasan, konflik versi, review/approval, boundary tetap, label awal tidak hilang, ekspor, media ranges, kode per orang, dan pencabutan akses menggunakan adapter SQLite/R2 lokal. Pengujian browser end-to-end dan deploy pada akun Cloudflare Anda belum dilakukan. Schema migrations di `drizzle/` harus diaplikasikan sebelum menjalankan app.

Sumber dan UI tidak menggunakan library eksternal/CDN. Jangan commit `wrangler.jsonc` milik akun pribadi, `.dev.vars`, `.env`, audio, kode akses, atau hasil anotasi ke GitHub.


## Tim Juan, Jordan, dan Otniel; upload batch dari Mac

Deployment Sites kini mendukung tautan undangan langsung, tanpa login untuk anotator. Pengelola tetap masuk dengan akun pemilik. Di runtime gunakan INVITE_AUTH=1, HOSTED_SITES=1, OWNER_EMAIL yang benar, dan TEAM_LINK_KEY sebagai secret acak. Jangan mengaktifkan HOSTED_SITES pada deployment Cloudflare mandiri karena header identitas pada host mandiri tidak diverifikasi oleh Sites.

TEAM_LINK_KEY menghasilkan tautan stabil berbeda untuk Juan, Jordan, dan Otniel. Rahasiakan tautan; siapa pun yang memegangnya dapat bertindak sebagai anotator tersebut. Tautan disimpan hanya di sessionStorage tab, fragment dihapus dari address bar setelah diproses. Cabut melalui Tim & hasil jika diperlukan; revokasi bertahan setelah refresh. Mengganti TEAM_LINK_KEY mencabut kompatibilitas kode lama dan memerlukan pemutakhiran hash/konfigurasi, jadi jangan merotasinya sembarangan.

Lagu baru otomatis ditugaskan ke anggota dengan jumlah tugas paling sedikit, urutan tie-break Juan → Jordan → Otniel. Hanya lagu pending yang belum punya assignee yang dialokasikan ulang; hasil submitted/approved atau pekerjaan orang lain tidak dipindahkan. Untuk 323 lagu baru, target jumlah adalah 108/108/107, bukan klaim seluruh file sudah ada di cloud.

Pengelola: buka Tim & hasil → Hubungkan uploader dari Mac. Browser mengunduh songstruct-upload.local.json. Simpan di Downloads atau root repo Python, lalu jalankan dari root repo:

```bash
python3 upload_dataset.py --dry-run
python3 upload_dataset.py
```

Skrip memakai DATASET_PATH dari check_dataset.py dan JSON dari outputs.local/labeled. Hanya pasangan yang sudah punya label SongFormer yang dikirim. File MP3 tidak dibaca oleh situs dari path lokal secara otomatis tanpa menjalankan skrip atau memilih folder browser. Kode uploader dibatasi untuk impor JSON, melihat metadata pasangan, dan mengirim audio; tidak dapat mengubah anotasi, membaca audio, mengekspor, atau mengundang anggota. Jangan bagikan file konfigurasi uploader.

Run ulang melewati audio cloud yang sudah lengkap; boundary dan prediksi awal diverifikasi sebelum upload audio. Jika cloud berbeda dari lokal, file dilewati dengan error. Koreksi teman tidak ditimpa. Upload berlangsung berurutan dan dapat dilanjutkan setelah koneksi terputus. Daftar upload tidak sama dengan jumlah MP3 jika proses SongFormer belum selesai untuk seluruh lagu.


## JSON sumber besar

Batas 2 MB berlaku untuk payload anotasi jaringan, bukan ukuran JSON mentah Apple di disk. Uploader Python dan browser mengambil hanya metadata model, track ID, durasi, serta sections (boundary, label awal/manual, skor). File asli dan raw_result di Mac tetap utuh. Setelah pembaruan uploader, ulangi dry-run dan upload untuk file yang sebelumnya ditolak. Tidak perlu menjalankan ulang model.
