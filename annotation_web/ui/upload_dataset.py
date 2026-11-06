#!/usr/bin/env python3
"""Batch-upload local SongFormer JSON and matching MP3 to Annotation Studio."""
import argparse
import json
from pathlib import Path
import ssl
import sys
import time
from urllib import error, parse, request

BASE = Path(__file__).resolve().parent
APP_URL = 'https://songstruct-annotation-studio.slowcake.chatgpt.site'
LABELED_PATH = BASE / 'outputs.local' / 'labeled'
CONFIG_PATH = BASE / 'songstruct-upload.local.json'
LABELS = {'intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'inst', 'outro', 'silence'}


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an access code to a redirected host or login page.
        return None


class Client:
    def __init__(self, origin, token):
        url = parse.urlsplit(origin.strip())
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
            raise ValueError('APP_URL harus origin HTTPS, tanpa path, query, atau kode akses.')
        self.origin = parse.urlunsplit((url.scheme, url.netloc, '', '', ''))
        self.token = token
        self.opener = request.build_opener(NoRedirect(), request.HTTPSHandler(context=ssl.create_default_context()))

    def call(self, path, data=None, audio=None):
        headers = {'Authorization': 'Bearer ' + self.token, 'X-Requested-With': 'SongStruct',
                   'Origin': self.origin, 'Accept': 'application/json', 'User-Agent': 'SongStructUploader/1.0'}
        method = 'PUT' if audio else 'POST' if data is not None else 'GET'
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8') if data is not None else None
        if encoded is not None:
            headers.update({'Content-Type': 'application/json', 'Content-Length': str(len(encoded))})
        if audio:
            headers.update({'Content-Type': 'audio/mpeg', 'Content-Length': str(audio.stat().st_size)})
        for attempt in range(3):
            handle = None
            try:
                if audio:
                    handle = audio.open('rb')
                req = request.Request(self.origin + '/api' + path, data=handle if audio else encoded,
                                      headers=headers, method=method)
                with self.opener.open(req, timeout=180) as response:
                    if 'application/json' not in response.headers.get('Content-Type', ''):
                        raise ValueError('Server meminta login browser. Unduh konfigurasi uploader dari Tim & hasil di versi web terbaru.')
                    return json.loads(response.read(2_000_001))
            except error.HTTPError as exc:
                if exc.code in (429, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                detail = 'Server menolak permintaan.'
                try:
                    payload = json.loads(exc.read(20000))
                    detail = str(payload.get('error', detail))
                except (ValueError, AttributeError):
                    if exc.code in (301, 302, 303, 307, 308, 401, 403):
                        detail = 'Kode uploader tidak valid atau situs masih meminta login. Gunakan konfigurasi uploader dari web terbaru.'
                raise ValueError(f'HTTP {exc.code}: {detail}') from None
            except (error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise ValueError('Koneksi upload gagal. Jalankan ulang untuk melanjutkan file yang belum terunggah.') from None
                time.sleep(2 ** attempt)
            finally:
                if handle:
                    handle.close()


def plan_uploads(dataset, labeled):
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError('DATASET_PATH di check_dataset.py tidak ditemukan.')
    items, errors, seen = [], [], set()
    for file in sorted(labeled.rglob('*.json')):
        try:
            if file.stat().st_size > 2_000_000:
                raise ValueError('JSON melebihi 2 MB.')
            doc = json.loads(file.read_text(encoding='utf-8'))
            if doc.get('label_source') != 'ASLP-lab/SongFormer' or not doc.get('sections'):
                raise ValueError('Belum ada label SongFormer. Gunakan outputs.local/labeled, bukan outputs.local/apple.')
            if any(s.get('label_initial', s.get('label')) not in LABELS for s in doc['sections']):
                raise ValueError('Label awal tidak valid.')
            track_id = doc.get('track_id')
            if not isinstance(track_id, str) or not track_id or track_id in seen:
                raise ValueError('Track ID kosong atau duplikat.')
            seen.add(track_id)
            # Resolve by relative ID first, so moving the dataset does not break upload.
            audio = (dataset / track_id).resolve()
            if not audio.is_file():
                audio = Path(doc.get('audio_path', '')).expanduser().resolve()
            try:
                audio.relative_to(dataset)
            except ValueError:
                raise ValueError('Audio berada di luar DATASET_PATH.') from None
            if not audio.is_file() or audio.suffix.lower() != '.mp3':
                raise ValueError('MP3 pasangan tidak ditemukan.')
            if not 0 < audio.stat().st_size <= 90 * 1024 * 1024:
                raise ValueError('MP3 harus berukuran 1 byte sampai 90 MB.')
            items.append((file, doc, audio))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append((str(file), str(exc)))
    return items, errors


def matches(local, remote):
    return (abs(float(local['duration']) - float(remote['duration'])) < 0.001
            and len(local['sections']) == len(remote['sections'])
            and all(a['start'] == b['start'] and a['end'] == b['end']
                    and a.get('label_initial', a.get('label')) == b['label_initial']
                    for a, b in zip(local['sections'], remote['sections'])))


def main():
    from check_dataset import DATASET_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Cek pasangan audio/JSON tanpa jaringan atau upload.')
    parser.add_argument('--limit', type=int, default=0, help='0 = semua JSON berlabel yang tersedia.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit harus >= 0')
    items, problems = plan_uploads(DATASET_PATH, LABELED_PATH)
    if args.limit:
        items = items[:args.limit]
    print(f'JSON berlabel siap: {len(items)} | tidak valid: {len(problems)}', flush=True)
    print(f'Audio yang direncanakan: {sum(a.stat().st_size for _, _, a in items) / 1024**3:.2f} GiB', flush=True)
    for file, message in problems:
        print(f'LEWATI {Path(file).name}: {message}', flush=True)
    if not items:
        print('Belum ada pasangan berlabel. Jalankan exporter Apple dan label_apple_sections.py terlebih dahulu.')
        return 1
    if args.dry_run:
        for _, doc, _ in items:
            print(f"SIAP: {doc['track_id']} ({len(doc['sections'])} section berlabel)")
        return 1 if problems else 0
    config = None
    for candidate in (CONFIG_PATH, Path.home() / 'Downloads' / 'songstruct-upload.local.json'):
        if candidate.is_file():
            config = json.loads(candidate.read_text(encoding='utf-8'))
            break
    if config is None:
        print('Di web: Tim & hasil → Hubungkan uploader. Simpan file konfigurasi di Downloads atau folder repo.')
        return 1
    configured = parse.urlsplit(str(config.get('app_url', '')))
    expected = parse.urlsplit(APP_URL)
    if (configured.scheme, configured.netloc) != (expected.scheme, expected.netloc):
        raise ValueError('URL konfigurasi berbeda dari APP_URL di skrip. Periksa dan sesuaikan APP_URL terlebih dahulu.')
    token = str(config.get('upload_code', ''))
    if len(token) < 20:
        raise ValueError('Konfigurasi tidak memuat kode uploader yang valid.')
    client = Client(APP_URL, token)
    identity = client.call('/me')
    if identity.get('role') not in ('uploader', 'admin'):
        raise ValueError('Gunakan konfigurasi uploader milik pengelola, bukan tautan anotator.')
    print('Terhubung. Upload berurutan dimulai; file cloud dan koreksi yang sudah ada tidak ditimpa.', flush=True)
    uploaded, skipped, failed = 0, 0, 0
    for index, (_, doc, audio) in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {doc['track_id']}", flush=True)
        try:
            result = client.call('/import', data={'document': doc, 'title': audio.stem})
            remote = client.call('/tracks/' + result['id'])
            if not matches(doc, remote):
                raise ValueError('Boundary/prediksi cloud berbeda dari JSON lokal. Tidak diubah otomatis.')
            if remote.get('audio_ready'):
                skipped += 1
                print('  Sudah lengkap di cloud; dilewati.', flush=True)
                continue
            result = client.call('/tracks/' + result['id'] + '/audio', audio=audio)
            if result.get('skipped'):
                skipped += 1
            else:
                uploaded += 1
            print('  Audio + label awal siap untuk anotasi.', flush=True)
        except (ValueError, KeyError, OSError) as exc:
            failed += 1
            print(f'  GAGAL: {exc}', flush=True)
    print(f'Selesai: {uploaded} audio baru, {skipped} sudah ada, {failed} gagal.', flush=True)
    print('Hanya file yang sudah memiliki JSON SongFormer yang diunggah; ini bukan proses inferensi baru.')
    return 1 if failed or problems else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
