# -*- coding: utf-8 -*-
"""Скачивание данных и моделей для сервиса: Google Drive (архивы) + HuggingFace (энкодер).

Реализация задачи «развернуть на другом сервере»: код и конфиги едут в git, а данные
(`data/`, `filtered/`), модели (`models/`) и дамп БД (pgvector) лежат в облаке и
подтягиваются этим скриптом. Большая модель SigLIP2 на облако не выгружается — она
публичная, качается напрямую из HuggingFace.

Только стандартная библиотека: ни gdown, ни requests, ни unzip в образе не нужны.

    python3 deploy/fetch_data.py                     # всё: данные, модели, дамп, энкодер
    python3 deploy/fetch_data.py --only data,dump    # только часть
    python3 deploy/fetch_data.py --dest /tmp/test    # распаковать в другой корень (проверка)
    python3 deploy/fetch_data.py --check             # ничего не качать, только проверить ссылки
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- архивы в Google Drive (id — из ссылки вида drive.google.com/file/d/<ID>/view) ---
# expect — ожидаемый размер архива в байтах (раннее предупреждение об обрыве загрузки)
ARCHIVES = [
    {
        'name': 'data',
        'comment': 'каталог, эталонные фото, OCR-поля (data/)',
        'gid': '19nvLvudN4DH9whIVgrXJSQQmSAaYpydH',
        'expect': 259_466_482,
    },
    {
        'name': 'filtered',
        'comment': 'фото каталога <slug>/NN.webp + catalog.csv (filtered/)',
        'gid': '1IoQAlH4jdIboyxQZKChe8dClbUEhkoAL',
        'expect': 136_886_641,
    },
    {
        'name': 'models',
        'comment': 'YOLO-детекторы: label_det_best.pt, yolo11n.pt (models/)',
        'gid': '1th7GbXvoUeI-PmfcZn9Q0ke4JlPpPhYa',
        'expect': 11_850_390,
    },
    {
        'name': 'dump',
        'comment': 'дамп pgvector (векторы индекса) — в deploy/archives/',
        'gid': '1FRC2oUEeV8TYstCJ6x7OqSEPKA31z7_L',
        'expect': 31_591_750,
    },
]

# --- энкодер SigLIP2 (публичный репозиторий HuggingFace) ---
ENCODER_REPO = 'google/siglip2-base-patch16-256'
ENCODER_DIR = REPO / 'models' / 'siglip2-base-patch16-256'
ENCODER_FILES = [
    # имя файла -> ожидаемый размер (чтобы не качать заново и ловить обрыв)
    ('config.json', 276),
    ('preprocessor_config.json', 394),
    ('special_tokens_map.json', 636),
    ('tokenizer_config.json', 47_164),
    ('tokenizer.json', 34_363_039),
    ('model.safetensors', 1_500_985_224),
]
UA = 'Mozilla/5.0 (compatible; lct-vino-fetch/1.0)'


def log(msg: str) -> None:
    print(msg, flush=True)


def human(n) -> str:
    if not isinstance(n, (int, float)):
        return '—'
    for unit in ('Б', 'КБ', 'МБ', 'ГБ'):
        if n < 1024 or unit == 'ГБ':
            return f'{n:.0f} {unit}' if unit == 'Б' else f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} ГБ'


def _open(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    return urllib.request.urlopen(req, timeout=timeout)


def _confirm_token(html: str) -> str:
    """Достаёт uuid со страницы подтверждения Google Drive (для больших файлов)."""
    for marker in ('name="uuid" value="', "name='uuid' value='"):
        i = html.find(marker)
        if i >= 0:
            rest = html[i + len(marker):]
            return rest.split(rest[0], 1)[0]
    i = html.find('confirm=')
    if i >= 0:
        rest = html[i + len('confirm='):]
        return ''.join(c for c in rest if c.isalnum() or c in '-_')[:64]
    return ''


def _stream_to(resp, out_path: Path, what: str) -> int:
    total = int(resp.headers.get('Content-Length') or 0)
    done, last = 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.part')
    with open(tmp, 'wb') as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done - last >= (20 << 20) or (total and done == total):
                last = done
                pct = f' ({done * 100 // total}%)' if total else ''
                log(f'    {what}: {human(done)} / {human(total) if total else "—"}{pct}')
    tmp.replace(out_path)
    return done


def drive_download(file_id: str, out_path: Path) -> int:
    """Качает файл с Google Drive по id, обходя страницу подтверждения для больших файлов."""
    base = 'https://drive.usercontent.google.com/download'
    url = f'{base}?id={file_id}&export=download'
    with _open(url) as resp:
        ctype = (resp.headers.get('Content-Type') or '').lower()
        if 'text/html' not in ctype:
            return _stream_to(resp, out_path, out_path.name)
        html = resp.read().decode('utf-8', 'replace')
    if 'request access' in html.lower() or 'Запросить доступ' in html:
        raise RuntimeError('файл закрыт: включите доступ «всем, у кого есть ссылка» '
                           '(Anyone with the link)')
    token = _confirm_token(html)
    url2 = f'{base}?id={file_id}&export=download&confirm=t'
    if token:
        url2 += f'&uuid={token}'
    with _open(url2) as resp:
        ctype = (resp.headers.get('Content-Type') or '').lower()
        if 'text/html' in ctype:
            raise RuntimeError('Google Drive вернул страницу вместо файла (проверьте доступ и id)')
        return _stream_to(resp, out_path, out_path.name)


def download_archive(arc: dict, dest_zips: Path, force: bool) -> Path:
    zip_path = dest_zips / f'lct-{arc["name"]}.zip'
    log(f'  [{arc["name"]}] {arc["comment"]}')
    if zip_path.is_file() and not force and zip_path.stat().st_size == arc['expect']:
        log(f'    архив уже скачан: {zip_path.name} ({human(zip_path.stat().st_size)})')
        return zip_path
    size = drive_download(arc['gid'], zip_path)
    log(f'    скачано {zip_path.name}: {human(size)}')
    if arc['expect'] and size != arc['expect']:
        log(f'    ВНИМАНИЕ: ожидалось {human(arc["expect"])} — архив могли перезалить '
            f'или загрузка обрезана')
    with open(zip_path, 'rb') as f:
        if f.read(4) != b'PK\x03\x04':
            raise RuntimeError(f'{zip_path.name}: это не zip-архив (проверьте ссылку)')
    return zip_path


def extract_zip(zip_path: Path, dest_root: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f'{zip_path.name}: битый файл внутри архива: {bad}')
        zf.extractall(dest_root)


def _encoder_url(name: str) -> str:
    return f'https://huggingface.co/{ENCODER_REPO}/resolve/main/{name}?download=true'


def fetch_encoder(dest: Path) -> None:
    """Скачивает файлы SigLIP2 из HuggingFace в <dest>/models/siglip2-base-patch16-256."""
    out_dir = dest / 'models' / 'siglip2-base-patch16-256'
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f'  [encoder] {ENCODER_REPO} -> {out_dir}')
    for name, size in ENCODER_FILES:
        target = out_dir / name
        if target.is_file() and target.stat().st_size == size:
            log(f'    {name}: уже есть ({human(size)})')
            continue
        log(f'    {name}: качаю ({human(size)})')
        with _open(_encoder_url(name), timeout=120) as resp:
            got = _stream_to(resp, target, name)
        if got != size:
            raise RuntimeError(f'{name}: скачано {human(got)} вместо {human(size)} — '
                              f'повторите запуск (файл .part не используется)')
    log('    энкодер готов')


def check_only(only: list, with_encoder: bool) -> bool:
    """Проверяет доступность ссылок и размеры, ничего не записывая на диск."""
    ok = True
    log('  [проверка] архивы в Google Drive:')
    for arc in ARCHIVES:
        if only and arc['name'] not in only:
            continue
        try:
            size = drive_download_size(arc['gid'])
            mark = '✓' if (not arc['expect'] or size == arc['expect']) else '!'
            log(f'    {mark} {arc["name"]:9s} {human(size)}'
                f'{" (ожидалось " + human(arc["expect"]) + ")" if arc["expect"] and size != arc["expect"] else ""}')
        except Exception as e:                                       # noqa: BLE001
            ok = False
            log(f'    ✗ {arc["name"]:9s} ошибка: {e}')
    if with_encoder:
        log('  [проверка] энкодер в HuggingFace:')
        for name, size in ENCODER_FILES:
            try:
                got = _remote_size(_encoder_url(name))
                mark = '✓' if got == size else '!'
                log(f'    {mark} {name:24s} {human(got)}')
            except Exception as e:                                   # noqa: BLE001
                ok = False
                log(f'    ✗ {name:24s} ошибка: {e}')
    return ok


def drive_download_size(file_id: str) -> int:
    """Content-Length файла на Google Drive (тело ответа не читаем)."""
    base = 'https://drive.usercontent.google.com/download'
    with _open(f'{base}?id={file_id}&export=download') as resp:
        ctype = (resp.headers.get('Content-Type') or '').lower()
        if 'text/html' not in ctype:
            return int(resp.headers.get('Content-Length') or 0)
        html = resp.read().decode('utf-8', 'replace')
    token = _confirm_token(html)
    url2 = f'{base}?id={file_id}&export=download&confirm=t'
    if token:
        url2 += f'&uuid={token}'
    with _open(url2) as resp:
        if 'text/html' in (resp.headers.get('Content-Type') or '').lower():
            raise RuntimeError('страница вместо файла (нет доступа)')
        return int(resp.headers.get('Content-Length') or 0)


def _remote_size(url: str) -> int:
    with _open(url) as resp:
        return int(resp.headers.get('Content-Length') or 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='скачать данные/модели для сервиса')
    ap.add_argument('--only', default='', help='через запятую: data,filtered,models,dump')
    ap.add_argument('--dest', default=None, help='корень распаковки (по умолчанию репозиторий)')
    ap.add_argument('--force', action='store_true', help='перекачать архивы заново')
    ap.add_argument('--keep-zips', action='store_true', help='не удалять скачанные архивы')
    ap.add_argument('--no-encoder', action='store_true', help='не качать SigLIP2 из HuggingFace')
    ap.add_argument('--only-encoder', action='store_true', help='качать только SigLIP2')
    ap.add_argument('--check', action='store_true', help='только проверить ссылки и размеры')
    args = ap.parse_args(argv)

    dest = Path(args.dest).resolve() if args.dest else REPO
    only = [s.strip() for s in args.only.split(',') if s.strip()]
    log(f'[fetch] корень распаковки: {dest}')

    if args.check:
        ok = check_only(only, with_encoder=not args.no_encoder)
        log('[fetch] проверка ' + ('пройдена ✓' if ok else 'не пройдена (см. выше)'))
        return 0 if ok else 1

    if not args.only_encoder:
        fetch_archives(dest, only, args.force, args.keep_zips)
    if not args.no_encoder:
        fetch_encoder(dest)
    log('[fetch] готово ✓')
    return 0


def _run() -> int:
    """Обёртка для запуска из консоли: понятные коды выхода вместо трейсбеков."""
    try:
        return main()
    except urllib.error.URLError as e:                                # сеть/доступ
        log(f'[fetch] ОШИБКА сети: {e}')
        return 2
    except (RuntimeError, zipfile.BadZipFile, OSError) as e:          # данные/диск
        log(f'[fetch] ОШИБКА: {e}')
        return 3


def fetch_archives(dest_root: Path, only: list, force: bool, keep_zips: bool) -> list:
    """Качает архивы и раскладывает их по местам. Возвращает список обработанных имён."""
    dest_zips = dest_root / 'deploy' / 'archives'
    dest_zips.mkdir(parents=True, exist_ok=True)
    done = []
    for arc in ARCHIVES:
        if only and arc['name'] not in only:
            continue
        zip_path = download_archive(arc, dest_zips, force)
        extract_zip(zip_path, dest_root)
        log(f'    распаковано в {dest_root}/')
        if arc['name'] == 'dump':                      # внутри архива pgvector.dump в корне
            raw = dest_root / 'pgvector.dump'
            target = dest_zips / 'pgvector.dump'
            if raw.is_file() and raw != target:
                shutil.move(str(raw), str(target))
            log(f'    дамп БД: {target}')
        if not keep_zips:
            zip_path.unlink(missing_ok=True)
        done.append(arc['name'])
    return done


if __name__ == '__main__':
    sys.exit(_run())
