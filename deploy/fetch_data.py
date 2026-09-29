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
import time
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
        # что должно лежать на диске, чтобы шаг считался выполненным
        'artifacts': ['data/found_in_catalog_corrected.csv', 'data/eval.csv',
                      'data/catalog_ocr_fields.csv', 'data/start_photos'],
    },
    {
        'name': 'filtered',
        'comment': 'фото каталога <slug>/NN.webp + catalog.csv (filtered/)',
        'gid': '1IoQAlH4jdIboyxQZKChe8dClbUEhkoAL',
        'expect': 136_886_641,
        'artifacts': ['filtered/catalog.csv', 'filtered'],
    },
    {
        'name': 'models',
        'comment': 'YOLO-детекторы: label_det_best.pt, yolo11n.pt (models/)',
        'gid': '1th7GbXvoUeI-PmfcZn9Q0ke4JlPpPhYa',
        'expect': 11_850_390,
        'artifacts': ['models/label_det_best.pt', 'models/yolo11n.pt'],
    },
    {
        'name': 'dump',
        'comment': 'дамп pgvector (векторы индекса) — в deploy/archives/',
        'gid': '1FRC2oUEeV8TYstCJ6x7OqSEPKA31z7_L',
        'expect': 31_591_750,
        'artifacts': ['deploy/archives/pgvector.dump'],
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


def _open(url: str, timeout: int = 60, headers: dict | None = None):
    hdrs = {'User-Agent': UA}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
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
    """Дописывает ответ в ``out_path`` (пишем в ``*.part``, потом атомарно заменяем).

    ``resp`` может быть ответом на Range-запрос (206) — тогда файл дописывается, а не
    переписывается; это позволяет докачивать большие файлы (модель SigLIP2 1.4 ГБ)
    после обрыва сети.
    """
    append = getattr(resp, 'status', 200) == 206 and out_path.suffix == '.part' and out_path.exists()
    total = int(resp.headers.get('Content-Length') or 0)
    base = out_path.stat().st_size if append else 0
    done, last = base, base
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'ab' if append else 'wb') as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done - last >= (50 << 20) or (total and done - base == total):
                last = done
                pct = f' ({done * 100 // (base + total)}%)' if total else ''
                log(f'    {what}: {human(done)} / {human(base + total) if total else "—"}{pct}')
    return done


def download_file(url: str, out_path: Path, expect: int = 0, tries: int = 5, what: str = '') -> int:
    """Скачивает файл с докачкой и ретраями. Возвращает размер.

    Сеть до HuggingFace/Google Drive бывает нестабильной: большой файл (1.4 ГБ) обрывается
    на середине. Держим ``*.part`` и на следующей попытке запрашиваем ``Range: bytes=N-``,
    поэтому каждая повторная попытка продолжает загрузку, а не начинает её заново.
    """
    tmp = out_path.with_suffix(out_path.suffix + '.part')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    name = what or out_path.name
    last_err = None
    for attempt in range(1, tries + 1):
        if tmp.exists() and expect and tmp.stat().st_size == expect:
            break
        already = tmp.stat().st_size if tmp.exists() else 0
        headers = {'User-Agent': UA}
        if already:
            headers['Range'] = f'bytes={already}-'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                if already and getattr(resp, 'status', 200) != 206:
                    already = 0                     # сервер не умеет Range — качаем заново
                    tmp.unlink(missing_ok=True)
                    tmp = out_path.with_suffix(out_path.suffix + '.part')
                got = _stream_to(resp, tmp, name)
            if expect and got != expect:
                raise RuntimeError(f'скачано {human(got)} вместо {human(expect)}')
            tmp.replace(out_path)
            return got
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            last_err = e
            done = tmp.stat().st_size if tmp.exists() else 0
            log(f'    {name}: попытка {attempt}/{tries} не удалась ({e}); скачано {human(done)} — '
                f'продолжаю с этого места')
            time.sleep(min(10, 2 * attempt))
    raise RuntimeError(f'{name}: не скачалось за {tries} попыток ({last_err})')


def drive_download(file_id: str, out_path: Path, expect: int = 0, tries: int = 5) -> int:
    """Качает файл с Google Drive по id: подтверждение для больших файлов + докачка и ретраи."""
    base = 'https://drive.usercontent.google.com/download'
    tmp = out_path.with_suffix(out_path.suffix + '.part')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, tries + 1):
        already = tmp.stat().st_size if tmp.exists() else 0
        headers = {'Range': f'bytes={already}-'} if already else None
        try:
            with _open(f'{base}?id={file_id}&export=download') as resp:
                ctype = (resp.headers.get('Content-Type') or '').lower()
                if 'text/html' in ctype:
                    html = resp.read().decode('utf-8', 'replace')
                    if 'request access' in html.lower() or 'Запросить доступ' in html:
                        raise RuntimeError('файл закрыт: включите доступ «всем, у кого есть ссылка» '
                                           '(Anyone with the link)')
                    token = _confirm_token(html)
                    url2 = f'{base}?id={file_id}&export=download&confirm=t'
                    if token:
                        url2 += f'&uuid={token}'
                    with _open(url2, timeout=120, headers=headers) as r2:
                        if 'text/html' in (r2.headers.get('Content-Type') or '').lower():
                            raise RuntimeError('Google Drive вернул страницу вместо файла '
                                               '(проверьте доступ и id)')
                        got = _stream_to(r2, tmp, out_path.name)
                else:
                    got = _stream_to(resp, tmp, out_path.name)
            if expect and got != expect:
                raise RuntimeError(f'скачано {human(got)} вместо {human(expect)}')
            tmp.replace(out_path)
            return got
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            last_err = e
            done = tmp.stat().st_size if tmp.exists() else 0
            log(f'    {out_path.name}: попытка {attempt}/{tries} не удалась ({e}); '
                f'скачано {human(done)} — продолжаю с этого места')
            time.sleep(min(10, 2 * attempt))
    raise RuntimeError(f'{out_path.name}: не скачалось за {tries} попыток ({last_err})')


def download_archive(arc: dict, dest_zips: Path, force: bool) -> Path:
    zip_path = dest_zips / f'lct-{arc["name"]}.zip'
    log(f'  [{arc["name"]}] {arc["comment"]}')
    if zip_path.is_file() and not force and zip_path.stat().st_size == arc['expect']:
        log(f'    архив уже скачан: {zip_path.name} ({human(zip_path.stat().st_size)})')
        return zip_path
    size = drive_download(arc['gid'], zip_path, expect=arc['expect'])
    log(f'    скачано {zip_path.name}: {human(size)}')
    if arc['expect'] and size != arc['expect']:
        log(f'    ВНИМАНИЕ: ожидалось {human(arc["expect"])} — архив могли перезалить '
            f'или загрузка обрезана')
    with open(zip_path, 'rb') as f:
        if f.read(4) != b'PK\x03\x04':
            raise RuntimeError(f'{zip_path.name}: это не zip-архив (проверьте ссылку)')
    return zip_path


def _clear_path_conflicts(zf: zipfile.ZipFile, dest_root: Path) -> None:
    """Убирает каталоги, которые стоят на месте файлов из архива.

    Типичная причина: docker создаёт пустой каталог под bind-mount файла (например
    `./filtered/catalog.csv` монтируется в sommelier), если файла на хосте ещё нет.
    После этого распаковка архива падала с `[Errno 21] Is a directory`.
    """
    for info in zf.infolist():
        if info.is_dir():
            continue
        p = dest_root / info.filename
        if p.is_dir():
            log(f'    конфликт путей: {info.filename} — на диске каталог, '
                f'удаляю (его создал docker под bind-mount файла)')
            shutil.rmtree(p)


def extract_zip(zip_path: Path, dest_root: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f'{zip_path.name}: битый файл внутри архива: {bad}')
        _clear_path_conflicts(zf, dest_root)
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
        part = target.with_suffix(target.suffix + '.part')
        resume = f' (докачиваю с {human(part.stat().st_size)})' if part.exists() else ''
        log(f'    {name}: качаю ({human(size)}){resume}')
        got = download_file(_encoder_url(name), target, expect=size, what=name)
        if got != size:
            raise RuntimeError(f'{name}: скачано {human(got)} вместо {human(size)}')
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
    ap.add_argument('--status', action='store_true', help='показать, что уже на диске')
    args = ap.parse_args(argv)

    dest = Path(args.dest).resolve() if args.dest else REPO
    only = [s.strip() for s in args.only.split(',') if s.strip()]
    log(f'[fetch] корень распаковки: {dest}')

    if args.status:
        ok = status(dest, with_encoder=not args.no_encoder)
        log('[fetch] состояние: ' + ('всё на месте ✓' if ok else 'не всё (см. выше)'))
        return 0 if ok else 1

    if args.check:
        ok = check_only(only, with_encoder=not args.no_encoder)
        log('[fetch] проверка ' + ('пройдена ✓' if ok else 'не пройдена (см. выше)'))
        return 0 if ok else 1

    if not args.only_encoder:
        fetch_archives(dest, only, args.force, args.keep_zips)
    if not args.no_encoder:
        fetch_encoder(dest)
    ok = status(dest, with_encoder=not args.no_encoder)
    log('[fetch] готово ' + ('✓' if ok else '— но чего-то не хватает (см. выше)'))
    return 0 if ok else 1


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


def _artifact_ok(dest: Path, rel: str) -> bool:
    """Есть ли артефакт на месте (папка — непустая, файл — непустой)."""
    p = dest / rel
    if p.is_dir():
        return any(p.iterdir())
    return p.is_file() and p.stat().st_size > 0


def _archive_done(arc: dict, dest: Path) -> bool:
    """Все артефакты архива уже на диске (значит, качать/распаковывать не нужно)."""
    return all(_artifact_ok(dest, a) for a in arc.get('artifacts', []))


def status(dest: Path, with_encoder: bool = True) -> bool:
    """Печатает, что уже лежит на диске, а чего не хватает (ничего не качает)."""
    log(f'  [состояние] {dest}')
    ok = True
    for arc in ARCHIVES:
        miss = [a for a in arc.get('artifacts', []) if not _artifact_ok(dest, a)]
        mark = '✓' if not miss else '✗'
        ok = ok and not miss
        log(f'    {mark} {arc["name"]:9s} {arc["comment"]}'
            + (f' — не хватает: {", ".join(miss)}' if miss else ''))
    if with_encoder:
        enc_dir = dest / 'models' / 'siglip2-base-patch16-256'
        miss = [n for n, sz in ENCODER_FILES
                if not (enc_dir / n).is_file() or (enc_dir / n).stat().st_size != sz]
        ok = ok and not miss
        log(f'    {"✓" if not miss else "✗"} encoder   SigLIP2 ({ENCODER_REPO})'
            + (f' — не хватает: {", ".join(miss)}' if miss else ''))
    return ok


def fetch_archives(dest_root: Path, only: list, force: bool, keep_zips: bool) -> list:
    """Качает архивы и раскладывает их по местам. Возвращает список обработанных имён."""
    dest_zips = dest_root / 'deploy' / 'archives'
    dest_zips.mkdir(parents=True, exist_ok=True)
    todo = [a for a in ARCHIVES if not only or a['name'] in only]
    done = []
    for idx, arc in enumerate(todo, 1):
        if not force and _archive_done(arc, dest_root):
            log(f'[{idx}/{len(todo)}] {arc["name"]}: уже на месте — пропускаю')
            done.append(f'{arc["name"]}(skip)')
            continue
        log(f'[{idx}/{len(todo)}] {arc["name"]}')
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
