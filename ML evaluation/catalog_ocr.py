# -*- coding: utf-8 -*-
"""OCR-разметка КАТАЛОЖНЫХ фото: slug+фото -> поля этикетки -> CSV.

Зачем: в каталоге организаторов много пропусков/несоответствий (нет сахара, сорт
указан только в названии), из-за чего OCR-мэтчинг не различает near-dup «сёстер»
(отчёт 20: `87.88`, `96.27`). Прогоняем VLM по каталожным фото и сохраняем прочитанные
поля; матчер подмешивает их к записям каталога (`csv_match.enrichment_file`),
НЕ меняя исходную разметку организаторов.

Подготовка кропа — как в проде: бутылочный кроп -> кроп этикетки (`ocr.label_crop`);
если поля вышли «бедные» (`ocr_rerank._fields_poor`), читаем ещё раз по кропу бутылки.

    python3 catalog_ocr.py --limit 20      # смоук (20 фото)
    python3 catalog_ocr.py                 # все каталожные фото (с докачкой)
    python3 catalog_ocr.py --aggregate     # + агрегат по slug

Артефакты:
  * `data/catalog_ocr_enrichment.csv` — построчно (slug, photo, поля, тайминг)
  * `data/catalog_ocr_fields.csv`     — по slug (агрегат по фото, `--aggregate`)
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO), str(REPO / 'ML service')]
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import DATA, FILTERED                                          # noqa: E402
from pipeline_config import load_config, apply_retrieval_env              # noqa: E402
import pipeline as P                                                      # noqa: E402
import ocr_rerank as ocr                                                  # noqa: E402

IMG_EXT = {'.webp', '.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tif', '.tiff'}
FIELDS = ('year', 'winery', 'grape', 'color', 'sugar', 'line', 'sparkling',
          'wine_type', 'additional_text', 'raw_text')
OUT = DATA / 'catalog_ocr_enrichment.csv'
AGG = DATA / 'catalog_ocr_fields.csv'
COLS = ('slug', 'photo', *FIELDS, 'source', 'seconds', 'error')


def catalog_photos() -> list:
    """[(slug, path)] — все фото каталога (`filtered/<slug>/*`), как в индексе."""
    out = []
    for d in sorted(FILTERED.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                out.append((d.name, p))
    return out


def poor_fields(fields: dict, vocab: set, mode: str) -> bool:
    """Нужен ли добор полей по кропу бутылки (для каталожных фото — по режиму)."""
    if mode == 'off':
        return False
    if mode == 'strict':                       # только когда не прочитано ничего важного
        return not (fields.get('winery') or '').strip() \
            and not (fields.get('grape') or '').strip()
    return bool(fields) and ocr._fields_poor(fields, vocab)


def crops_for(path: Path, crop_mod):
    """Прод-препроцесс: (кроп для VLM, кроп бутылки) — считается последовательно (YOLO)."""
    from PIL import Image
    img = Image.open(path).convert('RGB')
    bottle, _ = crop_mod.maybe_crop(img)
    ocr_img, _found = crop_mod.ocr_label_crop(bottle)
    return ocr_img, bottle


def extract_fields(extractor, img) -> tuple:
    """-> (fields, error) — один вызов VLM (без исключений наружу)."""
    try:
        out = extractor.extract(img)
    except Exception as e:                                        # noqa: BLE001
        return {}, f'extract_failed: {e}'
    if out.get('error'):
        return {}, str(out['error'])
    return (out.get('fields') or {}), ''


def load_done(out: Path) -> set:
    """Уже обработанные (slug, photo) — для докачки после падения."""
    if not out.is_file():
        return set()
    with open(out, encoding='utf-8-sig', newline='') as f:
        return {(r['slug'], r['photo']) for r in csv.DictReader(f, delimiter=';')}


def append_rows(out: Path, rows: list) -> None:
    new = not out.is_file()
    with open(out, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter=';')
        if new:
            w.writeheader()
        w.writerows(rows)


def aggregate(out: Path, agg: Path) -> int:
    """По slug: самое частое непустое значение каждого поля (по всем фото винa)."""
    by_slug = {}
    with open(out, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            by_slug.setdefault(r['slug'], []).append(r)
    with open(agg, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=('slug', 'photos', *FIELDS), delimiter=';')
        w.writeheader()
        for slug, rows in sorted(by_slug.items()):
            rec = {'slug': slug, 'photos': len(rows)}
            for k in FIELDS:
                vals = [(r.get(k) or '').strip() for r in rows]
                vals = [v for v in vals if v]
                if vals:
                    rec[k] = Counter(vals).most_common(1)[0][0]
            w.writerow(rec)
    print(f'агрегат по slug -> {agg} ({len(by_slug)} вин)')
    return len(by_slug)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='OCR-разметка каталожных фото')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--agg', default=str(AGG))
    ap.add_argument('--workers', type=int, default=4, help='параллельных вызовов VLM')
    ap.add_argument('--retry-mode', choices=('config', 'strict', 'off'), default='strict',
                    help='добор полей по кропу бутылки: strict (нет бренда и сорта) — дефолт')
    ap.add_argument('--batch', type=int, default=64, help='сколько кропов готовим за раз')
    ap.add_argument('--limit', type=int, default=0, help='обработать первые N фото (смоук)')
    ap.add_argument('--aggregate', action='store_true', help='собрать агрегат по slug')
    ap.add_argument('--rebuild', action='store_true', help='игнорировать готовые строки')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    # ключ OpenRouter: <repo>/.env (в .gitignore), иначе легаси-файл вне репо
    P.load_env_file(REPO / '.env')
    if not os.environ.get('OPENROUTER_API_KEY'):
        P.load_env_file(REPO.parent / 'code' / '.env')
    apply_retrieval_env(cfg)
    import crop
    crop.reset_policy()                        # каталог = индексная политика
    extractor = ocr.OcrExtractor(cfg)
    matcher = ocr.CsvMatcher(cfg)              # нужен только словарь сортов
    out = Path(args.out)
    done = set() if args.rebuild else load_done(out)
    todo = [(s, p) for s, p in catalog_photos()
            if (s, str(Path(p).relative_to(REPO))) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'фото к обработке: {len(todo)} (готово ранее {len(done)}) | воркеров {args.workers}')
    t0, processed, errors = time.time(), 0, 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        t_batch = time.time()
        crops = []                             # кропы готовим последовательно (YOLO)
        for slug, p in chunk:
            try:
                crops.append((slug, p, *crops_for(p, crop), ''))
            except Exception as e:             # noqa: BLE001
                crops.append((slug, p, None, None, f'prepare_failed: {e}'))
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            first = list(pool.map(
                lambda c: extract_fields(extractor, c[2]) if c[2] is not None else ({}, c[4]),
                crops))
            poor = [j for j, (f, _err) in enumerate(first)
                    if poor_fields(f, matcher.grape_vocab, args.retry_mode)]
            second = {}
            if poor:
                second = dict(zip(poor, pool.map(
                    lambda j: extract_fields(extractor, crops[j][3]), poor)))
        rows = []
        for j, (slug, p, _c, _b, err) in enumerate(crops):
            fields, e1 = first[j]
            source, err = 'label', err or e1
            if j in second:
                f2, e2 = second[j]
                if f2:
                    fields = ocr._merge_fields(fields, f2, matcher.grape_vocab)
                    source = 'label+bottle'
                err = err or e2
            rows.append({'slug': slug, 'photo': str(Path(p).relative_to(REPO)),
                         **{k: fields.get(k, '') for k in FIELDS},
                         'source': source, 'seconds': round(time.time() - t_batch, 2),
                         'error': err})
        append_rows(out, rows)
        processed += len(chunk)
        errors += sum(1 for r in rows if r['error'])
        total = len(todo) + (0 if args.rebuild else len(done))
        print(f'  {processed + (0 if args.rebuild else len(done))}/{total} '
              f'({(time.time() - t0) / max(processed, 1):.1f} с/фото, ошибок {errors}, '
              f'добор по бутылке {len(second)})', flush=True)
    print(f'готово -> {out} | ошибок {errors}')
    if args.aggregate:
        aggregate(out, Path(args.agg))
    return 0


if __name__ == '__main__':
    sys.exit(main())

