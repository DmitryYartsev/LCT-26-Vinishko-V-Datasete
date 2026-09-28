# -*- coding: utf-8 -*-
"""P0-1: аудит эталонов и разметки (данные как источник ошибок recall).

Часть «недостижимых» винов (правильный slug не попадает даже в top-30) — это
неверно привязанные фото каталога или шумная разметка. Здесь проверяем:

  1. **Целостность эталонов**: у каждого slug есть фото в ``start_photos``.
  2. **Согласованность разметки**: винодельня, прочитанная VLM с фото, ⟷
     винодельня карточки ``true_slug`` (на eval-фото). Расхождение — либо
     неверное фото у slug, либо ошибка ground-truth → кандидат в ручной аудит.

    python3 audit_refs.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import DATA, DATA_REPORTS, EVAL_REPORTS, REFERENCE_CSV, START_PHOTOS  # noqa: E402
import ocr_rerank as O                                                           # noqa: E402


def load_reference() -> dict:
    """slug -> {winery, photo_file, paired: bool}."""
    out = {}
    with open(REFERENCE_CSV, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            slug = (r.get('Slug') or '').strip()
            if not slug or slug in out:
                continue
            out[slug] = {'winery': (r.get('Винодельня') or '').strip(),
                         'photo_file': (r.get('photo_file') or '').strip()}
    return out


def resolve_photo(slug: str, photo_file: str) -> Path | None:
    if photo_file:
        rel = photo_file.replace('start_photos/', '')
        p = START_PHOTOS / rel
        if p.exists():
            return p
    for ext in ('.webp', '.jpg', '.jpeg', '.png'):
        p = START_PHOTOS / f'{slug}{ext}'
        if p.exists():
            return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='P0-1: аудит эталонов/разметки')
    ap.add_argument('--ocr-report', default=str(DATA_REPORTS / 'ocr_rerank_report.json'))
    args = ap.parse_args(argv)

    ref = load_reference()
    missing = [s for s, e in ref.items() if resolve_photo(s, e['photo_file']) is None]

    report = json.loads(Path(args.ocr_report).read_text(encoding='utf-8'))
    mismatches, no_ocr = [], 0
    for p in report['predictions']:
        true = p['true_slug']
        fields = p.get('ocr_fields') or {}
        ocr_w = (fields.get('winery') or '').strip()
        cat_w = ref.get(true, {}).get('winery', '')
        if not ocr_w:
            no_ocr += 1
            continue
        sim = O._sym_cover(O._norm_tokens(ocr_w), O._norm_tokens(cat_w)) if cat_w else None
        if sim is None or sim < 0.5:
            mismatches.append({'image': p['image'], 'true_slug': true,
                               'ocr_winery': ocr_w, 'catalog_winery': cat_w,
                               'winery_sim': None if sim is None else round(sim, 3)})

    summary = {
        'n_reference_slugs': len(ref),
        'reference_photos_missing': len(missing),
        'reference_photos_missing_slugs': missing[:200],
        'eval_n': len(report['predictions']),
        'eval_ocr_winery_empty': no_ocr,
        'eval_winery_mismatch': len(mismatches),
    }
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    (EVAL_REPORTS / 'audit_refs.json').write_text(
        json.dumps({'summary': summary, 'mismatches': mismatches}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    with open(EVAL_REPORTS / 'audit_refs_suspicious.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'true_slug', 'ocr_winery', 'catalog_winery', 'winery_sim'])
        for m in mismatches:
            w.writerow([m['image'], m['true_slug'], m['ocr_winery'],
                        m['catalog_winery'], m['winery_sim']])

    print('=== Аудит эталонов / разметки (P0-1) ===')
    print(f'референсных slug: {len(ref)} | без фото: {len(missing)}')
    print(f'eval фото: {summary["eval_n"]} | OCR без винодельни: {no_ocr} | '
          f'расхождение винодельни: {len(mismatches)}')
    if mismatches[:10]:
        print('примеры расхождений:')
        for m in mismatches[:10]:
            print(f'  {m["image"][:34]:34} true={m["true_slug"][:34]:34} '
                  f'ocr={m["ocr_winery"]!r} cat={m["catalog_winery"]!r} sim={m["winery_sim"]}')
    print('отчёты ->', EVAL_REPORTS / 'audit_refs.json', '|', EVAL_REPORTS / 'audit_refs_suspicious.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
