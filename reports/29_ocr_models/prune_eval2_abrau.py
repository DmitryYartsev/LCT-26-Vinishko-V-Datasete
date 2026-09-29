# -*- coding: utf-8 -*-
"""Сбалансировать eval2: убрать половину вин Абрау-Дюрсо, где OCR ничего не изменил.

Что делает (по умолчанию — только показывает план, `--apply` — выполняет):
* считает по отчёту gpt-4o-mini (reports/26_recall_metrics/report_eval2.json), у каких abrau-вин
  OCR не менял ответ (`final_slug == pre_ocr_slug`);
* выбирает половину таких вин (по числу фото, детерминированно) и переносит их:
  папки `data/eval2/<slug>/` -> `data/eval2_extracted/<slug>/`,
  плоские файлы `data/eval2_images/<image>` -> `data/eval2_extracted/images/<image>` (только с
  `--move-images`, чтобы не сломать идущий прогон);
* переписывает `data/eval2.csv` без этих строк и сохраняет убранные строки в
  `data/eval2_extracted.csv`; детали — в `reports/29_ocr_models/eval2_pruning.json`.

    python3 reports/29_ocr_models/prune_eval2_abrau.py [--apply] [--move-images]
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_IN = REPO / 'data' / 'eval2.csv'
CSV_OUT_REMOVED = REPO / 'data' / 'eval2_extracted.csv'
FOLDERS = REPO / 'data' / 'eval2'
IMAGES = REPO / 'data' / 'eval2_images'
EXTRACTED = REPO / 'data' / 'eval2_extracted'
REPORT = REPO / 'reports' / '26_recall_metrics' / 'report_eval2.json'
DETAILS = REPO / 'reports' / '29_ocr_models' / 'eval2_pruning.json'


def main(apply, move_images):
    rows = list(csv.DictReader(open(CSV_IN, encoding='utf-8')))
    preds = {p['image']: p for p in json.load(open(REPORT, encoding='utf-8'))['predictions']}
    by_slug = defaultdict(list)
    for r in rows:
        by_slug[r['true_slug']].append(r)

    abrau = {s: rs for s, rs in by_slug.items() if 'abrau' in s.lower()}
    ocr_flat = {s: all((preds.get(r['image']) or {}).get('final_slug')
                       == (preds.get(r['image']) or {}).get('pre_ocr_slug') for r in rs)
                for s, rs in abrau.items()}
    cand = {s: rs for s, rs in abrau.items() if ocr_flat[s]}
    # «половина вин»: берём ровно половину кандидатов, начиная с самых многолюдных
    n_wines = (len(cand) + 1) // 2
    picked = [s for s, _ in sorted(cand.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n_wines]]
    remove_rows = [r for s in picked for r in by_slug[s]]

    print(f'eval2: {len(rows)} фото / {len(by_slug)} вин; abrau: '
          f'{sum(len(v) for v in abrau.values())} фото / {len(abrau)} вин')
    print(f'  из них OCR ничего не менял: {sum(len(v) for v in cand.values())} фото / {len(cand)} вин')
    print(f'  убираем: {len(picked)} вин из {len(cand)} кандидатов '
          f'(половина вин) / {len(remove_rows)} фото; останется {len(rows) - len(remove_rows)} фото')
    print('  abrau-доля: '
          f'{sum(len(v) for v in abrau.values()) / len(rows):.1%} -> '
          f'{(sum(len(v) for v in abrau.values()) - len(remove_rows)) / (len(rows) - len(remove_rows)):.1%}')

    DETAILS.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'removed_wines': picked, 'removed_photos': [r['image'] for r in remove_rows],
               'kept_photos': len(rows) - len(remove_rows),
               'abrau_share_before': round(sum(len(v) for v in abrau.values()) / len(rows), 4),
               'abrau_share_after': round(
                   (sum(len(v) for v in abrau.values()) - len(remove_rows))
                   / (len(rows) - len(remove_rows)), 4)},
              open(DETAILS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if not apply:
        print('\nэто план (без --apply ничего не меняется);  детали:', DETAILS)
        return

    kept = [r for r in rows if r not in remove_rows]
    with open(CSV_IN, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['image', 'true_slug'])
        w.writeheader()
        w.writerows(kept)
    with open(CSV_OUT_REMOVED, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['image', 'true_slug'])
        w.writeheader()
        w.writerows(remove_rows)

    for slug in picked:
        src, dst = FOLDERS / slug, EXTRACTED / slug
        if src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    if move_images:
        (EXTRACTED / 'images').mkdir(parents=True, exist_ok=True)
        for r in remove_rows:
            src = IMAGES / r['image']
            if src.exists():
                shutil.move(str(src), str(EXTRACTED / 'images' / r['image']))
    print(f'\nприменено: eval2.csv — {len(kept)} строк, убранное — '
          f'{CSV_OUT_REMOVED.relative_to(REPO)} ({len(remove_rows)}), папки — в '
          f'{EXTRACTED.relative_to(REPO)}/, картинки перенесены: {move_images}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--move-images', action='store_true')
    a = ap.parse_args()
    main(a.apply, a.move_images)
