# -*- coding: utf-8 -*-
"""P0-2: контроль потолка объединённого пула (image top-K ∪ text top-T).

Отвечает на ключевой вопрос плана: докидывает ли полнокаталожный ТЕКСТОВЫЙ путь
кандидатов, которых нет в визуальном шортлисте. Метрика — **покрытие пула**
(истина внутри пула), а не точность: именно покрытие ограничивает потолок
любого ре-ранкера.

Использует кэш ``reports/image_topk.json`` (``recall_at_k.py``) и сохранённые
``ocr_fields`` (``data/reports/ocr_rerank_report.json``) — VLM повторно не зовём.

    python3 union_text_eval.py                 # сетка image_k × text_k
    python3 union_text_eval.py --no-e5         # только CsvMatcher (без e5)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import DATA_REPORTS, EVAL_REPORTS          # noqa: E402
from pipeline_config import load_config               # noqa: E402
import pipeline as P                                  # noqa: E402
import rerank_fusion as RF                            # noqa: E402
from text_retrieval import TextRetriever              # noqa: E402


def _branch(ce: dict):
    return [r['slug'] for r in (ce.get(ce.get('winner') or 'bottle') or [])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='P0-2: потолок union-пула')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--image-ks', default='6,10,30,60')
    ap.add_argument('--text-ks', default='0,5,10,20,30')
    ap.add_argument('--no-e5', action='store_true')
    args = ap.parse_args(argv)
    image_ks = [int(x) for x in args.image_ks.split(',') if x]
    text_ks = [int(x) for x in args.text_ks.split(',') if x]

    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    cache = json.loads((EVAL_REPORTS / 'image_topk.json').read_text(encoding='utf-8'))['images']
    report = json.loads((DATA_REPORTS / 'ocr_rerank_report.json').read_text(encoding='utf-8'))
    tr = TextRetriever(cfg, use_e5=not args.no_e5)
    tmax = max(text_ks) if text_ks else 0

    per = []
    for p in report['predictions']:
        ce = cache.get(p['image'])
        if not ce:
            continue
        rows, _ = tr.full_scores(p.get('ocr_fields') or {})
        per.append({'image': p['image'], 'true': p['true_slug'],
                    'image_slugs': _branch(ce),
                    'text_slugs': [r['slug'] for r in rows[:tmax]]})
    n = len(per) or 1
    text_top1 = round(sum(1 for r in per if r['text_slugs'] and r['text_slugs'][0] == r['true']) / n, 4)

    grid = {}
    for ik in image_ks:
        for tk in text_ks:
            cov = sum(1 for r in per
                      if r['true'] in RF.build_pool(r['image_slugs'], r['text_slugs'], ik, tk))
            grid[f'img{ik}_txt{tk}'] = round(cov / n, 4)

    # сколько уникальных кейсов текст добавляет ВНЕ визуального пула
    ik_max = max(image_ks)
    text_only = [r for r in per
                 if r['true'] not in r['image_slugs'][:ik_max]
                 and r['true'] in r['text_slugs']]
    summary = {'n': n, 'e5': tr.use_e5, 'text_top1': text_top1,
               'image_ks': image_ks, 'text_ks': text_ks, 'pool_coverage': grid,
               f'text_fixes_outside_img{ik_max}': len(text_only),
               'text_fix_images': [r['image'] for r in text_only]}
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    (EVAL_REPORTS / 'union_text_eval.json').write_text(
        json.dumps({'summary': summary, 'predictions': per}, ensure_ascii=False, indent=2),
        encoding='utf-8')

    print(f'n={n}  e5={tr.use_e5}  text-only top1={text_top1}')
    print('покрытие пула (строки image top-K, столбцы + text top-T):')
    header = 'image_k | ' + ' '.join(f'K={tk:<3}' for tk in text_ks)
    print(header)
    for ik in image_ks:
        row = ' '.join(f'{grid[f"img{ik}_txt{tk}"]:.3f}' for tk in text_ks)
        print(f'{ik:>7} | {row}')
    print(f'текст добавляет ВНЕ image top-{ik_max}: {len(text_only)}')
    print('отчёт ->', EVAL_REPORTS / 'union_text_eval.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
