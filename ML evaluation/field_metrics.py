# -*- coding: utf-8 -*-
"""Полевые метрики качества: exact + winery/grape/color/category/line + near-dup.

Отвечает «по тому, что важно» (организаторы: винодельня/сорт/сахар/тип/линейка;
год не критичен). Работает по отчёту пайплайна (``ocr_rerank_report.json``):
``final_slug`` ⟷ ``true_slug`` сверяются по каталогу + основа слага (линейка).

    python3 field_metrics.py                       # data/reports/ocr_rerank_report.json
    python3 field_metrics.py --report ../reports/ocr_rerank_report.json --tag before62
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import REFERENCE_CSV, DATA_REPORTS, EVAL_REPORTS   # noqa: E402
import ocr_rerank as O                                        # noqa: E402


def load_catalog() -> dict:
    out = {}
    with open(REFERENCE_CSV, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f, delimiter=';')
        next(rd, None)
        for row in rd:
            if len(row) <= 8:
                continue
            slug = row[8].strip()
            if slug and slug not in out:
                out[slug] = {'winery': row[7].strip(), 'grape': row[5].strip(),
                             'cat': row[2].strip(), 'color': row[3].strip()}
    return out


def base_key(slug: str) -> str:
    b, _ = O.parse_slug(slug or '')
    return '-'.join(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Полевые метрики качества')
    ap.add_argument('--report', default=str(DATA_REPORTS / 'ocr_rerank_report.json'))
    ap.add_argument('--tag', default='field_metrics')
    args = ap.parse_args(argv)

    ref = load_catalog()
    rep = json.loads(Path(args.report).read_text(encoding='utf-8'))
    pr = rep['predictions']
    n = len(pr) or 1

    def attr(a, b, k):
        va = ref.get(a, {}).get(k)
        return bool(va) and va == ref.get(b, {}).get(k)

    def frac(f):
        return round(sum(1 for p in pr if f(p.get('final_slug'), p['true_slug'])) / n, 4)

    # near-dup группы: одна (winery, grape, cat) на несколько slug
    grp = Counter()
    for s, r in ref.items():
        key = (r['winery'], r['grape'], r['cat'])
        if all(key):
            grp[key] += 1
    tg = [p for p in pr if grp.get(tuple(ref.get(p['true_slug'], {}).get(k, '')
                                         for k in ('winery', 'grape', 'cat')), 0) > 1]
    err = [p for p in pr if p.get('final_slug') != p['true_slug']]

    summary = {
        'n': len(pr),
        'report_summary': rep.get('summary', {}),
        'exact': frac(lambda a, b: a == b),
        'winery': frac(lambda a, b: attr(a, b, 'winery')),
        'grape': frac(lambda a, b: attr(a, b, 'grape')),
        'category': frac(lambda a, b: attr(a, b, 'cat')),
        'line_base': frac(lambda a, b: base_key(a) == base_key(b)),
        'errors': len(err),
        'on_errors_winery_kept': sum(1 for p in err if attr(p.get('final_slug'), p['true_slug'], 'winery')),
        'on_errors_cat_kept': sum(1 for p in err if attr(p.get('final_slug'), p['true_slug'], 'cat')),
        'on_errors_grape_kept': sum(1 for p in err if attr(p.get('final_slug'), p['true_slug'], 'grape')),
        'near_dup_group_n': len(tg),
        'near_dup_exact_ok': sum(1 for p in tg if p.get('final_slug') == p['true_slug']),
        'ocr_present': {k: sum(1 for p in pr if (p.get('ocr_fields') or {}).get(k))
                        for k in ('grape', 'line', 'sparkling', 'additional_text', 'winery', 'color')},
    }
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    (EVAL_REPORTS / f'{args.tag}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'=== field metrics [{args.tag}] n={summary["n"]} ===')
    for k in ('exact', 'winery', 'grape', 'category', 'line_base'):
        print(f'  {k:10s}: {summary[k]}')
    print(f'  errors={summary["errors"]} (winery_kept={summary["on_errors_winery_kept"]}, '
          f'cat_kept={summary["on_errors_cat_kept"]}, grape_kept={summary["on_errors_grape_kept"]})')
    print(f'  near_dup_group={summary["near_dup_group_n"]} exact_ok={summary["near_dup_exact_ok"]}')
    print(f'  ocr_present={summary["ocr_present"]}')
    print('отчёт ->', EVAL_REPORTS / f'{args.tag}.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
