# -*- coding: utf-8 -*-
"""Метрики прогона OCR-rerank по отчёту ``ML service/pipeline.py --eval-csv``.

Читает ``ocr_rerank_report.json`` (пишет pipeline.py) и считает:

  * accuracy retrieval (до OCR) vs final (с OCR-rerank) и дельту OCR;
  * распределение ``stage``/``reason`` — сколько раз OCR реально изменил ответ;
  * исход каждого override: fixed / broken / neutral (в т.ч. список broken);
  * ошибки, сгруппированные по «семейству» slug (общий префикс винодельни) —
    near-dup «сёстры» vs прочие;
  * статистику ``csv_margin``/``visual_similarity`` по решающим срабатываниям.

CLI::

    cd "ML evaluation"
    python analyze_ocr_eval.py \\
        --report ../data/reports/ocr_rerank_report.json \\
        --out ../data/reports/OCR_report_run.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'ML service'))
sys.path.insert(0, str(REPO))

try:                                     # переиспользуем разбор слага из пайплайна
    from ocr_rerank import parse_slug
except Exception:                        # noqa: BLE001 — скрипт должен работать автономно
    def parse_slug(slug: str):
        parts = (slug or '').lower().split('-')
        digits = ''
        while parts and parts[-1].isdigit():
            digits = parts.pop() + digits
        return [p for p in parts if p], digits


def slug_family(slug: str, depth: int = 2) -> str:
    """«Семейство» вина: первые ``depth`` токенов основы слага (обычно винодельня)."""
    base, _ = parse_slug(slug or '')
    return '-'.join(base[:depth]) if base else ''


def _fmt(x, nd=3):
    return '—' if x is None else f'{float(x):.{nd}f}'


def analyze(report: dict, top_errors: int = 15) -> tuple:
    summary = report.get('summary', {})
    preds = report.get('predictions', [])
    n = len(preds)
    m = {'n': n, **summary}

    m['stages'] = dict(Counter(p.get('stage') for p in preds))
    m['reasons'] = dict(Counter(p.get('reason') for p in preds))

    changed = [p for p in preds if p.get('final_slug') != p.get('pre_ocr_slug')]
    fixed = [p for p in changed if p.get('final_slug') == p.get('true_slug')]
    broken = [p for p in changed if p.get('pre_ocr_slug') == p.get('true_slug')
              and p.get('final_slug') != p.get('true_slug')]
    neutral = [p for p in changed if p not in fixed and p not in broken]
    m['ocr_changed'] = len(changed)
    m['ocr_fixed'] = len(fixed)
    m['ocr_broken'] = len(broken)
    m['ocr_neutral'] = len(neutral)
    m['net_gain'] = len(fixed) - len(broken)

    errors = [p for p in preds if p.get('final_slug') != p.get('true_slug')]
    near = [p for p in errors
            if p.get('final_slug') and slug_family(p['final_slug']) == slug_family(p.get('true_slug'))]
    m['errors'] = len(errors)
    m['errors_near_dup'] = len(near)
    m['errors_other'] = len(errors) - len(near)

    per_cfg = Counter()
    margins, visuals = [], []
    for p in preds:
        if p.get('ocr_input'):
            per_cfg[p['ocr_input']] += 1
        if p.get('csv_margin') is not None:
            margins.append(float(p['csv_margin']))
        if p.get('visual_similarity') is not None:
            visuals.append(float(p['visual_similarity']))
    m['ocr_input'] = dict(per_cfg)
    if margins:
        m['csv_margin'] = {'min': min(margins), 'median': round(statistics.median(margins), 4),
                           'max': max(margins)}
    if visuals:
        m['visual_similarity'] = {'min': min(visuals),
                                  'median': round(statistics.median(visuals), 4),
                                  'max': max(visuals)}
    return m, (fixed, broken, neutral, errors, near)


def build_markdown(m: dict, fixed, broken, errors, top_errors: int = 15) -> str:
    acc_r = m.get('accuracy_retrieval')
    acc_f = m.get('accuracy_final')
    delta = (acc_f - acc_r) if (acc_r is not None and acc_f is not None) else None
    lines = [
        '# Прогон OCR-rerank (data/eval)', '',
        f'- Изображений: **{m["n"]}**',
        f'- accuracy retrieval (до OCR): **{acc_r}**',
        f'- accuracy final (с OCR-rerank): **{acc_f}**',
        f'- Δ OCR: **{"%+.4f" % delta if delta is not None else "—"}**',
        f'- Время: {m.get("seconds")} c', '',
        '## stage / reason', '',
        f'- stage: `{m["stages"]}`',
        f'- reason: `{m["reasons"]}`', '',
        '## Вклад OCR', '',
        f'- OCR изменил ответ: **{m["ocr_changed"]}**',
        f'  - исправил (fixed): **{m["ocr_fixed"]}**',
        f'  - сломал (broken): **{m["ocr_broken"]}**',
        f'  - нейтрально: {m["ocr_neutral"]}',
        f'- net gain: **{m["net_gain"]:+d}**', '',
        '## Ошибки финального ответа', '',
        f'- всего: **{m["errors"]}**',
        f'- near-dup «сёстры» (то же семейство slug): **{m["errors_near_dup"]}**',
        f'- прочие (другая винодельня/вино): {m["errors_other"]}', '',
        '## Диагностика решений', '',
        f'- ocr_input: `{m["ocr_input"]}`',
        f'- csv_margin: `{m.get("csv_margin", {})}`',
        f'- visual_similarity: `{m.get("visual_similarity", {})}`', '',
    ]
    if broken:
        lines += ['## Примеры broken (OCR испортил верный retrieval)', '',
                  '| image | true | retrieval (верно) | final (неверно) | csv_margin | reason |',
                  '|---|---|---|---|---|---|']
        for p in broken[:top_errors]:
            lines.append(f'| {p.get("image")} | `{p.get("true_slug")}` | '
                         f'`{p.get("pre_ocr_slug")}` | `{p.get("final_slug")}` | '
                         f'{_fmt(p.get("csv_margin"))} | {p.get("reason")} |')
        lines.append('')
    if fixed:
        lines += ['## Примеры fixed (OCR исправил retrieval)', '',
                  '| image | true = final | было (retrieval) | csv_margin |',
                  '|---|---|---|---|']
        for p in fixed[:top_errors]:
            lines.append(f'| {p.get("image")} | `{p.get("true_slug")}` | '
                         f'`{p.get("pre_ocr_slug")}` | {_fmt(p.get("csv_margin"))} |')
        lines.append('')
    if errors:
        lines += [f'## Ошибки (первые {top_errors})', '',
                  '| image | true | final | семейство | ocr_input | csv_margin | reason |',
                  '|---|---|---|---|---|---|---|']
        for p in errors[:top_errors]:
            lines.append(f'| {p.get("image")} | `{p.get("true_slug")}` | `{p.get("final_slug")}` | '
                         f'{slug_family(p.get("true_slug"))} | {p.get("ocr_input")} | '
                         f'{_fmt(p.get("csv_margin"))} | {p.get("reason")} |')
        lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', default=str(REPO / 'data/reports/ocr_rerank_report.json'))
    ap.add_argument('--out', default=str(REPO / 'data/reports/OCR_report_run.md'))
    ap.add_argument('--top-errors', type=int, default=15)
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    m, extra = analyze(report, args.top_errors)
    fixed, broken, _, errors, _ = extra
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_markdown(m, fixed, broken, errors, args.top_errors), encoding='utf-8')

    print(f'accuracy_retrieval={m.get("accuracy_retrieval")} '
          f'accuracy_final={m.get("accuracy_final")}')
    print(f'stage={m["stages"]}')
    print(f'reason={m["reasons"]}')
    print(f'OCR changed={m["ocr_changed"]} fixed={m["ocr_fixed"]} '
          f'broken={m["ocr_broken"]} net_gain={m["net_gain"]:+d}')
    print(f'errors={m["errors"]} (near-dup sisters={m["errors_near_dup"]}, '
          f'other={m["errors_other"]})')
    print(f'ocr_input={m["ocr_input"]}')
    print(f'csv_margin={m.get("csv_margin")}')
    print('отчёт ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

