# -*- coding: utf-8 -*-
"""Офлайн-replay OCR-rerank: метрики без повторных вызовов VLM.

Зачем: единственная дорогая/нестабильная часть OCR-тракта — сетевой вызов VLM.
Поля, прочитанные моделью, пайплайн сохраняет в отчёте (``ocr_fields``), а
short-list retrieval — в ``retrieval_topk``. Значит, любые изменения мэтчера,
весов и гейта можно перебирать мгновенно и офлайн, полностью прогоняя РЕАЛЬНЫЙ
код ``ocr_rerank.rerank`` (подменяется только ``OcrExtractor`` — заглушкой,
возвращающей сохранённые поля, и ``VisualVerifier``).

Так метрики replay совпадают с живым прогоном при той же конфигурации.

Запуск::

    cd "ML evaluation"
    python replay_ocr.py                             # метрики текущего конфига
    python replay_ocr.py --pool 5 --margin 0.15      # A/B гейта
    python replay_ocr.py --sweep                     # сетка margin x pool
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'ML service'))
sys.path.insert(0, str(REPO))

from pipeline_config import load_config          # noqa: E402
from pipeline import apply_host_paths            # noqa: E402
import ocr_rerank as ocr                         # noqa: E402


class StubExtractor:
    """Возвращает ранее сохранённые OCR-поля вместо сетевого вызова VLM."""

    def __init__(self, fields):
        self.fields = fields

    def extract(self, image):  # noqa: ARG002 — имитируем интерфейс OcrExtractor
        return {'fields': self.fields}


class StubVerifier:
    """Визуальная проверка не блокирует (в проде она мягкая, см. config.visual)."""

    def verify(self, *a, **kw):
        return 1.0


def load_rows(report_path: Path) -> list:
    rep = json.loads(Path(report_path).read_text(encoding='utf-8'))
    return rep['predictions']


def candidates_of(p, retrieval_path, pool: int) -> list:
    """Short-list retrieval для rerank: из дампа top-N либо из самого отчёта."""
    if retrieval_path and Path(retrieval_path).is_file():
        dump = json.loads(Path(retrieval_path).read_text(encoding='utf-8'))
        v = dump.get(p['image'])
        if v:
            return [r['slug'] for r in v['retrieval'][:pool]]
    return (p.get('retrieval_topk') or [])[:pool]


def evaluate(cfg, matcher, rows, retrieval_path, pool: int) -> dict:
    """Прогон реального ``rerank`` на сохранённых полях -> accuracy."""
    n = len(rows)
    retr_ok = sum(1 for p in rows if p.get('pre_ocr_slug') == p.get('true_slug'))
    final_ok = switched = fixed = broken = 0
    for p in rows:
        fields = p.get('ocr_fields') or {}
        cands = candidates_of(p, retrieval_path, pool)
        out = ocr.rerank(cfg, None, None, p.get('pre_ocr_slug'),
                         p.get('pre_ocr_score') or 0.0,
                         StubExtractor(fields), matcher, StubVerifier(),
                         ocr_image=object(), pre_ocr_candidates=cands)
        fin = out['final_slug']
        final_ok += int(fin == p.get('true_slug'))
        if fin != p.get('pre_ocr_slug'):
            switched += 1
            if p.get('pre_ocr_slug') == p.get('true_slug'):
                broken += 1
            elif fin == p.get('true_slug'):
                fixed += 1
    return {'n': n, 'retrieval_correct': retr_ok, 'final_correct': final_ok,
            'accuracy_retrieval': round(retr_ok / n, 4) if n else 0.0,
            'accuracy_final': round(final_ok / n, 4) if n else 0.0,
            'switched': switched, 'fixed': fixed, 'broken': broken}


MARGINS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', default=str(REPO / 'data/reports/ocr_rerank_report.json'))
    ap.add_argument('--retrieval', default='/tmp/retrieval_top20.json',
                    help='дамп top-N retrieval (опционально; иначе из отчёта)')
    ap.add_argument('--config', default=str(REPO / 'config/pipeline.yaml'))
    ap.add_argument('--pool', type=int, default=None, help='сколько top-k retrieval дать OCR')
    ap.add_argument('--margin', type=float, default=None, help='override decision_margin')
    ap.add_argument('--min-conf', type=float, default=None, help='override порога CSV-уверенности')
    ap.add_argument('--sweep', action='store_true', help='сетка margin x pool')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    apply_host_paths(cfg)
    matcher = ocr.CsvMatcher(cfg)
    rows = load_rows(Path(args.report))
    retr_path = Path(args.retrieval) if args.retrieval else None
    pool0 = int(args.pool if args.pool is not None else cfg.retrieval.get('top_k', 5))

    if args.margin is not None:
        cfg.csv_match.decision_margin = args.margin
    if args.min_conf is not None:
        cfg.csv_match.min_score = args.min_conf

    if args.sweep:
        print('pool | ' + ' | '.join('m=%.2f' % m for m in MARGINS))
        for pool in (3, 4, 5, 6, 8, 10):
            cells = []
            for m in MARGINS:
                cfg.csv_match.decision_margin = m
                r = evaluate(cfg, matcher, rows, retr_path, pool)
                cells.append('%+d' % (r['final_correct'] - r['retrieval_correct']))
            print('%4d | ' % pool + ' | '.join(cells))
        return 0

    r = evaluate(cfg, matcher, rows, retr_path, pool0)
    print('pool(top-k retrieval для OCR) = %d' % pool0)
    print('decision_margin = %.2f, min_score(CSV) = %.2f, min_confidence(OCR) = %.2f'
          % (float(cfg.csv_match.decision_margin), float(cfg.csv_match.min_score),
             float(cfg.ocr.min_confidence)))
    print('retrieval  %d/%d = %.4f' % (r['retrieval_correct'], r['n'], r['accuracy_retrieval']))
    print('final(OCR) %d/%d = %.4f' % (r['final_correct'], r['n'], r['accuracy_final']))
    print('switched=%d fixed=%d broken=%d (net %+d)'
          % (r['switched'], r['fixed'], r['broken'],
             r['final_correct'] - r['retrieval_correct']))
    return 0


if __name__ == '__main__':
    sys.exit(main())

