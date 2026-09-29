# -*- coding: utf-8 -*-
"""Подбор порогов гейта «есть в каталоге» на 185 позитивах и 11 негативах.

Позитивы: 62 + eval2(123) — берём score/margin победившей ветки из кэша top-K,
          верность финального ответа (карточки) — из отчётов pipeline.run_eval.
Негативы: data/negatives (pos-режим, исключать из индекса нечего).

Метрика решения «отдать карточку»: TP = позитив принят и ответ верный, FP = принят и неверный
или это негатив, FN = позитив отклонён (ушёл на экран «нет совпадений»).

    python3 gate_sweep.py <out.json>
"""
import json
import sys

POS = [
    ("62", "reports/26_recall_metrics/topk62.json",
     "reports/23_ml_dev2_pipeline_eval/mlweb_check/report62.json"),
    ("eval2", "reports/26_recall_metrics/topkeval2.json",
     "reports/23_ml_dev2_pipeline_eval/mlweb_check/report_eval2_partial.json"),
]
NEG = "reports/27_gate_negatives/neg_scores.json"


def branch_top(entry):
    b, l = entry.get('bottle') or [], entry.get('label') or []
    sb = b[0]['score'] if b else 0.0
    sl = l[0]['score'] if l else 0.0
    arr = b if sb >= sl else l
    s = [x['score'] for x in arr]
    return (s[0] if s else 0.0), (s[0] - s[1] if len(s) > 1 else s[0] if s else 0.0)


def load_positives():
    rows = []
    for tag, topk_path, report_path in POS:
        topk = json.load(open(topk_path, encoding='utf-8'))['images']
        preds = json.load(open(report_path, encoding='utf-8'))['predictions']
        for p in preds:
            score, margin = branch_top(topk[p['image']])
            rows.append({'set': tag, 'image': p['image'], 'score': score, 'margin': margin,
                         'correct': p['final_slug'] == p['true_slug'],
                         'stage': p['stage'], 'reason': p.get('reason'),
                         'csv_conf': p.get('csv_confidence')})
    return rows


def load_negatives():
    out = []
    for r in json.load(open(NEG, encoding='utf-8')):
        pos = r['pos']
        out.append({'image': r['image'], 'score': pos['top1'], 'margin': pos['margin'],
                    'branch': pos['branch'], 'closest': pos['top1_slug']})
    return out


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(pos, neg, tau_score, tau_margin, use_ocr=False):
    """use_ocr: принимать также, если OCR уверенно переставил ответ (stage=ocr_rerank)."""
    accept_pos = [((p['score'] >= tau_score and p['margin'] >= tau_margin)
                   or (use_ocr and p['stage'] == 'ocr_rerank')) for p in pos]
    accept_neg = [(n['score'] >= tau_score and n['margin'] >= tau_margin)
                  or (use_ocr and n.get('stage') == 'ocr_rerank') for n in neg]
    tp = sum(1 for a, p in zip(accept_pos, pos) if a and p['correct'])
    fp = sum(1 for a, p in zip(accept_pos, pos) if a and not p['correct']) + sum(accept_neg)
    fn = sum(1 for a in accept_pos if not a)
    return tp, fp, fn, f1(tp, fp, fn), sum(accept_neg)


def main(out_path):
    pos, neg = load_positives(), load_negatives()
    n_pos = len(pos)
    rows = []
    for use_ocr in (False, True):
        for tau_score in (0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82):
            for tau_margin in (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030):
                tp, fp, fn, (p_, r_, f_), neg_acc = evaluate(pos, neg, tau_score, tau_margin, use_ocr)
                rows.append({'tau_score': tau_score, 'tau_margin': tau_margin, 'ocr': use_ocr,
                             'tp': tp, 'fp': fp, 'fn': fn, 'prec': round(p_, 4),
                             'rec': round(r_, 4), 'f1': round(f_, 4),
                             'neg_accepted': neg_acc, 'neg_total': len(neg),
                             'pos_accepted': tp + fp - neg_acc})
    best = sorted(rows, key=lambda x: (-x['f1'], x['neg_accepted']))[:10]
    cur = [r for r in rows if r['tau_score'] == 0.75 and r['tau_margin'] == 0.0 and not r['ocr']][0]
    print(f'позитивов {n_pos}, негативов {len(neg)}')
    print(f'текущее правило (score>=0.75, margin>=0): TP={cur["tp"]} FP={cur["fp"]} FN={cur["fn"]} '
          f'P={cur["prec"]} R={cur["rec"]} F1={cur["f1"]} | негативов принято {cur["neg_accepted"]}/{cur["neg_total"]}')
    print('\nтоп-10 по F1:')
    for r in best:
        print(f'  score>={r["tau_score"]:.2f} margin>={r["tau_margin"]:.3f} ocr={int(r["ocr"])} '
              f'→ TP={r["tp"]} FP={r["fp"]} FN={r["fn"]} P={r["prec"]} R={r["rec"]} F1={r["f1"]} '
              f'| негативов принято {r["neg_accepted"]}/{r["neg_total"]}')
    json.dump({'current': cur, 'grid': rows, 'best': best,
               'positives': len(pos), 'negatives': len(neg)},
              open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/27_gate_negatives/sweep.json')
