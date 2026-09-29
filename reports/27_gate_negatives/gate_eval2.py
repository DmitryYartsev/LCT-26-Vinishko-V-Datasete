# -*- coding: utf-8 -*-
"""Подбор гейта с учётом OCR-сигналов (позитивы 185 + негативы 11).

Правило:
    карточка, если  top1 >= TAU_HI
    или            top1 >= TAU_LO и OCR уверенно ПОДТВЕРДИЛ карточку
                   (stage=ocr_rerank и csv_confidence >= C_OCR, либо csv_agrees и csv >= C_AGR)
Оптимум: максимум принятых негативов-отклонений при потере позитивов <= допуска.

    python3 gate_eval2.py <out.json>
"""
import itertools
import json
import sys

POS = [("62", "reports/26_recall_metrics/topk62.json",
        "reports/23_ml_dev2_pipeline_eval/mlweb_check/report62.json"),
       ("eval2", "reports/26_recall_metrics/topkeval2.json",
        "reports/23_ml_dev2_pipeline_eval/mlweb_check/report_eval2_partial.json")]
NEG_SCORES = "reports/27_gate_negatives/neg_scores.json"
NEG_OCR = "reports/27_gate_negatives/neg_ocr.json"


def winner_top(entry):
    b, l = entry.get('bottle') or [], entry.get('label') or []
    sb = b[0]['score'] if b else 0.0
    sl = l[0]['score'] if l else 0.0
    return sb if sb >= sl else sl


def load_pos():
    rows = []
    for tag, topk_path, report_path in POS:
        topk = json.load(open(topk_path, encoding='utf-8'))['images']
        for p in json.load(open(report_path, encoding='utf-8'))['predictions']:
            rows.append({'set': tag, 'image': p['image'], 'top1': winner_top(topk[p['image']]),
                         'stage': p['stage'], 'reason': p.get('reason'),
                         'csv': p.get('csv_confidence') or 0.0,
                         'correct': p['final_slug'] == p['true_slug']})
    return rows


def load_neg():
    scores = {r['image']: r['pos']['top1'] for r in json.load(open(NEG_SCORES, encoding='utf-8'))}
    ocr = {r['image']: r for r in json.load(open(NEG_OCR, encoding='utf-8'))['predictions']}
    rows = []
    for img, top1 in scores.items():
        p = ocr.get(img, {})
        rows.append({'image': img, 'top1': top1, 'stage': p.get('stage'),
                     'reason': p.get('reason'), 'csv': p.get('csv_confidence') or 0.0})
    return rows


def accept(row, tau_hi, tau_lo, c_ocr, c_agr):
    if row['top1'] >= tau_hi:
        return True
    if row['top1'] < tau_lo:
        return False
    if row['stage'] == 'ocr_rerank' and row['csv'] >= c_ocr:
        return True
    if row['reason'] == 'csv_agrees' and row['csv'] >= c_agr:
        return True
    return False


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(pos, neg, tau_hi, tau_lo, c_ocr, c_agr):
    acc_p = [accept(r, tau_hi, tau_lo, c_ocr, c_agr) for r in pos]
    acc_n = [accept(r, tau_hi, tau_lo, c_ocr, c_agr) for r in neg]
    tp = sum(1 for a, r in zip(acc_p, pos) if a and r['correct'])
    fp = sum(1 for a, r in zip(acc_p, pos) if a and not r['correct']) + sum(acc_n)
    fn = sum(1 for a in acc_p if not a)
    return dict(tp=tp, fp=fp, fn=fn, prec=round(f1(tp, fp, fn)[0], 4),
                rec=round(f1(tp, fp, fn)[1], 4), f1=round(f1(tp, fp, fn)[2], 4),
                pos_acc=sum(acc_p), pos_total=len(pos), pos_lost=len(pos) - sum(acc_p),
                neg_acc=sum(acc_n), neg_total=len(neg))


def main(out_path):
    pos, neg = load_pos(), load_neg()
    print(f'позитивов {len(pos)}, негативов {len(neg)}')
    grid = []
    for tau_hi, tau_lo, c_ocr, c_agr in itertools.product(
            (0.80, 0.81, 0.82, 0.83, 0.85), (0.70, 0.75, 0.78), (0.75, 0.80, 0.85, 0.90), (0.60, 0.70, 0.80)):
        r = evaluate(pos, neg, tau_hi, tau_lo, c_ocr, c_agr)
        r.update(tau_hi=tau_hi, tau_lo=tau_lo, c_ocr=c_ocr, c_agr=c_agr)
        grid.append(r)
    cur = evaluate(pos, neg, 0.75, 9.9, 0.80, 0.80)     # текущее правило: только top1>=0.75
    good = [r for r in grid if r['neg_acc'] == 0 and r['pos_lost'] <= 10]
    print('текущее правило (top1>=0.75):', {k: cur[k] for k in ('tp', 'fp', 'fn', 'prec', 'rec', 'f1',
                                                               'pos_acc', 'neg_acc')})
    print(f'\nварианты без принятых негативов (0/{len(neg)}) и потерей позитивов <= 10:')
    for r in sorted(good, key=lambda x: (-x['f1'], x['pos_lost']))[:12]:
        print(f"  HI>={r['tau_hi']} LO>={r['tau_lo']} c_ocr>={r['c_ocr']} c_agr>={r['c_agr']} → "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']} P={r['prec']} R={r['rec']} F1={r['f1']} | "
              f"позитивов {r['pos_acc']}/{r['pos_total']} (потеряно {r['pos_lost']})")
    best = sorted(grid, key=lambda x: (-x['neg_acc'] * -1, x['pos_lost']))[:0] or \
        sorted(grid, key=lambda x: (x['neg_acc'], x['pos_lost']))[:12]
    print('\nминимум принятых негативов:')
    for r in best:
        print(f"  HI>={r['tau_hi']} LO>={r['tau_lo']} c_ocr>={r['c_ocr']} c_agr>={r['c_agr']} → "
              f"негативов принято {r['neg_acc']}/{r['neg_total']}, позитивов потеряно {r['pos_lost']}, "
              f"F1={r['f1']}")
    json.dump({'current': cur, 'grid': grid, 'good': good}, open(out_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('сохранено:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/27_gate_negatives/sweep_ocr.json')
