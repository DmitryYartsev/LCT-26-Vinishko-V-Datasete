# -*- coding: utf-8 -*-
"""Гейт v3: позитивы 185 + 3 «то же вино, другой год», негативы 8 настоящих.

Верность карточки считается по правилу организаторов (`same_wine`: год/крепость не влияют).

    python3 gate_eval3.py <out.json>
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ML evaluation'))
from wine_identity import same_wine  # noqa: E402

POS = [("62", "reports/26_recall_metrics/topk62.json",
        "reports/23_ml_dev2_pipeline_eval/mlweb_check/report62.json"),
       ("eval2", "reports/26_recall_metrics/topkeval2.json",
        "reports/23_ml_dev2_pipeline_eval/mlweb_check/report_eval2_partial.json")]
NEG_S = "reports/27_gate_negatives/neg_scores.json"
NEG_O = "reports/27_gate_negatives/neg_ocr.json"
NEG_C = "reports/27_gate_negatives/neg_classified.json"


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
                         'correct': same_wine(p['final_slug'] or '', p['true_slug'] or '')})
    return rows


def load_neg():
    scores = {r['image']: r['pos'] for r in json.load(open(NEG_S, encoding='utf-8'))}
    ocr = {r['image']: r for r in json.load(open(NEG_O, encoding='utf-8'))['predictions']}
    cls = {r['image']: r for r in json.load(open(NEG_C, encoding='utf-8'))}
    pos_extra, neg = [], []
    for img, s in scores.items():
        p = ocr.get(img, {})
        row = {'image': img, 'top1': s['top1'], 'stage': p.get('stage'),
               'reason': p.get('reason'), 'csv': p.get('csv_confidence') or 0.0}
        if cls.get(img, {}).get('verdict') == 'positive_vintage':
            pos_extra.append({**row, 'set': 'negatives→positive', 'correct': True})
        else:
            neg.append(row)
    return pos_extra, neg


def accept(row, hi, lo, c_ocr, c_agr):
    if row['top1'] >= hi:
        return True
    if row['top1'] < lo:
        return False
    if row['stage'] == 'ocr_rerank' and row['csv'] >= c_ocr:
        return True
    if row['reason'] == 'csv_agrees' and row['csv'] >= c_agr:
        return True
    return False


def evaluate(pos, neg, hi, lo, c_ocr, c_agr):
    ap = [accept(r, hi, lo, c_ocr, c_agr) for r in pos]
    an = [accept(r, hi, lo, c_ocr, c_agr) for r in neg]
    tp = sum(1 for a, r in zip(ap, pos) if a and r['correct'])
    fp = sum(1 for a, r in zip(ap, pos) if a and not r['correct']) + sum(an)
    fn = sum(1 for a in ap if not a)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'prec': round(p, 4), 'rec': round(r, 4),
            'f1': round(2 * p * r / (p + r), 4) if p + r else 0.0,
            'pos_acc': sum(ap), 'pos_total': len(pos), 'pos_lost': len(pos) - sum(ap),
            'neg_acc': sum(an), 'neg_total': len(neg), 'hi': hi, 'lo': lo,
            'c_ocr': c_ocr, 'c_agr': c_agr}


def main(out_path):
    pos = load_pos()
    extra, neg = load_neg()
    pos_all = pos + extra
    grid = [evaluate(pos_all, neg, hi, lo, co, ca)
            for hi, lo, co, ca in itertools.product((0.78, 0.80, 0.81, 0.82), (0.65, 0.70, 0.75),
                                                    (0.70, 0.80, 0.85), (0.60, 0.70, 0.80))]
    cur = evaluate(pos_all, neg, 0.75, 0.0, 9.9, 9.9)
    print(f'позитивов {len(pos_all)} (185 + {len(extra)} «то же вино, другой год»), '
          f'негативов {len(neg)}')
    print(f'было (top1>=0.75): TP={cur["tp"]} FP={cur["fp"]} FN={cur["fn"]} P={cur["prec"]} '
          f'R={cur["rec"]} F1={cur["f1"]} | негативов принято {cur["neg_acc"]}/{cur["neg_total"]}')
    zero = [g for g in grid if g['neg_acc'] == 0]
    print(f'\nбез принятых негативов (0/{len(neg)}):')
    for g in sorted(zero, key=lambda x: (-x['f1'], x['pos_lost']))[:8]:
        print(f"  HI>={g['hi']} LO>={g['lo']} c_ocr>={g['c_ocr']} c_agr>={g['c_agr']} → "
              f"TP={g['tp']} FP={g['fp']} FN={g['fn']} P={g['prec']} R={g['rec']} F1={g['f1']} | "
              f"позитивы {g['pos_acc']}/{g['pos_total']} (потеряно {g['pos_lost']})")
    json.dump({'positives': len(pos_all), 'positives_core': len(pos), 'positives_vintage': extra,
               'negatives': neg, 'current': cur, 'grid': grid},
              open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/27_gate_negatives/sweep_v3.json')
