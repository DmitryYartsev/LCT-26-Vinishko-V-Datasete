# -*- coding: utf-8 -*-
"""Сколько реально даёт OCR: реранк ответов, сигнал гейта и контрфакты.

    python3 reports/28_ocr_value/ocr_value.py [out.json]

Считается по артефактам отчётов 26/27 (предсказания 185 фото + сигналы негативов),
истина — из `data/eval62.csv` и `data/eval2.csv`, тождество вина — `ML evaluation/wine_identity.py`.
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'reports' / '27_gate_negatives'))
sys.path.insert(0, str(ROOT / 'ML evaluation'))
from gate_eval3 import accept, load_neg, load_pos  # noqa: E402
from wine_identity import same_wine  # noqa: E402

HI, LO, C_OCR, C_AGR = 0.81, 0.70, 0.80, 0.80
SETS = [('62', 'reports/26_recall_metrics/report62.json'),
        ('eval2', 'reports/26_recall_metrics/report_eval2.json')]


def load_gt():
    gt = {}
    for path in ('data/eval62.csv', 'data/eval2.csv'):
        for row in csv.DictReader(open(path, encoding='utf-8')):
            if row['true_slug']:
                gt[row['image']] = row['true_slug']
    return gt


def hit(slug, truth, mode):
    return same_wine(slug or '', truth or '') if mode == 'same' else (slug or '') == (truth or '')


def rerank_stats(gt):
    """Что делает OCR как реранкер: переставляет ответы, чинит и ломает их."""
    out = {}
    for mode in ('exact', 'same'):
        total = {'n': 0, 'ret': 0, 'fin': 0, 'fixed': 0, 'broken': 0, 'switched': 0}
        per_set = {}
        for tag, path in SETS:
            preds = json.load(open(path, encoding='utf-8'))['predictions']
            st = {k: 0 for k in total}
            st['n'] = len(preds)
            for p in preds:
                truth = gt.get(p['image']) or p['true_slug']
                pre, final = p['pre_ocr_slug'], p['final_slug']
                pre_ok, fin_ok = hit(pre, truth, mode), hit(final, truth, mode)
                st['ret'] += pre_ok
                st['fin'] += fin_ok
                st['switched'] += (final != pre)
                st['fixed'] += (not pre_ok and fin_ok)
                st['broken'] += (pre_ok and not fin_ok)
            per_set[tag] = st
            for k in total:
                total[k] += st[k]
        out[mode] = {'per_set': per_set, 'total': total}
    return out


def gate_stats(pos_all, neg):
    """Карточки, которые держит подтверждение OCR, и контрфакты гейта."""
    strict = [r for r in pos_all if accept(r, HI, LO, C_OCR, 9.9)]          # HI + только переранк
    soft = [r for r in pos_all if accept(r, HI, LO, C_OCR, C_AGR)]           # HI + любые подтверждения
    marginal = [r for r in soft if r not in strict]
    variants = {}
    for label, hi, lo, co, ca in (
            ('визуальный top1>=0.75 (было)', 0.75, 0.0, 9.9, 9.9),
            ('визуальный top1>=0.81 (только HI)', HI, 0.0, 9.9, 9.9),
            ('HI/LO + подтверждение OCR', HI, LO, C_OCR, C_AGR)):
        ap = [accept(r, hi, lo, co, ca) for r in pos_all]
        an = [accept(r, hi, lo, co, ca) for r in neg]
        tp = sum(1 for a, r in zip(ap, pos_all) if a and r['correct'])
        fp = sum(1 for a, r in zip(ap, pos_all) if a and not r['correct']) + sum(an)
        fn = sum(1 for a in ap if not a)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        variants[label] = {'cards': sum(ap), 'cards_correct': tp, 'negatives_accepted': sum(an),
                           'precision': round(p, 4), 'recall': round(rec, 4),
                           'f1': round(2 * p * rec / (p + rec), 4) if p + rec else 0.0}
    return {'marginal_cards': len(marginal),
            'marginal_cards_correct': sum(1 for r in marginal if r['correct']),
            'marginal_cards_answer_changed_by_ocr':
                sum(1 for r in marginal if r['stage'] == 'ocr_rerank'),
            'marginal': [{'image': r['image'], 'set': r['set'], 'top1': round(r['top1'], 4),
                          'csv': round(r['csv'], 4), 'stage': r['stage'],
                          'correct': r['correct']} for r in marginal],
            'variants': variants}


def main(out_path):
    gt = load_gt()
    rerank = rerank_stats(gt)
    pos_all = load_pos() + load_neg()[0]
    neg = load_neg()[1]
    gate = gate_stats(pos_all, neg)
    result = {'note': ('pre_ocr_slug -> final_slug = эффект OCR как реранкера; '
                       'затем карточки, которые держит подтверждение OCR, и контрфакты гейта'),
              'rerank': rerank, 'gate': gate}
    json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    for mode in ('exact', 'same'):
        t = rerank[mode]['total']
        print(f'[{mode}] 185 фото: retrieval {t["ret"]} -> с OCR {t["fin"]} (net {t["fin"] - t["ret"]:+d}; '
              f'чинит {t["fixed"]}, ломает {t["broken"]}, переставил {t["switched"]})')
        for tag, st in rerank[mode]['per_set'].items():
            print(f'    {tag}: {st["ret"]} -> {st["fin"]} (net {st["fin"] - st["ret"]:+d}; '
                  f'чинит {st["fixed"]}, ломает {st["broken"]}, переставил {st["switched"]})')
    print(f'карточек держит подтверждение OCR: {gate["marginal_cards"]} '
          f'(верных {gate["marginal_cards_correct"]}, из них OCR менял ответ '
          f'{gate["marginal_cards_answer_changed_by_ocr"]})')
    for label, v in gate['variants'].items():
        print(f'  {label}: карточек {v["cards"]}, верных {v["cards_correct"]}, '
              f'негативов {v["negatives_accepted"]}, P={v["precision"]} R={v["recall"]} F1={v["f1"]}')
    print('сохранено:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/28_ocr_value.json')
