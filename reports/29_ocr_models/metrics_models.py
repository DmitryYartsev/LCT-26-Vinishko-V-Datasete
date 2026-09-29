# -*- coding: utf-8 -*-
"""Метрики сравнения OCR-моделей: точность, recall/precision/F1@k и метрики гейта.

    python3 reports/29_ocr_models/metrics_models.py [out.json]

Наборы (по задаче):
* **233 (62 + 171)** — старая выборка `data/eval.csv` целиком: `62` — `data/eval62.csv`,
  `rest171` — `data/eval233_rest.csv` (пул 233 без 62, фото в `data/eval/`);
* **eval2** — `data/eval2.csv` после сбалансирования (убрана половина вин Абрау-Дюрсо, где OCR
  ничего не менял; см. `reports/29_ocr_models/prune_eval2_abrau.py`).

Модели: `openai/gpt-4o-mini` (оффлайн по артефактам) и две Gemini (прогоны отчёта 29).
Метрики: accuracy (retrieval и с OCR, `exact`/`same`), recall/precision/F1@1/5/10/30 по пулу
retrieval и по порядку после OCR-реранка, решение гейта «карточка или похожие» на порогах прода,
секунды на фото (скорость).
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
from gate_eval3 import accept  # noqa: E402
from wine_identity import same_wine  # noqa: E402

HI, LO, C_OCR, C_AGR = 0.81, 0.70, 0.80, 0.80
K_LIST = (1, 5, 10, 30)
SETS_ORDER = ('62', 'rest171', 'eval2')
GROUPS = {'233 (62+171)': ('62', 'rest171'), 'eval2 (после прунинга)': ('eval2',),
          'ИТОГО': SETS_ORDER}
GT_FILES = {'62': 'data/eval62.csv', 'rest171': 'data/eval233_rest.csv', 'eval2': 'data/eval2.csv'}
TOPKS = {'62': 'reports/26_recall_metrics/topk62.json',
         'rest171': 'reports/29_ocr_models/topk_rest171.json',
         'eval2': 'reports/26_recall_metrics/topkeval2.json'}
MODELS = {
    'openai/gpt-4o-mini': {'62': 'reports/26_recall_metrics/report62.json',
                           'rest171': 'reports/29_ocr_models/gpt_rest171.json',
                           'eval2': 'reports/26_recall_metrics/report_eval2.json',
                           'neg': 'reports/27_gate_negatives/neg_ocr.json'},
    'google/gemini-2.5-flash': {'62': 'reports/29_ocr_models/gf62.json',
                                'rest171': 'reports/29_ocr_models/gf_rest171.json',
                                'eval2': 'reports/29_ocr_models/gf_eval2.json',
                                'neg': 'reports/29_ocr_models/gf_neg.json'},
    'google/gemini-2.5-flash-lite': {'62': 'reports/29_ocr_models/gfl62.json',
                                     'rest171': 'reports/29_ocr_models/gfl_rest171.json',
                                     'eval2': 'reports/29_ocr_models/gfl_eval2.json',
                                     'neg': 'reports/29_ocr_models/gfl_neg.json'},
}
NEG_SCORES = 'reports/27_gate_negatives/neg_scores.json'
NEG_CLASSIFIED = 'reports/27_gate_negatives/neg_classified.json'


def load_gt():
    """Истина по наборам: ключ — имя файла фото (BOM в старых CSV учтён)."""
    out = {}
    for name, path in GT_FILES.items():
        out[name] = {}
        if not Path(path).exists():
            continue
        for row in csv.DictReader(open(path, encoding='utf-8-sig')):
            if row['true_slug']:
                out[name][row['image']] = row['true_slug']
    return out


def hit(slug, truth, mode):
    return same_wine(slug or '', truth or '') if mode == 'same' else (slug or '') == (truth or '')


def winner_top(entry):
    b, l = entry.get('bottle') or [], entry.get('label') or []
    sb = b[0]['score'] if b else 0.0
    sl = l[0]['score'] if l else 0.0
    return sb if sb >= sl else sl


def pool_order(entry):
    """Порядок пула = список ветки, победившей по top-1 (как в app._rank и отчёте 26)."""
    b, l = entry.get('bottle') or [], entry.get('label') or []
    sb = b[0]['score'] if b else 0.0
    sl = l[0]['score'] if l else 0.0
    return [x['slug'] for x in (b if sb >= sl else l)]


def prf(hits_at_k, n):
    """P/R/F1@k в задаче с одним релевантным объектом: R@k=hits/n, P@k=hits/(n*k)."""
    out = {}
    for k, hits in hits_at_k.items():
        P = hits / (n * k * 1.0)
        R = hits / (n * 1.0)
        out[f'@{k}'] = {'hits': hits, 'precision': round(P, 4), 'recall': round(R, 4),
                        'f1': round(2 * P * R / (P + R), 4) if P + R else 0.0}
    return out



def accuracy_and_prf(paths, gt):
    """accuracy (retrieval / с OCR) + recall/precision/F1@k по наборам и группам."""
    out = {}
    for mode in ('exact', 'same'):
        sets = {}
        for name in SETS_ORDER:
            path = paths.get(name)
            if not path or not Path(path).exists():
                continue
            rep = json.load(open(path, encoding='utf-8'))
            topk = json.load(open(TOPKS[name], encoding='utf-8'))['images']
            st = {'n': 0, 'ret_ok': 0, 'fin_ok': 0,
                  'hits_ret': {k: 0 for k in K_LIST}, 'hits_ocr': {k: 0 for k in K_LIST}}
            for p in rep['predictions']:
                truth = gt[name].get(p['image'])
                if not truth:      # фото вне текущего набора (напр. убранное из eval2)
                    continue
                order = pool_order(topk[p['image']])
                ocr_order = [p['final_slug']] + [s for s in order if s != p['final_slug']]
                st['n'] += 1
                st['ret_ok'] += hit(p['pre_ocr_slug'], truth, mode)
                st['fin_ok'] += hit(p['final_slug'], truth, mode)
                for k in K_LIST:
                    st['hits_ret'][k] += any(hit(s, truth, mode) for s in order[:k])
                    st['hits_ocr'][k] += any(hit(s, truth, mode) for s in ocr_order[:k])
            sec = (rep.get('summary') or {}).get('seconds')
            # Секунды считаем на все фото прогона, а не на оценённые: длинный прогон мог идти
            # по полному CSV (eval2 гонялся по 123 фото, а в метрику после прунинга попало 105).
            run_n = len(rep['predictions']) or st['n']
            st['run_n'] = run_n
            st['sec'] = sec or 0.0
            st['seconds_per_photo'] = round(sec / run_n, 2) if sec and run_n else None
            st['ocr_errors'] = (rep.get('summary') or {}).get('ocr_errors')
            sets[name] = st
        groups = {}
        for gname, members in GROUPS.items():
            blocks = [sets[m] for m in members if m in sets]
            if not blocks:
                continue
            agg = {'n': 0, 'ret_ok': 0, 'fin_ok': 0,
                   'hits_ret': {k: 0 for k in K_LIST}, 'hits_ocr': {k: 0 for k in K_LIST},
                   'seconds_per_photo': None, 'ocr_errors': None, 'sec': 0.0, 'run_n': 0}
            for b in blocks:
                for key in ('n', 'ret_ok', 'fin_ok', 'sec', 'run_n'):
                    agg[key] += b[key]
                for k in K_LIST:
                    agg['hits_ret'][k] += b['hits_ret'][k]
                    agg['hits_ocr'][k] += b['hits_ocr'][k]
            agg['seconds_per_photo'] = (round(agg['sec'] / agg['run_n'], 2)
                                       if agg['run_n'] else None)
            errors = [b['ocr_errors'] for b in blocks if b.get('ocr_errors') is not None]
            agg['ocr_errors'] = sum(errors) if errors else None
            groups[gname] = agg
        for block in (*sets.values(), *groups.values()):
            block['accuracy_retrieval'] = round(block['ret_ok'] / block['n'], 4)
            block['accuracy_final'] = round(block['fin_ok'] / block['n'], 4)
            block['prf_retrieval'] = prf(block['hits_ret'], block['n'])
            block['prf_ocr_order'] = prf(block['hits_ocr'], block['n'])
        out[mode] = {'sets': sets, 'groups': groups}
    return out


def gate_metrics(paths, gt):
    """Метрики решения «карточка или похожие» на порогах прода."""
    neg_scores = {r['image']: r['pos']['top1']
                  for r in json.load(open(NEG_SCORES, encoding='utf-8'))}
    cls = {r['image']: r['verdict']
           for r in json.load(open(NEG_CLASSIFIED, encoding='utf-8'))}
    neg_preds = {p['image']: p
                 for p in json.load(open(paths['neg'], encoding='utf-8'))['predictions']}
    pos, neg = [], []
    for name in SETS_ORDER:
        path = paths.get(name)
        if not path or not Path(path).exists():
            continue
        rep = json.load(open(path, encoding='utf-8'))
        topk = json.load(open(TOPKS[name], encoding='utf-8'))['images']
        for p in rep['predictions']:
            truth = gt[name].get(p['image'])
            if not truth:
                continue
            pos.append({'set': name, 'image': p['image'], 'top1': winner_top(topk[p['image']]),
                        'stage': p['stage'], 'reason': p.get('reason'),
                        'csv': p.get('csv_confidence') or 0.0,
                        'correct': same_wine(p['final_slug'] or '', truth or '')})
    for image, top1 in neg_scores.items():
        p = neg_preds.get(image) or {}
        row = {'set': 'negatives', 'image': image, 'top1': top1, 'stage': p.get('stage'),
               'reason': p.get('reason'), 'csv': p.get('csv_confidence') or 0.0}
        (pos if cls.get(image) == 'positive_vintage' else neg).append(
            {**row, 'correct': True} if cls.get(image) == 'positive_vintage' else row)
    ap = [accept(r, HI, LO, C_OCR, C_AGR) for r in pos]
    an = [accept(r, HI, LO, C_OCR, C_AGR) for r in neg]
    tp = sum(1 for a, r in zip(ap, pos) if a and r['correct'])
    fp = sum(1 for a, r in zip(ap, pos) if a and not r['correct']) + sum(an)
    fn = sum(1 for a in ap if not a)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {'positives': len(pos), 'negatives': len(neg), 'cards': sum(ap),
            'cards_correct': tp, 'negatives_accepted': sum(an), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(P, 4), 'recall': round(R, 4),
            'f1': round(2 * P * R / (P + R), 4) if P + R else 0.0,
            'agree_signals': sum(1 for r in pos if r.get('reason') == 'csv_agrees'),
            'negatives_detail': [{'image': r['image'], 'top1': round(r['top1'], 4),
                                  'stage': r['stage'], 'reason': r['reason'],
                                  'csv': round(r['csv'], 4), 'accepted': a}
                                 for r, a in zip(neg, an)]}


def _row(tag, b, mode_exact, mode_same, gate=None):
    ex, sm = b['accuracy_prf'][mode_exact], b['accuracy_prf'][mode_same]
    g = gate if gate is not None else {}
    secs = b.get('seconds_per_photo')
    return (f"{tag:16s} {ex['n']:>4d} {ex['accuracy_retrieval']:>8.4f} {ex['accuracy_final']:>8.4f} "
            f"{sm['accuracy_retrieval']:>9.4f} {sm['accuracy_final']:>8.4f} "
            f"{ex['prf_retrieval']['@1']['precision']:>6.4f} "
            f"{ex['prf_retrieval']['@5']['recall']:>6.4f} {ex['prf_retrieval']['@5']['f1']:>6.4f} "
            f"{ex['prf_retrieval']['@5']['precision']:>6.4f} "
            f"{ex['prf_retrieval']['@30']['f1']:>6.4f} "
            f"{ex['prf_ocr_order']['@5']['f1']:>7.4f} "
            f"{g.get('cards', 0):>5d} {str(g.get('negatives_accepted', '-')) + '/' + str(g.get('negatives', '-')):>7s} "
            f"{g.get('f1', 0):>7.4f} {str(secs):>7s}")


def main(out_path):
    gt = load_gt()
    result = {}
    for model, paths in MODELS.items():
        if not Path(paths['neg']).exists():
            print(f'!! {model}: нет артефакта негативов {paths["neg"]} — пропускаю', flush=True)
            continue
        missing = [p for p in paths.values() if not Path(p).exists()]
        if missing:
            print(f'!! {model}: наборы ещё не готовы, считаю без них: '
                  f'{[Path(p).name for p in missing]}', flush=True)
        result[model] = {'accuracy_prf': accuracy_and_prf(paths, gt),
                         'gate': gate_metrics(paths, gt)}
    json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    head = ('набор              n acc retr acc OCR same retr same OCR    P@1    R@5   F1@5'
            '    P@5  F1@30 F1@5ocr  карт. негат. F1 гейт  с/фото')
    for model, m in result.items():
        print(f'\n=== {model}')
        print(head)
        for gname, gblock in m['accuracy_prf']['exact']['groups'].items():
            row = {'accuracy_prf': {'exact': gblock,
                                    'same': m['accuracy_prf']['same']['groups'][gname]},
                   'seconds_per_photo': gblock.get('seconds_per_photo')}
            print(_row(gname, row, 'exact', 'same',
                       m['gate'] if gname == 'ИТОГО' else None))
        print('  по наборам:')
        for sname, sblock in m['accuracy_prf']['exact']['sets'].items():
            row = {'accuracy_prf': {'exact': sblock,
                                    'same': m['accuracy_prf']['same']['sets'][sname]},
                   'seconds_per_photo': sblock.get('seconds_per_photo')}
            print('  ' + _row(sname, row, 'exact', 'same'))
    print('\nсохранено:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/29_ocr_models.json')
