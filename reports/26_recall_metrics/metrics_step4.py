# -*- coding: utf-8 -*-
"""Метрики ветки ml-web на 62 + eval2 (185 фото) в двух семантиках истины.

* **exact** — совпал сам слаг;
* **same**  — «то же вино» по правилу организаторов: год/крепость (хвостовое число слага)
  не влияют, а сорт/цвет/сахар/линейка влияют (`ML evaluation/wine_identity.py`).

recall@K (retrieval) — истина попала в top-K пула; recall@K (с OCR) — OCR ставит свой ответ
первым, дальше идёт пул без него. Всё считается из кэша top-K (`image_topk*.json`) и отчётов
`pipeline.run_eval`.

    python3 metrics_step4.py <out.json>
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ML evaluation'))
from wine_identity import same_wine  # noqa: E402

SETS = [
    ("62", "reports/26_recall_metrics/topk62.json", "reports/26_recall_metrics/report62.json"),
    ("eval2 (123)", "reports/26_recall_metrics/topkeval2.json",
     "reports/26_recall_metrics/report_eval2.json"),
]
K_LIST = (1, 5, 10, 30)


def pool_order(entry):
    """Порядок пула = список ветки, победившей по top-1 (как в app._rank)."""
    b, l = entry.get('bottle') or [], entry.get('label') or []
    sb = b[0]['score'] if b else 0.0
    sl = l[0]['score'] if l else 0.0
    return [x['slug'] for x in (b if sb >= sl else l)]


def hit(slug, true_slug, mode):
    if not slug:
        return False
    return slug == true_slug if mode == 'exact' else same_wine(slug, true_slug)


def counts_for(preds, topk, mode, gt):
    n = len(preds)
    out = {'n': n, 'correct_retrieval': 0, 'correct_final': 0,
           'recall_retrieval': {k: 0 for k in K_LIST}, 'recall_with_ocr': {k: 0 for k in K_LIST}}
    for p in preds:
        true = gt.get(p['image']) or p['true_slug']
        order = pool_order(topk[p['image']])
        ocr_order = [p['final_slug']] + [s for s in order if s != p['final_slug']]
        out['correct_retrieval'] += hit(p['pre_ocr_slug'], true, mode)
        out['correct_final'] += hit(p['final_slug'], true, mode)
        for k in K_LIST:
            out['recall_retrieval'][k] += any(hit(s, true, mode) for s in order[:k])
            out['recall_with_ocr'][k] += any(hit(s, true, mode) for s in ocr_order[:k])
    return out


def load_gt():
    """Истина берётся из манифестов датасетов (единый источник: правки GT подхватываются)."""
    gt = {}
    for path in ('data/eval62.csv', 'data/eval2.csv'):
        rows = list(csv.DictReader(open(path, encoding='utf-8')))
        gt.update({r['image']: r['true_slug'] for r in rows if r['true_slug']})
    return gt


def block(c):
    n = c['n']
    return {'n': n,
            'accuracy_retrieval': round(c['correct_retrieval'] / n, 4),
            'accuracy_final_with_ocr': round(c['correct_final'] / n, 4),
            'recall_retrieval': {f'@{k}': round(c['recall_retrieval'][k] / n, 4) for k in K_LIST},
            'recall_with_ocr': {f'@{k}': round(c['recall_with_ocr'][k] / n, 4) for k in K_LIST},
            'counts': c}


def main(out_path):
    gt = load_gt()
    sets, total = {}, None
    for name, topk_path, report_path in SETS:
        topk = json.load(open(topk_path, encoding='utf-8'))['images']
        preds = json.load(open(report_path, encoding='utf-8'))['predictions']
        sets[name] = {mode: counts_for(preds, topk, mode, gt) for mode in ('exact', 'same')}
    n_all = sum(v['exact']['n'] for v in sets.values())
    total = {}
    for mode in ('exact', 'same'):
        agg = {'n': n_all, 'correct_retrieval': 0, 'correct_final': 0,
               'recall_retrieval': {k: 0 for k in K_LIST}, 'recall_with_ocr': {k: 0 for k in K_LIST}}
        for v in sets.values():
            c = v[mode]
            agg['correct_retrieval'] += c['correct_retrieval']
            agg['correct_final'] += c['correct_final']
            for k in K_LIST:
                agg['recall_retrieval'][k] += c['recall_retrieval'][k]
                agg['recall_with_ocr'][k] += c['recall_with_ocr'][k]
        total[mode] = agg

    result = {'note': ('exact — совпал слаг; same — «то же вино» (год/крепость не влияют, '
                       'сорт/цвет/сахар/линейка влияют); recall «с OCR»: [final_slug] + пул без '
                       'него; истина берётся из data/eval62.csv и data/eval2.csv'),
              'sets': {name: {mode: block(v[mode]) for mode in ('exact', 'same')}
                       for name, v in sets.items()},
              'total': {mode: block(total[mode]) for mode in ('exact', 'same')}}
    json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    for tag, blk in list(result['sets'].items()) + [('ИТОГО', result['total'])]:
        print(f'\n{tag}: n={blk["exact"]["n"]}')
        for mode in ('exact', 'same'):
            b = blk[mode]
            print(f'  [{mode:5s}] accuracy: без OCR {b["accuracy_retrieval"]:.4f} | '
                  f'с OCR {b["accuracy_final_with_ocr"]:.4f}')
            print('            recall retrieval @1/5/10/30: '
                  + ' / '.join(f'{b["recall_retrieval"][f"@{k}"]:.4f}' for k in K_LIST))
            print('            recall с OCR   @1/5/10/30: '
                  + ' / '.join(f'{b["recall_with_ocr"][f"@{k}"]:.4f}' for k in K_LIST))
    print('\nсохранено:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/26_recall_metrics.json')
