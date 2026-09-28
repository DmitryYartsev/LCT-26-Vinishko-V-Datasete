# -*- coding: utf-8 -*-
"""P0-3: обучение калиброванного fusion-ранкера (2-fold CV) + отчёт.

Собирает признаковый датасет из кэша image top-K (``reports/image_topk.json``,
делает ``recall_at_k.py``) и сохранённых ``ocr_fields`` (``data/reports/
ocr_rerank_report.json``), обучает pointwise логистическую регрессию
(``rerank_fusion.LogisticRanker``) и честно оценивает её 2-fold CV.

Пул кандидатов = image top-``--image-k`` ∪ text top-``--text-k``
(``text_retrieval.TextRetriever``). Метрики: accuracy top-1, ``broken``
(сломанные ответы относительно image top-1), прирост над baseline.

Порядок запуска:
    1) python3 ../"ML evaluation"/recall_at_k.py      # кэш image topk
    2) python3 train_fusion.py                        # обучение + сдача весов
    3) python3 union_text_eval.py                     # контроль потолка пула

Артефакты: ``ML service/fusion_weights.json`` (веса), ``reports/fusion_*.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import DATA_REPORTS, EVAL_REPORTS, FUSION_WEIGHTS   # noqa: E402
from pipeline_config import load_config                        # noqa: E402
import pipeline as P                                           # noqa: E402
import rerank_fusion as RF                                     # noqa: E402
from text_retrieval import TextRetriever                       # noqa: E402


def _image_branch(cache_entry: dict):
    """Победившая ветка кэша -> (slugs, {slug: score})."""
    res = cache_entry.get(cache_entry.get('winner') or 'bottle') or []
    return [r['slug'] for r in res], {r['slug']: float(r['score']) for r in res}


def load_dataset(cfg, image_k: int, text_k: int, use_e5: bool = True, rebuild: bool = False):
    """-> (samples, matcher, tr). Датасет кэшируется в reports/fusion_dataset_*.pkl."""
    cache = EVAL_REPORTS / f'fusion_dataset_i{image_k}_t{text_k}_e{int(use_e5)}.pkl'
    if cache.exists() and not rebuild:
        d = pickle.loads(cache.read_bytes())
        print(f'датасет из кэша: {len(d["samples"])} фото ({cache.name})')
        return d['samples'], None, types.SimpleNamespace(use_e5=d['use_e5'])
    cache_img = json.loads((EVAL_REPORTS / 'image_topk.json').read_text(encoding='utf-8'))['images']
    report = json.loads((DATA_REPORTS / 'ocr_rerank_report.json').read_text(encoding='utf-8'))
    tr = TextRetriever(cfg, use_e5=use_e5)
    matcher = tr.matcher
    samples, skipped = [], 0
    for p in report['predictions']:
        name, true = p['image'], p['true_slug']
        ce = cache_img.get(name)
        if not ce:
            skipped += 1
            continue
        fields = p.get('ocr_fields') or {}
        image_slugs, image_scores = _image_branch(ce)
        image_top1 = image_slugs[0] if image_slugs else None
        rows, cos_map = tr.full_scores(fields)
        text_slugs = [r['slug'] for r in rows[:text_k]]
        raw_text = fields.get('raw_text') or ''
        raw_map = tr.raw_cos_map(raw_text)
        pool = RF.build_pool(image_slugs, text_slugs, image_k=image_k, text_k=text_k)
        slugs, X = RF.features_for_pool(cfg, matcher, fields, pool, image_scores,
                                        image_top1, cos_map, raw_map=raw_map, raw_text=raw_text)
        y = np.asarray([1.0 if s == true else 0.0 for s in slugs], dtype=np.float32)
        samples.append({'image': name, 'true': true, 'image_top1': image_top1,
                        'slugs': slugs, 'X': X, 'y': y,
                        'true_in_pool': int(true in slugs)})
    print(f'датасет: {len(samples)} фото (пропущено {skipped}), '
          f'image_k={image_k} text_k={text_k} e5={tr.use_e5}')
    cache.write_bytes(pickle.dumps({'samples': samples, 'use_e5': tr.use_e5}))
    return samples, matcher, tr


def top1_eval(ranker, samples) -> dict:
    """Ранжировать каждый пул -> accuracy top-1 + fixed/broken/switched vs image top-1."""
    acc = fixed = broken = switched = 0
    for s in samples:
        if not s['slugs']:
            continue
        pred = s['slugs'][int(np.argmax(ranker.score(s['X'])))]
        acc += int(pred == s['true'])
        switched += int(pred != s['image_top1'])
        broken += int(s['image_top1'] == s['true'] and pred != s['true'])
        fixed += int(s['image_top1'] != s['true'] and pred == s['true'])
    n = max(len(samples), 1)
    return {'n': len(samples), 'accuracy': round(acc / n, 4), 'switched': switched,
            'fixed': fixed, 'broken': broken}


def fit_ranker(samples, l2: float, iters: int) -> RF.LogisticRanker:
    X = np.concatenate([s['X'] for s in samples]) if samples else np.zeros((0, RF.N_FEATURES))
    y = np.concatenate([s['y'] for s in samples]) if samples else np.zeros((0,))
    return RF.LogisticRanker().fit(X, y, l2=l2, iters=iters)


def diagnose(samples) -> None:
    """Разбор «починимых» кейсов: где сигнал есть, а ранкер его не берёт."""
    idx = {n: i for i, n in enumerate(RF.FEATURE_NAMES)}
    fix = [s for s in samples if s['image_top1'] != s['true'] and s['true_in_pool']]
    in_img = [s for s in fix if s['image_top1'] in s['slugs'] and
              (s['X'][:, idx['in_image_pool']] > 0.5).sum() > 0]
    print(f'починимых (image wrong, true в пуле): {len(fix)}')
    print(f'  из них true пришёл из image-пула (rank<=image_k): '
          f'{sum(1 for s in fix if s["X"][:, idx["in_image_pool"]].max() > 0.5)}')
    for f in ('image_cos', 'e5_cos', 'csv_conf', 'year_match', 'color_match',
              'grape_match', 'sugar_match', 'title_match', 'winery_match'):
        i = idx[f]
        acc = sum(1 for s in fix
                  if s['slugs'][int(np.argmax(s['X'][:, i]))] == s['true'])
        print(f'  oracle argmax[{f:12s}]: {acc}/{len(fix)}')


def sweep(samples) -> None:
    """Сравнить простые скореры/гиперпараметры LR на кэшированном датасете."""
    idx = {n: i for i, n in enumerate(RF.FEATURE_NAMES)}

    def ev(score_fn):
        acc = broken = fixed = switch = 0
        for s in samples:
            pred = s['slugs'][int(np.argmax(score_fn(s)))]
            acc += pred == s['true']
            switch += pred != s['image_top1']
            broken += s['image_top1'] == s['true'] and pred != s['true']
            fixed += s['image_top1'] != s['true'] and pred == s['true']
        n = max(len(samples), 1)
        return round(acc / n, 4), switch, fixed, broken

    def col(s, i):
        return s['X'][:, i]

    print('baseline image top1 :', round(sum(s['image_top1'] == s['true'] for s in samples) / len(samples), 4))
    print('argmax csv_conf     :', ev(lambda s: col(s, idx['csv_conf'])))
    print('argmax e5_cos       :', ev(lambda s: col(s, idx['e5_cos'])))
    for w in (0.2, 0.3, 0.5, 0.8):
        print(f'csv + {w}*e5        :', ev(lambda s, w=w: col(s, idx['csv_conf']) + w * col(s, idx['e5_cos'])))
    for w in (0.3, 0.6, 1.0, 1.5):
        print(f'csv + {w}*top1_prior:', ev(lambda s, w=w: col(s, idx['csv_conf']) + w * col(s, idx['is_image_top1'])))

    def margin_rule(s, m):
        c = col(s, idx['csv_conf'])
        t1 = int(np.argmax(col(s, idx['is_image_top1'])))
        best = int(np.argmax(c))
        pick = best if (best != t1 and c[best] - c[t1] >= m) else t1
        return np.asarray([1.0 if i == pick else 0.0 for i in range(len(c))])

    for m in (0.0, 0.05, 0.10, 0.15, 0.20):
        print(f'margin rule m={m:<4} :', ev(lambda s, m=m: margin_rule(s, m)))
    for l2, it in ((1.0, 8000), (0.1, 8000), (0.01, 15000)):
        r = fit_ranker(samples, l2, it)
        print(f'LR l2={l2} it={it:<5}:', ev(lambda s, r=r: r.score(s['X'])))


def eval_weights(samples, w):
    """Точность/switch/fixed/broken. ``w`` — массив весов ИЛИ объект с ``.score``."""
    scorer = (lambda X: w.score(X)) if hasattr(w, 'score') else (lambda X: X @ w)
    acc = broken = fixed = sw = 0
    for s in samples:
        pred = s['slugs'][int(np.argmax(scorer(s['X'])))]
        acc += pred == s['true']
        sw += pred != s['image_top1']
        broken += s['image_top1'] == s['true'] and pred != s['true']
        fixed += s['image_top1'] != s['true'] and pred == s['true']
    n = max(len(samples), 1)
    return round(acc / n, 4), sw, fixed, broken


def hill_climb(samples, seed=None, rounds: int = 5,
               deltas=(0.05, 0.1, 0.25, 0.5, 1.0)):
    """Greedy hill-climb линейных весов, максимизируя accuracy (тай-брейк: меньше broken).

    Точность — порядковая метрика, log-loss её оптимизирует косвенно, поэтому
    pointwise-LR недобирает (см. ``--sweep``). Hill-climb даёт лучший операндум.
    """
    F = RF.N_FEATURES
    w = np.zeros(F) if seed is None else np.asarray(seed, dtype=np.float64).copy()
    best = eval_weights(samples, w)

    def key(m):
        return (m[0], -m[3])

    for _ in range(rounds):
        improved = False
        for i in range(F):
            for d in deltas:
                for sgn in (1.0, -1.0):
                    w2 = w.copy()
                    w2[i] += sgn * d
                    m = eval_weights(samples, w2)
                    if key(m) > key(best):
                        best, w, improved = m, w2, True
        if not improved:
            break
    return w, best


def weights_ranker(w) -> RF.LogisticRanker:
    """Линейные веса -> ранкер без стандартизации (score = X @ w)."""
    return RF.LogisticRanker(coef=w, intercept=0.0,
                             mean=[0.0] * RF.N_FEATURES, std=[1.0] * RF.N_FEATURES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='P0-3: обучение fusion-ранкера (2-fold CV)')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--image-k', type=int, default=30, help='ширина визуального пула')
    ap.add_argument('--text-k', type=int, default=10, help='ширина текстового пула')
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--no-e5', action='store_true')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--rebuild', action='store_true', help='пересобрать датасет (игнор кэш)')
    ap.add_argument('--diagnose', action='store_true', help='разбор сигнала на починимых кейсах')
    ap.add_argument('--sweep', action='store_true', help='сравнить скореры/LR-гиперпараметры')
    ap.add_argument('--tag', default='fusion')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    samples, matcher, tr = load_dataset(cfg, args.image_k, args.text_k,
                                        use_e5=not args.no_e5, rebuild=args.rebuild)
    if args.diagnose:
        diagnose(samples)
        return 0
    if args.sweep:
        sweep(samples)
        return 0
    n = len(samples)
    image_top1_acc = round(sum(1 for s in samples if s['image_top1'] == s['true']) / max(n, 1), 4)
    pool_cov = round(sum(s['true_in_pool'] for s in samples) / max(n, 1), 4)
    print(f'baseline image top-1: {image_top1_acc} | истина в пуле: {pool_cov}')

    default_eval = top1_eval(RF.default_ranker(), samples)
    print('дефолтные веса:', default_eval)

    idx = {nm: i for i, nm in enumerate(RF.FEATURE_NAMES)}
    seed = np.zeros(RF.N_FEATURES)
    seed[idx['csv_conf']] = 1.0
    seed[idx['is_image_top1']] = 0.3
    folds = [[s for i, s in enumerate(samples) if i % 2 == f] for f in (0, 1)]

    lr_cv = [eval_weights(folds[f], fit_ranker(folds[1 - f], args.l2, args.iters)) for f in (0, 1)]
    hc_cv = [eval_weights(folds[f], hill_climb(folds[1 - f], seed=seed)[0]) for f in (0, 1)]
    lr_acc = round(sum(m[0] for m in lr_cv) / 2, 4)
    hc_acc = round(sum(m[0] for m in hc_cv) / 2, 4)
    print(f'2-fold CV  logreg     : acc={lr_acc} {lr_cv}')
    print(f'2-fold CV  hill-climb : acc={hc_acc} {hc_cv}')

    lr_full = fit_ranker(samples, args.l2, args.iters)
    w_hc, hc_full = hill_climb(samples, seed=seed)
    lr_full_m = eval_weights(samples, lr_full)
    if hc_acc >= lr_acc:
        method, model, full_m = 'hill_climb', weights_ranker(w_hc), hc_full
    else:
        method, model, full_m = 'logreg', lr_full, lr_full_m
    if not args.no_save:
        model.save(FUSION_WEIGHTS)
        print('веса ->', FUSION_WEIGHTS)
    print(f'выбран метод: {method} | full-fit: {full_m}')

    summary = {
        'n': n, 'image_k': args.image_k, 'text_k': args.text_k, 'e5': tr.use_e5,
        'baseline_image_top1': image_top1_acc, 'true_in_pool': pool_cov, 'method': method,
        'default_weights': default_eval,
        'cv': {'logreg': {'accuracy': lr_acc, 'folds': lr_cv},
               'hill_climb': {'accuracy': hc_acc, 'folds': hc_cv}},
        'full_fit': {'logreg': lr_full_m, 'hill_climb': hc_full},
        'coef': dict(zip(model.feature_names, [round(float(c), 4) for c in model.coef])),
    }
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    (EVAL_REPORTS / f'{args.tag}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    with open(EVAL_REPORTS / f'{args.tag}_predictions.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['image', 'true_slug', 'image_top1', 'fusion_top1', 'hit', 'broken'])
        for s in samples:
            pred = s['slugs'][int(np.argmax(model.score(s['X'])))] if s['slugs'] else None
            wr.writerow([s['image'], s['true'], s['image_top1'], pred,
                         int(pred == s['true']),
                         int(s['image_top1'] == s['true'] and pred != s['true'])])
    print('отчёт ->', EVAL_REPORTS / f'{args.tag}.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

