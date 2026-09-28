# -*- coding: utf-8 -*-
"""Калиброванный fusion-ранкер над объединённым пулом кандидатов (P0-3).

Заменяет жёсткий гейт ``ocr_rerank.decide()`` (ручной ``decision_margin``):
ранжирует пул ``image top-K ∪ text top-T`` по признакам и выбирает #1.

Почему нельзя просто «расширить пул» — см. ``Plans/План до 90.md`` §0: с текущим
хард-гейтом более широкий пул добавляет ``broken`` (метрика падает). Нужен
обучаемый ранкер.

Признаки на пару (запрос, кандидат) — ``FEATURE_NAMES``. Модель — pointwise
логистическая регрессия на numpy (без sklearn): признаки стандартизуются,
дисбаланс классов компенсируется весом позитивов. Обучение/2-fold CV — в
``ML evaluation/train_fusion.py``; веса кладутся в ``paths.FUSION_WEIGHTS``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import ocr_rerank as _o

FEATURE_NAMES = [
    'image_cos',        # косинус image-retrieval (0, если кандидата нет в image-пуле)
    'in_image_pool',    # кандидат пришёл из визуального шортлиста
    'is_image_top1',    # совпадает с текущим ответом retrieval
    'e5_cos',           # косинус текстового эмбеддера (OCR-поля ⟷ карточка)
    'csv_conf',         # уверенность CsvMatcher
    'csv_margin',       # csv_conf − csv_conf(image top1)
    'winery_match',
    'grape_match',
    'color_match',
    'sugar_match',
    'year_match',
    'title_match',
    'visual_sim',       # косинус кропа запроса и фото каталога (0.5 если нет)
    'raw_e5',           # косинус СЫРОЙ транскрипции этикетки ⟷ карточка (P0-4)
    'raw_match',        # покрытие токенов raw_text в title+slug записи (P0-4)
    'line_match',       # покрытие токенов линейки (line) в title+slug записи
    'spark_match',      # класс: игристое/шампанское ⟷ тихое
]
N_FEATURES = len(FEATURE_NAMES)


def _cover(q, t):
    return _o._token_cover(q, t) if (q and t) else None


def _sym(q, t):
    return _o._sym_cover(q, t) if (q and t) else None


def query_features(cfg, matcher, fields: dict) -> dict:
    """Предвычисляемое на ЗАПРОС (одинаково для всех кандидатов) — нормированные поля."""
    return matcher._query(fields)


def candidate_features(q: dict, entry: dict, image_cos: float, in_image_pool: int,
                       image_top1: str, csv_conf: float, csv_conf_top1: float,
                       e5_cos: float, visual_sim: float,
                       raw_e5: float = 0.0, raw_match: float | None = None,
                       line_match: float | None = None, spark_match: float | None = None) -> np.ndarray:
    """Вектор признаков для пары (запрос ``q``, запись каталога ``entry``)."""
    winery = _sym(q.get('winery'), entry['winery_tok'])
    grape = _sym(q.get('grape'), entry['grape_tok'])
    color = _cover(q.get('color'), entry['color_tok'])
    sugar = _cover(q.get('wine_type'), entry['sweet_tok'])
    title = _cover(q.get('additional_text'), entry['title_tok'])
    if q.get('year') and entry['year']:
        year = _o.year_sim(entry['year'], q['year'])
    else:
        year = None
    credit = 0.5  # отсутствующее у записи поле — нейтрально (как в CsvMatcher)
    vals = [
        float(image_cos),
        float(in_image_pool),
        float(entry['slug'] == image_top1),
        float(e5_cos),
        float(csv_conf),
        float(csv_conf) - float(csv_conf_top1),
        float(winery if winery is not None else credit),
        float(grape if grape is not None else credit),
        float(color if color is not None else credit),
        float(sugar if sugar is not None else credit),
        float(year if year is not None else credit),
        float(title if title is not None else credit),
        float(visual_sim),
        float(raw_e5),
        float(raw_match if raw_match is not None else 0.0),
        float(line_match if line_match is not None else 0.0),
        float(spark_match if spark_match is not None else 0.0),
    ]
    return np.asarray(vals, dtype=np.float32)


class LogisticRanker:
    """Pointwise логистическая регрессия (numpy) с L2 и стандартизацией признаков."""

    def __init__(self, feature_names=None, coef=None, intercept: float = 0.0,
                 mean=None, std=None):
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.coef = None if coef is None else np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    # ---------------------------------------------------------------- стандартизация
    def _scale(self, X):
        if self.mean is None or self.std is None:
            return X
        return (X - self.mean) / self.std

    def fit(self, X, y, l2: float = 1.0, iters: int = 4000, lr: float = 0.2):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        Xs = self._scale(X)
        n, d = Xs.shape
        pos = max(y.sum(), 1.0)
        neg = max(n - pos, 1.0)
        w_pos = 0.5 * n / pos                     # балансируем классы
        w_neg = 0.5 * n / neg
        sw = np.where(y > 0.5, w_pos, w_neg)
        w = np.zeros(d)
        b = 0.0
        for _ in range(int(iters)):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = (p - y) * sw
            grad_w = Xs.T @ err / n + l2 * w / n
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b
        self.coef, self.intercept = w, float(b)
        return self

    def score(self, X):
        Xs = self._scale(np.asarray(X, dtype=np.float64))
        return 1.0 / (1.0 + np.exp(-np.clip(Xs @ self.coef + self.intercept, -30, 30)))

    # ---------------------------------------------------------------- персистенция
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            'feature_names': self.feature_names, 'coef': list(map(float, self.coef)),
            'intercept': self.intercept, 'mean': list(map(float, self.mean)),
            'std': list(map(float, self.std))}, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(d.get('feature_names'), d['coef'], d.get('intercept', 0.0),
                   d.get('mean'), d.get('std'))



# «Резервные» веса: применяются, пока не обучен ``fusion_weights.json``.
# Признаки сырые (без стандартизации) — коэффициенты отражают ожидаемую важность:
# точное совпадение года/сорта/цвета/типа и близость текста важнее косинуса image.
DEFAULT_COEF = {
    'image_cos': 3.0, 'in_image_pool': 0.2, 'is_image_top1': 0.8,
    'e5_cos': 2.0, 'csv_conf': 4.0, 'csv_margin': 2.0,
    'winery_match': 1.5, 'grape_match': 1.5, 'color_match': 1.5,
    'sugar_match': 1.5, 'year_match': 2.5, 'title_match': 1.0,
    'visual_sim': 0.5, 'raw_e5': 2.0, 'raw_match': 1.5,
    'line_match': 2.0, 'spark_match': 1.0,
}
DEFAULT_INTERCEPT = -8.0


def default_ranker() -> LogisticRanker:
    coef = [DEFAULT_COEF[n] for n in FEATURE_NAMES]
    return LogisticRanker(coef=coef, intercept=DEFAULT_INTERCEPT,
                          mean=[0.0] * N_FEATURES, std=[1.0] * N_FEATURES)


def load_or_default(path) -> LogisticRanker:
    try:
        return LogisticRanker.load(path)
    except Exception:  # noqa: BLE001 — нет весов -> работаем на дефолтных
        return default_ranker()


def build_pool(image_slugs, text_slugs, image_k: int = 30, text_k: int = 10) -> list:
    """Объединённый пул: image top-K (в порядке) + text top-T (которых нет в image)."""
    pool = list(dict.fromkeys(list(image_slugs or [])[:image_k]))
    for s in (text_slugs or [])[:text_k]:
        if s not in pool:
            pool.append(s)
    return pool


def features_for_pool(cfg, matcher, fields: dict, pool, image_scores: dict,
                      image_top1, text_scores: dict, visual_sims: dict | None = None,
                      raw_map: dict | None = None, raw_text: str | None = None):
    """Собрать матрицу признаков для пула. -> (slugs, X[N, F]). Общий код train/infer."""
    visual_sims = visual_sims or {}
    raw_map = raw_map or {}
    q = matcher._query(fields)
    if any(q.values()):
        conf_by = {c['slug']: float(c['confidence']) for c in matcher.match(fields, only=list(pool))}
    else:
        conf_by = {}
    csv_top1 = conf_by.get(image_top1, 0.0)
    img_set = set(image_scores or {})
    raw_tok = _o._norm_tokens(raw_text) if raw_text else None
    slugs, feats = [], []
    for slug in pool:
        entry = matcher._by_slug.get(slug)
        if entry is None:
            continue
        raw_targets = (entry['title_tok'] or []) + (entry['slug_tok'] or [])
        raw_match = _o._token_cover(raw_tok, raw_targets) if (raw_tok and raw_targets) else None
        line_match = _o._token_cover(q.get('line'), raw_targets) if (q.get('line') and raw_targets) else None
        spark_match = (_o._token_cover(q.get('sparkling'), entry.get('spark_tok') or [])
                       if q.get('sparkling') else None)
        feats.append(candidate_features(
            q, entry,
            image_cos=float((image_scores or {}).get(slug, 0.0)),
            in_image_pool=int(slug in img_set),
            image_top1=image_top1 if image_top1 else '',
            csv_conf=conf_by.get(slug, 0.0), csv_conf_top1=csv_top1,
            e5_cos=float((text_scores or {}).get(slug, 0.0)),
            visual_sim=float(visual_sims.get(slug, 0.5)),
            raw_e5=float(raw_map.get(slug, 0.0)), raw_match=raw_match,
            line_match=line_match, spark_match=spark_match))
        slugs.append(slug)
    if not slugs:
        return [], np.zeros((0, N_FEATURES), dtype=np.float32)
    return slugs, np.stack(feats)


def rerank_pool(cfg, matcher, ranker, fields: dict, pool, image_scores: dict,
                image_top1, text_scores: dict, visual_sims: dict | None = None,
                raw_map: dict | None = None, raw_text: str | None = None) -> list:
    """Ранжировать пул кандидатов обученным ранкером. -> [{slug, prob, x}]."""
    slugs, X = features_for_pool(cfg, matcher, fields, pool, image_scores,
                                 image_top1, text_scores, visual_sims, raw_map, raw_text)
    if not slugs:
        return []
    probs = ranker.score(X)
    rows = [{'slug': s, 'x': x, 'prob': float(p)} for s, x, p in zip(slugs, X, probs)]
    rows.sort(key=lambda r: r['prob'], reverse=True)
    return rows


class FusionReranker:
    """Оркестратор: image top-K + текстовый пул -> fusion-ранжирование -> top-1."""

    def __init__(self, cfg, matcher, text_retriever=None, weights_path=None):
        self.cfg = cfg
        self.matcher = matcher
        self.text = text_retriever
        self.ranker = load_or_default(weights_path) if weights_path else default_ranker()

    def rerank(self, fields: dict, image_results, query_crop=None, verifier=None,
               image_k: int = 30, text_k: int = 10) -> dict:
        """image_results: [{slug, score}] best-first (ответ prod-ветки). -> dict с пулом."""
        image_scores = {r['slug']: float(r['score']) for r in (image_results or [])}
        image_top1 = image_results[0]['slug'] if image_results else None
        text_rows = self.text.search(fields, k=text_k) if self.text else []
        text_scores = {r['slug']: r['e5_cos'] for r in text_rows}
        raw_text = (fields or {}).get('raw_text') or ''
        raw_map = self.text.raw_cos_map(raw_text) if self.text else {}
        pool = build_pool([r['slug'] for r in (image_results or [])],
                          [r['slug'] for r in text_rows], image_k=image_k, text_k=text_k)
        visual = {}
        if verifier is not None and query_crop is not None:
            for slug in pool:
                entry = self.matcher._by_slug.get(slug)
                if not entry:
                    continue
                try:
                    visual[slug] = verifier.verify(query_crop, entry['photo_file'], slug)
                except Exception:  # noqa: BLE001
                    pass
        ranked = rerank_pool(self.cfg, self.matcher, self.ranker, fields, pool,
                             image_scores, image_top1, text_scores, visual, raw_map, raw_text)
        return {'final_slug': ranked[0]['slug'] if ranked else image_top1,
                'image_top1': image_top1, 'pool_size': len(pool),
                'top_candidates': [{'slug': r['slug'], 'prob': round(r['prob'], 4)} for r in ranked[:8]]}
