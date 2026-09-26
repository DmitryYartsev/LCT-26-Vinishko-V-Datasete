# -*- coding: utf-8 -*-
"""Метрики качества retrieval: отдельный, заменяемый метод подсчёта на тесте.

Сейчас метрика одна — **точное совпадение slug** (accuracy@1/@k): для каждого
тестового кропа берём ближайший эталон из референсной базы и сравниваем slug'и.
Это ТОЧКА ЗАМЕНЫ: когда понадобится другая метрика (partial match по названию,
F1, mAP, near-dup серии и т.п.) — достаточно переписать ``ExactSlugAccuracy.__call__``
(или подсунуть свою реализацию класса с тем же интерфейсом), остальной пайплайн
(``evaluate.run_test``, ``MetricModel.on_validation_epoch_end``) не меняется.

Интерфейс метрики::

    metric(query_emb, query_slugs, gallery_emb, gallery_slugs) -> dict

где ``*_emb`` — [N, D] тензоры (L2-нормируются внутри), ``*_slugs`` — список строк.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- базовые
def cosine_scores(query_emb: torch.Tensor, gallery_emb: torch.Tensor) -> torch.Tensor:
    """[Q, D] × [G, D] -> [Q, G] косинусные близости (векторы нормируются внутри)."""
    q = F.normalize(query_emb.float(), dim=-1)
    g = F.normalize(gallery_emb.float(), dim=-1)
    return q @ g.t()


def topk_indices(scores: torch.Tensor, k: int = 5) -> torch.Tensor:
    """[Q, G] -> [Q, min(k, G)] индексов ближайших эталонов (по убыванию близости)."""
    k = max(1, min(int(k), scores.shape[1]))
    return scores.topk(k, dim=1).indices


def nearest_slugs(query_emb: torch.Tensor, gallery_emb: torch.Tensor,
                  gallery_slugs: list[str], k: int = 5) -> list[list[str]]:
    """Для каждого query — список slug'ов эталонов от ближайшего к дальнему."""
    idx = topk_indices(cosine_scores(query_emb, gallery_emb), k=k)
    return [[gallery_slugs[i] for i in row.tolist()] for row in idx]


# ------------------------------------------------------- метрика теста (accuracy)
@dataclass
class ExactSlugAccuracy:
    """Точные совпадения slug + accuracy (черновая метрика теста).

    Parameters
    ----------
    k : int
        До какого ранга считаем попадание (accuracy@1 и accuracy@k).
    only_covered : bool
        True — считать accuracy только по query, чьи slug есть в референсной базе
        (для остальных попадание невозможно по построению); в отчёте всегда
        отдельно возвращаются ``coverage`` и ``n_covered``.
    """

    k: int = 5
    only_covered: bool = True
    name: str = "exact_slug_accuracy"
    last_predictions: list[dict] = field(default_factory=list, repr=False)

    def __call__(self, query_emb: torch.Tensor, query_slugs: list[str],
                 gallery_emb: torch.Tensor, gallery_slugs: list[str],
                 query_paths: list[str] | None = None) -> dict:
        n_q = len(query_slugs)
        gallery_emb = gallery_emb.reshape(len(gallery_slugs), -1)
        k_eff = max(1, min(int(self.k), gallery_emb.shape[0]))
        top = nearest_slugs(query_emb, gallery_emb, gallery_slugs, k=k_eff)

        gallery_set = set(gallery_slugs)
        covered = [s in gallery_set for s in query_slugs]
        paths = query_paths or [""] * n_q

        hit1 = hitk = 0
        self.last_predictions = []
        for i, (slug, preds) in enumerate(zip(query_slugs, top)):
            ok1 = bool(preds) and preds[0] == slug
            okk = slug in preds
            hit1 += int(ok1)
            hitk += int(okk)
            self.last_predictions.append({
                "image_path": paths[i], "true_slug": slug,
                "pred_slug": preds[0] if preds else None, "topk": preds,
                "hit@1": int(ok1), "hit@k": int(okk), "covered": int(covered[i]),
            })

        idx = [i for i, c in enumerate(covered) if c] if self.only_covered else list(range(n_q))
        n_eval = len(idx)
        acc1 = hit1 / n_eval if n_eval else 0.0
        acck = hitk / n_eval if n_eval else 0.0

        return {
            "metric": self.name,
            "acc@1": round(acc1, 4),
            f"acc@{k_eff}": round(acck, 4),
            "k": k_eff,
            "n_queries": n_q,
            "n_queries_covered": sum(covered),
            "coverage": round(sum(covered) / n_q, 4) if n_q else 0.0,
            "n_evaluated": n_eval,
            "only_covered": self.only_covered,
            "gallery_size": len(gallery_slugs),
            "n_gallery_slugs": len(gallery_set),
        }


# --------------------------------------------------------------------- сводка
def format_report(metrics: dict, title: str = "ТЕСТ") -> str:
    """Текстовая сводка метрик для stdout/логов."""
    lines = [f"=== {title} ==="]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
