# -*- coding: utf-8 -*-
"""Лоссы для metric learning: triplet-лоссы (batch-hard / semi-hard / all).

Задача — стянуть эмбеддинги одинаковых этикеток (один slug) и расталкивать разные.
Все лоссы работают на L2-нормированных (или произвольных) эмбеддингах и метках-индексах
классов, поэтому их можно применять и к батчу, и ко всему val-сету целиком
(см. ``MetricModel.on_validation_epoch_end``).

Дистанции:
  cosine     — 1 - cos(a, b), диапазон [0, 2] (деф: так считает и поиск в сервисе);
  euclidean  — обычная евклидова дистанция (на нормированных векторах монотонна с cosine).

Майнинг:
  batch_hard — для каждого якоря самый далёкий позитив и самый близкий негатив;
  semi_hard  — самый далёкий позитив + ближайший негатив, который дальше позитива
               (нарушающие margin; у кого таких нет — вклад 0);
  all        — все валидные тройки (медленно, O(B^3) по памяти; для отладки).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DISTANCES = ("cosine", "euclidean")
MININGS = ("batch_hard", "semi_hard", "all")


def _num(x) -> float:
    """Тензор -> float без графа (только для статистики логов)."""
    return float(x.detach().cpu()) if hasattr(x, "detach") else float(x)


def pairwise_distance(emb: torch.Tensor, distance: str = "cosine") -> torch.Tensor:
    """[B, D] -> [B, B] матрица попарных дистанций (диагональ = 0)."""
    if distance == "cosine":
        # эмбеддинги могут быть не нормированы (normalize=False) — нормируем локально
        z = F.normalize(emb, dim=-1)
        return 1.0 - z @ z.t()
    if distance == "euclidean":
        return torch.cdist(emb, emb, p=2)
    raise ValueError(f"неизвестная дистанция {distance!r}, ждём одну из {DISTANCES}")


def _triplet_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """По меткам [B] -> (pos_mask, neg_mask, valid_anchor) без учёта диагонали."""
    labels = labels.reshape(-1)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    pos = same & ~eye
    neg = ~same
    valid = pos.any(dim=1) & neg.any(dim=1)   # якорь без пары не даёт тройки
    return pos, neg, valid


def triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3,
                 distance: str = "cosine", mining: str = "batch_hard",
                 ) -> tuple[torch.Tensor, dict]:
    """Triplet loss + статистика для логов.

    Returns
    -------
    (loss, stats) : (Tensor-скаляр, {'frac_valid', 'd_pos', 'd_neg', 'loss'})
        Если валидных троек в батче нет (в батче нет двух объектов одного класса),
        возвращает 0-тензор с графом (чтобы backward не падал) и frac_valid=0.
    """
    if mining not in MININGS:
        raise ValueError(f"неизвестный майнинг {mining!r}, ждём одну из {MININGS}")
    zero = emb.sum() * 0.0
    empty = {"frac_valid": 0.0, "d_pos": 0.0, "d_neg": 0.0, "loss": 0.0}
    if labels.reshape(-1).numel() < 3:
        return zero, dict(empty)

    d = pairwise_distance(emb, distance)
    pos, neg, valid = _triplet_masks(labels)
    if not bool(valid.any()):
        return zero, dict(empty)

    if mining == "all":
        # все тройки (a, p, n): pos[a, p] и neg[a, n]; [B, B, B] по памяти — для отладки
        trip = pos[:, :, None] & neg[:, None, :] & valid[:, None, None]
        if not bool(trip.any()):
            return zero, dict(empty)
        d_ap = d[:, :, None]                     # дистанция якорь-позитив  [B, B, 1]
        d_an = d[:, None, :]                     # дистанция якорь-негатив  [B, 1, B]
        losses = F.relu(d_ap - d_an + margin)[trip]   # broadcast до [B, B, B]
        return losses.mean(), {
            "frac_valid": _num(valid.float().mean()),
            "d_pos": _num(d[pos].mean()) if bool(pos.any()) else 0.0,
            "d_neg": _num(d[neg].mean()) if bool(neg.any()) else 0.0,
            "loss": _num(losses.mean()),
        }

    # самый далёкий позитив (max по позитивам) и самый близкий негатив (min по негативам)
    d_pos = d.masked_fill(~pos, float("-inf")).max(dim=1).values
    cand = neg & (d > d_pos[:, None]) if mining == "semi_hard" else neg
    d_neg = d.masked_fill(~cand, float("inf")).min(dim=1).values
    ok = valid & torch.isfinite(d_neg) & torch.isfinite(d_pos)
    if not bool(ok.any()):
        return zero, dict(empty)
    losses = F.relu(d_pos[ok] - d_neg[ok] + margin)
    loss = losses.mean()
    stats = {
        "frac_valid": _num(ok.float().mean()),
        "d_pos": _num(d_pos[ok].mean()),
        "d_neg": _num(d_neg[ok].mean()),
        "loss": _num(loss),
    }
    return loss, stats


def pairwise_distance_cross(a: torch.Tensor, b: torch.Tensor,
                            distance: str = "cosine") -> torch.Tensor:
    """[A, D] × [R, D] -> [A, R] матрица дистанций."""
    if distance == "cosine":
        return 1.0 - F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()
    if distance == "euclidean":
        return torch.cdist(a, b, p=2)
    raise ValueError(f"неизвестная дистанция {distance!r}, ждём одну из {DISTANCES}")


def cross_modal_triplet_loss(anchor_emb: torch.Tensor, anchor_labels, ref_emb: torch.Tensor,
                             ref_labels, margin: float = 0.3, distance: str = "cosine",
                             ) -> tuple[torch.Tensor, dict]:
    """Triplet «кроп против референсной базы» (cross-modal).

    Якорь — кроп из train/val, позитивы/негативы — только картинки референсной базы
    (эталоны каталога). Для каждого якоря берём самый далёкий позитив (эталон того же
    slug'а) и самый близкий негатив (эталон другого вина). Именно это поведение
    эксплуатирует поиск в сервисе, поэтому лосс удобно мониторить отдельно
    (``losses/val_ref``).

    Метки — индексы классов; ``-1`` = «нет такого класса в трейне» (якорь пропускается,
    но как негатив для других якорей такой эталон годится — это просто другое вино).
    """
    zero = anchor_emb.sum() * 0.0
    empty = {"frac_valid": 0.0, "d_pos": 0.0, "d_neg": 0.0, "loss": 0.0, "n_valid": 0}
    if anchor_emb.numel() == 0 or ref_emb.numel() == 0:
        return zero, dict(empty)

    a_lab = torch.as_tensor(anchor_labels, dtype=torch.long,
                            device=anchor_emb.device).reshape(-1)
    r_lab = torch.as_tensor(ref_labels, dtype=torch.long, device=ref_emb.device).reshape(-1)
    known = a_lab[:, None] >= 0
    pos = known & (a_lab[:, None] == r_lab[None, :])
    neg = known & (r_lab[None, :] >= 0) & (a_lab[:, None] != r_lab[None, :])
    valid = pos.any(dim=1) & neg.any(dim=1)
    if not bool(valid.any()):
        return zero, dict(empty)

    d = pairwise_distance_cross(anchor_emb, ref_emb, distance)
    d_pos = d.masked_fill(~pos, float("-inf")).max(dim=1).values
    d_neg = d.masked_fill(~neg, float("inf")).min(dim=1).values
    ok = valid & torch.isfinite(d_pos) & torch.isfinite(d_neg)
    if not bool(ok.any()):
        return zero, dict(empty)

    losses = F.relu(d_pos[ok] - d_neg[ok] + margin)
    return losses.mean(), {
        "frac_valid": _num(ok.float().mean()),
        "d_pos": _num(d_pos[ok].mean()),
        "d_neg": _num(d_neg[ok].mean()),
        "loss": _num(losses.mean()),
        "n_valid": int(_num(ok.sum())),
    }


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3,

                            distance: str = "cosine", ) -> tuple[torch.Tensor, dict]:
    """Удобный алиас: batch-hard triplet loss."""
    return triplet_loss(emb, labels, margin=margin, distance=distance, mining="batch_hard")


class CosineAuxClassifier(nn.Module):
    """Вспомогательный классификатор (cosine-логиты) для aux-CE по slug'ам трейна.

    Тройки — основной сигнал; CE-голова (``ce_weight > 0``) стабилизирует обучение
    на старте и помогает, когда в батче мало позитивных пар.
    """

    def __init__(self, n_classes: int, embed_dim: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)
        self.scale = scale

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        z = F.normalize(emb, dim=-1)
        return self.scale * (z @ F.normalize(self.weight, dim=-1).t())
