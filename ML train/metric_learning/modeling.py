# -*- coding: utf-8 -*-
"""Энкодер для metric learning: HF-модель (SigLIP2/DINOv2/…) + проекционная голова.

Отличие от ``ML service/encoder.py``: тот обслуживает инференс (``torch.no_grad``,
eval), а здесь модель — часть вычислительного графа, её дообучаем. Логика извлечения
признаков повторяет ``encoder._feats`` (get_image_features -> pooler_output ->
mean по токенам), чтобы после обучения вектор доставался так же, как в сервисе.

Бэкбоны:
  * HF-id или локальная папка (``models/siglip2-base-patch16-256``) — ``AutoModel``;
  * ``tiny`` — случайный маленький SigLIP-vision без скачивания (для смоук-тестов,
    см. ``make_dummy_data.py``): быстро, офлайн, никаких весов в git.
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from losses import CosineAuxClassifier, cross_modal_triplet_loss, triplet_loss

TINY_NAMES = {"tiny", "debug", "random"}


# ------------------------------------------------------------------ backbone
@dataclass
class BackboneSpec:
    """Загруженный бэкбон + его процессор (препроцесс обязан совпадать с сервисом)."""

    name: str
    model: nn.Module
    processor: Optional[object]
    hidden_size: int
    image_size: int

    @property
    def is_tiny(self) -> bool:
        return self.name in TINY_NAMES


def _tiny_backbone(name: str = "tiny") -> BackboneSpec:
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(image_size=64, patch_size=16, num_channels=3,
                             hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=64)
    model = SiglipVisionModel(cfg)
    print(f"[model] игрушечный бэкбон '{name}' (dim={cfg.hidden_size}, "
          f"img={cfg.image_size}) — для смоук-тестов, без скачивания весов", flush=True)
    return BackboneSpec(name=name, model=model, processor=None,
                        hidden_size=cfg.hidden_size, image_size=cfg.image_size)


def _cfg_image_size(model: nn.Module, processor) -> int:
    """Размер входа: из конфига модели, иначе из процессора, иначе 224."""
    for obj in (getattr(model, "config", None),
                getattr(getattr(model, "config", None), "vision_config", None)):
        v = getattr(obj, "image_size", None) if obj is not None else None
        if isinstance(v, int) and v > 0:
            return v
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        for key in ("height", "width", "shortest_edge"):
            if isinstance(size.get(key), int):
                return int(size[key])
    if isinstance(size, int) and size > 0:
        return int(size)
    return 224


def build_backbone(name: str) -> BackboneSpec:
    """HF-id / локальная папка / ``tiny`` -> ``BackboneSpec`` (веса + процессор)."""
    if name in TINY_NAMES:
        return _tiny_backbone(name)

    from transformers import AutoModel

    model = AutoModel.from_pretrained(name)
    cfg = getattr(model, "config", None)
    hidden = (getattr(cfg, "hidden_size", None)
              or getattr(getattr(cfg, "vision_config", None), "hidden_size", None)
              or getattr(getattr(cfg, "text_config", None), "hidden_size", None) or 0)
    hidden = int(hidden)

    processor = None
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(name)
    except Exception as e:  # noqa: BLE001 — препроцессор не критичен, есть fallback
        print(f"[model] AutoImageProcessor не загрузился ({e}); использую resize+normalize",
              flush=True)

    image_size = _cfg_image_size(model, processor)
    print(f"[model] бэкбон '{name}': dim={hidden}, img={image_size}, "
          f"препроцессор={'HF' if processor is not None else 'resize+normalize'}", flush=True)
    return BackboneSpec(name=name, model=model, processor=processor,
                        hidden_size=hidden, image_size=image_size)


def image_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] -> [B, D]. Логика как в ``ML service/encoder.py::_feats``."""
    if hasattr(model, "get_image_features"):
        try:
            out = model.get_image_features(pixel_values=pixel_values)
        except TypeError:
            out = model.get_image_features(pixel_values)
        return out[0] if isinstance(out, tuple) else out
    tower = getattr(model, "vision_model", None)
    try:
        out = (tower(pixel_values=pixel_values) if tower is not None
               else model(pixel_values=pixel_values))
    except TypeError:                     # кастомные модели: только позиционный аргумент
        out = tower(pixel_values) if tower is not None else model(pixel_values)
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0] if hidden.ndim == 3 else hidden
    return out


def freeze_module(module: nn.Module) -> None:
    """Замораживает все параметры модуля."""
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_last_blocks(model: nn.Module, n: int) -> int:
    """Разморозить последние ``n`` блоков энкодера (имена ``encoder.layer[s].<i>``).

    Возвращает число размороженных блоков (0, если слои не нашлись). Работает и для
    CLIP/SigLIP (``vision_model.encoder.layers``), и для DINOv2 (``encoder.layer``).
    """
    if n <= 0:
        return 0
    found = [(name, int(m.group(1)))
             for name, _ in model.named_parameters()
             if (m := re.search(r"encoder\.layer[s]?\.(\d+)\.", name))]
    if not found:
        return 0
    last = max(i for _, i in found)
    first = max(0, last - n + 1)
    params = dict(model.named_parameters())
    for name, idx in found:
        if idx >= first:
            params[name].requires_grad_(True)
    return last - first + 1


# ------------------------------------------------------------------ эмбеддинги
@dataclass
class Embeddings:
    """Результат прогона лоадера через модель (порядок совпадает с порядком CSV)."""

    emb: torch.Tensor
    slugs: list[str]
    labels: list[int] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slugs)


def _accepts_normalize(model: nn.Module) -> bool:
    try:
        return "normalize" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


@torch.no_grad()
def embed_loader(model: nn.Module, loader, device=None, normalize: bool = True) -> Embeddings:
    """Прогон лоадера -> ``Embeddings`` (CPU-тензор). Так считаются галерея и тест."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    embs, slugs, labels, paths = [], [], [], []
    for batch in loader:
        px = batch["pixel_values"].to(device, non_blocking=True)
        z = model(px, normalize=normalize) if _accepts_normalize(model) else model(px)
        embs.append(z.detach().float().cpu())
        slugs.extend(list(batch["slug"]))
        labels.extend([int(x) for x in batch["label"]])
        paths.extend(list(batch["path"]))
    emb = torch.cat(embs, dim=0) if embs else torch.zeros(0, 0)
    return Embeddings(emb=emb, slugs=slugs, labels=labels, paths=paths)


class MetricModel(pl.LightningModule):
    """Metric learning: дообучаем энкодер + проекционную голову triplet-лоссом.

    Тренеруются голова и (опционально) последние блоки бэкбона
    (``freeze_backbone`` / ``unfreeze_last_n``). На валидации считаем:
      * ``losses/val`` — batch-hard triplet по всему val-сету;
      * ``losses/val_ref`` — cross-modal triplet: кропы val против референсной базы;
      * ``metrics/val_acc@1/@k``, ``metrics/val_coverage`` — accuracy по эталонам.

    :param backbone: ``BackboneSpec`` (веса + процессор), загружается в ``train.py``.
    :param n_classes: число slug'ов в трейне (для aux-CE головы).
    :param gallery_loader: лоадер референсной базы (валидация/тест против эталонов).
    :param test_metric: метрика теста (``metrics.ExactSlugAccuracy``) — та же на валидации.
    """

    def __init__(self, backbone: BackboneSpec, n_classes: int, *,
                 backbone_name: Optional[str] = None, embed_dim: int = 256,
                 head: str = "linear", normalize: bool = True, dropout: float = 0.0,
                 freeze_backbone: bool = False, unfreeze_last_n: int = 0,
                 lr: float = 1e-4, backbone_lr_scale: float = 0.1,
                 weight_decay: float = 1e-4, warmup_frac: float = 0.05,
                 min_lr_scale: float = 0.02, triplet_margin: float = 0.3,
                 distance: str = "cosine", mining: str = "batch_hard",
                 ce_weight: float = 0.0, ce_scale: float = 16.0, val_k: int = 5,
                 gallery_loader=None, test_metric=None):
        super().__init__()
        # сначала авто-сбор всех аргументов __init__, затем докидываем служебное имя бэкбона
        self.save_hyperparameters(ignore=["backbone", "gallery_loader", "test_metric"])
        self.hparams["backbone_name"] = backbone_name or backbone.name

        self.backbone = backbone.model
        self.normalize_emb = bool(normalize)
        self.triplet_margin = float(triplet_margin)
        self.distance = distance
        self.mining = mining
        self.ce_weight = float(ce_weight)
        self.val_k = int(val_k)
        self.min_lr_scale = float(min_lr_scale)
        self.n_classes = int(n_classes)

        if freeze_backbone:
            freeze_module(self.backbone)
        elif unfreeze_last_n > 0:
            freeze_module(self.backbone)
            n_unfrozen = unfreeze_last_blocks(self.backbone, unfreeze_last_n)
            print(f"[model] разморожено блоков бэкбона: {n_unfrozen}", flush=True)

        in_dim = backbone.hidden_size
        if head == "mlp":
            layers = [nn.Linear(in_dim, embed_dim), nn.ReLU(inplace=True),
                      nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim)]
        elif head == "linear":
            layers = ([nn.Dropout(dropout)] if dropout else []) + [nn.Linear(in_dim, embed_dim)]
        elif head == "identity":
            layers, embed_dim = [], in_dim
        else:
            raise ValueError(f"неизвестная голова {head!r}: linear|mlp|identity")
        self.head = nn.Sequential(*layers)
        self.embed_dim = int(embed_dim)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.aux = (CosineAuxClassifier(self.n_classes, self.embed_dim, scale=ce_scale)
                    if self.ce_weight > 0 and self.n_classes > 1 else None)

        self.gallery_loader = gallery_loader          # от референсной базы (val/test)
        self.test_metric = test_metric
        self.label_space = None                       # ставит train.py (нужен для val_ref)
        self._reset_val()
        self._train_loss_sum, self._train_ce_sum, self._train_n = 0.0, 0.0, 0

    # -------------------------------------------------------------- forward
    def forward(self, pixel_values: torch.Tensor, normalize: Optional[bool] = None) -> torch.Tensor:
        z = self.head(image_features(self.backbone, pixel_values).float())
        norm = self.normalize_emb if normalize is None else normalize
        return F.normalize(z, dim=-1) if norm else z

    # ------------------------------------------ параметры для оптимизатора
    def _param_groups(self):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        head = [p for p in self.head.parameters() if p.requires_grad]
        if self.aux is not None:
            head += [p for p in self.aux.parameters() if p.requires_grad]
        groups = []
        if bb:
            groups.append({"params": bb, "name": "backbone",
                           "lr": self.hparams.lr * self.hparams.backbone_lr_scale})
        if head:
            groups.append({"params": head, "name": "head", "lr": self.hparams.lr})
        if not groups:                     # всё заморожено — не роняем обучение
            groups.append({"params": list(self.parameters()), "name": "head",
                           "lr": self.hparams.lr})
        return groups

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self._param_groups(), weight_decay=self.hparams.weight_decay)
        try:
            total = max(1, int(self.trainer.estimated_stepping_batches))
        except Exception:  # noqa: BLE001 — без max_epochs число шагов может быть неизвестно
            total = 10 ** 6
        warmup = max(1, int(total * self.hparams.warmup_frac)) if self.hparams.warmup_frac else 0

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            prog = (step - warmup) / max(1, total - warmup)
            prog = min(1.0, max(0.0, prog))
            return self.min_lr_scale + (1 - self.min_lr_scale) * 0.5 * (1 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    # ---------------------------------------------------------------- train
    def on_train_epoch_start(self) -> None:
        self._train_loss_sum, self._train_ce_sum, self._train_n = 0.0, 0.0, 0

    def training_step(self, batch, batch_idx):
        z = self(batch["pixel_values"])
        loss, stats = triplet_loss(z, batch["label"], margin=self.triplet_margin,
                                   distance=self.distance, mining=self.mining)
        if self.aux is not None:
            ce = F.cross_entropy(self.aux(z), batch["label"])
            loss = loss + self.ce_weight * ce
            self.log("losses/train_ce_step", ce.detach(), on_step=True, on_epoch=False,
                     batch_size=len(batch["slug"]))
            self._train_ce_sum += float(ce)
        self._train_loss_sum += float(loss.detach())
        self._train_n += 1

        bs = len(batch["slug"])
        self.log("losses/train_step", loss.detach(), on_step=True, on_epoch=False,
                 prog_bar=True, batch_size=bs)
        self.log("train/d_pos", stats["d_pos"], on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/d_neg", stats["d_neg"], on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/frac_valid", stats["frac_valid"], on_step=True, on_epoch=False, batch_size=bs)
        return loss

    def on_train_epoch_end(self) -> None:
        if self._train_n:
            self.log("losses/train", self._train_loss_sum / self._train_n,
                     on_epoch=True, on_step=False, prog_bar=True)
            if self.aux is not None:
                self.log("losses/train_ce", self._train_ce_sum / self._train_n,
                         on_epoch=True, on_step=False)
        # lr по группам (эпоховый срез — попадает в графики и ClearML)
        opt = self.optimizers()
        groups = (opt[0] if isinstance(opt, (list, tuple)) else opt).param_groups
        for i, g in enumerate(groups):
            self.log(f"lr/{g.get('name', i)}", float(g["lr"]), on_epoch=True, on_step=False)

    # ----------------------------------------------------------- validation
    def _reset_val(self) -> None:
        self._val_emb: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._val_slugs: list[str] = []
        self._val_paths: list[str] = []

    def on_validation_epoch_start(self) -> None:
        self._reset_val()

    def validation_step(self, batch, batch_idx):
        z = self(batch["pixel_values"])
        self._val_emb.append(z.detach().float().cpu())
        self._val_labels.append(batch["label"].detach().cpu())
        self._val_slugs.extend(list(batch["slug"]))
        self._val_paths.extend(list(batch["path"]))
        return None

    def on_validation_epoch_end(self) -> None:
        if not self._val_emb:
            return
        emb = torch.cat(self._val_emb, dim=0)
        labels = torch.cat(self._val_labels, dim=0)

        loss_val, stats = triplet_loss(emb, labels, margin=self.triplet_margin,
                                       distance=self.distance, mining=self.mining)
        self.log("losses/val", loss_val, prog_bar=True)
        self.log("val/d_pos", stats["d_pos"])
        self.log("val/d_neg", stats["d_neg"])
        self.log("val/frac_valid", stats["frac_valid"])

        if self.gallery_loader is None:
            return
        gal = embed_loader(self, self.gallery_loader, device=self.device,
                           normalize=self.normalize_emb)
        if not len(gal):
            return

        # 1) cross-modal: кропы val против референсной базы (ровно то, что нужно в прод)
        loss_ref, ref_stats = cross_modal_triplet_loss(
            emb, labels.tolist(), gal.emb, [self._slug_to_label(s) for s in gal.slugs],
            margin=self.triplet_margin, distance=self.distance)
        self.log("losses/val_ref", loss_ref)
        self.log("val_ref/d_pos", ref_stats["d_pos"])
        self.log("val_ref/d_neg", ref_stats["d_neg"])
        self.log("val_ref/frac_valid", ref_stats["frac_valid"])

        # 2) accuracy по эталонам — та же метрика, что и на тесте
        if self.test_metric is not None:
            res = self.test_metric(emb, self._val_slugs, gal.emb, gal.slugs,
                                   query_paths=self._val_paths)
            self.log("metrics/val_acc@1", res["acc@1"], prog_bar=True)
            self.log(f"metrics/val_acc@{res['k']}", res[f"acc@{res['k']}"])
            self.log("metrics/val_coverage", res["coverage"])

    def _slug_to_label(self, slug: str) -> int:
        """slug -> индекс класса трейна (нет в трейне -> -1, такие позитивы игнорируются)."""
        space = self.label_space
        return space.index(slug) if space is not None and slug in space else -1



