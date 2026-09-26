# -*- coding: utf-8 -*-
"""Общие аргументы CLI и фабрики (данные/модель/чекпоинт) для ``train.py`` и ``evaluate.py``.

Все пути по умолчанию берутся из корневого ``paths.py`` (без хардкода). Данные задаются
папкой ``--data-dir`` с четырьмя файлами (``reference.csv``, ``train.csv``, ``val.csv``,
``test.csv``) либо явными путями к каждому CSV.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))                           # repo root (paths.py)

from paths import SIGLIP_LOCAL, TRAIN_ARTIFACTS, TRAIN_DUMMY          # noqa: E402
import dataset as ds                                                 # noqa: E402
import metrics as metrics_mod                                        # noqa: E402
from modeling import MetricModel, build_backbone                      # noqa: E402

CSV_NAMES = ("reference", "train", "val", "test")


def default_backbone() -> str:
    """SEARCH_MODEL/TRAIN_BACKBONE -> локальная копия SigLIP 2 -> HF-id."""
    env = os.environ.get("TRAIN_BACKBONE") or os.environ.get("SEARCH_MODEL")
    if env:
        return env
    return str(SIGLIP_LOCAL) if SIGLIP_LOCAL.exists() else "google/siglip2-base-patch16-256"


# ------------------------------------------------------------------ аргументы
def add_data_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("данные")
    g.add_argument("--data-dir", default=str(TRAIN_DUMMY),
                   help="папка с reference.csv/train.csv/val.csv/test.csv")
    for name in CSV_NAMES:
        g.add_argument(f"--{name}-csv", default=None,
                       help=f"путь к {name}.csv (перекрывает --data-dir)")
    g.add_argument("--path-col", default=None, help="имя колонки с путём (иначе авто)")
    g.add_argument("--slug-col", default=None, help="имя колонки со slug (иначе авто)")
    g.add_argument("--sep", default=None, help="разделитель CSV (иначе авто: , ; tab |)")
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--include-reference", choices=["none", "train"], default="train",
                   help="подмешивать эталоны референсной базы в трейн как позитивы")
    g.add_argument("--p", type=int, default=8, help="классов в батче (PK-семплер)")
    g.add_argument("--k", type=int, default=4, help="картинок одного класса в батче")
    g.add_argument("--num-batches", type=int, default=None,
                   help="батчей в эпоху (деф: ceil(классы / P))")
    g.add_argument("--aug-size", type=int, default=None, help="размер кропа аугментации")
    g.add_argument("--rrc-scale", type=float, nargs=2, default=(0.65, 1.0),
                   metavar=("MIN", "MAX"), help="RandomResizedCrop scale")
    g.add_argument("--degrees", type=float, default=6.0, help="поворот аугментации")
    g.add_argument("--color-jitter", type=float, default=0.2)
    g.add_argument("--hflip", action="store_true", help="зеркалить (текст этикетки!)")


def add_model_args(ap: argparse.ArgumentParser, backbone_default: str | None = "") -> None:
    g = ap.add_argument_group("модель")
    default = default_backbone() if backbone_default == "" else backbone_default
    g.add_argument("--backbone", default=default,
                   help="HF-id, локальная папка или 'tiny' (для evaluate: из чекпоинта)")
    g.add_argument("--embed-dim", type=int, default=256)
    g.add_argument("--head", choices=["linear", "mlp", "identity"], default="linear")
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="не L2-нормировать эмбеддинги")
    g.add_argument("--freeze-backbone", action="store_true",
                   help="учить только голову (linear probe)")
    g.add_argument("--unfreeze-last-n", type=int, default=0,
                   help="разморозить последние N блоков энкодера")
    g.add_argument("--triplet-margin", type=float, default=0.3)
    g.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    g.add_argument("--mining", choices=["batch_hard", "semi_hard", "all"], default="batch_hard")
    g.add_argument("--ce-weight", type=float, default=0.0,
                   help="вес aux-CE по slug'ам (0 — только triplet)")
    g.add_argument("--ce-scale", type=float, default=16.0)
    g.add_argument("--val-k", type=int, default=5, help="k для accuracy на валидации")
    ap.set_defaults(normalize=True)


def add_optim_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("оптимизация")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--lr", type=float, default=1e-4, help="lr головы")
    g.add_argument("--backbone-lr-scale", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-frac", type=float, default=0.05)
    g.add_argument("--min-lr-scale", type=float, default=0.02)
    g.add_argument("--grad-clip", type=float, default=1.0, help="0 — выключить")
    g.add_argument("--accumulate-grad-batches", type=int, default=1)
    g.add_argument("--precision", default="32-true")
    g.add_argument("--accelerator", default="auto", help="auto|cpu|mps|cuda")
    g.add_argument("--devices", default="auto")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--limit-batches", type=float, default=1.0,
                   help="доля батчей (для смоук-теста: 0.1)")


def add_clearml_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("ClearML")
    g.add_argument("--clearml", choices=["on", "offline", "off"], default="on",
                   help="on — сервер; offline — локальная сессия; off — без трекинга")
    g.add_argument("--project", default="lct26-wine/ML train")
    g.add_argument("--task-name", default=None)
    g.add_argument("--tags", nargs="*", default=None)
    g.add_argument("--plot-every", type=int, default=5,
                   help="каждые N эпох сохранять графики (0 — только в конце)")
    g.add_argument("--log-every", type=int, default=20, help="шагов между батчевыми скалярами")


def add_output_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("выход и тест")
    g.add_argument("--out-dir", default=str(TRAIN_ARTIFACTS))
    g.add_argument("--test-k", type=int, default=5, help="k для accuracy на тесте")
    g.add_argument("--test-all-queries", action="store_true",
                   help="считать accuracy по всем query (а не только покрытым эталонами)")
    g.add_argument("--no-run-test", dest="run_test", action="store_false",
                   help="не прогонять тестовый датасет после обучения")
    g.add_argument("--no-plot-embeddings", dest="plot_embeddings", action="store_false")
    ap.set_defaults(run_test=True, plot_embeddings=True)


# -------------------------------------------------------------------- фабрики
def resolve_csvs(args) -> dict:
    """``--data-dir`` + явные пути -> ``{'reference': Path, ...}`` (с проверкой наличия)."""
    out = {}
    for name in CSV_NAMES:
        explicit = getattr(args, f"{name}_csv", None)
        path = Path(explicit) if explicit else Path(args.data_dir) / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"нет {name}.csv: {path}\n"
                f"  положите 4 файла (reference.csv/train.csv/val.csv/test.csv) в --data-dir "
                f"или задайте --{name}-csv; для смоук-теста: python make_dummy_data.py")
        out[name] = path
    return out


def make_datamodule(args, spec, csvs: dict | None = None) -> ds.VinoDataModule:
    """Собирает ``VinoDataModule`` по аргументам CLI и загруженному бэкбону."""
    csvs = csvs or resolve_csvs(args)
    augment = {"size": args.aug_size, "rrc_scale": tuple(args.rrc_scale),
               "degrees": args.degrees, "color_jitter": args.color_jitter, "hflip": args.hflip}
    return ds.VinoDataModule(
        csvs["reference"], csvs["train"], csvs["val"], csvs["test"],
        processor=spec.processor, image_size=spec.image_size,
        batch_size=args.p * args.k, p=args.p, k=args.k, num_workers=args.num_workers,
        augment=augment, include_reference=args.include_reference,
        path_col=args.path_col, slug_col=args.slug_col, sep=args.sep,
        num_batches=args.num_batches, seed=args.seed)


def make_test_metric(args):
    """Метрика теста/валидации (сейчас — exact slug accuracy; см. ``metrics.py``)."""
    return metrics_mod.ExactSlugAccuracy(k=args.test_k,
                                         only_covered=not args.test_all_queries)


def make_model(args, spec, n_classes: int, gallery_loader=None, test_metric=None) -> MetricModel:
    """``MetricModel`` по аргументам CLI (та же конфигурация, что сохранится в чекпоинт)."""
    return MetricModel(
        spec, n_classes, embed_dim=args.embed_dim, head=args.head, normalize=args.normalize,
        dropout=args.dropout, freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n, lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale, weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac, min_lr_scale=args.min_lr_scale,
        triplet_margin=args.triplet_margin, distance=args.distance, mining=args.mining,
        ce_weight=args.ce_weight, ce_scale=args.ce_scale, val_k=args.val_k,
        gallery_loader=gallery_loader, test_metric=test_metric)


def load_model_from_ckpt(ckpt_path, spec, gallery_loader=None, test_metric=None,
                         label_space=None, map_location="cpu") -> MetricModel:
    """Восстанавливает ``MetricModel`` из чекпоинта (гиперпараметры — из чекпоинта).

    Не используем ``LightningModule.load_from_checkpoint``: он прокидывает hparams в
    ``__init__`` целиком и спотыкается на служебных ключах (``backbone_name``).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    for key in ("backbone_name", "gallery_loader", "test_metric"):
        hp.pop(key, None)
    model = MetricModel(spec, gallery_loader=gallery_loader, test_metric=test_metric, **hp)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[ckpt] не загружено ключей: {len(missing)}, лишних: {len(unexpected)}",
              flush=True)
    model.label_space = label_space
    return model


def make_backbone(args):
    """``--backbone`` -> ``(BackboneSpec, имя)``. ``tiny`` — игрушечный бэкбон без весов."""
    name = getattr(args, "backbone", None) or default_backbone()
    return build_backbone(name), name


