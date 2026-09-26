# -*- coding: utf-8 -*-
"""Прогон по тестовому датасету: accuracy по slug (через ``metrics.ExactSlugAccuracy``).

Это отдельный шаг пайплайна (и отдельный CLI): считаем эмбеддинги кропов теста,
ищем ближайшие эталоны референсной базы и меряем accuracy. Сама метрика живёт
в ``metrics.py`` — меняется там, без правок этого файла.

Запуск по чекпоинту::

    python evaluate.py --ckpt artifacts/run/checkpoints/best.ckpt --data-dir dummy_data

Может вызываться и как функция ``run_test`` — так делает ``train.py`` в конце
обучения («в конце должен быть прогон по тестовому датасету с замером accuracy»).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))                       # repo root (paths.py)

import pytorch_lightning as pl                                   # noqa: E402
import torch                                                     # noqa: E402

import config                                                    # noqa: E402
import tracking                                                  # noqa: E402
from metrics import ExactSlugAccuracy, format_report             # noqa: E402
from modeling import MetricModel, embed_loader                   # noqa: E402
from paths import TRAIN_ARTIFACTS                                # noqa: E402


def pick_device(explicit: str | None = None) -> str:
    """cuda -> mps -> cpu (если не задано явно)."""
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_test(model: MetricModel, dm, *, metric=None, out_dir=None, task=None,
             device: str | None = None, plot_embeddings: bool = True,
             tag: str = "test", verbose: bool = True) -> dict:
    """Тестовый прогон: эмбеддинги теста -> ближайшие эталоны -> accuracy.

    Возвращает словарь метрик (``acc@1``, ``acc@k``, ``coverage``, ...) и сохраняет
    отчёт ``reports/{tag}.json`` + предсказания ``reports/{tag}_predictions.csv``.
    """
    out_dir = Path(out_dir or TRAIN_ARTIFACTS)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    device = device or str(next(model.parameters()).device)
    metric = metric or ExactSlugAccuracy(k=5)
    normalize = bool(getattr(model, "normalize_emb", True))

    model.eval().to(device)
    t0 = time.time()
    gallery = embed_loader(model, dm.gallery_dataloader(), device=device, normalize=normalize)
    test = embed_loader(model, dm.test_dataloader(), device=device, normalize=normalize)
    if verbose:
        print(f"[test] эталонов {len(gallery)}, тестовых кропов {len(test)} "
              f"({time.time() - t0:.1f} c)", flush=True)

    res = metric(test.emb, test.slugs, gallery.emb, gallery.slugs, query_paths=test.paths)
    res.update({"tag": tag, "backbone": getattr(model.hparams, "backbone_name", None),
                "n_test_images": len(test), "n_reference_images": len(gallery),
                "device": device, "seconds": round(time.time() - t0, 1)})

    (reports / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    if metric.last_predictions:
        pd.DataFrame(metric.last_predictions).to_csv(
            reports / f"{tag}_predictions.csv", index=False, encoding="utf-8")

    if plot_embeddings:
        plot_path = tracking.plot_embeddings(
            test.emb.numpy(), test.slugs, reports / f"{tag}_embeddings.png",
            ref_emb=gallery.emb.numpy(), ref_slugs=gallery.slugs,
            title=f"PCA тестовых эмбеддингов vs эталоны [{tag}]")
        if plot_path is not None and task is not None:
            tracking.upload_image(task, plot_path, title="test", series=plot_path.stem)

    if task is not None:
        for key, value in res.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                task.get_logger().report_scalar(title="test", series=f"test/{key}",
                                                value=float(value), iteration=0)
        for path in (reports / f"{tag}.json", reports / f"{tag}_predictions.csv"):
            if path.exists():
                tracking.upload_artifact(task, path)

    res["report_path"] = str(reports / f"{tag}.json")
    if verbose:
        print("\n" + format_report(res, title=f"ТЕСТ [{tag}]"), flush=True)
        print(f"[test] отчёт: {res['report_path']}", flush=True)
    return res


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Прогон по test.csv: accuracy по точному совпадению slug.")
    config.add_data_args(ap)
    config.add_model_args(ap, backbone_default=None)   # деф: бэкбон из чекпоинта
    config.add_clearml_args(ap)
    config.add_output_args(ap)
    ap.add_argument("--ckpt", default=None,
                    help="чекпоинт обученной модели (без него — энкодер как есть + случайная голова)")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (деф: авто)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False) if args.ckpt else None
    hp = (ckpt or {}).get("hyper_parameters", {})
    name = args.backbone or hp.get("backbone_name") or config.default_backbone()
    spec, backbone_name = config.make_backbone(argparse.Namespace(backbone=name))

    run_name = args.task_name or f"test-{Path(args.data_dir).name}-{time.strftime('%m%d-%H%M%S')}"
    task = tracking.init_task(args.clearml, args.project, run_name, args.tags, vars(args))
    if ckpt is not None and task is not None:
        tracking.upload_artifact(task, args.ckpt)

    dm = config.make_datamodule(args, spec)
    dm.setup()
    metric = config.make_test_metric(args)
    gallery_loader = dm.gallery_dataloader()

    if args.ckpt:
        model = config.load_model_from_ckpt(args.ckpt, spec, gallery_loader=gallery_loader,
                                           test_metric=metric, label_space=dm.label_space)
    else:
        print("[test] --ckpt не задан: считаю эмбеддинги «как есть» "
              "(baseline-энкодер + случайная голова)", flush=True)
        model = config.make_model(args, spec, n_classes=len(dm.label_space),
                                  gallery_loader=gallery_loader, test_metric=metric)
    model.label_space = dm.label_space

    run_test(model, dm, metric=metric, out_dir=Path(args.out_dir) / "eval", task=task,
             device=args.device, plot_embeddings=args.plot_embeddings)
    if task is not None:
        task.close()


if __name__ == "__main__":
    main()

