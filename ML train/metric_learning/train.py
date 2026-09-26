# -*- coding: utf-8 -*-
"""Обучение энкодера методом metric learning (triplet loss) на pytorch-lightning.

Пайплайн:

1. ``--data-dir`` -> ``reference.csv`` (референсная база/эталоны) + ``train/val/test.csv``
   (кропы этикеток со slug'ами);
2. трейн: PK-семплер (P классов × K картинок) -> triplet loss (batch-hard) стягивает
   кропы одного slug'а и расталкивает разные; опционально эталоны референсной базы
   подмешиваются в трейн как «опорные» позитивы (``--include-reference train``);
3. валидация: ``losses/val``, ``losses/val_ref`` (кропы против эталонов) и accuracy
   по референсной базе — всё уходит в ClearML скалярами, плюс графики лоссов/метрик;
4. в конце — прогон по ``test.csv`` с замером accuracy (``evaluate.run_test``,
   метрика — ``metrics.ExactSlugAccuracy``: точное совпадение slug).

Примеры::

    # смоук-тест на синтетике (без весов и без сервера ClearML)
    python make_dummy_data.py
    python train.py --backbone tiny --data-dir dummy_data --epochs 6 --clearml offline

    # реальное обучение SigLIP 2 из локальной папки
    python train.py --data-dir /path/to/csv --backbone ../../models/siglip2-base-patch16-256 \\
        --epochs 15 --p 8 --k 4 --unfreeze-last-n 2 --clearml on
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))                       # repo root (paths.py)

import pytorch_lightning as pl                                   # noqa: E402
import torch                                                     # noqa: E402

import config                                                    # noqa: E402
import evaluate                                                  # noqa: E402
import tracking                                                  # noqa: E402
from paths import TRAIN_ARTIFACTS                                # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Metric learning (triplet) для энкодера этикеток.")
    config.add_data_args(ap)
    config.add_model_args(ap)
    config.add_optim_args(ap)
    config.add_clearml_args(ap)
    config.add_output_args(ap)
    ap.add_argument("--resume", default=None, help="чекпоинт для продолжения обучения")
    ap.add_argument("--ckpt-metric", default="metrics/val_acc@1",
                    help="метрика для ModelCheckpoint (mode=max)")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="0 — EarlyStopping выключен")
    ap.add_argument("--fast-dev-run", action="store_true",
                    help="1 трейн- и 1 валидационный батч (проверка пайплайна)")
    return ap


def _slugify(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Zа-яА-Я._-]+", "-", str(name)).strip("-") or "run"


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    run_name = args.task_name or f"metric-{Path(args.data_dir).name}-{time.strftime('%m%d-%H%M%S')}"
    run_dir = Path(args.out_dir) / _slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    task = tracking.init_task(args.clearml, args.project, run_name, args.tags, vars(args))

    # ---------------------------------------------------------------- данные
    spec, backbone_name = config.make_backbone(args)
    dm = config.make_datamodule(args, spec)
    dm.setup()
    stats = dm.stats()
    print("[data] сводка: " + json.dumps(stats, ensure_ascii=False), flush=True)
    (run_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    (run_dir / "label_space.json").write_text(
        json.dumps(dm.label_space.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if task is not None:
        task.connect(stats, name="dataset")

    # ------------------------------------------------------ модель и метрика
    metric = config.make_test_metric(args)
    gallery_loader = dm.gallery_dataloader()
    model = config.make_model(args, spec, n_classes=len(dm.label_space),
                              gallery_loader=gallery_loader, test_metric=metric)
    model.label_space = dm.label_space
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] обучаемых параметров: {n_trainable:,} | голова={args.head} "
          f"dim={args.embed_dim} | бэкбон '{backbone_name}'", flush=True)

    # ------------------------------------------------------------- трекинг
    logger = tracking.ClearMLLogger(task, log_every_n_steps=args.log_every) if task else False
    history_cb = tracking.HistoryCallback(run_dir / "plots", plot_every=args.plot_every, task=task)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=run_dir / "checkpoints", filename="best", monitor=args.ckpt_metric,
        mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
    callbacks = [history_cb, ckpt_cb]
    if args.early_stop_patience:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor=args.ckpt_metric, mode="max", patience=args.early_stop_patience))

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=args.devices,
        precision=args.precision, logger=logger, callbacks=callbacks,
        gradient_clip_val=args.grad_clip or None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_batches, limit_val_batches=args.limit_batches,
        log_every_n_steps=args.log_every, fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=0 if args.fast_dev_run else 1,
        default_root_dir=str(run_dir))

    # ------------------------------------------------------------- обучение
    t0 = time.time()
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    print(f"[train] обучение завершено за {time.time() - t0:.1f} c "
          f"(эпох: {trainer.current_epoch})", flush=True)

    # графики лоссов/метрик на диск + в ClearML
    history_cb.dump_plots()
    for path in (run_dir / "dataset_stats.json", run_dir / "label_space.json"):
        tracking.upload_artifact(task, path)
    best_ckpt = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    if best_ckpt is not None and best_ckpt.exists() and task is not None:
        tracking.upload_artifact(task, best_ckpt, name="best.ckpt")

    # -------------------------------------------------- тест: accuracy по slug
    test_res = None
    if args.run_test:
        if best_ckpt is not None and best_ckpt.exists():
            print(f"[test] беру лучший чекпоинт: {best_ckpt}", flush=True)
            eval_model = config.load_model_from_ckpt(
                best_ckpt, spec, gallery_loader=gallery_loader, test_metric=metric,
                label_space=dm.label_space)
        else:
            print("[test] чекпоинта нет — тестирую модель из памяти", flush=True)
            eval_model = model
        test_res = evaluate.run_test(
            eval_model, dm, metric=metric, out_dir=run_dir, task=task,
            plot_embeddings=args.plot_embeddings, tag="test")

    # ------------------------------------------------------------------ итог
    print("\n=== ИТОГ ===", flush=True)
    print(f"  run dir        : {run_dir}")
    print(f"  бэкбон         : {backbone_name}")
    print(f"  эпох           : {trainer.current_epoch}")
    print(f"  losses/val     : {history_cb.history.last('losses/val')}")
    print(f"  metrics/val_acc@1: {history_cb.history.last('metrics/val_acc@1')}")
    if test_res is not None:
        k = test_res["k"]
        print(f"  ТЕСТ accuracy@1: {test_res['acc@1']} | accuracy@{k}: {test_res[f'acc@{k}']} | "
              f"coverage: {test_res['coverage']} (n={test_res['n_evaluated']})")
    if task is not None:
        task.close()


if __name__ == "__main__":
    main()

