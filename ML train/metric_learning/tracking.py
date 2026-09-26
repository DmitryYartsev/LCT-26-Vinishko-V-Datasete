# -*- coding: utf-8 -*-
"""Трекинг в ClearML: Task, адаптер Lightning-логгера и графики лоссов/метрик.

Зачем отдельный адаптер: в установленной версии ``clearml`` нет готового биндинга
для Lightning (``clearml.binding.lightning`` отсутствует), поэтому ``ClearMLLogger``
реализует интерфейс ``pytorch_lightning.loggers.Logger`` самостоятельно и пишет
скаляры в ClearML. Если clearml не установлен/недоступен — всё деградирует мягко
(как в ``code/clearml_logger.py``), обучение не падает.

Дополнительно ``HistoryCallback`` ведёт историю метрик по эпохам и рисует графики
(matplotlib) — лоссы отдельно, метрики отдельно — и кладёт их в ClearML как картинки
и на диск (``artifacts/plots/``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytorch_lightning as pl

PLOT_PREFIXES = ("losses/", "metrics/", "lr/", "train/", "val/", "val_ref/")


# ------------------------------------------------------------------ ClearML Task
def init_task(mode: str = "on", project: str = "lct26-wine/ML train",
              task_name: str = None, tags=None, params: dict | None = None):
    """``mode``: ``on`` — обычный ClearML; ``offline`` — локальная offline-сессия;
    ``off`` — вообще без ClearML. Возвращает ``Task`` или ``None``."""
    if mode == "off":
        return None
    try:
        from clearml import Task
    except ImportError:
        print("[clearml] пакет не установлен — трекинг выключен (pip install clearml)",
              flush=True)
        return None
    try:
        if mode == "offline":
            Task.set_offline(True)
        task = Task.init(project_name=project, task_name=task_name,
                         auto_connect_frameworks=False,
                         tags=list(tags) if tags else None)
        if params:
            task.connect(dict(params))          # аргументы в Configuration задачи
        print(f"[clearml] task {task.id} | project '{project}' | mode={mode}", flush=True)
        return task
    except Exception as e:  # noqa: BLE001 — трекинг не должен ломать обучение
        print(f"[clearml] не удалось инициализировать ({e}) — логирование выключено",
              flush=True)
        return None


# --------------------------------------------------------------------- история
class History:
    """История метрик по эпохам: ``{key: {epoch: value}}`` + графики/CSV/JSON."""

    def __init__(self, out_dir: Path | None = None):
        self.rows: dict[str, dict[int, float]] = {}
        self.out_dir = Path(out_dir) if out_dir else None

    def update(self, metrics: dict, epoch: int) -> None:
        for k, v in metrics.items():
            try:
                self.rows.setdefault(k, {})[int(epoch)] = float(v)
            except (TypeError, ValueError):
                continue

    def keys(self, prefix: str | None = None) -> list[str]:
        keys = sorted(self.rows)
        return [k for k in keys if k.startswith(prefix)] if prefix else keys

    def series(self, key: str) -> tuple[list[int], list[float]]:
        items = sorted(self.rows.get(key, {}).items())
        return [e for e, _ in items], [v for _, v in items]

    def last(self, key: str):
        """Последнее значение метрики (для финальных отчётов)."""
        items = sorted(self.rows.get(key, {}).items())
        return items[-1][1] if items else None

    def save(self, out_dir: Path | None = None) -> list[Path]:
        """Пишет ``history.json``, ``history.csv`` и графики в ``out_dir``."""
        out = Path(out_dir or self.out_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        written = []

        json_path = out / "history.json"
        json_path.write_text(json.dumps(
            {k: {str(e): v for e, v in sorted(self.rows[k].items())} for k in self.rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(json_path)

        csv_path = out / "history.csv"
        epochs = sorted({e for d in self.rows.values() for e in d})
        keys = sorted(self.rows)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("epoch," + ",".join(keys) + "\n")
            for e in epochs:
                row = [str(self.rows.get(k, {}).get(e, "")) for k in keys]
                f.write(f"{e}," + ",".join(row) + "\n")
        written.append(csv_path)

        for prefix in PLOT_PREFIXES:
            if self.keys(prefix):
                written.append(self.plot(prefix, out / f"plot_{_slug(prefix)}.png"))
        return written

    def plot(self, prefix: str, out_path: Path):
        """График всех метрик с данным префиксом (носless: matplotlib Agg)."""
        return plot_series({k: self.series(k) for k in self.keys(prefix)}, out_path,
                           ylabel=prefix.rstrip("/"))


def _slug(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Zа-яА-Я_]+", "-", str(name)).strip("-").lower() or "plot"


def plot_series(series: dict, out_path, ylabel: str = "", title: str = ""):
    """``{label: (xs, ys)}`` -> PNG. Возвращает путь (или None, если нечего рисовать)."""
    series = {k: v for k, v in series.items() if v[0]}
    if not series:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(alpha=0.3)
    if len(series) > 1 or len(str(next(iter(series)))) > 12:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_embeddings(emb, slugs, out_path, ref_emb=None, ref_slugs=None,
                    title: str = "эмбеддинги (PCA-2)") -> Path | None:
    """Скаттер 2D-PCA тестовых эмбеддингов (+ эталонов референсной базы).

    Чистая визуальная диагностика metric learning: если одинаковые slug'и собрались
    в кучки, а эталоны легли рядом со своими кропами — обучение работает.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = np.asarray(emb, dtype=float)
    if emb.ndim != 2 or emb.shape[0] < 2:
        return None
    ref = np.asarray(ref_emb, dtype=float) if ref_emb is not None and len(ref_emb) else None
    stack = emb if ref is None else np.vstack([emb, ref])
    stack = stack - stack.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(stack, full_matrices=False)
    coords = stack @ vt[:2].T
    q, r = coords[:len(emb)], coords[len(emb):]

    labels = list(dict.fromkeys(list(slugs) + (list(ref_slugs) if ref is not None else [])))
    colors = plt.get_cmap("tab20").resampled(max(len(labels), 1))
    cmap = {s: colors(i) for i, s in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(q[:, 0], q[:, 1], c=[cmap[s] for s in slugs], marker="o", s=36,
               edgecolors="white", linewidths=0.4)
    if ref is not None and len(r):
        ax.scatter(r[:, 0], r[:, 1], c=[cmap[s] for s in ref_slugs], marker="*", s=190,
                   edgecolors="black", linewidths=0.6)
    if len(labels) <= 20:
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=cmap[s], label=s)
                   for s in labels]
        handles.append(plt.Line2D([], [], marker="*", linestyle="", color="gray",
                                  label="эталоны"))
        ax.legend(handles=handles, fontsize=7, loc="best")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------ callbacks
class HistoryCallback(pl.Callback):
    """Копит ``trainer.callback_metrics`` по эпохам, рисует графики и льёт их в ClearML."""

    def __init__(self, out_dir, plot_every: int = 0, task=None):
        super().__init__()
        self.history = History(out_dir)
        self.plot_every = int(plot_every or 0)
        self.task = task

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
            except (TypeError, ValueError):
                continue
        epoch = trainer.current_epoch
        self.history.update(metrics, epoch)
        if self.plot_every and (epoch + 1) % self.plot_every == 0:
            self.dump_plots()

    def dump_plots(self) -> list[Path]:
        """Графики (+CSV/JSON истории) на диск и в ClearML. Возвращает пути."""
        paths = self.history.save()
        if self.task is not None:
            for p in paths:
                if p.suffix == ".png":
                    upload_image(self.task, p, title="plots", series=p.stem)
                else:
                    upload_artifact(self.task, p)
        return paths


# -------------------------------------------------------------- ClearML logger
class ClearMLLogger(pl.loggers.Logger):
    """Адаптер ``pl.loggers.Logger`` -> ClearML scalars (title = префикс до ``/``).

    Метрики с суффиксом ``_step`` (батчевые) отправляются раз в ``log_every_n_steps``
    шагов, чтобы не спамить сервер; эпоховые — всегда.
    """

    def __init__(self, task, log_every_n_steps: int = 20):
        super().__init__()
        self.task = task
        self._logger = task.get_logger()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    # --- pl.loggers.Logger API ---
    @property
    def name(self) -> str:
        return "clearml"

    @property
    def version(self) -> str:
        return str(self.task.id)

    @property
    def experiment(self):
        return self._logger

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        # гиперпараметры уже ушли в Configuration через Task.connect(vars(args))
        return None

    def log_metrics(self, metrics: dict, step=None) -> None:
        for key, value in metrics.items():
            try:
                value = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            except (TypeError, ValueError):
                continue
            if step is not None and str(key).endswith("_step") and int(step) % self.log_every_n_steps:
                continue
            title = str(key).split("/", 1)[0] or "metrics"
            self._logger.report_scalar(title=title, series=str(key), value=value,
                                       iteration=int(step or 0))

    def save(self) -> None:
        return None

    def finalize(self, status: str) -> None:
        try:
            self._logger.flush()
        except Exception:  # noqa: BLE001 — не падаем на закрытии
            pass

    def log_graph(self, model, input_array=None) -> None:
        return None


def upload_image(task, path, title: str = "plots", series: str = None) -> None:
    """Отправить PNG в ClearML как debug-sample."""
    if task is None:
        return
    try:
        task.get_logger().report_image(title=title, series=series or Path(path).stem,
                                       local_path=str(path))
    except Exception as e:  # noqa: BLE001
        print(f"[clearml] не удалось загрузить {path}: {e}", flush=True)


def upload_artifact(task, path, name: str = None) -> None:
    """Загрузить артефакт (чекпоинт/отчёт) в ClearML."""
    if task is None:
        return
    try:
        task.upload_artifact(name=name or Path(path).name, artifact_object=str(path))
    except Exception as e:  # noqa: BLE001
        print(f"[clearml] не удалось загрузить артефакт {path}: {e}", flush=True)


