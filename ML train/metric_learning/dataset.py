# -*- coding: utf-8 -*-
"""Данные для metric learning: ``reference.csv`` + ``train/val/test.csv``.

Формат у всех CSV одинаковый: колонка с путём до картинки + колонка со slug'ом.
Имена колонок определяются автоматически (или задаются через ``--path-col`` /
``--slug-col``), а также автоматически подбирается разделитель (``,`` / ``;`` / tab).

* ``reference.csv`` — референсная база (эталоны каталога). По ней считается accuracy
  на валидации/тесте, и её картинки могут подмешиваться в трейн как «опорные»
  позитивы к тем же slug'ам (``--include-reference train``).
* ``train.csv`` / ``val.csv`` / ``test.csv`` — кропы этикеток со своими slug'ами.

Здесь же: аугментации/препроцесс, ``PKBatchSampler`` (P классов × K объектов
в батче — гарантирует позитивные пары для triplet loss) и ``VinoDataModule``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

IMG_EXT = {".webp", ".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".tif", ".tiff", ".heic"}

PATH_COL_HINTS = ("image_path", "path", "img_path", "image", "img", "photo", "filepath", "file", "filename")
SLUG_COL_HINTS = ("slug", "true_slug", "wine_slug", "label", "class", "class_name")


# ------------------------------------------------------------------ чтение CSV
def _sniff_sep(path: Path, sample: int = 8192) -> str:
    head = path.read_text(encoding="utf-8", errors="replace")[:sample].splitlines()
    line = next((l for l in head if l.strip()), "")
    counts = {s: line.count(s) for s in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def _pick_col(columns: list[str], hints: tuple[str, ...], explicit, what: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"колонки {explicit!r} нет в CSV; есть: {columns}")
        return explicit
    lower = {c.lower(): c for c in columns}
    for hint in hints:
        if hint in lower:
            return lower[hint]
    for c in columns:
        if what in c.lower():
            return c
    raise ValueError(f"не нашёл колонку с {what} среди {columns} "
                     f"(подсказка: --{'path' if what == 'path' else 'slug'}-col)")


def read_index_csv(path, path_col=None, slug_col=None, sep=None) -> list[dict]:
    """CSV -> ``[{'image_path': Path(abs), 'slug': str}, ...]``.

    Относительные пути разрешаются от папки самого CSV. Строки без slug'а или с
    несуществующей картинкой пропускаются (с предупреждением в stdout).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет CSV: {path}")
    sep = sep or _sniff_sep(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    pc = _pick_col(cols, PATH_COL_HINTS, path_col, "path")
    sc = _pick_col(cols, SLUG_COL_HINTS, slug_col, "slug")

    base = path.parent
    records, missing, empty = [], 0, 0
    for slug, raw in zip(df[sc].tolist(), df[pc].tolist()):
        slug, raw = str(slug).strip(), str(raw).strip()
        if not slug or not raw:
            empty += 1
            continue
        img = Path(raw)
        if not img.is_absolute():
            cand = base / img
            img = cand if cand.exists() else img
        if not img.exists():
            missing += 1
            continue
        records.append({"image_path": img, "slug": slug})
    print(f"[data] {path.name}: колонки '{pc}' (path) + '{sc}' (slug), sep={sep!r} -> "
          f"{len(records)} записей из {len(df)} строк"
          f"{f', пропущено {missing} без файла' if missing else ''}"
          f"{f', пустых {empty}' if empty else ''}", flush=True)
    return records


class LabelSpace:
    """Соответствие slug <-> индекс класса (по трейну)."""

    def __init__(self, slugs):
        self.slugs = sorted({str(s) for s in slugs})
        self._idx = {s: i for i, s in enumerate(self.slugs)}

    def __len__(self) -> int:
        return len(self.slugs)

    def __contains__(self, slug) -> bool:
        return slug in self._idx

    def index(self, slug) -> int:
        return self._idx[slug]

    def __getitem__(self, idx) -> str:
        return self.slugs[idx]

    def as_dict(self) -> dict:
        return {"slugs": self.slugs}


# ------------------------------------------------------------------ препроцесс
class HFPreprocess:
    """PIL -> Tensor[C, H, W] через HF image processor (как в сервисе).

    Осознанно класс, а не замыкание: датасет пиклится при ``num_workers > 0``
    (macOS spawn), а замыкания pickle не переживают.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, pil: Image.Image) -> torch.Tensor:
        out = self.processor(images=pil, return_tensors="pt")
        px = out["pixel_values"] if not isinstance(out, torch.Tensor) else out
        return px[0]


class FallbackPreprocess:
    """Resize + ToTensor + Normalize — когда у бэкбона нет HF-процессора (tiny)."""

    def __init__(self, image_size: int, mean: float = 0.5, std: float = 0.5):
        from torchvision import transforms as T
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(),
                             T.Normalize([mean] * 3, [std] * 3)])

    def __call__(self, pil: Image.Image) -> torch.Tensor:
        return self.tf(pil)


class Preprocess:
    """Итоговый transform бэкбона: аугментации (если есть) -> препроцесс."""

    def __init__(self, post, aug=None):
        self.post, self.aug = post, aug

    def __call__(self, pil: Image.Image) -> torch.Tensor:
        return self.post(self.aug(pil) if self.aug is not None else pil)


def _hf_preprocess(processor):
    """Фабрика препроцесса HF (пиклуемый класс вместо замыкания)."""
    return HFPreprocess(processor)


def _fallback_preprocess(image_size: int, mean: float = 0.5, std: float = 0.5):
    """Фабрика fallback-препроцесса (пиклуемый класс вместо замыкания)."""
    return FallbackPreprocess(image_size, mean, std)


def build_transforms(processor, image_size: int, augment: dict | None = None,
                     train: bool = True):
    """Собирает ``callable(PIL) -> Tensor[C, H, W]``.

    Train: аугментации (torchvision v2) -> препроцесс бэкбона. Eval: только препроцесс
    (он же используется сервисом при инференсе — важно, чтобы совпадало). Всё —
    пиклуемые классы, чтобы работал ``num_workers > 0``.
    """
    post = _hf_preprocess(processor) if processor is not None else _fallback_preprocess(image_size)
    if not train:
        return Preprocess(post)

    from torchvision.transforms import v2
    cfg = dict(augment or {})
    size = int(cfg.get("size") or image_size)
    ops = [v2.RandomResizedCrop(size=(size, size),
                                scale=tuple(cfg.get("rrc_scale", (0.65, 1.0))),
                                ratio=tuple(cfg.get("rrc_ratio", (0.9, 1.1))))]
    if cfg.get("degrees"):
        ops.append(v2.RandomAffine(degrees=float(cfg["degrees"]),
                                   translate=tuple(cfg.get("translate", (0.04, 0.04))),
                                   shear=float(cfg.get("shear", 0.0))))
    cj = float(cfg.get("color_jitter", 0.0) or 0.0)
    if cj > 0:
        ops.append(v2.ColorJitter(brightness=cj, contrast=cj, saturation=cj,
                                  hue=min(0.5, cj / 4)))
    if cfg.get("grayscale", 0.0):
        ops.append(v2.RandomGrayscale(p=float(cfg["grayscale"])))
    if cfg.get("hflip", False):
        ops.append(v2.RandomHorizontalFlip(0.5))
    return Preprocess(post, aug=v2.Compose(ops))


# ------------------------------------------------------------------- датасеты
class ImageDataset(Dataset):
    """Записи CSV -> ``{'pixel_values': [C,H,W], 'label': int, 'slug': str, 'path': str}``.

    ``label`` — индекс slug'а в ``LabelSpace`` трейна (``-1`` если slug'а там нет,
    например тестовое вино вне трейна).
    """

    def __init__(self, records: list[dict], transform, label_space: LabelSpace | None = None):
        self.records = list(records)
        self.transform = transform
        self.label_space = label_space

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:  # noqa: BLE001 — битый файл не должен ронять эпоху
            print(f"[data] не читается {rec['image_path']}: {e}", flush=True)
            img = Image.new("RGB", (224, 224), "black")
        slug = str(rec["slug"])
        label = (self.label_space.index(slug)
                 if self.label_space is not None and slug in self.label_space else -1)
        return {"pixel_values": self.transform(img), "label": label,
                "slug": slug, "path": str(rec["image_path"])}


class PKBatchSampler(Sampler):
    """P классов × K объектов в батче (batch-hard triplets без пустых позитивов).

    Класс выбирается равномерно (борьба с long-tail), внутри класса — K объектов.
    Если у класса меньше K картинок, они берутся с повтором. ``num_batches`` по
    умолчанию = ``ceil(n_classes / P)`` — примерно одна эпоха на проход по классам.
    """

    def __init__(self, labels, p: int, k: int, num_batches: int | None = None, seed: int = 0):
        from collections import defaultdict
        self.p, self.k = max(1, int(p)), max(1, int(k))
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            self.by_label[int(lab)].append(i)
        self.classes = sorted(self.by_label)
        if not self.classes:
            raise ValueError("PKBatchSampler: пустой набор меток")
        self.num_batches = (int(num_batches) if num_batches
                            else max(1, -(-len(self.classes) // self.p)))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        p = min(self.p, len(self.classes))
        for _ in range(self.num_batches):
            chosen = rng.choice(self.classes, size=p, replace=False)
            batch: list[int] = []
            for cls in chosen:
                idx = self.by_label[int(cls)]
                replace = len(idx) < self.k
                pick = rng.choice(idx, size=self.k, replace=replace)
                batch.extend(int(x) for x in pick)
            yield batch


# ------------------------------------------------------------------ datamodule
class VinoDataModule(pl.LightningDataModule):
    """Данные пайплайна: ``reference.csv`` (эталоны) + ``train/val/test.csv`` (кропы).

    :param processor: HF image processor бэкбона (или None — тогда resize+normalize);
        тот же препроцесс, что и в сервисе (``ML service/encoder.py``).
    :param include_reference: ``none`` — трейн только по кропам; ``train`` — в трейн
        дополнительно подмешиваются эталоны тех slug'ов, что есть в трейне
        (cross-modal позитивы: «кроп должен стянуться к своей картинке референса»).
    :param p, k: параметры ``PKBatchSampler`` (P классов × K картинок в батче).
    """

    def __init__(self, reference_csv, train_csv, val_csv, test_csv, *, processor=None,
                 image_size: int = 224, batch_size: int = 32, p: int = 8, k: int = 4,
                 num_workers: int = 4, augment: dict | None = None,
                 include_reference: str = "none", path_col=None, slug_col=None, sep=None,
                 num_batches: int | None = None, seed: int = 0, pin_memory: bool = True):
        super().__init__()
        self.csvs = {"reference": reference_csv, "train": train_csv,
                     "val": val_csv, "test": test_csv}
        self.processor, self.image_size = processor, image_size
        self.batch_size, self.p, self.k = batch_size, p, k
        self.num_workers, self.augment = num_workers, augment or {}
        self.include_reference = include_reference
        self.path_col, self.slug_col, self.sep = path_col, slug_col, sep
        self.num_batches, self.seed = num_batches, seed
        self.pin_memory = pin_memory

        self.reference: list[dict] = []
        self.train: list[dict] = []
        self.val: list[dict] = []
        self.test: list[dict] = []
        self.train_pool: list[dict] = []
        self.label_space: LabelSpace | None = None
        self.train_sampler: PKBatchSampler | None = None
        self._ready = False

    # ------------------------------------------------------------------ setup
    def setup(self, stage=None) -> None:
        if self._ready:
            return
        kw = dict(path_col=self.path_col, slug_col=self.slug_col, sep=self.sep)
        self.reference = read_index_csv(self.csvs["reference"], **kw)
        self.train = read_index_csv(self.csvs["train"], **kw)
        self.val = read_index_csv(self.csvs["val"], **kw)
        self.test = read_index_csv(self.csvs["test"], **kw)
        if not self.train:
            raise ValueError(f"трейн пуст: {self.csvs['train']}")

        eval_tf = build_transforms(self.processor, self.image_size, train=False)
        train_tf = build_transforms(self.processor, self.image_size,
                                    augment=self.augment, train=True)

        pool = list(self.train)
        train_slugs = {r["slug"] for r in self.train}
        if self.include_reference == "train":
            extra = [r for r in self.reference if r["slug"] in train_slugs]
            pool += extra
            print(f"[data] к трейну подмешано {len(extra)} эталонов референсной базы "
                  f"(include_reference={self.include_reference})", flush=True)
        elif self.include_reference != "none":
            raise ValueError("include_reference: none|train")
        self.train_pool = pool
        self.label_space = LabelSpace(r["slug"] for r in pool)

        self.train_ds = ImageDataset(self.train_pool, train_tf, self.label_space)
        self.eval_ds = {"val": ImageDataset(self.val, eval_tf, self.label_space),
                        "test": ImageDataset(self.test, eval_tf, self.label_space)}
        self.gallery_ds = ImageDataset(self.reference, eval_tf, None)

        labels = [self.label_space.index(r["slug"]) for r in self.train_pool]
        self.train_sampler = PKBatchSampler(labels, self.p, self.k,
                                            num_batches=self.num_batches, seed=self.seed)
        self._ready = True
        n_extra = len(self.train_pool) - len(self.train)
        print(f"[data] трейн {len(self.train_pool)} ({len(self.train)} кропов"
              f"{f' + {n_extra} эталонов' if n_extra else ''}), "
              f"классов {len(self.label_space)}, val {len(self.val)}, test {len(self.test)}, "
              f"референс {len(self.reference)}", flush=True)

    def stats(self) -> dict:
        """Сводка по данным (в лог/артефакты)."""
        ref_slugs = {r["slug"] for r in self.reference}
        n_slugs = lambda recs: len({r["slug"] for r in recs})  # noqa: E731
        coverage = lambda recs: (round(sum(r["slug"] in ref_slugs for r in recs) / len(recs), 4)  # noqa: E731
                                 if recs else 0.0)
        return {
            "n_train_images": len(self.train), "n_train_pool": len(self.train_pool),
            "n_train_slugs": n_slugs(self.train), "n_classes": len(self.label_space or []),
            "n_val_images": len(self.val), "n_val_slugs": n_slugs(self.val),
            "n_test_images": len(self.test), "n_test_slugs": n_slugs(self.test),
            "n_reference_images": len(self.reference), "n_reference_slugs": len(ref_slugs),
            "val_reference_coverage": coverage(self.val),
            "test_reference_coverage": coverage(self.test),
            "include_reference": self.include_reference, "p": self.p, "k": self.k,
            "batch_size": self.p * self.k,
            "num_batches_per_epoch": len(self.train_sampler or []),
        }

    # -------------------------------------------------------------- лоадеры
    def on_train_epoch_start(self) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.trainer.current_epoch)

    def _loader(self, dataset, *, shuffle=False, batch_sampler=None, batch_size=None):
        workers = int(self.num_workers or 0)
        common = dict(num_workers=workers, pin_memory=bool(self.pin_memory and workers > 0))
        if batch_sampler is not None:      # PK-семплер задаёт батчи сам
            return DataLoader(dataset, batch_sampler=batch_sampler, **common)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)

    def train_dataloader(self):
        return self._loader(self.train_ds, batch_sampler=self.train_sampler)

    def val_dataloader(self):
        return self._loader(self.eval_ds["val"], batch_size=self.batch_size)

    def test_dataloader(self):
        return self._loader(self.eval_ds["test"], batch_size=self.batch_size)

    def gallery_dataloader(self):
        """Лоадер референсной базы — галерея для accuracy и cross-modal лосса."""
        return self._loader(self.gallery_ds, batch_size=self.batch_size)



