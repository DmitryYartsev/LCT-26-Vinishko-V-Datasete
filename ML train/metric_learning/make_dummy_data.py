# -*- coding: utf-8 -*-
"""Синтетика для смоук-теста пайплайна (пока реальные CSV с данными не готовы).

Генерит «этикетки», которые различаются между slug'ами и слегка варьируются внутри
slug'а: у каждого вина свой цвет-фон и свой узор из блоков, а картинки внутри slug'а —
это тот же узор со сдвигом/поворотом/яркостью/шумом. Так задача не вырождена и
triplet-обучение имеет шанс реально снизить лосс и поднять accuracy.

Создаёт в ``--out`` (по умолчанию ``paths.TRAIN_DUMMY`` = ``ML train/metric_learning/dummy_data``):

* ``images/<slug>/<name>.png`` — картинки;
* ``reference.csv`` — эталоны (пути ОТНОСИТЕЛЬНО папки csv — проверка резолвера),
* ``train.csv`` / ``val.csv`` / ``test.csv`` — «кропы» (абсолютные пути);
* один slug присутствует только в ``test.csv`` (не в reference) — проверка coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))                       # repo root (paths.py)
from paths import TRAIN_DUMMY                                     # noqa: E402

CSV_COLUMNS = ["image_path", "slug"]


def _rng_for(slug: str, seed: int) -> np.random.Generator:
    digest = hashlib.md5(f"{slug}:{seed}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:12], 16))


def make_pattern(slug: str, size: int = 96, grid: int = 6, seed: int = 0) -> Image.Image:
    """Базовый «дизайн этикетки» slug'а: цвет-фон + узор из блоков."""
    rng = _rng_for(slug, seed)
    base = tuple(int(v) for v in rng.integers(25, 235, 3))
    blocks = rng.integers(20, 240, size=(grid, grid, 3))
    keep = rng.random((grid, grid)) < 0.7
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    cell = size / grid
    for gy in range(grid):
        for gx in range(grid):
            if not keep[gy, gx]:
                continue
            box = (gx * cell, gy * cell, (gx + 1) * cell, (gy + 1) * cell)
            draw.rectangle(box, fill=tuple(int(v) for v in blocks[gy, gx]))
    return img


def jitter(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """«Фото с полки»: небольшой сдвиг/поворот, яркость/контраст, шум."""
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.8:                       # поворот на кратный шаг сетки
        arr = np.rot90(arr, k=int(rng.integers(0, 4)))
        h, w = arr.shape[:2]
        new = np.full((h, w, 3), arr.reshape(-1, 3).mean(axis=0), dtype=np.float32)
        dy, dx = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
        ys, xs = max(0, dy), max(0, dx)
        ye, xe = min(h, h + dy), min(w, w + dx)
        new[ys:ye, xs:xe] = arr[ys - dy:ye - dy, xs - dx:xe - dx]
        arr = new
    arr = arr * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-12, 12))
    arr = arr + rng.normal(0, 6, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Синтетические CSV+картинки для смоук-теста.")
    ap.add_argument("--out", default=str(TRAIN_DUMMY))
    ap.add_argument("--slugs", type=int, default=12)
    ap.add_argument("--n-reference", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="перезаписать существующее")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "train.csv").exists() and not args.force:
        print(f"[dummy] {out} уже заполнено (--force чтобы перезаписать)")
        return
    slugs = [f"dummy-wine-{i:02d}" for i in range(args.slugs)]
    test_only = ["dummy-wine-only-in-test"]          # нет в reference.csv -> coverage < 1
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = {"reference": [], "train": [], "val": [], "test": []}
    for slug in slugs + test_only:
        pattern = make_pattern(slug, size=args.size, seed=args.seed)
        counts = {"reference": args.n_reference, "train": args.n_train,
                  "val": args.n_val, "test": args.n_test}
        if slug in test_only:
            counts = {"reference": 0, "train": 0, "val": 0, "test": args.n_test}
        (img_dir / slug).mkdir(exist_ok=True)
        for split, n in counts.items():
            for i in range(n):
                rng = _rng_for(f"{slug}:{split}:{i}", args.seed)
                img = jitter(pattern, rng) if (split != "reference" or i > 0) else pattern
                path = img_dir / slug / f"{split}_{i:02d}.png"
                img.save(path)
                # reference пишем относительными путями — проверка резолвера в read_index_csv
                csv_path = path.relative_to(out) if split == "reference" else path
                rows[split].append({"image_path": str(csv_path), "slug": slug})

    for split, recs in rows.items():
        df = pd.DataFrame(recs, columns=CSV_COLUMNS)
        df.to_csv(out / f"{split}.csv", index=False)
        print(f"[dummy] {split}.csv: {len(df)} строк, "
              f"{df['slug'].nunique()} slug'ов -> {out / f'{split}.csv'}")
    print(f"[dummy] картинки: {img_dir} ({sum(len(v) for v in rows.values())} шт.)")
    print(f"[dummy] готово: python train.py --backbone tiny --data-dir {out} "
          f"--epochs 6 --clearml offline")


if __name__ == "__main__":
    main()
