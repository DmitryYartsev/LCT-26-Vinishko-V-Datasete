# -*- coding: utf-8 -*-
"""Собирает наборы query для валидации в querysets/*.jsonl.

Формат строки: {query_id, image_path(abs), true_slug, source, in_catalog}

Источники:
  dataset  image_labeling/dataset_v0/<slug>.<ext> — ОСНОВНОЙ тестовый набор (171 фото, 1 на вино).
  labeled  ручная разметка скрейпа (image_labeling/labels.csv, label=good) — ЧИСТЫЕ метки,
           ~1 фото на вино (220 шт).
  scrape   весь скрейп scraped_raw/<slug>/ (метка = имя папки) — метки ШУМНЫЕ (в папке
           бывают чужие кадры): абсолюты занижены, но для сравнения конфигов годится.

  uv run python build_queryset.py                         # dataset_v0 -> querysets/dataset.jsonl
  uv run python build_queryset.py --source labeled
  uv run python build_queryset.py --source scrape --max-per-wine 3
"""
import sys, json, argparse
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import SCRAPED, CATALOG_CSV, LABELS_CSV, DATASET_V0

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "querysets"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".webp", ".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


def _ok_image(p: Path) -> bool:
    try:
        return min(Image.open(p).size) >= 60
    except Exception:
        return False


def build_dataset():
    return [(p.stem, p) for p in sorted(DATASET_V0.iterdir()) if p.suffix.lower() in IMG_EXT]


def build_labeled():
    df = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
    df = df[df["label"].str.strip().str.lower() == "good"]
    return [(r["Slug"].strip(), SCRAPED / r["image"].strip()) for _, r in df.iterrows()]


def build_scrape(max_per_wine: int):
    pairs = []
    for wdir in sorted(d for d in SCRAPED.iterdir() if d.is_dir()):
        imgs = sorted(p for p in wdir.iterdir() if p.suffix.lower() in IMG_EXT)
        pairs += [(wdir.name, p) for p in imgs[:max_per_wine or None]]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dataset", "labeled", "scrape"], default="dataset")
    ap.add_argument("--max-per-wine", type=int, default=0, help="scrape: не больше N фото на вино (0 — все)")
    ap.add_argument("--out", help="имя файла в querysets/ (деф <source>.jsonl)")
    args = ap.parse_args()

    catalog_slugs = set(pd.read_csv(CATALOG_CSV, dtype=str, keep_default_na=False)["slug"])
    pairs = {"dataset": build_dataset, "labeled": build_labeled,
             "scrape": lambda: build_scrape(args.max_per_wine)}[args.source]()
    rows, skipped = [], 0
    for slug, img in pairs:
        if not _ok_image(img):
            skipped += 1; continue
        rows.append({"query_id": f"{args.source}-{len(rows):05d}", "image_path": str(img.resolve()),
                     "true_slug": slug, "source": args.source, "in_catalog": slug in catalog_slugs})
    out = OUT / (args.out or f"{args.source}.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_cat = sum(r["in_catalog"] for r in rows)
    print(f"{args.source}: {len(rows)} фото по {len({r['true_slug'] for r in rows})} винам "
          f"(в каталоге {n_cat}, пропущено битых/мелких {skipped}) -> {out}")


if __name__ == "__main__":
    main()
