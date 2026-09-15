# -*- coding: utf-8 -*-
"""Собирает наборы query для валидации в querysets/*.jsonl.

Формат строки: {query_id, image_path(abs), true_slug, source, in_catalog}

Пока источник — scrape (наскрейпленные веб-фото, метка = имя папки=slug).
ВАЖНО: метки скрейпа ШУМНЫЕ (в папке вина бывают чужие/мусорные кадры) — абсолютные
числа занижены, но для СРАВНЕНИЯ конфигов (crop on/off и т.п.) годится: шум одинаков.
Позже добавим synthetic (аугментации эталонов) и oocatalog (негативы)."""
import sys, json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import SCRAPED, CATALOG_CSV

sys.stdout.reconfigure(encoding="utf-8")
SCRAPE = SCRAPED
OUT = Path(__file__).resolve().parent / "querysets"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".webp", ".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


def build_scrape():
    import pandas as pd
    catalog_slugs = set(pd.read_csv(CATALOG_CSV, dtype=str, keep_default_na=False)["slug"])
    rows, skipped = [], 0
    for wdir in sorted(SCRAPE.iterdir()):
        if not wdir.is_dir():
            continue
        slug = wdir.name
        in_cat = slug in catalog_slugs
        for img in sorted(wdir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            try:
                w, h = Image.open(img).size
                if min(w, h) < 60:
                    skipped += 1; continue
            except Exception:
                skipped += 1; continue
            rows.append({"query_id": f"scrape-{len(rows):05d}",
                         "image_path": str(img), "true_slug": slug,
                         "source": "scrape", "in_catalog": in_cat})
    out = OUT / "scrape.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    wines = len({r["true_slug"] for r in rows})
    print(f"scrape queryset: {len(rows)} фото по {wines} винам (пропущено {skipped}) -> {out}")


if __name__ == "__main__":
    build_scrape()
