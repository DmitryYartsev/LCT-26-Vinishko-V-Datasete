# -*- coding: utf-8 -*-
"""Единые пути проекта. Импортируется всеми скриптами, чтобы не хардкодить абсолюты.

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import FILTERED, INDEX_DIR, ...
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- исходники организатора (gitignore) ---
ORG = ROOT / "org_files"
CATALOG_SRC = ORG / "strapi_output0709.csv"          # оригинальный дамп каталога (CSV)

# --- сырые данные изображений (gitignore) ---
UPLOADS = ROOT / "solution/images/_uploads_raw"       # распакованная медиатека Strapi (вход)
SCRAPED = ROOT / "solution/images/scraped_raw"        # веб-скрейп (в процессе; не трогать)

# --- обработанные данные (выход process_images.py, gitignore) ---
FILTERED = ROOT / "filtered"                          # <slug>/NN.webp — все фото по вину
CATALOG_CSV = FILTERED / "catalog.csv"                # инфа про slug (карточки + список фото)

# --- индекс для сервиса (gitignore) ---
INDEX_DIR = ROOT / "service" / "index"

# --- прочее ---
SERVICE_DIR = ROOT / "service"
EVAL_DIR = ROOT / "eval"                              # скрипт-оценщик + публичные query


def index_file(crop: bool) -> Path:
    return INDEX_DIR / ("catalog_crop.npz" if crop else "catalog.npz")


def meta_file(crop: bool) -> Path:
    return INDEX_DIR / ("meta_crop.json" if crop else "meta.json")
