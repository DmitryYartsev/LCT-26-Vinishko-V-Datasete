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
UPLOADS = ROOT / "raw_images"                         # распакованная медиатека Strapi (вход)
SCRAPED = ROOT / "solution/images/scraped_raw"        # веб-скрейп (в процессе; solution/ уберём позже)

# --- обработанные данные (выход process_images.py, gitignore) ---
FILTERED = ROOT / "filtered"                          # <slug>/NN.webp — все фото по вину
CATALOG_CSV = FILTERED / "catalog.csv"                # инфа про slug (карточки + список фото)

# --- данные для оценки/эталонов (внутри репо) ---
DATA = ROOT / "data"
EVAL_CSV = DATA / "eval.csv"                          # image,true_slug (233 фото)
EVAL_IMAGES = DATA / "eval"                           # папка с фото для EVAL_CSV
REFERENCE_CSV = DATA / "found_in_catalog_corrected.csv"  # каталог (2108 slug)
START_PHOTOS = DATA / "start_photos"                  # эталонные фото по slug
DATA_REPORTS = DATA / "reports"                       # отчёты прогонов пайплайна

# --- индекс для сервиса (gitignore) ---
INDEX_DIR = ROOT / "ML service" / "index"

# --- веса моделей (gitignore) ---
MODELS_DIR = ROOT / "models"                          # YOLO + локальная копия SigLIP 2
SIGLIP_LOCAL = MODELS_DIR / "siglip2-base-patch16-256"  # HF-совместимая папка энкодера

# --- дообучение энкодера: metric learning (ML train/) ---
TRAIN_DIR = ROOT / "ML train"
TRAIN_ARTIFACTS = TRAIN_DIR / "artifacts"             # чекпоинты/графики/отчёты (gitignore)
TRAIN_DUMMY = TRAIN_DIR / "dummy_data"                # синтетика для смоук-теста (gitignore)

# --- прочее ---
SERVICE_DIR = ROOT / "ML service"
EVAL_DIR = ROOT / "eval"                              # скрипт-оценщик + публичные query
EVAL_REPORTS = ROOT / "ML evaluation" / "reports"     # отчёты харнесса оценки (gitignore)
FUSION_WEIGHTS = SERVICE_DIR / "fusion_weights.json"  # веса калиброванного fusion-ранкера (gitignore)


def index_file(crop: bool) -> Path:
    return INDEX_DIR / ("catalog_crop.npz" if crop else "catalog.npz")


def meta_file(crop: bool) -> Path:
    return INDEX_DIR / ("meta_crop.json" if crop else "meta.json")
