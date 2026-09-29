# -*- coding: utf-8 -*-
"""Единые пути проекта (ROOT-relative, без хардкода абсолютов).

В контейнерах файл монтируется как /app/paths.py, данные — рядом (/app/filtered, ...).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- фото каталога и карточки (скачивает сервис fetch, gitignore) ---
FILTERED = ROOT / "filtered"                          # <slug>/NN.webp — все фото по вину
CATALOG_CSV = FILTERED / "catalog.csv"                # карточки вин (slug, name, winery, ...)

# --- атрибуты вин с vino-svoe.ru (блюда/крепость/подача; вход сомелье, в git) ---
SOMMELIER_DIR = ROOT / "sommelier service"
WINES_PARSED = SOMMELIER_DIR / "wines_parsed.jsonl"   # карточки сайта (по site-slug)
WINES_SLUG_MAP = SOMMELIER_DIR / "slug_map.csv"       # slug каталога -> site_slug (+ метод/score матча)
