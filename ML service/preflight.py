# -*- coding: utf-8 -*-
"""Предполётная проверка: есть ли данные и модели, без которых сервис не поднимется.

Реализация задачи «развернуть на другом сервере»: файлы кода/конфигов лежат в
репозитории, а данные (`data/`, `filtered/`), модели (`models/`) и ключ `.env`
подкладываются на сервере (скрипт загрузки + ручной перенос ключа). Если чего-то нет,
сервис должен сказать об этом ЧЕЛОВЕЧЕСКИ, а не падать с ошибкой HuggingFace.

    python3 preflight.py [--config ../config/pipeline.yaml]   # на хосте
    (в контейнере вызывается из app.startup до загрузки моделей)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def missing_items(cfg) -> list:
    """Список отсутствующих ОБЯЗАТЕЛЬНЫХ путей (пустой список = всё на месте)."""
    miss = []

    def need(path, what: str, is_dir: bool = False) -> None:
        p = Path(str(path))
        ok = p.is_dir() if is_dir else p.exists()
        if not ok:
            miss.append(f'{what}: {p}')

    # --- данные каталога/эталонов (data/, монтируется как /app/ref) ---
    need(cfg.paths.reference_csv, 'каталог для OCR-матчера (CSV)')
    need(cfg.paths.start_photos, 'эталонные фото каталога (start_photos)', is_dir=True)
    need(cfg.paths.prompt_file, 'промт сопоставления (prompts_wine_match.txt)')

    # --- индекс: фото каталога и их карточки (filtered/) ---
    from paths import FILTERED, CATALOG_CSV          # noqa: E402 (пути проекта)
    need(CATALOG_CSV, 'карточки каталога (filtered/catalog.csv)')
    need(FILTERED, 'фото каталога (filtered/<slug>/*)', is_dir=True)

    # --- модели (models/, монтируется как /app/models) ---
    for what, val in (('модель кропа бутылки (YOLO)', cfg.retrieval.crop_model),
                      ('модель кропа этикетки (YOLO)', cfg.retrieval.label_model)):
        p = Path(str(val))
        if p.is_absolute() or str(val).endswith('.pt'):
            need(p, what)
    model = str(cfg.retrieval.model)
    if model.startswith('/') or model.startswith('.'):          # локальная папка энкодера
        need(model, 'энкодер SigLIP2 (локальная папка)', is_dir=True)
    return miss


def warnings(cfg) -> list:
    """Не фатальные, но важные предупреждения (без них сервис работает хуже)."""
    out = []
    if bool(getattr(cfg.ocr, 'enabled', False)) and not os.environ.get('OPENROUTER_API_KEY'):
        out.append('нет OPENROUTER_API_KEY: OCR-rerank работать не будет (только '
                   'retrieval). Положите .env в корень репозитория (OPENROUTER_API_KEY='
                   'sk-or-...) или задайте переменную окружения')
    return out


def check(cfg, where: str = 'сервис') -> list:
    """Печатает понятный отчёт; возвращает список ФАТАЛЬНЫХ проблем (пусто = ok)."""
    miss = missing_items(cfg)
    warns = warnings(cfg)
    for w in warns:
        print(f'[preflight] ПРЕДУПРЕЖДЕНИЕ: {w}', flush=True)
    if not miss:
        if not warns:
            print('[preflight] всё на месте ✓', flush=True)
        return []
    print(f'\n[preflight] НЕ ХВАТАЕТ ФАЙЛОВ для запуска ({where}):', flush=True)
    for m in miss:
        print(f'  - {m}', flush=True)
    print('  Данные/модели кладутся на сервере скриптом загрузки (data/ -> /app/ref, '
          'models/ -> /app/models, filtered/ -> /app/filtered); ключ — в .env.',
          flush=True)
    return miss


def _load_repo_env() -> None:
    """Простой парсер <repo>/.env (KEY=value или KEY=\"value\") — как в pipeline.py."""
    envf = Path(__file__).resolve().parents[1] / '.env'
    if not envf.is_file():
        return
    for line in envf.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ap = argparse.ArgumentParser(description='проверка готовности окружения к запуску')
    ap.add_argument('--config', default=None)
    args = ap.parse_args(argv)
    _load_repo_env()
    from pipeline_config import load_config
    cfg = load_config(args.config)
    if args.config is None:                 # локальный прогон: пути из репозитория
        try:
            from pipeline import apply_host_paths
            apply_host_paths(cfg)
        except Exception:                   # noqa: BLE001 — конфиг и так контейнерный
            pass
    miss = check(cfg, where='хостовый прогон')
    if miss:
        return 1
    print('[preflight] всё на месте ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
