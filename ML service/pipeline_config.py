# -*- coding: utf-8 -*-
"""Загрузка OmegaConf-конфига пайплайна и прокидка retrieval-параметров в env.

Единая точка для ``app.py`` (сервис) и ``pipeline.py`` (standalone-прогон).
Путь к YAML: env ``PIPELINE_CONFIG`` либо ``<repo>/config/pipeline.yaml``.
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

_DEFAULT = Path(__file__).resolve().parent.parent / 'config' / 'pipeline.yaml'


def load_config(path: str | os.PathLike | None = None):
    path = path or os.environ.get('PIPELINE_CONFIG', str(_DEFAULT))
    return OmegaConf.load(path)


def apply_retrieval_env(cfg) -> None:
    """retrieval.* -> env, который читают encoder/crop/db (исторически env-driven)."""
    r = cfg.retrieval
    os.environ['SEARCH_MODEL'] = str(r.model)
    os.environ['SEARCH_ENCODER'] = str(r.encoder)
    os.environ['SEARCH_BACKEND'] = str(r.backend)
    os.environ['SEARCH_API_URL'] = str(r.api_url)
    os.environ['SEARCH_API_TOKEN'] = str(r.api_token)
    os.environ['CROP_ENABLED'] = '1' if r.crop_enabled else '0'
    os.environ['CROP_MODEL'] = str(r.crop_model)
    os.environ['CROP_MARGIN'] = str(r.crop_margin)
    os.environ['CROP_MIN_CONF'] = str(r.crop_min_conf)
    os.environ['USE_LABEL_BRANCH'] = '1' if r.label_enabled else '0'
    os.environ['LABEL_MODEL'] = str(r.label_model)
    os.environ['LABEL_MIN_CONF'] = str(r.label_min_conf)
    os.environ['LABEL_MARGIN'] = str(r.label_margin)
    os.environ['LABEL_ALIGN'] = '1' if r.label_align else '0'
    os.environ['SEARCH_PIPELINE'] = str(r.pipeline)
    # OCR-вход: что подаём VLM (crop.py: ocr_label_crop). align = историческое поведение.
    o = cfg.ocr
    os.environ['OCR_LABEL_CROP'] = str(getattr(o, 'label_crop', 'align'))
    os.environ['LABEL_CROP_MIN_CONF'] = str(getattr(o, 'label_crop_min_conf', 0.45))
    os.environ['LABEL_CROP_MIN_AREA'] = str(getattr(o, 'label_crop_min_area', 0.02))
    os.environ['LABEL_CROP_MAX_ASPECT'] = str(getattr(o, 'label_crop_max_aspect', 3.0))
    os.environ['THRESH_SCORE'] = str(r.thresh_score)
    os.environ['THRESH_SCORE_LO'] = str(getattr(r, 'thresh_score_lo', 0.70))
    os.environ['THRESH_MARGIN'] = str(r.thresh_margin)
    os.environ['EVAL_ABSTAIN'] = '1' if r.eval_abstain else '0'
    # Гейт «есть в каталоге»: между LO и HI карточка показывается только если OCR
    # подтвердил кандидата (см. reports/27_gate_negatives.md).
    os.environ['OCR_CONFIRM_CONF'] = str(getattr(cfg.ocr, 'confirm_confidence', 0.80))
    os.environ['OCR_AGREE_CONF'] = str(getattr(cfg.csv_match, 'agree_confidence', 0.60))
    os.environ['DATABASE_URL'] = str(r.database_url)
    # политика кропа (аудит Reports/15_Crop_audit.md). Исторические значения по
    # умолчанию: старые конфиги и уже собранный индекс остаются валидными.
    os.environ['CROP_PICK'] = str(getattr(r, 'crop_pick', 'big_center'))
    os.environ['CROP_FALLBACK'] = str(getattr(r, 'crop_fallback', 'orig'))
    os.environ['CROP_MIN_AREA'] = str(getattr(r, 'crop_min_area', 0.0))
    os.environ['LABEL_PICK'] = str(getattr(r, 'label_pick', 'conf'))
    os.environ['LABEL_MIN_W_FRAC'] = str(getattr(r, 'label_min_w_frac', 0.0))
    # Политика ТОЛЬКО для запроса (v3-рецепт аудита Reports/15_Crop_audit.md): индекс
    # собирается `crop_*`, а кроп запроса — `query_crop_*` (crop.py: use_query_policy()).
    # Пустое/отсутствующее значение = оверрайда нет, запрос идёт политикой индекса.
    for key, env in (('query_crop_pick', 'QUERY_CROP_PICK'),
                     ('query_crop_fallback', 'QUERY_CROP_FALLBACK'),
                     ('query_crop_min_conf', 'QUERY_CROP_MIN_CONF'),
                     ('query_crop_min_area', 'QUERY_CROP_MIN_AREA'),
                     ('query_label_pick', 'QUERY_LABEL_PICK'),
                     ('query_label_min_w_frac', 'QUERY_LABEL_MIN_W_FRAC')):
        val = getattr(r, key, None)
        os.environ[env] = '' if val is None else str(val)
