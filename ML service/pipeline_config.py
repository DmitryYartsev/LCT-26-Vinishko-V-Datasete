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
    os.environ['THRESH_SCORE'] = str(r.thresh_score)
    os.environ['THRESH_MARGIN'] = str(r.thresh_margin)
    os.environ['EVAL_ABSTAIN'] = '1' if r.eval_abstain else '0'
    os.environ['DATABASE_URL'] = str(r.database_url)
