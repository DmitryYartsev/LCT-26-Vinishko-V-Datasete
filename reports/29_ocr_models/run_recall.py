# -*- coding: utf-8 -*-
"""Контейнерная обёртка для `ML evaluation/recall_at_k.py`.

`paths.py` в контейнере считает DATA='/app/data' (в хост-раскладке), а данные смонтированы
в `/app/ref`. Обёртка подменяет константы ДО импорта харнесса и запускает его как `__main__`.

    python /app/work29/run_recall.py --config /app/config/pipeline.yaml \
        --eval-csv /app/work29/eval233_rest.csv --images-dir /app/ref/eval --k 30 --tag rest171
"""
import sys
from pathlib import Path

sys.path.insert(0, '/app')
import paths  # noqa: E402

paths.REFERENCE_CSV = Path('/app/ref/found_in_catalog_corrected.csv')
paths.EVAL_REPORTS = Path('/app/work29/reports')
paths.EVAL_CSV = Path('/app/ref/eval.csv')
paths.EVAL_IMAGES = Path('/app/ref/eval')
paths.EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, '/app/ml')
# db.py берёт DSN из своего модульного DB_URL (в образе /app/ml/paths.py с хостовым дефолтом)
import os  # noqa: E402
os.environ['DATABASE_URL'] = 'postgresql://vino:vino@db:5432/vino'
import db  # noqa: E402
db.DB_URL = 'postgresql://vino:vino@db:5432/vino'

import runpy  # noqa: E402

runpy.run_path('/app/work29/recall_at_k.py', run_name='__main__')
