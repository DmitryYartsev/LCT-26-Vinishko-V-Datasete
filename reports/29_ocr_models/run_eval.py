# -*- coding: utf-8 -*-
"""Прогон оценки внутри контейнера ml с выбором конфига (для сравнения OCR-моделей).

Отличие от `reports/23_ml_dev2_pipeline_eval/container_eval.py` — параметр `--config`,
поэтому можно гонять варианты пайплайна (`config/pipeline.gemini-flash.yaml` и т.п.)
рядом с дефолтным, не трогая рабочий конфиг приложения.

    docker cp reports/29_ocr_models/run_eval.py <ml>:/app/work29/run_eval.py
    docker exec <ml> python /app/work29/run_eval.py --config /app/config/pipeline.gemini-flash.yaml \
        --eval-csv /app/ref/eval62.csv --images-dir /app/ref/real_photo --out-dir /app/tmp/29/gf62
"""
import argparse
import sys

sys.path.insert(0, '/app/ml')
sys.path.insert(0, '/app')

import pipeline as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='/app/config/pipeline.yaml')
ap.add_argument('--eval-csv', required=True)
ap.add_argument('--images-dir', required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

cfg = P.load_config(a.config)
P.apply_retrieval_env(cfg)
print(f'[run-eval] config={a.config} ocr.model={cfg.ocr.model} '
      f'pipeline={cfg.retrieval.pipeline} top_k={cfg.retrieval.get("top_k")}', flush=True)

rank, enc, conn = P.build_rank(cfg)
extractor, matcher, verifier, fusion = P.build_ocr_components(cfg, enc)
P.run_eval(cfg, rank, enc, extractor, matcher, verifier, a, fusion)
