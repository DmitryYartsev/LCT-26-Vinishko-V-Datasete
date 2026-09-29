# -*- coding: utf-8 -*-
"""Изолированный замер скорости OCR-моделей (одна и та же картинка и тот же промпт).

Использует `OcrExtractor` из сервиса, поэтому payload/prompt/параметры ровно те же,
что в пайплайне; отличается только `ocr.model` из конфига.

    docker cp reports/29_ocr_models/latency_bench.py <ml>:/app/work29/
    docker exec <ml> python /app/work29/latency_bench.py --calls 3 \
        --out /app/tmp/29/latency.json
"""
import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, '/app/ml')
sys.path.insert(0, '/app')

from PIL import Image  # noqa: E402
import pipeline as P  # noqa: E402
from ocr_rerank import OcrExtractor  # noqa: E402

MODELS = {
    'openai/gpt-4o-mini': '/app/config/pipeline.yaml',
    'google/gemini-2.5-flash': '/app/config/pipeline.gemini-flash.yaml',
    'google/gemini-2.5-flash-lite': '/app/config/pipeline.gemini-flash-lite.yaml',
}
IMAGES = ['/app/ref/real_photo/48.98_02-09-2026_18-34-15.webp',
          '/app/ref/real_photo/68.81_05-09-2026_15-46-31.webp']

ap = argparse.ArgumentParser()
ap.add_argument('--calls', type=int, default=3, help='вызовов на модель и картинку')
ap.add_argument('--out', default='/app/tmp/29/latency.json')
a = ap.parse_args()

result = {}
for model, cfg_path in MODELS.items():
    cfg = P.load_config(cfg_path)
    P.apply_retrieval_env(cfg)
    ex = OcrExtractor(cfg)
    per_image, tokens = {}, []
    for img_path in IMAGES:
        img = Image.open(img_path)
        times, usages = [], []
        for _ in range(a.calls):
            t0 = time.time()
            r = ex.extract(img)
            times.append(round(time.time() - t0, 2))
            usage = r.get('usage') or {}
            usages.append({'prompt_tokens': usage.get('prompt_tokens', 0),
                           'completion_tokens': usage.get('completion_tokens', 0),
                           'total_tokens': usage.get('total_tokens', 0)})
            if r.get('error'):
                print(f'  [{model}] ошибка: {r["error"]}', flush=True)
        tokens.extend(usages)
        per_image[img_path.split('/')[-1]] = {'times_s': times,
                                              'mean_s': round(statistics.mean(times), 2),
                                              'tokens': usages,
                                              'fields_ok': bool(r.get('fields'))}
        print(f'[{model}] {img_path.split("/")[-1][:24]}: {times} с, '
              f'токены {usages[-1]}', flush=True)
    all_times = [t for v in per_image.values() for t in v['times_s']]
    result[model] = {'per_image': per_image,
                     'mean_s': round(statistics.mean(all_times), 2),
                     'median_s': round(statistics.median(all_times), 2),
                     'min_s': min(all_times), 'max_s': max(all_times),
                     'tokens_mean': {k: round(statistics.mean(t[k] for t in tokens), 1)
                                     for k in ('prompt_tokens', 'completion_tokens',
                                               'total_tokens')}}
    print(f'[{model}] итог: среднее {result[model]["mean_s"]} с, '
          f'медиана {result[model]["median_s"]} с, токенов {result[model]["tokens_mean"]}',
          flush=True)

json.dump({'note': f'изолированный замер: {len(IMAGES)} картинки x {a.calls} вызова на модель, '
                   'тот же промпт/параметры, что в пайплайне',
           'models': result}, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('сохранено:', a.out, flush=True)
