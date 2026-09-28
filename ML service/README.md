# ML service — инференс-сервис сканера

FastAPI-сервис: фото этикетки → карточка вина. Ядро — кроп бутылки (YOLO) → эмбеддинг
(SigLIP 2) → косинусный поиск по индексу каталога (`index/catalog[_crop].npz`) →
(опционально) реранк top-K мультимодальным реранкером по фото эталонов.

## Файлы

| Файл | Что |
|---|---|
| `app.py` | FastAPI: эндпоинты, склейка препроцесс→энкодер→поиск |
| `encoder.py` | SigLIP 2: картинка → L2-нормированный вектор (`get_encoder()`) |
| `crop.py` | кроп бутылки (COCO-YOLO), env-выключатель `CROP_ENABLED` |
| `rerank.py` | реранк top-K: фото запроса vs эталоны через Jina m0 / Qwen3-VL-Rerank (env `RERANK_BACKEND`) |
| `search.py` | индекс в памяти + косинусный поиск + карточки (`CatalogIndex`) |
| `build_index.py` | строит `index/catalog[_crop].npz` из `filtered/` (эмбеддинги + crop) |
| `static/index.html` | мини-UI в стиле «Своё вино» |
| `index/` | `catalog.npz` / `catalog_crop.npz` (+ meta) — строит `build_index.py` |

## Запуск

```bash
cd "ML service"
HF_HUB_OFFLINE=1 uv run uvicorn app:app --host 127.0.0.1 --port 8080
# открыть http://127.0.0.1:8080/
```

Индекс должен быть построен заранее (`build_index.py`; эталоны готовит
`../eda and image filtering/process_images.py`). Кроп в сервисе и в индексе
обязан совпадать — управляется одним `CROP_ENABLED`.

```bash
cd "ML service"
uv run python build_index.py            # crop по CROP_ENABLED (деф 1)
uv run python build_index.py --both     # оба индекса (crop off + on)
```

## Эндпоинты

| Метод | Путь | Ответ |
|---|---|---|
| POST | `/v1/eval/predict` | `{"slug","score","margin"}` — для скрипта-оценщика |
| POST | `/v1/search?k=5` | top-k (деф 5, макс 50) + score/margin/in_catalog + карточка |
| GET | `/` | мини-UI |
| GET | `/ref/{slug}` | эталонное фото вина (из `filtered/<slug>/`) |
| GET | `/wine/{slug}` | карточка по slug |
| GET | `/health` | статус (модель, размер индекса, crop) |

## Переменные окружения

| ENV | Деф. | Смысл |
|---|---|---|
| `SIGLIP_MODEL` | `google/siglip2-base-patch16-256` | энкодер (GPU — `...so400m-patch16-384`) |
| `CROP_ENABLED` | `1` | кроп бутылки; переключает и препроцесс, и файл индекса |
| `CROP_MODEL` / `CROP_MARGIN` / `CROP_MIN_CONF` | `yolo11n.pt` / `0.06` / `0.25` | параметры кропа |
| `THRESH_SCORE` / `THRESH_MARGIN` | `0.75` / `0.015` | пороги `in_catalog` (черновые, калибровать) |
| `EVAL_ABSTAIN` | `0` | `1` — отдавать `null` при низкой уверенности |
| `RERANK_BACKEND` | `none` | `jina` (jina-reranker-m0) / `dashscope` (qwen3-vl-rerank) / `none` |
| `RERANK_TOP_K` | `10` | сколько кандидатов из pgvector отдавать реранкеру |
| `RERANK_WEIGHT` | `0.5` | доля реранкера в итоговом скоре (min-max внутри top-K); `1.0` — только реранкер |
| `RERANK_TIMEOUT` / `RERANK_IMG_SIDE` | `5` / `448` | таймаут, с / макс. сторона картинок (влияет на расход токенов) |
| `JINA_API_KEY` | — | ключ Jina (для `jina`) |
| `DASHSCOPE_API_KEY` / `DASHSCOPE_RERANK_URL` | — | ключ и полный URL rerank-эндпоинта Model Studio (с WorkspaceId и регионом) |
| `HF_HUB_OFFLINE` | — | `1` — модель из кэша, без похода в HuggingFace |

## Реранк

После pgvector берётся top-`RERANK_TOP_K`, и реранкер попарно сравнивает кропнутое фото
запроса с эталоном каждого кандидата (первое фото из `filtered/<slug>/`, base64, кэш в памяти).
Итог: `final = w·norm(rerank) + (1−w)·norm(visual)`. В `/v1/search` у результатов появляются
`rerank_score` / `final_score`, а в ответе — блок `rerank` (backend, ms, changed_top1, error).
`in_catalog`, `top1_score` и `margin` считаются по визуальному скору (до реранка). Если реранкер
упал или не ответил за таймаут, остаётся визуальный порядок.

## Замечания

- Индекс **мультивекторный**: на вино может быть несколько фото (несколько строк в npz
  с одним slug); матч = max cosine по slug (`CatalogIndex.search`).
- Пути берутся из корневого `paths.py` — хардкода абсолютов нет.
