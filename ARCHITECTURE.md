# Архитектура

Пайплайн: **фото этикетки → нормализация → эмбеддинг → поиск по каталогу (pgvector) → карточка JSON**.

```
[web : Nuxt]  --/api/scan-->  [ml : FastAPI]  --SQL <=>-->  [db : Postgres + pgvector]
 mobile-first UI                SigLIP/ViT + YOLO-кроп          wines + wine_vectors
 (стиль «Своё Вино»)            warm-up строит индекс
```

## Слои

| Слой | Где | Что делает |
|---|---|---|
| Нормализация фото | `ML service/crop.py` | YOLO-кроп бутылки (env `CROP_ENABLED`), изолирует этикетку от фона/соседей |
| Извлечение признаков | `ML service/encoder.py` | картинка → L2-вектор. Модель/бэкенд через env (`SEARCH_MODEL`, `SEARCH_BACKEND=local\|remote`) |
| Индекс | `ML service/build_index.py` + `db.py` | warm-up: сид каталога + эмбеддинг `filtered/` → `wine_vectors` (pgvector, мультивектор) |
| Поиск | `ML service/db.py` `search()` | косинус `emb <=> q`, агрегация max по slug, `margin` = top1−top2, порог `in_catalog` |
| Выдача | `ML service/app.py` | `/v1/eval/predict` (плоский `{slug}`), `/v1/search` (карточка + уверенность), `/wine`, `/ref` |
| Бизнес-бэкенд + UI | `web/` (Nuxt) | Nitro-прокси к `ml`, mobile-first карточка вина в стилистике портала |

## Данные

- `wines(slug, name, winery, category, color, region, grape, description, rating)` — карточки.
- `wine_vectors(slug, model, emb vector)` — по вектору на каждое фото вина (мультивектор). Колонка
  `model` позволяет держать вектора **нескольких моделей одновременно**; `emb` — dimensionless
  (у разных моделей разный dim), ANN-индекс не строим (каталог мал, full scan быстрый). Поиск
  фильтрует по текущей модели (`WHERE model=…`).
- `models(model, dim, backend, n_vectors, built_at)` — какие модели проиндексированы.
- Источник карточек/фото: `filtered/` + `filtered/catalog.csv` (готовит `eda and image filtering/process_images.py`).

## Границы / решения

- **Индекс живёт в pgvector** (source of truth), поиск делает сам `ml` (у него и эмбеддинг запроса, и коннект к БД).
- **Свап модели без кода**: сменил `SEARCH_MODEL`/`SEARCH_BACKEND` → warm-up (`--force`) пересобирает индекс под новую размерность. `remote` — чтобы во время разработки не держать GPU-машину.
- **Near-duplicates** (одна серия, разный год/категория) — главный источник ошибок; рычаги: кроп, разрешение/модель посильнее (so400m), позже OCR-реранк.
- **Метрики**: F1/recall меряются офлайн в `ML evaluation/evaluate.py` (HTTP к `ml`); в API отдаём `score`+`margin` как уверенность top-1/top-5.
- **Retention-фича после поиска** (аналоги / сомелье) — планируется отдельным шагом.
