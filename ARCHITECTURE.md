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
| Нормализация фото | `ML service/crop.py` | YOLO-кроп бутылки (env `CROP_ENABLED`), изолирует этикетку от фона/соседей. Политика кропа **разная для индекса и запроса**: индекс — `crop_*` (стабильность pgvector-векторов), запрос — `query_crop_*` (UGC-фото полки; `reports/15_Crop_audit.md`, 16) |
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
- **Near-duplicates** (одна серия, разный год/категория) — главный источник ошибок. Реализован
  **OCR-rerank** (`ML service/ocr_rerank.py`): VLM читает выпрямленный (rectify) кроп этикетки,
  параметры матчатся по CSV с высоким весом года/цвета/типа, а решение принимается по разрыву
  CSV top1↔top2 (retrieval — fallback; косинус и token-overlap не сравниваются). Прочие рычаги:
  кроп, разрешение/модель посильнее (so400m).
- **Кроп запроса ≠ кроп индекса** (осознанное отступление от «препроцесс одинаков»): на
  студийных фото каталога снижение порога бутылочного детектора и fallback на этикетку
  по полному фото ломают уже верные пары, а на UGC-фото полки — чинят. Поэтому политика
  индекса (`crop_*`, `build_index.py`) и запроса (`query_crop_*`, query-тракт: `app.py`,
  `pipeline.py`, `recall_at_k.py`) разведены; переключение — явное
  (`crop.use_query_policy()`), кропы эталонов всегда под `index_policy()`. Даёт
  recall@1 0.677 → 0.758 без пересборки индекса.
- **Потолок ре-ранка** (P0): recall@30 визуального пула = 87.6%, 12.4% фото правильный slug
  не попадает даже в top-30 → добавлен **полнокаталожный текстовый путь**
  (`ML service/text_retrieval.py`: CsvMatcher + multilingual-e5) и **fusion-ранкер**
  (`ML service/rerank_fusion.py`) над объединённым пулом image+text. Union поднимает потолок
  до ~94%; ранкер включается флагом `fusion.enabled` и калибруется офлайн
  (`ML evaluation/train_fusion.py`). Метрики и харнесс — в `ML evaluation/` (`recall_at_k.py`,
  `union_text_eval.py`, `audit_refs.py`).
- **Метрики**: F1/recall меряются офлайн в `ML evaluation/evaluate.py` (HTTP к `ml`); в API отдаём `score`+`margin` как уверенность top-1/top-5.
- **Retention-фича после поиска** (аналоги / сомелье) — планируется отдельным шагом.
