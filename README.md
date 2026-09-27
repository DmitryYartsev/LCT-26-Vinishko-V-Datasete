# Сканер российских вин «Своё вино» (кейс РСХБ, ЛЦТ-2026)

Сервис распознаёт вино по фотографии этикетки (снятой у полки — под углом, с бликами)
и возвращает **одну** карточку из каталога «Своё вино». Ядро — image retrieval:
кроп бутылки (YOLO) → эмбеддинг (SigLIP 2) → поиск ближайшего эталона по косинусной близости.

Формат для скрипта-оценщика: `POST /v1/eval/predict`, multipart-поле `image` → `{"slug":"..."}`.
Есть мини-UI: `GET /` (загрузка фото → карточка + top-5), открывается на `http://127.0.0.1:8080/`.

**Доп. фича — цифровой сомелье** (`sommelier service/`): пользователь заранее рассказывает текстом или
голосом, к чему/для какого повода берёт вино; LLM собирает профиль предпочтений, сервис подбирает вина,
а на каждой отсканированной этикетке показывает «под ваш запрос: N%» с причинами. Сервис stateless —
переписка живёт на фронте.

---

## Данные (что уже разобрано)

| Что | Значение |
|---|---|
| Каталог `strapi_output0709.csv` | 4147 строк → **2103 уникальные позиции** (по `Slug`) |
| Категории | Белое 999 / Красное 795 / Розовое 293 / Игристое 16 |
| Near-duplicate серии | 131 серия (один дизайн, разный год/категория) = 521 позиция |
| Медиатека Strapi (`*.rar`) | 15 770 картинок → 5810 уникальных (после снятия size-вариантов и контент-дедупа) |
| **Эталонов сопоставлено** | **1977 / 2103 (94%)** → `solution/images/reference/<slug>.webp` |
| Публичный eval (`eval.zip`) | 3 фото (2 из них — вина ВНЕ каталога → тест на «не найдено») |

Подробности EDA — в `solution/eda/` (`summary.json`, `dedupe_summary.json`, `reference_map_full.csv`).

---

## Установка (uv)

```bash
uv venv --python 3.13
uv pip install -e .
uv run playwright install chromium   # если браузер ещё не установлен
```

Проверка:
```bash
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

**GPU (опционально).** По умолчанию ставится CPU-сборка torch. Для GPU (CUDA 12.1):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Структура

Три раздела в корне + центральный `paths.py` (все пути в одном месте, без хардкода):

```
eda and image filtering/    # EDA + фильтрация мусора (офлайн, не рантайм)   [README]
  eda.ipynb                 #   разведочный анализ каталога
  process_images.py         #   дамп -> filtered/<slug>/*.webp + catalog.csv
  scrape_images.py          #   сбор доп. фото из веба (Yandex+Playwright)
ML service/                 # инференс-сервис + построение индекса           [README]
  app.py encoder.py crop.py search.py build_index.py  static/index.html  index/*.npz
ML evaluation/              # валидация качества (recall@1/@5, ...)          [README]
sommelier service/          # цифровой сомелье: LLM-диалог, STT, подбор/скоринг [README]
web/                        # Nuxt: mobile-first UI (сканер + сомелье), Nitro-прокси к ml/sommelier
  build_queryset.py evaluate.py  querysets/ reports/
filtered/                   # выход process_images: фото по slug + catalog.csv  (gitignore)
raw_images/                 # распакованная медиатека Strapi (вход process_images) (gitignore)
org_files/                  # исходники организатора: pdf/pptx/csv/zip/rar     (gitignore)
models/                     # веса YOLO                                        (gitignore)
solution/images/scraped_raw # веб-скрейп (временно; solution/ уберём позже)   (gitignore)
eval/                       # публичный eval + participant_test.sh (скрипт оценщика)
paths.py  pyproject.toml  README.md  .gitignore
```

У каждого раздела свой README. Данные (`filtered/`, `org_files/`, `solution/`,
`ML service/index/`) в git не попадают.

---

## Быстрый старт (Docker)

Весь сервис (`db` + `ml` + `sommelier` + `web`) поднимается одной командой. Нужен готовый `filtered/`
(шаг 1 ниже) — из него warm-up строит индекс в pgvector на первом старте `ml`. Атрибуты вин для
сомелье лежат в git (`sommelier service/wines_parsed.jsonl` + `slug_map.csv`). Ключ OpenRouter — в `.env`
(см. `.env.example`).

```bash
cp .env.example .env   # вписать OPENROUTER_API_KEY
docker compose up --build
# web (UI):   http://localhost:3000
# ml (API):   http://localhost:8080/health
# sommelier:  http://localhost:8090/health
# оценщик:    POST http://localhost:8080/v1/eval/predict  (multipart image -> {"slug": ...})
```

Свап модели без правки кода — через env (см. таблицу ниже), например лёгкая/удалённая на время разработки:
```bash
SEARCH_MODEL=google/vit-base-patch16-224 SEARCH_BACKEND=remote SEARCH_API_URL=https://... \
  docker compose up --build ml
```

## Запуск шагов (локально, без Docker)

**1. Фильтрация изображений** (дамп → `filtered/<slug>/*.webp` + `catalog.csv`):
```bash
cd "eda and image filtering"
uv run python process_images.py
```

**2. Построение индекса** (warm-up: сид каталога + эмбеддинги → pgvector). Обычно это делает
сам `ml` на старте; вручную (нужен поднятый Postgres и `DATABASE_URL`):
```bash
cd "ML service"
uv run python build_index.py           # собрать, если пусто
uv run python build_index.py --force   # пересобрать (после смены модели)
```

**3. EDA** — `eda and image filtering/eda.ipynb`.

**4. Сбор доп. фото из веба** (для мульти-вектора и валидации):
```bash
cd "eda and image filtering"
uv run python scrape_images.py --all --per-wine 6 --sleep 3   # весь каталог
uv run python scrape_images.py --slugs slug-a,slug-b           # точечно
```
Yandex Images через Playwright, resume-safe. При капче — `--proxy http://user:pass@host:port`.

**5. Инференс-сервис** (нужен Postgres+pgvector и `DATABASE_URL`; проще через Docker выше):
```bash
cd "ML service"
DATABASE_URL=postgresql://vino:vino@localhost:5432/vino uv run uvicorn app:app --port 8080
```

**6. Валидация:**
```bash
cd "ML evaluation"
uv run python build_queryset.py
CROP_ENABLED=1 uv run python evaluate.py --queryset querysets/scrape.jsonl --tag scrape_crop
```

**Прогон скрипта-оценщика** (нужен `jq`, сервис поднят):
```bash
cd eval
bash participant_test.sh --images-dir ./queries --manifest ./queries.tsv \
  --endpoint 'http://127.0.0.1:8080/v1/eval/predict' --output ./predictions.jsonl
```

Подробности по каждому разделу — в его `README.md`.

**Переменные окружения:**

| ENV | По умолч. | Смысл |
|---|---|---|
| `SEARCH_MODEL` | `google/siglip2-base-patch16-256` | энкодер поиска (любая HF image-модель: SigLIP/CLIP/ViT) |
| `SEARCH_BACKEND` | `local` | `local` — transformers; `remote` — HTTP-эндпоинт (не держать GPU в dev) |
| `SEARCH_API_URL` / `SEARCH_API_TOKEN` | — | адрес/токен инференс-эндпоинта для `remote` |
| `DATABASE_URL` | `postgresql://vino:vino@db:5432/vino` | Postgres+pgvector |
| `CROP_ENABLED` | `1` | кроп бутылки (YOLO); нормализация этикетки |
| `CROP_MODEL` | `yolo11n.pt` | веса YOLO (скачиваются автоматически) |
| `CROP_MARGIN` / `CROP_MIN_CONF` | `0.06` / `0.25` | паддинг вокруг бокса / порог детекции |
| `THRESH_SCORE` / `THRESH_MARGIN` | `0.75` / `0.015` | пороги флага `in_catalog` (черновые, калибровать) |
| `EVAL_ABSTAIN` | `0` | `1` — отдавать `null` при низкой уверенности в eval-эндпоинте |
| `HF_HUB_OFFLINE` | — | `1` — не ходить в HuggingFace (модель из кэша) |
| `OPENROUTER_API_KEY` | — | ключ для сомелье (LLM + STT); `LLM_MODEL`/`STT_MODEL` — см. `sommelier service/README.md` |

> `SIGLIP_MODEL` ещё читается как алиас `SEARCH_MODEL` (обратная совместимость).

---

## Дорожная карта

- [x] EDA каталога, разбор формата оценки
- [x] Приведение эталонов в порядок (1977/94% в `reference/`)
- [x] Скрейпер доп. фото (Yandex+Playwright), протестирован
- [x] `encoder.py`: SigLIP 2 → эмбеддинги, индекс каталога (`build_index.py` → `index/catalog.npz`)
- [x] FastAPI-сервис (`app.py`) + мини-UI: `/v1/eval/predict` + `/v1/search` + `/` — прогон grader'а end-to-end OK (~340 мс/фото CPU)
- [x] Кроп бутылки (`crop.py`, COCO-YOLO) с env-выключателем — изолирует бутылку от фона/соседей
- [ ] **Валидация (ML evaluation)** (синтетика + студийные held-out + out-of-catalog негативы) → F1 top-1/top-5 ← следующее
- [ ] so400m + разрешение 384/512 на GPU (различение near-dups)
- [ ] OCR-реранк near-dups (читать название/год с этикетки)
- [ ] Калибровка порога «нет в каталоге / аналоги»
- [ ] Чистка скрейпа (dHash-дедуп + верификация по эталону) → мульти-вектор галерея
- [x] Мобильная карточка (Nuxt), Docker Compose
- [x] Цифровой сомелье: диалог (LLM, OpenRouter) + голос (STT) → профиль → подборка + «под ваш запрос N%» на скане

---

## Ограничения / заметки

- **Покрытие эталонами 94%:** у 126 вин фото в CSV — CDN-ключи/`DSC…`, которых нет в
  локальном дампе. Добираются веб-скрейпом.
- **Валидация:** размеченного полевого сета нет (3 публичных фото — без ground-truth на руках).
  F1 меряем на синтетике + студийных held-out + скрейпе; абсолютные числа оптимистичны,
  относительные сравнения — надёжны.
- **Модель по умолчанию** (CPU-сессия): `google/siglip2-base-patch16-256`.
  На GPU 16 ГБ — `google/siglip2-so400m-patch16-384` (лучше на near-duplicates).
- **Кроп vs near-dups (важно):** кроп изолирует бутылку и поднимает абсолютные score, но у
  «серийных» вин (напр. классическая линейка Массандры) этикетки почти идентичны — отличается
  лишь мелкий текст названия, который `base-256` не читает. Т.е. узкое место near-dups —
  **разрешение/текст**, а не фон. Рычаги: so400m + 384/512, OCR-реранк. Чистый эффект кропа
  на F1 меряется в «ML evaluation» (по 3 публичным фото судить нельзя).
- Bing/DuckDuckGo для скрейпа не годятся (отдают decoy-мусор на кириллицу) — только Yandex.
```
