# Сканер российских вин «Своё вино» (кейс РСХБ, ЛЦТ-2026)

Сервис распознаёт вино по фотографии этикетки (снятой у полки — под углом, с бликами)
и возвращает **одну** карточку из каталога «Своё вино». Ядро — image retrieval:
кроп бутылки (YOLO) → эмбеддинг (SigLIP 2) → поиск ближайшего эталона по косинусной близости.

Формат для скрипта-оценщика: `POST /v1/eval/predict`, multipart-поле `image` → `{"slug":"..."}`.
Есть мини-UI: `GET /` (загрузка фото → карточка + top-5), открывается на `http://127.0.0.1:8080/`.

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
eda_and_image_processing/   # EDA + подготовка данных (офлайн, не рантайм)   [README]
  eda.ipynb                 #   разведочный анализ каталога
  process_images.py         #   дамп -> filtered/<slug>/*.webp + catalog.csv + npz
  scrape_images.py          #   сбор доп. фото из веба (Yandex+Playwright)
service/                    # инференс-сервис                               [README]
  app.py encoder.py crop.py search.py  static/index.html  index/*.npz
harness/                    # валидация качества (recall@1/@5, ...)         [README]
  build_queryset.py evaluate.py  querysets/ reports/
filtered/                   # выход process_images: фото по slug + catalog.csv  (gitignore)
org_files/                  # исходники организатора: pdf/pptx/csv/zip/rar     (gitignore)
solution/images/            # сырые данные: _uploads_raw, scraped_raw          (gitignore)
eval/                       # публичный eval + participant_test.sh (скрипт оценщика)
paths.py  pyproject.toml  README.md  .gitignore
```

У каждого раздела свой README. Данные (`filtered/`, `org_files/`, `solution/`,
`service/index/`) в git не попадают.

---

## Запуск шагов

**1. Подготовка изображений** (дамп → `filtered/<slug>/*.webp` + `catalog.csv` + индекс):
```bash
cd eda_and_image_processing
uv run python process_images.py --both          # оба индекса (crop off + on)
```

**2. EDA** — `eda_and_image_processing/eda.ipynb`.

**3. Сбор доп. фото из веба** (для мульти-вектора и валидации):
```bash
cd eda_and_image_processing
uv run python scrape_images.py --all --per-wine 6 --sleep 3   # весь каталог
uv run python scrape_images.py --slugs slug-a,slug-b           # точечно
```
Yandex Images через Playwright, resume-safe. При капче — `--proxy http://user:pass@host:port`.

**4. Инференс-сервис + мини-UI:**
```bash
cd service
HF_HUB_OFFLINE=1 uv run uvicorn app:app --host 127.0.0.1 --port 8080
# затем открыть http://127.0.0.1:8080/
```

**5. Валидация:**
```bash
cd harness
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
| `SIGLIP_MODEL` | `google/siglip2-base-patch16-256` | энкодер (на GPU — `...so400m-patch16-384`) |
| `CROP_ENABLED` | `1` | кроп бутылки (YOLO); переключает и препроцесс, и файл индекса |
| `CROP_MODEL` | `yolo11n.pt` | веса YOLO (скачиваются автоматически) |
| `CROP_MARGIN` / `CROP_MIN_CONF` | `0.06` / `0.25` | паддинг вокруг бокса / порог детекции |
| `THRESH_SCORE` / `THRESH_MARGIN` | `0.75` / `0.015` | пороги флага `in_catalog` (черновые, калибровать) |
| `EVAL_ABSTAIN` | `0` | `1` — отдавать `null` при низкой уверенности в eval-эндпоинте |
| `HF_HUB_OFFLINE` | — | `1` — не ходить в HuggingFace (модель из кэша; убирает SSL-зависания) |

---

## Дорожная карта

- [x] EDA каталога, разбор формата оценки
- [x] Приведение эталонов в порядок (1977/94% в `reference/`)
- [x] Скрейпер доп. фото (Yandex+Playwright), протестирован
- [x] `encoder.py`: SigLIP 2 → эмбеддинги, индекс каталога (`build_index.py` → `index/catalog.npz`)
- [x] FastAPI-сервис (`app.py`) + мини-UI: `/v1/eval/predict` + `/v1/search` + `/` — прогон grader'а end-to-end OK (~340 мс/фото CPU)
- [x] Кроп бутылки (`crop.py`, COCO-YOLO) с env-выключателем — изолирует бутылку от фона/соседей
- [ ] **Валидационный harness** (синтетика + студийные held-out + out-of-catalog негативы) → F1 top-1/top-5 ← следующее
- [ ] so400m + разрешение 384/512 на GPU (различение near-dups)
- [ ] OCR-реранк near-dups (читать название/год с этикетки)
- [ ] Калибровка порога «нет в каталоге / аналоги»
- [ ] Чистка скрейпа (dHash-дедуп + верификация по эталону) → мульти-вектор галерея
- [ ] Мобильная карточка (Nuxt) + фича после поиска, Docker Compose

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
  на F1 меряется на harness (по 3 публичным фото судить нельзя).
- Bing/DuckDuckGo для скрейпа не годятся (отдают decoy-мусор на кириллицу) — только Yandex.
```
