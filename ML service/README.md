# ML service — распознавание вина по фото

FastAPI-сервис (:8080): фото этикетки → карточка вина. Ретривер: кроп бутылки (YOLO) и
этикетки (дообученный YOLO) → эмбеддинг (SigLIP 2) → косинусный поиск в pgvector (две ветки,
ответ по максимальному top-1 score). Поверх — OCR-rerank near-duplicates. Все параметры —
в `config/pipeline.yaml` (OmegaConf; `pipeline_config.py` прокидывает их в env модулей).

## Файлы

| Файл | Что |
|---|---|
| `app.py` | FastAPI: эндпоинты, склейка кроп → энкодер → поиск → OCR-rerank |
| `encoder.py` | SigLIP 2 / DINOv2 / удалённые эмбеддинги: картинка → L2-вектор (`get_encoder()`) |
| `crop.py` | кроп бутылки (COCO-YOLO) и этикетки; политики кропа индекса и запроса |
| `label_align.py` | выравнивание этикетки по 4 углам (rectify, как в Adobe Scan) |
| `ocr_rerank.py` | OCR-rerank: VLM читает кроп этикетки → мэтч по CSV → гейт top1↔top2 |
| `db.py` | pgvector: схема, апсерт каталога/векторов, поиск (max cosine по slug) |
| `build_index.py` | warm-up: сид каталога + эмбеддинги `filtered/` → pgvector (если пусто) |
| `pipeline_config.py` | загрузка `config/pipeline.yaml` |
| `preflight.py` | проверка данных/моделей до старта (понятная ошибка вместо traceback) |
| `prompts_wine_match.txt` | промпт извлечения полей этикетки для VLM |

## Запуск

Сервис поднимается в составе стека (`docker compose up` в корне, см. корневой README).
Код смонтирован в контейнер — после правок `docker compose restart ml`.

```bash
docker compose logs -f ml                              # старт / запросы
docker compose exec ml python preflight.py             # есть ли все данные и модели
docker compose exec ml python build_index.py --force   # пересобрать индекс (~40 мин на CPU)
```

Индекс пересобирать нужно только при смене энкодера или политики кропа индекса
(`crop_*`, `label_*`, `label_align`); без `--force` warm-up пропускает уже собранные ветки.

## Эндпоинты

| Метод | Путь | Ответ |
|---|---|---|
| POST | `/v1/eval/predict` | `{"slug", ...}` — для скрипта-оценщика (`docker compose run --rm eval`) |
| POST | `/v1/search` | top-5 + score/margin/in_catalog + карточка |
| GET | `/ref/{slug}` | эталонное фото вина (из `filtered/<slug>/`) |
| GET | `/wine/{slug}` | карточка по slug |
| GET | `/health` | статус (модель, размер индекса, crop) |

## Параметры

Источник правды — `config/pipeline.yaml`: на старте `pipeline_config.apply_retrieval_env()`
выставляет из него env ниже (поэтому задавать их в `docker-compose.yml` бесполезно —
правьте YAML). Таблица — справочник по смыслу параметров.

| ENV | Деф. | Смысл |
|---|---|---|
| `SIGLIP_MODEL` | `google/siglip2-base-patch16-256` | энкодер (GPU — `...so400m-patch16-384`) |
| `SEARCH_ENCODER` | `auto` | `siglip` / `dinov2` / `auto` (по `SEARCH_MODEL`: есть `dinov2` → DINOv2) |
| `SEARCH_MODEL` | `google/siglip2-base-patch16-256` | HF-id или локальный путь (`facebook/dinov2-base` / `/app/models/dinov2-base`) |
| `CROP_ENABLED` | `1` | кроп бутылки; переключает и препроцесс, и файл индекса |
| `CROP_MODEL` / `CROP_MARGIN` / `CROP_MIN_CONF` | `yolo11n.pt` / `0.06` / `0.25` | параметры кропа |
| `CROP_MIN_AREA` | `0.0` | мин. площадь бокса бутылки (доля кадра): ниже — детекция отбрасывается, и работают `CROP_FALLBACK`/`crop_margin`. Нужен при низком `CROP_MIN_CONF`: тот вытаскивает и целевой бокс, и крошечные ложные (6–9% кадра) |
| `OCR_LABEL_CROP` | `align` | что подаём VLM как «кроп этикетки»: `align` (истор. — bbox детектора + rectify) │ `bbox` (без выравнивания) │ `bottle` (страховка: детектор этикетки не используется, читаем кроп бутылки) │ `auto` (bbox+rectify, но при ненадёжном боксе — кроп бутылки) |
| `LABEL_CROP_MIN_CONF` / `LABEL_CROP_MIN_AREA` / `LABEL_CROP_MAX_ASPECT` | `0.45` / `0.02` / `3.0` | пороги «надёжности» бокса этикетки для `OCR_LABEL_CROP=auto` (уверенность, доля площади кропа бутылки, аспект h/w) |
| `ocr.retry_on_poor_fields` | `false` | страховка на ошибки детектора этикетки: если VLM не прочитал бренд и сорт (или сорт вне словаря каталога) — второй проход VLM **по кропу бутылки** и слияние полей (+1 вызов только для таких фото) |
| `csv_match.enrichment_file` | `auto` | OCR-разметка каталожных фото (`data/catalog_ocr_fields.csv`): токены сорта/линейки/сахара и пустой год **дополняются** к записям каталога (исходная разметка не меняется). `auto` = `paths.catalog_ocr_fields` (`/app/ref/…`), `""`/`none` = выключено, иначе — путь. Мусорные чтения отсекает проверка бренда |
| `LABEL_ALIGN` | `0` | `1` — выравнивать кроп этикетки по её 4 углам (rectify). Включать симметрично для индекса и запроса (`build_index.py --force`). OCR-тракт выпрямляет свой кроп независимо (см. `ocr.use_label_crop`) |
| `LABEL_ALIGN_MARGIN` / `LABEL_ALIGN_PAD` | `0.06` / `0.02` | паддинг окна поиска углов / fallback-кропа bbox |
| `LABEL_ALIGN_MIN_AREA` / `LABEL_ALIGN_MAX_SIDE` | `0.15` / `0.35` | пороги доверия к 4-угольнику (площадь от окна / дисбаланс сторон) |
| `LABEL_ALIGN_WORK` / `LABEL_AUTO_ORIENT` | `800` / `0` | длинная сторона рабочей копии для поиска углов; доворот результата в портрет |
| `THRESH_SCORE` / `THRESH_MARGIN` | `0.75` / `0.015` | пороги `in_catalog` (черновые, калибровать) |
| `EVAL_ABSTAIN` | `0` | `1` — отдавать `null` при низкой уверенности |
| `HF_HUB_OFFLINE` | `1` (compose) | не ходить в HuggingFace: энкодер лежит в `models/` |

### Политика выбора кропа (аудит `reports/15_Crop_audit.md`)

Выбор бокса бутылки/этикетки вынесен в env-флаги; дефолты = историческое поведение.

| ENV | Деф. | Смысл |
|---|---|---|
| `CROP_PICK` | `big_center` | `big_center` — площадь − 0.6·удалённость от центра кадра; `big_center_border` — плюс штраф за срезанную рамкой сторону и небутылочные пропорции |
| `CROP_FALLBACK` | `orig` | бутылка не найдена: `orig` — вернуть фото как есть, `label` — искать этикетку прямо на фото |
| `LABEL_PICK` | `conf` | выбор бокса этикетки: `conf` — максимальная уверенность, `conf_size_center` — с учётом площади и центральности |
| `LABEL_MIN_W_FRAC` | `0.0` | отсечь боксы этикетки уже этой доли ширины кропа |
| `BORDER_PENALTY` / `ASPECT_PENALTY` / `BOTTLE_MIN_ASPECT` / `CROP_CENTER_W` / `CROP_TOUCH_EPS` | `0.15` / `0.10` / `1.3` / `0.6` / `0.015` | параметры скоринга бокса бутылки |

### Асимметрия «индекс vs запрос» (`query_crop_*`)

Гипотеза аудита: политику кропа выгодно менять **только на запросе** — на студийных
фото каталога та же политика кроп этикетки портит (`reports/16_Query_crop_asymmetry.md`).
Индекс собирается `crop_*`/`label_*`, запрос — `query_crop_*`/`query_label_*` из
`config/pipeline.yaml` (`crop.py: use_query_policy()`, переключение вызывает только
query-тракт `app.py`).
Кропы эталонов (`VisualVerifier`) — всегда `index_policy()`. Пустое значение
`query_*` = запрос идёт политикой индекса (прежнее поведение, индекс не трогается).

Симметричный режим (один препроцесс на индекс и запрос) давал recall@1 0.7419 против
0.7581 у асимметрии (`reports/16_Query_crop_asymmetry.md`).


## Выравнивание этикетки — `label_align.py`

YOLO-детектор этикеток отдаёт только axis-aligned bbox, а на UGC-фото этикетка
наклонена и снята под углом — bbox подмешивает фон и «скашивает» этикетку.
Модуль повторяет приём сканеров документов (Adobe Scan): внутри bbox ищет 4 угла
этикетки (Canny/Оцу → контуры → `approxPolyDP`, fallback `minAreaRect`),
упорядочивает их (tl,tr,br,bl) и перспективной гомографией распрямляет в
прямоугольник (`cv2.warpPerspective`). Углы не нашлись → мягкий fallback на
обычный кроп bbox (модуль не падает на «плохом» входе).

```bash
docker compose exec ml python label_align.py   # самопроверка на синтетике (10 проверок)
# включить в пайплайне (индекс + запрос): retrieval.label_align: true в config/pipeline.yaml,
# затем пересобрать индекс под новый препроцесс:
docker compose exec ml python build_index.py --force && docker compose restart ml
```

Как библиотека: `align_label(img, box)`, `maybe_align_label(img, box)`
(с учётом `LABEL_ALIGN`), `LabelAligner().find_quad(img, box)` (только углы).
Зависимость — `opencv-python` (идёт транзитивно с `ultralytics`, добавлена в
`requirements.txt` явно).

## OCR-rerank — `ocr_rerank.py`

Второй проход поверх retrieval для near-dup «сестёр» (одна линейка, разный
сорт/цвет/тип/год) — именно их SigLIP не различает. Один вызов VLM
(`gpt-4o-mini` через OpenRouter) на фото:

1. **VLM читает выпрямленный кроп этикетки**, а не сырое полочное фото:
   при `ocr.use_label_crop: true` кроп этикетки форсированно выравнивается по
   4 углам (`crop.ocr_label_crop` → `label_align`) и апскейлится до
   `ocr.min_side` (cap `ocr.max_side`) — иначе мелкий наклонный год/цвет не
   вычитывается. Препроцесс retrieval-индекса при этом НЕ меняется.
2. **Мэтч по `found_in_catalog_corrected.csv`** (`CsvMatcher`, без модели),
   как **re-rank короткого списка retrieval** (`csv_match.shortlist_only: true`):
   правильный ответ почти всегда уже в top-k retrieval, а поиск по всем 2108
   записям каталога создаёт ложные «ничьи». Поля и веса: `winery 0.16`,
   `grape 0.18`, `color 0.18`, `wine_type 0.18`, `year 0.18`,
   `additional_text 0.12` (год+цвет+тип ≈ 0.54 — они и различают near-dup
   «сестёр»). Год берётся из поля `year` ИЛИ из суффикса слага
   (`-13`/`-135` → 2013) и учитывается **всегда**
   (`year_only_if_vintage_conflict: false`). Поле, отсутствующее в записи
   каталога, даёт `csv_match.missing_credit` (0.5, нейтрально), а не 0/1 —
   иначе записи без цвета/типа набирали ложную уверенность.
3. **Гейт решения — в ОДНОЙ шкале**: `csv_conf` OCR-кандидата против
   `csv_conf` действующего ответа retrieval (обе — покрытие полей от
   фиксированного набора, заданного запросом). Замена происходит только с
   запасом `csv_match.decision_margin` (0.20 ≈ вес одного поля-дискриминатора).
   Прежний гейт сравнивал `csv_conf` с косинусом SigLIP (0.7–0.85) — разные
   шкалы — и блокировал OCR почти всегда.
   `VisualVerifier` (косинус к `start_photos`) — мягкое подтверждение:
   жёстко блокирует только при `visual.block_on_mismatch: true`.

Диагностика — в ответе `/v1/search`:
`ocr_input` (`label_crop`/`full_image`), `ocr_fields`, `csv_confidence`,
`csv_margin`, `visual_similarity`. Все параметры — в `config/pipeline.yaml`
(`ocr.*`, `csv_match.*`, `visual.*`).

## Замечания

- Индекс **мультивекторный**: на вино может быть несколько фото (несколько строк в
  `wine_vectors` с одним slug); матч = max cosine по slug. Вектора разных моделей
  сосуществуют (колонка `model`), поиск фильтрует по текущей.
- Данные (каталог, эталоны, OCR-разметка каталога) скачивает сервис `fetch` в `data/` —
  в контейнере это `/app/ref`. Ключ OpenRouter — в `<repo>/.env`.
- **Отказы OCR.** `402/401/403` от OpenRouter (кредиты / неверный ключ) не ретраятся:
  запрос отвечает результатом retrieval.
