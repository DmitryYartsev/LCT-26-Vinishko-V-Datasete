# ML service — инференс-сервис сканера

FastAPI-сервис: фото этикетки → карточка вина. Ядро — кроп бутылки (YOLO) → эмбеддинг
(SigLIP 2) → косинусный поиск по индексу каталога (`index/catalog[_crop].npz`).

## Файлы

| Файл | Что |
|---|---|
| `app.py` | FastAPI: эндпоинты, склейка препроцесс→энкодер→поиск |
| `encoder.py` | SigLIP 2 / DINOv2 / **OpenRouter-эмбеддинги**: картинка → L2-вектор (`get_encoder()`) |
| `crop.py` | кроп бутылки (COCO-YOLO), env-выключатель `CROP_ENABLED` |
| `label_align.py` | постобработка этикетки: выравнивание по 4 углам (rectify, как в Adobe Scan) |
| `ocr_rerank.py` | OCR-rerank: VLM читает кроп этикетки → мэтч по CSV → гейт top1↔top2 |
| `text_retrieval.py` | полнокаталожный текстовый поиск (CsvMatcher + e5) — пул кандидатов (P0-2) |
| `rerank_fusion.py` | fusion-ранкер над пулом image+text: признаки + калиброванный линейный скор (P0-3) |
| `pipeline.py` / `pipeline_config.py` | standalone-прогон (одиночное фото / eval-CSV) и загрузка YAML |
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
| POST | `/v1/search` | top-5 + score/margin/in_catalog + карточка |
| GET | `/` | мини-UI |
| GET | `/ref/{slug}` | эталонное фото вина (из `filtered/<slug>/`) |
| GET | `/wine/{slug}` | карточка по slug |
| GET | `/health` | статус (модель, размер индекса, crop) |

## Переменные окружения

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
| `csv_match.enrichment_file` | `auto` | OCR-разметка каталожных фото (`ML evaluation/catalog_ocr.py` → `data/catalog_ocr_fields.csv`): токены сорта/линейки/сахара и пустой год **дополняются** к записям каталога (исходная разметка не меняется). `auto` = `paths.catalog_ocr_fields` (Docker `/app/ref/…`, хост `<repo>/data/…`), `""`/`none` = выключено, иначе — путь. Мусорные чтения отсекает проверка бренда |
| `LABEL_ALIGN` | `0` | `1` — выравнивать кроп этикетки по её 4 углам (rectify). Включать симметрично для индекса и запроса (`build_index.py --force`). OCR-тракт выпрямляет свой кроп независимо (см. `ocr.use_label_crop`) |
| `LABEL_ALIGN_MARGIN` / `LABEL_ALIGN_PAD` | `0.06` / `0.02` | паддинг окна поиска углов / fallback-кропа bbox |
| `LABEL_ALIGN_MIN_AREA` / `LABEL_ALIGN_MAX_SIDE` | `0.15` / `0.35` | пороги доверия к 4-угольнику (площадь от окна / дисбаланс сторон) |
| `LABEL_ALIGN_WORK` / `LABEL_AUTO_ORIENT` | `800` / `0` | длинная сторона рабочей копии для поиска углов; доворот результата в портрет |
| `THRESH_SCORE` / `THRESH_MARGIN` | `0.75` / `0.015` | пороги `in_catalog` (черновые, калибровать) |
| `EVAL_ABSTAIN` | `0` | `1` — отдавать `null` при низкой уверенности |
| `HF_HUB_OFFLINE` | — | `1` — модель из кэша, без похода в HuggingFace |

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
query-тракт: `app.py`, `pipeline.py`, `recall_at_k.py`, `encoder_ab.py`).
Кропы эталонов (`VisualVerifier`) — всегда `index_policy()`. Пустое значение
`query_*` = запрос идёт политикой индекса (прежнее поведение, индекс не трогается).

Симметричный режим (один препроцесс на индекс и запрос, без асимметрии) —
`config/pipeline.symmetric_v3.yaml` + `build_index.py --force`; он даёт recall@1 0.7419
против 0.7581 у асимметрии (`reports/16_Query_crop_asymmetry.md`).


## Выравнивание этикетки — `label_align.py`

YOLO-детектор этикеток отдаёт только axis-aligned bbox, а на UGC-фото этикетка
наклонена и снята под углом — bbox подмешивает фон и «скашивает» этикетку.
Модуль повторяет приём сканеров документов (Adobe Scan): внутри bbox ищет 4 угла
этикетки (Canny/Оцу → контуры → `approxPolyDP`, fallback `minAreaRect`),
упорядочивает их (tl,tr,br,bl) и перспективной гомографией распрямляет в
прямоугольник (`cv2.warpPerspective`). Углы не нашлись → мягкий fallback на
обычный кроп bbox (модуль не падает на «плохом» входе).

```bash
cd "ML service"
python label_align.py                        # самопроверка на синтетике (10 проверок)
python label_align.py --image photo.jpg --box 120,300,460,900 --out ./_scratch
# включить в пайплайне (индекс + запрос) и пересобрать индекс под новый препроцесс:
LABEL_ALIGN=1 uv run python build_index.py --force
LABEL_ALIGN=1 uv run uvicorn app:app --host 127.0.0.1 --port 8080
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

Диагностика в ответе `/v1/search` и в отчёте `pipeline.py --eval-csv`:
`ocr_input` (`label_crop`/`full_image`), `ocr_fields`, `csv_confidence`,
`csv_margin`, `visual_similarity`. Все параметры — в `config/pipeline.yaml`
(`ocr.*`, `csv_match.*`, `visual.*`).

```bash
cd "ML service"
OPENROUTER_API_KEY=... python pipeline.py --config ../config/pipeline.yaml \
  --eval-csv data/real_data_with_slug_eval.csv --images-dir "data/Реальные фото"
```

## Текстовый путь и fusion-ранкер (P0-2 / P0-3)

Ретривер физически не может дать 90%: 12.4% eval-фото правильный slug не
попадает даже в **top-30**, поэтому потолок любого ре-ранкера по визуальному
шортлисту ≈ 88% (`ML evaluation/recall_at_k.py`). Лечится **объединением**
визуального пула с полнокаталожным текстовым поиском.

**`text_retrieval.py` (P0-2).** `TextRetriever.search(fields, k)` ищет по ВСЕМ
2108 записям каталога двумя сигналами: `csv_conf` (тот же `CsvMatcher`) и
`e5_cos` (косинус `intfloat/multilingual-e5-small`, transformers напрямую).
Поля VLM → строка запроса (`query_text`), карточка → строка документа
(`catalog_text`). Union(image top-30, text top-10) покрывает ~94% истины
(`ML evaluation/union_text_eval.py`); текст добавляет 16 кейсов вне image top-60.

**`rerank_fusion.py` (P0-3).** Ранжирует объединённый пул по 13 признакам
(`FEATURE_NAMES`: `image_cos`, `e5_cos`, `csv_conf`, `csv_margin`,
`year/color/sugar/grape/winery/title_match`, …). Веса обучает
`ML evaluation/train_fusion.py` (2-fold CV; hill-climb по accuracy даёт лучше
pointwise-логистики) в `fusion_weights.json`. Ключевое: **наивное расширение
пула со старым жёстким гейтом УХУДШАЕТ метрику** (`broken=8`), поэтому нужен
именно калиброванный ранкер с запасом над `image top1`.

Включается флагом `fusion.enabled: true` в `config/pipeline.yaml` (по умолчанию
`false` — работает прежний `decide()`). Повторный вызов VLM не нужен: fusion
работает на уже извлечённых `ocr_fields`. Секции конфига: `text.*`, `fusion.*`.

```bash
cd "ML evaluation"
python3 recall_at_k.py        # recall@K + кэш image top-K (нужен для fusion)
python3 union_text_eval.py    # потолок union-пула
python3 train_fusion.py       # обучение весов (2-fold CV) -> ML service/fusion_weights.json
python3 audit_refs.py         # аудит эталонов/разметки
python3 selftest_p0.py        # самопроверки P0 (без сети)
```

## Замечания

- Индекс **мультивекторный**: на вино может быть несколько фото (несколько строк в npz
  с одним slug); матч = max cosine по slug (`CatalogIndex.search`).
- Пути берутся из корневого `paths.py` — хардкода абсолютов нет.
- Данные (каталог, эталоны, eval-фото, OCR-разметка каталога) лежат **внутри репозитория**
  в `data/`: смонтированы в контейнер как `/app/ref`, а для прогонов на хосте
  `pipeline.apply_host_paths()` переписывает пути конфига на `<repo>/...`. Ключ OpenRouter —
  в `<repo>/.env` (в `.gitignore`).
- **Отказы OCR.** `402/401/403` от OpenRouter (недостаточно кредитов / неверный ключ) не
  ретраятся, а `pipeline.run_eval` при массовом отказе OCR (≥10 ошибок и ≥50% фото)
  останавливает прогон, пишет причину в `ocr_rerank_predictions.csv` (`ocr_error`) и
  помечает отчёт `degraded: true` — иначе длинный eval молча выдаёт метрику, посчитанную
  на неработающем OCR.
- **Прерываемый eval.** `run_eval` каждые `EVAL_DUMP_EVERY` фото (по умолчанию 10) атомарно
  пишет `ocr_rerank_report.json` с `partial: true`: прерванный прогон (вручную, кредиты,
  сеть) сохраняет уже посчитанные фото — их можно переиспользовать
  (`_scratch/merge_eval.py`, `_scratch/split62.py`) вместо повторных вызовов VLM.
- `_scratch/check_names.py` — статическая проверка «имя используется, но не импортировано»
  (такой баг однажды уронил `app.py` в 500).
