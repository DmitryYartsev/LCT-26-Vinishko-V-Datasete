# ML service — инференс-сервис сканера

FastAPI-сервис: фото этикетки → карточка вина. Ядро — кроп бутылки (YOLO) → эмбеддинг
(SigLIP 2) → косинусный поиск по индексу каталога (`index/catalog[_crop].npz`).

## Файлы

| Файл | Что |
|---|---|
| `app.py` | FastAPI: эндпоинты, склейка препроцесс→энкодер→поиск |
| `encoder.py` | SigLIP 2: картинка → L2-нормированный вектор (`get_encoder()`) |
| `crop.py` | кроп бутылки (COCO-YOLO), env-выключатель `CROP_ENABLED` |
| `label_align.py` | постобработка этикетки: выравнивание по 4 углам (rectify, как в Adobe Scan) |
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
| `LABEL_ALIGN` | `0` | `1` — выравнивать кроп этикетки по её 4 углам (rectify). Включать симметрично для индекса и запроса (`build_index.py --force`) |
| `LABEL_ALIGN_MARGIN` / `LABEL_ALIGN_PAD` | `0.06` / `0.02` | паддинг окна поиска углов / fallback-кропа bbox |
| `LABEL_ALIGN_MIN_AREA` / `LABEL_ALIGN_MAX_SIDE` | `0.15` / `0.35` | пороги доверия к 4-угольнику (площадь от окна / дисбаланс сторон) |
| `LABEL_ALIGN_WORK` / `LABEL_AUTO_ORIENT` | `800` / `0` | длинная сторона рабочей копии для поиска углов; доворот результата в портрет |
| `THRESH_SCORE` / `THRESH_MARGIN` | `0.75` / `0.015` | пороги `in_catalog` (черновые, калибровать) |
| `EVAL_ABSTAIN` | `0` | `1` — отдавать `null` при низкой уверенности |
| `HF_HUB_OFFLINE` | — | `1` — модель из кэша, без похода в HuggingFace |

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

## Замечания

- Индекс **мультивекторный**: на вино может быть несколько фото (несколько строк в npz
  с одним slug); матч = max cosine по slug (`CatalogIndex.search`).
- Пути берутся из корневого `paths.py` — хардкода абсолютов нет.
