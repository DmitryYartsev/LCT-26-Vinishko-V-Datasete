# Справочник: данные, модели и схема БД

## Что скачивает `fetch`

Контейнер `fetch` проверяет артефакты при каждом запуске и докачивает только отсутствующее,
поэтому повторный `docker compose up` ничего не тянет.

| Артефакт | Куда | Объём | Что это |
|---|---|---|---|
| `data` | `data/` | ≈300 МБ | карточки каталога, эталонные фото, OCR-разметка, наборы для оценки |
| `filtered` | `filtered/` | ≈130 МБ | фото каталога `<slug>/NN.webp` + `catalog.csv` |
| `models` | `models/` | ≈1.4 ГБ | веса: энкодер и два детектора |
| `dump` | `deploy/archives/pgvector.dump` | ≈30 МБ | дамп pgvector с готовыми векторами индекса |

Ручные режимы:

```bash
docker compose run --rm fetch --status        # что уже есть
docker compose run --rm fetch --check         # только проверить
docker compose run --rm fetch --force         # перекачать заново
docker compose run --rm fetch --only data,dump
```

## Модели

| Файл | Что делает | Где используется |
|---|---|---|
| `models/siglip2-base-patch16-256` | энкодер: изображение → L2-нормированный вектор (768 чисел) | и сборка индекса, и поиск |
| `models/yolo11n.pt` | детектор бутылки | кроп перед эмбеддингом |
| `models/label_det_best.pt` | детектор этикетки | вторая ветка поиска и кроп для чтения этикетки |

Чтение этикетки выполняет внешняя модель на OpenRouter (`ocr.model` в конфиге) — локальных весов
для неё не нужно, нужен только `OPENROUTER_API_KEY`.

## Что лежит в `data/`

| Файл / папка | Что это |
|---|---|
| `found_in_catalog_corrected.csv` | карточки каталога: название, винодельня, цвет, регион, сорт, описание, slug |
| `start_photos/<slug>/NN.webp` | эталонные фото каталога — по ним считается визуальное сходство |
| `catalog_ocr_fields.csv` | поля этикеток каталожных фото, прочитанные моделью заранее (дополняют записи каталога) |
| `catalog_ocr_enrichment.csv` | та же разметка в развёрнутом виде |
| `negatives/` | фотографии вин, которых нет в каталоге — для проверки решения «не найдено» |
| `real_photo/`, `eval/`, `eval2/` | наборы фотографий для оценки |
| `eval62.csv`, `eval233_rest.csv`, `eval2.csv`, `eval2_batch2.csv`, `real_data_with_slug_eval.csv` | манифесты: имя файла → правильный slug |
| `eval2_extracted.csv` | вынесенные из `eval2` фото (использовались для балансировки набора) |

## Что лежит в `filtered/`

| Путь | Что это |
|---|---|
| `catalog.csv` | карточки каталога, из них заполняется таблица `wines` |
| `<slug>/NN.webp` | все фото этого вина — по ним строится индекс и отдаётся `/ref/{slug}` |

## Схема базы (Postgres + pgvector)

| Таблица | Колонки | Смысл |
|---|---|---|
| `wines` | `slug`, `name`, `winery`, `category`, `color`, `region`, `grape`, `description`, `rating` | карточки вин |
| `wine_vectors` | `id`, `slug`, `model`, `emb vector` | по вектору на каждое фото; `model` — ключ индекса |
| `models` | `model`, `dim`, `backend`, `n_vectors`, `built_at` | какие индексы собраны |
| `wine_profiles` | `slug`, `site_slug`, `in_catalog`, `title`, `manufacturer`, `category`, `color`, `sweetness`, `sparkling`, `alcohol`, `temperature`, `grapes[]`, `dishes[]`, `rating`, `description`, `tsv` | атрибуты для сомелье, `tsv` — полнотекстовый поиск по описанию |

Особенности:

* В `wine_vectors` у одного вина несколько строк (мультивектор), поиск агрегирует по slug лучший
  результат. ANN-индекс не строится: каталог небольшой, обычного сканирования достаточно.
* Ключей индекса два: `model` и `model#label` — обычный кроп и кроп этикетки. Активный ключ задаёт
  `retrieval.index_model`; поиск смотрит только векторы с этим ключом, поэтому векторы разных
  моделей могут сосуществовать.
* В каталоге 2 107 вин; на каждый ключ индекса приходится около 2 108 векторов.
* Кросс-проверка состояния:

```bash
docker compose exec db psql -U vino -d vino -c \
  "select model, count(*) from wine_vectors group by 1 order by 1;"
docker compose exec db psql -U vino -d vino -c "select * from models;"
docker compose exec db psql -U vino -d vino -c "select count(*) from wines;"
```

Ожидаемо: два ключа (`/app/models/siglip2-base-patch16-256` и `…#label`) с одинаковым числом
векторов и 2 107 вин в `wines`.
