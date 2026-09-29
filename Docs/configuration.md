# Конфигурация

Все параметры распознавания — `config/pipeline.yaml` (путь можно переопределить `PIPELINE_CONFIG`).
На старте `pipeline_config.apply_retrieval_env()` превращает YAML в env для модулей, поэтому
env-переменные `SEARCH_*`, `CROP_*`, `THRESH_*` задавать бесполезно — правьте YAML.

После правки: `docker compose restart ml`. Помеченные «индекс» параметры требуют пересборки:
`docker compose exec ml python build_index.py --force`.

## `retrieval`

| Параметр | Значение | Смысл |
|---|---|---|
| `model` | `/app/models/siglip2-base-patch16-256` | энкодер; он же ключ векторов в pgvector (индекс) |
| `encoder` / `backend` | `auto` / `local` | тип энкодера и где он считается (индекс) |
| `crop_enabled`, `crop_model`, `crop_margin`, `crop_min_conf` | `true`, `yolo11n.pt`, 0.06, 0.25 | кроп бутылки (индекс) |
| `label_enabled`, `label_model`, `label_min_conf`, `label_margin` | `true`, `label_det_best.pt`, 0.2, 0.02 | ветка этикетки (индекс) |
| `crop_pick`, `crop_fallback`, `crop_min_area`, `label_pick`, `label_min_w_frac` | `big_center`, `orig`, 0, `conf`, 0 | выбор боксов для индекса (индекс) |
| `query_crop_pick`, `query_crop_fallback`, `query_crop_min_conf`, `query_crop_min_area`, `query_label_*` | `big_center_border`, `label`, 0.10, 0.12, … | выбор боксов для запроса; пусто — как у индекса |
| `label_align` | `false` | выравнивание этикетки в retrieval (индекс). На OCR не влияет |
| `pipeline` | `combined` | `combined` / `bottle` / `label` — какие ветки искать |
| `top_k` | 6 | длина шорт-листа |
| `thresh_score` / `thresh_score_lo` | 0.81 / 0.70 | пороги гейта «есть в каталоге» |
| `thresh_margin` | 0.015 | только для отчёта в ответе |
| `eval_abstain` | `false` | `true` — `/v1/eval/predict` отдаёт `null` ниже `thresh_score_lo` |
| `database_url` | `postgresql://vino:vino@db:5432/vino` | |

## `ocr`

| Параметр | Значение | Смысл |
|---|---|---|
| `enabled` | `true` | `false` — только retrieval |
| `model` | `google/gemini-2.5-flash` | VLM на OpenRouter |
| `api_key_env`, `api_url` | `OPENROUTER_API_KEY`, OpenRouter chat/completions | |
| `temperature`, `max_tokens`, `timeout`, `retries` | 0.0, 700, 120, 3 | |
| `min_confidence` | 0.30 | ниже — перестановка не рассматривается |
| `confirm_confidence` | 0.80 | гейт: подтверждение перестановкой |
| `use_label_crop`, `label_crop` | `true`, `align` | что уходит в VLM: `align` (бокс + выравнивание), `bbox`, `bottle`, `auto` |
| `label_crop_min_conf/min_area/max_aspect` | 0.45 / 0.02 / 3.0 | надёжность бокса для `auto` |
| `retry_on_poor_fields` | `true` | второй проход по кропу бутылки |
| `min_side` / `max_side` | 800 / 1400 | размер картинки для VLM, px |

## `csv_match`

| Параметр | Значение | Смысл |
|---|---|---|
| `weights` | grape 0.24, line 0.20, winery/color/wine_type 0.15, sparkling 0.07, year/additional_text 0.02 | веса полей |
| `token_thresh` | 0.72 | порог нечёткого совпадения слова |
| `missing_credit` | 0.5 | оценка поля, которого нет в записи каталога |
| `min_score` | 0.35 | минимальная уверенность для перестановки |
| `decision_margin` | 0.35 | нужный отрыв от ответа retrieval |
| `agree_confidence` | 0.80 | гейт: подтверждение согласием |
| `shortlist_only` | `false` | `true` — сопоставлять только с шорт-листом retrieval |
| `grape_from_text`, `grape_contain`, `fuzzy_ck_norm` | `true` | сорт из доп. текста; все сорта этикетки должны быть в записи; `ц`≈`к` |
| `year_only_if_vintage_conflict` | `false` | год только если в каталоге есть то же вино другого года |
| `enrichment_file` | `auto` | дополнять каталог `paths.catalog_ocr_fields`; `""` — выключить |

## `visual`

| Параметр | Значение | Смысл |
|---|---|---|
| `enabled` | `true` | считать сходство с эталоном при перестановке |
| `min_similarity` | 0.70 | порог «похоже» |
| `block_on_mismatch` | `false` | `true` — отменять перестановку при низком сходстве |

## `paths`

Пути внутри контейнера `ml`: промпт, `found_in_catalog_corrected.csv`, `start_photos/`,
`catalog_ocr_fields.csv` (всё из `data/`, смонтирована в `/app/ref`), `models_dir`.

## Переменные окружения (`.env`)

| Переменная | Смысл |
|---|---|
| `OPENROUTER_API_KEY` | OCR в `ml`, диалог и STT в `sommelier` |
| `LLM_MODEL`, `STT_MODEL` | модели сомелье (по умолчанию `google/gemini-2.5-flash`) |
| `DEBUG_INFO` | `1` — показывать в UI score, margin, время |
| `PUBLIC_HOST`, `HTTP_PORT`, `HTTPS_PORT` | адрес и порты HTTPS-прокси |
| `EVAL_DIR` | набор для `docker compose run --rm eval` (по умолчанию `./eval`) |

Внутрь контейнера `eval` переменные передаются через `-e`: `EVAL_MAX_TIME` (таймаут на фото, 300 с),
`EVAL_ENDPOINT`, `EVAL_IMAGES_DIR`, `EVAL_MANIFEST`, `EVAL_OUTPUT`.

`.env` перечитывается только на `docker compose up -d`, не на `restart`.
