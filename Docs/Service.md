# Сервисы: что внутри и как с ними говорить

Четыре контейнера: `db` (Postgres + pgvector), `ml` (распознавание, :8080), `sommelier` (сомелье,
:8090), `web` (Nuxt UI + BFF, :3000). Плюс два одноразовых помощника: `fetch` (скачивает данные
и модель) и `db-init` (восстанавливает дамп индекса).

## `ml` — распознавание вина

Пайплайн одного запроса (`ML service/app.py::_rank` + `_rerank`):

```
фото → [crop.maybe_crop] кроп бутылки ───┐
                                         ├─► SigLIP2 emb ─► pgvector (2 ветки: model, model#label)
      → [crop.maybe_label_crop] кроп этикетки ┘                    │
                                                                   ▼
                             ветка с большим top-1 score = ответ retrieval (+margin внутри ветки)
                                                                   │
                                   [ocr_rerank.rerank] VLM читает ВЫПРЯМЛЕННЫЙ кроп этикетки
                                                                   │
                    поля (винодельня/сорт/цвет/сахар/тип/линейка/год) → матч по CSV каталога
                                                                   │
                       csv_decisive → final_slug = кандидат OCR; иначе остаётся ответ retrieval
```

Ключевые детали:

* **Кропы запроса и индекса различаются** (`crop.use_query_policy()`): индекс — `crop_*`,
  запрос — `query_crop_*`; кропы эталонов всегда под индексной политикой. Это дало recall@1
  0.677 → 0.758 без пересборки индекса.
* **OCR-rerank** (`ocr_rerank.py`) — не пересортировка косинусов, а *матчинг по данным каталога*:
  VLM читает этикетку, поля сравниваются с CSV (сорт/линейка весят больше, год почти нет),
  решение принимается по разрыву CSV top1↔top2 и подтверждается визуальной проверкой. Если поля
  прочитаны плохо (нет бренда/сорта), есть страховка — второй проход VLM по кропу бутылки.
* **Гейт «есть в каталоге»** (`in_catalog`) — сейчас `top1_score >= retrieval.thresh_score` (0.75)
  у ветки-победителя. Именно этот флаг веб использует, чтобы выбрать между карточкой и экраном
  «нет точного совпадения» (см. «Гейт и сценарий похожих»).

### Эндпоинты `ml`

| Метод | Путь | Вход | Ответ |
|---|---|---|---|
| POST | `/v1/search` | multipart `image` | `{in_catalog, elapsed_ms, pipeline, branch, branches, confidence{top1_score,margin,thresholds}, top1, results[], final_slug, stage, pre_ocr_slug, csv_confidence, csv_margin, visual_similarity, ocr_input, ocr_fields, top_candidates}` |
| POST | `/v1/eval/predict` | multipart `image` | `{slug, score, margin, branch, stage, pre_ocr_slug, csv_confidence, …}` — для скрипта-оценщика (`EVAL_ABSTAIN=1` → `{slug: null}` при низкой уверенности) |
| GET | `/wine/{slug}` | — | карточка из `wines` (`name, winery, category, color, region, grape, description, rating`) |
| GET | `/ref/{slug}?h=` | `h` — высота превью | эталонное фото (webp-превью или оригинал) |
| GET | `/health` | — | `{status, model, crop, wines, pipeline, label_branch, ocr_rerank, ocr_model, models[]}` |

`results[]` — шорт-лист пула (по умолчанию 6 элементов, `retrieval.top_k`): `{slug, score, card{…}}`.
BFF оставляет из него только `{slug, score}` и использует для блока «похожие по этикетке».

### Конфиг

`config/pipeline.yaml` (в контейнере — `/app/config/pipeline.yaml`), основные ручки:

| Секция | Параметр | Смысл |
|---|---|---|
| `retrieval` | `model`, `index_model` | папка энкодера и ключ индекса в pgvector |
| | `top_k` | ширина шорт-листа (ответ сервиса и пул для OCR) |
| | `crop_model`, `label_model` | веса YOLO (`models/*.pt`) |
| | `crop_*` / `query_crop_*` | политики кропа для индекса и для запроса |
| | `thresh_score`, `thresh_margin` | пороги гейта `in_catalog` |
| `ocr` | `enabled`, `model`, `min_confidence` | включение VLM-реранка и порог зачёта матча |
| | `label_crop`, `retry_on_poor_fields`, `min_side`/`max_side` | что подаём VLM и когда идёт второй проход |
| `csv_match` | `weights{winery,grape,color,wine_type,line,sparkling,year}`, `token_thresh`, `min_score` | веса и пороги матчинга по каталогу |
| | `grape_from_text`, `grape_contain`, `fuzzy_ck_norm`, `enrichment_file` | флаги добора полей (отчёты 20–21) |
| `text`, `fusion` | `enabled`, `top_k` | полнокаталожный текстовый путь и fusion-ранкер |

Переменные окружения (из compose): `DATABASE_URL`, `SEARCH_MODEL`, `SEARCH_BACKEND`,
`SEARCH_PIPELINE=combined|bottle|label`, `CROP_ENABLED`, `USE_LABEL_BRANCH`, `THRESH_SCORE`,
`THRESH_MARGIN`, `OPENROUTER_API_KEY`, `PIPELINE_CONFIG`, `EVAL_ABSTAIN`.

При старте `ml` печатает `[preflight]` (есть ли данные/модели — человеческим текстом),
`[warmup]` (считает индекс или пропускает, если он уже в БД) и `[startup] … ocr=True`.


### Гейт «есть в каталоге» и сценарий похожих

Пользователь фотографирует вино, которого в каталоге нет (иностранное, другой год или другая
линейка той же серии). Правильное поведение — не показывать карточку «ближайшего» вина, а
предложить сценарий «нет точного совпадения»: похожие по этикетке (из шорт-листа) + аналоги
других виноделен (сомелье).

Как это работает в коде: `in_catalog=false` в ответе `/v1/search` → BFF
`web/server/api/scan.post.ts` дёргает у сомелье `/v1/match_many` (по шорт-листу) и `/v1/analogs`
(по top-1) и отдаёт UI поля `similar` и `analogs` → UI открывает `/no-match`.
Пороги гейта подбираются на негативах `data/negatives`; правило и калибровка —
`reports/27_gate_negatives.md`. Гейт: `top1 >= HI (0.81)` — карточка сразу; `top1 < LO (0.70)` —
«нет в каталоге»; между ними карточка только при подтверждении от OCR (`stage=ocr_rerank` и
`csv_confidence >= 0.80` либо `reason=csv_agrees` и `>= 0.80`). В ответе `/v1/search` есть
`gate_reason` — почему принято решение.

## `sommelier` — цифровой сомелье

Stateless: переписка и профиль живут в браузере (localStorage) и приходят в каждом запросе.

| Метод | Путь | Вход → выход |
|---|---|---|
| POST | `/v1/chat` | `{messages, profile}` → `{reply, profile, picks, suggestions}` (LLM обновляет профиль) |
| POST | `/v1/recommend` | `{profile, k, exclude}` → `{profile, picks}` (без LLM) |
| POST | `/v1/match` | `{slug, profile}` → `{wine, score, reasons, checks, verdict, verdict_text}` |
| POST | `/v1/match_many` | `{slugs, profile}` → `{items[]}` (пакетно — для «похожих») |
| POST | `/v1/analogs` | `{slug, k}` → `{picks[]}` (похожие по стилю вина других виноделен) |
| POST | `/v1/stt` | multipart `audio` → `{text}` (голос → текст) |
| GET | `/v1/wine/{slug}`, `/v1/vocab`, `/health` | карточка / словари (цвета, сладость, блюда, регионы) / статус |

Профиль: `colors[], sweetness[], sparkling, dishes[], regions[], grapes[], notes[], exclude_*, alcohol_max, occasion, summary`.
Скоринг детерминированный (LLM в нём не участвует): взвешенная доля выполненных пожеланий, штраф
×0.3 за «не хочу», построчные `checks` (параметр → значение → ✓/—) и вердикт `ok|part|bad` с текстом.
Данные — таблица `wine_profiles`, сид из `sommelier service/wines_parsed.jsonl` + `slug_map.csv`.

## `web` — UI + BFF

SPA (Nuxt 3, `ssr:false`); состояние — localStorage (профиль/цель) + sessionStorage (сканы сессии).

| Экран | Что |
|---|---|
| `/` | загрузка фото (камера / галерея / drag&drop / «пример») → экран поиска |
| `/wine/:slug` | карточка: рейтинг, вердикт по цели с построчной проверкой, характеристики, «к чему подать»; ниже — «лучше под вашу цель» или аналоги других виноделен |
| `/no-match` | «точного совпадения нет»: похожие по этикетке (шорт-лист + сходство) + аналоги |
| `/session` | сканы за сессию, ранжированные под цель |
| панель сомелье | чат (текст/голос), цель сохраняется, доступна с любого экрана |

BFF (Nitro) — тонкий прокси: `/api/scan`, `/api/wine/[slug]`, `/api/wines`, `/api/chat`,
`/api/stt`, `/api/ref/[slug]`, `/api/stats`. Адреса бэкендов — `NUXT_ML_URL` / `NUXT_SOMMELIER_URL`.
Технические цифры в UI (время поиска, score/margin, «сходство N%») включаются `NUXT_PUBLIC_DEBUG_INFO=1`.

## `db` — данные

| Таблица | Что |
|---|---|
| `wines(slug, name, winery, category, color, region, grape, description, rating)` | карточки каталога |
| `wine_vectors(id, slug, model, emb vector)` | мультивектор: несколько фото на вино и несколько моделей одновременно; поиск `WHERE model=…`, матч = min-дистанция по slug |
| `models(model, dim, backend, n_vectors, built_at)` | какие модели проиндексированы |
| `wine_profiles(…, tsv)` | атрибуты сомелье (блюда/крепость/подача/рейтинг) + русский FTS по описанию |

Индекс собирает `ml` при старте (`build_index.py`): если векторов для текущего ключа модели нет —
считает эмбеддинги `filtered/<slug>/*` (долго, ~40 мин на CPU); если дамп уже восстановлен
(`db-init`) — стартует за секунды.

