# ML evaluation — валидация качества поиска

Численно меряет качество retrieval на наборах query, чтобы сравнивать конфиги
(кроп on/off, модель, реранк и т.д.), а не гадать по нескольким фото.
Работает **через HTTP к поднятому сервису** — что именно меряется (модель, кроп, реранк),
задаётся env **сервиса**, а не этого скрипта; конфиг записывается в отчёт из `/health`.

## Файлы

| Файл | Что |
|---|---|
| `build_queryset.py` | собирает набор query → `querysets/<source>.jsonl` |
| `evaluate.py` | queryset → запросы в `/v1/search` → `reports/<tag>.json` (сводка) + `reports/<tag>.csv` (построчно) |
| `querysets/` | наборы `{query_id, image_path, true_slug, source, in_catalog}` (gitignore) |
| `reports/` | результаты прогонов (gitignore) |

## Наборы (`--source`)

| source | Откуда | Размер | Метки |
|---|---|---|---|
| `dataset` (деф) | `image_labeling/dataset_v0/<slug>.<ext>` | 171 фото / 171 вино | **основной тестовый набор** |
| `labeled` | `image_labeling/labels.csv` (label=good) + `scraped_raw/` | 220 фото / 220 вин | ручная разметка, чистые |
| `scrape` | весь `scraped_raw/<slug>/` | ~11.7k фото | шумные (чужие кадры в папках) |

## Запуск

```bash
# 1. поднять сервис (docker compose up, или локально):
cd "ML service" && HF_HUB_OFFLINE=1 uv run uvicorn app:app --port 8080

# 2. собрать набор (один раз):
cd "ML evaluation"
uv run python build_queryset.py                                   # -> querysets/dataset.jsonl

# 3. прогнать:
uv run python evaluate.py --queryset querysets/dataset.jsonl --tag dataset_v0
```

Адрес сервиса — env `EVAL_URL` (деф `http://127.0.0.1:8080`). **Не используй `localhost`** на Windows:
он резолвится сначала в IPv6 `::1`, а проброс портов Docker Desktop по IPv6 даёт +10–20 с на каждый запрос.

Быстрая проверка на подвыборке (реранкер тратит токены — удобно, пока подбираешь параметры;
`--seed` фиксирует подвыборку, чтобы сравнивать конфиги на одних и тех же фото):

```bash
uv run python evaluate.py --queryset querysets/dataset.jsonl --tag quick --limit 30
```

Другие наборы:

```bash
uv run python build_queryset.py --source labeled                  # -> querysets/labeled.jsonl
uv run python build_queryset.py --source scrape --max-per-wine 3  # -> querysets/scrape.jsonl
```

Сравнение конфигов = перезапустить сервис с другим env и прогнать с другим `--tag`:

```bash
RERANK_BACKEND=none   # в env сервиса -> evaluate.py ... --tag dataset_norerank
RERANK_BACKEND=jina   # в env сервиса -> evaluate.py ... --tag dataset_jina
```

Для реранка отдельный прогон «без реранка» не обязателен: харнесс запрашивает
`/v1/search?k=max(--k, RERANK_TOP_K)` (все кандидаты реранкера) и восстанавливает порядок до
реранка по визуальному `score` — recall@K до и после считается в одном прогоне.

**Проверь `/health` перед прогоном** (харнесс печатает `rerank=...` в первой строке): с реранком
каждый запрос тратит токены. `docker compose restart` НЕ перечитывает `.env` — после правки env
нужен `docker compose up -d ml`.

## Метрики

- **recall@1 / @5 / @10** — доля, где правильный slug в top-1 / top-5 / top-10 (`--k` задаёт глубину).
- **winery@1** — угадана ли винодельня top-1 (показывает near-dup разрыв: бренд vs SKU).
- **mean top1 score / margin** — визуальная уверенность и отрыв от 2-го (до реранка).
- **latency_mean_s / latency_p95_s** — время ответа с точки зрения клиента (SLA 3 с).
- **rerank** (если реранк включён в сервисе):
  - `recall_visual` → `recall_final` — recall@1/5/10 до и после реранка;
    `recall_visual@RERANK_TOP_K` — потолок: вне top-K реранкер верное вино не найдёт;
  - `fixed_top1` / `broken_top1` — сколько top-1 реранк исправил / сломал;
  - `changed_top1`, `applied_share`, `errors`, `ms_mean`.

Построчный `reports/<tag>.csv` (true/pred/visual_top1, `rank`/`rank_visual` — позиция верного вина, путь к фото) — чтобы глазами
разобрать промахи.

## dataset_v0 (171 фото), SigLIP2-base-256 + crop, без реранка

| recall@1 | recall@5 | recall@10 | winery@1 | latency mean / p95 |
|---|---|---|---|---|
| 0.561 | 0.813 | 0.848 | 0.807 | 0.12 / 0.17 с |

## Первый замер (старый scrape-набор: 595 query, ~99 вин)

| | crop OFF | crop ON |
|---|---|---|
| recall@1 | 0.178 | **0.205** |
| recall@5 | 0.380 | **0.442** |
| winery@1 | 0.662 | **0.721** |

Вывод: кроп — чистый плюс; winery@1≫recall@1 = масштаб near-dup (бренд узнаётся, SKU нет).
