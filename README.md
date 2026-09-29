# Сканер российских вин «Своё вино» (кейс РСХБ, ЛЦТ-2026)

Сервис распознаёт вино по фотографии этикетки (снятой у полки — под углом, с бликами)
и возвращает **одну** карточку из каталога «Своё вино». Ядро — image retrieval:
кроп бутылки и этикетки (YOLO) → эмбеддинг (SigLIP 2) → поиск ближайшего эталона в pgvector,
поверх — OCR-переранжирование near-duplicates (VLM читает этикетку, мэтч по полям каталога).

**Доп. фича — цифровой сомелье** (`sommelier service/`): пользователь рассказывает текстом или
голосом, к чему/для какого повода берёт вино; LLM собирает профиль предпочтений, сервис подбирает
вина, а на каждой отсканированной этикетке показывает «под ваш запрос: N%» с причинами.

---

## Быстрый старт

Нужны Docker (Compose v2), интернет и ~4 ГБ свободного места.

```bash
cp .env.example .env     # 1. вписать OPENROUTER_API_KEY
docker compose up        # 2. всё остальное — автоматически
```

| Адрес | Что |
|---|---|
| http://localhost:3000 | UI (Nuxt, mobile-first) |
| http://localhost:8080 | ML API: `/health`, `POST /v1/eval/predict`, `POST /v1/search` |
| http://localhost:8090 | сомелье: `/health` |

Первый запуск сам скачивает то, чего нет в git (сервис `fetch`): каталог и эталоны `data/`,
фото каталога `filtered/`, веса YOLO `models/`, дамп pgvector с готовым индексом и
SigLIP 2 (~1.4 ГБ из HuggingFace). Прогресс — `docker compose logs -f fetch`. Повторные
запуски ничего не перекачивают. Цепочка: `fetch` → `db` → `db-init` (восстанавливает индекс
из дампа) → `ml` + `sommelier` → `web`. Готово, когда `docker compose ps` показывает
`ml ... (healthy)`.

Без `OPENROUTER_API_KEY` всё поднимается, но `ml` работает только в режиме retrieval
(без OCR-переранжирования), а сомелье не отвечает в чате.

## Прогон скрипта-оценщика (eval)

Папка `eval/` — пакет организаторов (`participant_test.sh`, `queries/`, `queries.tsv`).
Прогнать его против сервиса и получить предсказания:

```bash
docker compose run --rm eval
```

Результат — `eval/predictions.jsonl` (формат организаторов; прошлый файл переименовывается
в `*.bak`), в конце печатается сводка: сколько фото без slug и латентность. Если стек ещё
не поднят, команда сама его поднимет и дождётся готовности `ml`.

Другой набор той же структуры (`queries/` + `queries.tsv`, например закрытый набор жюри):

```bash
EVAL_DIR=/путь/к/набору docker compose run --rm eval    # -> /путь/к/набору/predictions.jsonl
```

Скрипт можно запустить и с хоста как в `eval/README.md` (нужны `bash`, `curl`, `jq`):
`cd eval && bash participant_test.sh --images-dir ./queries --manifest ./queries.tsv --endpoint http://127.0.0.1:8080/v1/eval/predict --output ./predictions.jsonl`.

> Каждое фото с включённым OCR — 1–2 вызова `openai/gpt-4o-mini` через OpenRouter (платно).

---

## Структура

```
ML service/          # FastAPI :8080 — кроп (YOLO) → SigLIP 2 → pgvector → OCR-rerank   [README]
sommelier service/   # FastAPI :8090 — сомелье: LLM-диалог, STT, скоринг вино↔профиль   [README]
web/                 # Nuxt 3 SPA :3000 + Nitro-BFF к ml/sommelier                       [README]
config/pipeline.yaml # все параметры пайплайна распознавания (модели, кроп, пороги, OCR)
deploy/              # fetch_data.py (скачивание данных), db_init.sh (дамп БД), run_eval.sh [README]
eval/                # пакет организаторов: participant_test.sh + публичные query
paths.py             # общие пути проекта
docker-compose.yml   # весь стек

# скачивается сервисом fetch (в git не входит):
data/                # каталог для OCR-матчера, эталонные фото start_photos/, OCR-поля каталога
filtered/            # фото каталога <slug>/NN.webp + catalog.csv (карточки)
models/              # YOLO (бутылка, этикетка) + SigLIP 2
deploy/archives/     # pgvector.dump — готовые векторы индекса
```

Архитектура и решения — [`ARCHITECTURE.md`](ARCHITECTURE.md), развёртывание и разбор
ошибок — [`deploy/README.md`](deploy/README.md).

## Полезные команды

```bash
docker compose logs -f ml              # логи распознавания (preflight / warm-up / запросы)
docker compose up -d                   # применить правку .env (restart НЕ перечитывает .env)
docker compose up --build              # после обновления кода web / Dockerfile'ов
docker compose run --rm fetch --status # что из данных/моделей уже на диске
docker compose exec db psql -U vino    # консоль БД (наружу порт не пробрасывается)
docker compose down                    # остановить (индекс остаётся в томе pgdata)
docker compose down -v                 # снести и том: индекс восстановится из дампа
```

Код `ml` и `sommelier` монтируется в контейнеры — после правки Python-кода достаточно
`docker compose restart ml` (или `sommelier`).

## Настройки

- `.env` — только секреты и модели сомелье (см. `.env.example`).
- `config/pipeline.yaml` — всё про распознавание: энкодер, политика кропа индекса/запроса,
  ширина short-list, пороги `in_catalog`, OCR (модель, веса полей, гейт решения). После правки —
  `docker compose restart ml`. Параметры, влияющие на индекс (энкодер, `crop_*`, `label_*`),
  требуют пересборки: `docker compose exec ml python build_index.py --force` (~40 мин на CPU).

Подробно — в [`ML service/README.md`](ML%20service/README.md).

---

## Данные

| Что | Значение |
|---|---|
| Каталог (`strapi_output0709.csv`) | 4147 строк → ~2100 уникальных позиций (по `slug`) |
| Эталонные фото | медиатека Strapi, после дедупа и сопоставления — фото для ~94% каталога |
| Near-duplicate серии | 131 серия (один дизайн, разный год/сорт/цвет) = 521 позиция |
| Публичный eval | 3 фото (2 из них — вина вне каталога) |

## Ограничения

- **Near-dups:** у «серийных» вин (напр. линейка Массандры) этикетки почти идентичны —
  отличается мелкий текст названия/года/цвета/типа, который `base-256` не читает. Рычаг —
  OCR-rerank (`ML service/ocr_rerank.py`): VLM читает выпрямленный кроп этикетки, решение
  принимается по разрыву CSV top1↔top2, retrieval остаётся fallback'ом.
- **CPU по умолчанию:** `google/siglip2-base-patch16-256`, ~0.5 с/фото без OCR.
