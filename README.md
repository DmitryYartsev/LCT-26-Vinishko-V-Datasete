# Сканер российских вин «Своё вино» (кейс РСХБ, ЛЦТ-2026)

- Презентация и демо: https://disk.yandex.kz/d/JYhlcbXM5Qgchw
- Демостенд: https://155-138-200-98.sslip.io/

По фото этикетки возвращает одну карточку вина из каталога «Своё вино» (~2100 позиций).
Пайплайн: YOLO-кроп бутылки → YOLO-кроп этикетки → выравнивание → SigLIP2 + pgvector →
OCR-rerank (VLM читает этикетку, CsvMatcher переставляет кандидатов). Подробно — [ARCHITECTURE.md](ARCHITECTURE.md).

Доп. фича — **цифровой сомелье**: по разговору (текст/голос) строит профиль вкуса и на каждом
скане показывает, подходит ли вино.

## Запуск

Нужны Docker Compose v2, интернет, ~6 ГБ диска.

```bash
cp .env.example .env     # вписать OPENROUTER_API_KEY
docker compose up -d --build
```

Первый запуск скачивает данные, модели и дамп индекса (сервис `fetch`). Готово, когда
`docker compose ps` показывает `ml ... (healthy)`.

| Адрес | Что |
|---|---|
| http://localhost:3000 | UI |
| https://localhost | UI по HTTPS (нужен для микрофона) |
| http://localhost:8080 | ML API: `/health`, `POST /v1/search`, `POST /v1/eval/predict` |
| http://localhost:8090 | сомелье |

Без `OPENROUTER_API_KEY` работает только retrieval (без OCR), чат сомелье молчит.

## Оценка

```bash
docker compose run --rm eval                              # eval/queries + eval/queries.tsv
EVAL_DIR=/путь/к/набору docker compose run --rm eval      # другой набор той же структуры
```

Результат — `predictions.jsonl` в папке набора. Каждое фото — 1–2 платных вызова VLM через OpenRouter.

## Структура

```
ML service/          FastAPI :8080 — распознавание
sommelier service/   FastAPI :8090 — сомелье
web/                 Nuxt 3 SPA + Nitro BFF :3000
config/pipeline.yaml все параметры распознавания
deploy/              fetch_data.py, db_init.sh, run_eval.sh, Caddyfile
eval/                пакет организаторов (participant_test.sh + публичные query)
Docs/                документация
data/ filtered/ models/   скачивает fetch (не в git)
```

## Документация

- [ARCHITECTURE.md](ARCHITECTURE.md) — пайплайн и сервисы
- [Docs/](Docs/README.md) — детали пайплайна, конфиг, API, эксплуатация, оценка
- READMEs сервисов: [ML service](ML%20service/README.md), [sommelier service](sommelier%20service/README.md), [web](web/README.md), [deploy](deploy/README.md)
