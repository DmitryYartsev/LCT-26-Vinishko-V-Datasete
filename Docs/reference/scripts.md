# Справочник: скрипты

## `deploy/fetch_data.py` — загрузка данных, моделей и дампа

Запускается автоматически при `docker compose up` (контейнер `fetch`) и вручную:

```bash
docker compose run --rm fetch                 # докачать отсутствующее
docker compose run --rm fetch --status        # что уже есть и что нет
docker compose run --rm fetch --check         # только проверка, без загрузки
docker compose run --rm fetch --force         # перекачать всё заново
docker compose run --rm fetch --only data,dump
```

Проверяет контрольные размеры, переживает обрывы сети и перезапускается с места сбоя,
повторный запуск не тянет уже скачанное.

## `deploy/db_init.sh` — инициализация базы

Выполняется контейнером `db-init` при каждом `up`, поэтому идемпотентен: если в базе уже есть
вина — просто выходит. Если база пустая и рядом лежит дамп (`deploy/archives/pgvector.dump`) —
восстанавливает его, чтобы сервису не пришлось считать векторы заново. Если дампа нет — выходит
успешно, а векторы посчитает сам `ml` при старте.

Логи смотреть так: `docker compose logs db-init`.

## `deploy/run_eval.sh` — прогон оценки внутри docker-сети

Обёртка, которая запускает скрипт прогона внутри контейнера `eval` и обращается к `ml`
по внутреннему адресу. Запуск:

```bash
docker compose run --rm eval                            # набор ./eval
EVAL_DIR=/путь/к/набору docker compose run --rm eval     # другой набор той же структуры
```

Набор — папка с `queries/` (фотографии) и `queries.tsv` (манифест). Результат — `predictions.jsonl`
внутри этой папки; предыдущий файл переименовывается в `*.bak`. Переопределяемые переменные:
`EVAL_DIR`, `EVAL_IMAGES_DIR`, `EVAL_MANIFEST`, `EVAL_OUTPUT`, `EVAL_ENDPOINT`.

## `eval/participant_test.sh` — прогон набора с записью предсказаний

Отправляет по одному фото из манифеста и пишет по строке на фотографию:

```json
{"query_id":"q-000001","image_path":"019c68d0.jpg","image_sha256":"c975b31e…",
 "predicted_slug":"pino-glyu-2025","latency_ms":15201}
```

| Параметр | Значение по умолчанию | Смысл |
|---|---|---|
| `--images-dir` | — | папка с фотографиями |
| `--manifest` | — | TSV-манифест `query_id<TAB>image_path` |
| `--endpoint` | `http://127.0.0.1:8080/v1/eval/predict` | куда обращаться |
| `--output` | `predictions.jsonl` | файл результата |
| `EVAL_MAX_TIME` | `300` | таймаут на одно фото, секунды |

Запросы идут строго по одному, без параллелизма и повторов. Доступные заголовки ответа —
`{"slug": "..."}` (плоский) или `[{"slug": "..."}]` (массив).

## `deploy/Caddyfile` — HTTPS перед интерфейсом

Отдаёт `web` по HTTPS: браузер разрешает доступ к микрофону (голосовой ввод) только на защищённом
соединении. Адрес задаётся переменной `PUBLIC_HOST` в `.env` (по умолчанию `localhost`),
порты — `HTTP_PORT` и `HTTPS_PORT`.

## Служебные команды сервиса распознавания

```bash
docker compose exec ml python build_index.py           # собрать векторы, если их нет
docker compose exec ml python build_index.py --force   # пересобрать индекс целиком
docker compose exec ml python preflight.py             # проверить наличие данных и моделей
```
