# Развёртывание сервиса на другом сервере

Код и конфиги едут в git (`git clone`), а «тяжёлое» подкладывается на сервере: данные
каталога (`data/`, `filtered/`), веса (`models/`) и дамп БД с векторами индекса
(`pgvector.dump`). Ключ OpenRouter переносится вручную в `.env` (в git не попадает).

Ниже — весь путь с нуля. Время: ~10 минут, из которых большая часть — скачивание
модели SigLIP2 (1.4 ГБ) и распаковка.

## Порядок действий

```bash
git clone <repo> && cd LCT-26-Vinishko-V-Datasete
bash deploy/fetch_data.sh           # данные + модели + дамп БД + SigLIP2 (1.4 ГБ из HuggingFace)
printf 'OPENROUTER_API_KEY=sk-or-...\n' > .env   # ключ без него: только retrieval, без OCR-rerank
docker compose up --build -d        # db -> db-init (восстановит дамп) -> ml -> web
curl -s localhost:8080/health       # {"status":"ok", ...}
```

Проверка перед запуском (ничего не качает): `python3 deploy/fetch_data.py --check`.

## Что именно скачивается

| Источник | Куда на сервере | Размер |
|---|---|---|
| Google Drive: `lct-data.zip` | `data/` (каталог, `start_photos/`, OCR-поля) | 247 МБ |
| Google Drive: `lct-filtered.zip` | `filtered/` (фото каталога `<slug>/NN.webp`, `catalog.csv`) | 131 МБ |
| Google Drive: `lct-models-small.zip` | `models/` (`label_det_best.pt`, `yolo11n.pt`) | 11 МБ |
| Google Drive: `lct-pgvector-dump.zip` | `deploy/archives/pgvector.dump` (дамп БД) | 30 МБ |
| HuggingFace (публично) `google/siglip2-base-patch16-256` | `models/siglip2-base-patch16-256/` | 1.4 ГБ |

Архивы после распаковки удаляются (остаётся только `pgvector.dump`), `--keep-zips` оставит их.
Модель энкодера из HuggingFace не выгружается в облако: она публичная и совпадает
байт-в-байт, а ключ индекса в БД (`/app/models/siglip2-base-patch16-256`) не меняется.

## Что делает `db-init`

Сервис `db-init` (образ pgvector, скрипт `deploy/db_init.sh`) запускается в `docker compose up`
перед `ml`:

* в БД уже есть вина → ничего не делает (идемпотентно);
* база пустая и есть дамп → `pg_restore` восстанавливает 2107 вин и векторы индекса, и
  сервису не нужно считать эмбеддинги (иначе warm-up ~40 минут);
* дампа нет → выходит успешно, `ml` построит индекс сам (в логе будет предупреждение).

## Параметры `fetch_data.sh` / `fetch_data.py`

```bash
bash deploy/fetch_data.sh                  # всё: data, filtered, models, дамп, SigLIP2
bash deploy/fetch_data.sh --no-encoder     # без SigLIP2 (если качаете отдельно/уже есть)
bash deploy/fetch_data.sh --only data,dump # только часть
bash deploy/fetch_data.sh --force          # перекачать архивы заново
bash deploy/fetch_data.sh --keep-zips      # не удалять скачанные архивы
python3 deploy/fetch_data.py --dest /tmp/x # распаковать в другой корень (проверка)
```

На сервере **без python3** на хосте скачивание можно выполнить в контейнере
(используется образ `python:3.11-slim`, только стандартная библиотека):

```bash
docker compose --profile fetch run --rm fetch
```

## Обновление данных на сервере

1. Перезалить архивы в облако (на машине, где данные есть):
   `zip -r -0 deploy/archives/lct-data.zip data` и т.п., затем обновить id/размеры в
   `deploy/fetch_data.py` (`ARCHIVES`).
2. На сервере: `bash deploy/fetch_data.sh --force --only data,filtered`.
3. Пересобрать индекс (новые фото каталога): удалить том БД `docker compose down -v`
   (осторожно: снесёт и дамп-состояние) или очистить таблицу `wine_vectors` и
   перезапустить `ml` — warm-up посчитает эмбеддинги заново.

## Если что-то не так

| Симптом | Причина / что делать |
|---|---|
| `файл закрыт: включите доступ «всем, у кого есть ссылка»` | у файла в Google Drive нет публичного доступа |
| `Google Drive вернул страницу вместо файла` | ссылка/`id` устарели — проверьте `ARCHIVES` в `fetch_data.py` |
| `это не zip-архив` | файл скачался не полностью: `--force` |
| `No space left on device` | нужно ~3 ГБ свободного места (1.4 ГБ модель + архивы + распаковка) |
| `[preflight] НЕ ХВАТАЕТ ФАЙЛОВ` | не выполнен `fetch_data.sh` (или упал на середине) |
| `OpenRouter` не отвечает / нет OCR-переранжирования | нет `.env` с `OPENROUTER_API_KEY` |
