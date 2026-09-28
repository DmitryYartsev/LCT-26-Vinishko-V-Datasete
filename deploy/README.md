# Развёртывание сервиса на другом сервере

Код и конфиги едут в git (`git clone`), а «тяжёлое» подкладывается на сервере: данные
каталога (`data/`, `filtered/`), веса (`models/`) и дамп БД с векторами индекса
(`pgvector.dump`). Ключ OpenRouter переносится вручную в `.env` (в git не попадает).

Ниже — весь путь с нуля. Время: ~10 минут, из которых большая часть — скачивание
модели SigLIP2 (1.4 ГБ) и распаковка.

## Порядок действий

На чистой машине достаточно двух команд — всё остальное делает сам `docker compose`:

```bash
git clone -b ml-dev2 git@github.com:DmitryYartsev/LCT-26-Vinishko-V-Datasete.git
cd LCT-26-Vinishko-V-Datasete
cp /путь/к/секрету/.env .     # единственный ручной шаг: OPENROUTER_API_KEY=sk-or-...
docker compose up --build     # fetch -> db -> db-init -> ml -> web
```

Что происходит на первом запуске:

1. `fetch` (образ `python:3.11-slim`) скачивает отсутствующее: `data/`, `filtered/`,
   `models/`, дамп pgvector и SigLIP2 из HuggingFace (~1.4 ГБ). Прогресс — `docker compose logs -f fetch`;
2. `db` поднимает Postgres+pgvector, `db-init` восстанавливает дамп (векторы индекса);
3. `ml` грузит SigLIP2 и отвечает, `web` отдаёт UI.

Готово, когда `curl -s localhost:8080/health` отвечает `{"status":"ok", ...}`.

Повторные запуски ничего не перекачивают: `fetch` проверяет, что нужные файлы уже лежат
(`docker compose run --rm fetch --status` — показать состояние, `--check` — проверить сами
ссылки в облаке, ничего не качая).

Скачивание можно запустить и вручную, без полного `up`:

```bash
bash deploy/fetch_data.sh                       # на хосте (нужен python3)
docker compose run --rm fetch                   # в контейнере (то же самое)
docker compose run --rm fetch --status          # что уже на месте
docker compose run --rm fetch --only data,dump  # только часть
docker compose run --rm fetch --force           # перекачать заново
```

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
bash deploy/fetch_data.sh --status         # что уже на диске, а чего не хватает
bash deploy/fetch_data.sh --no-encoder     # без SigLIP2 (если качаете отдельно/уже есть)
bash deploy/fetch_data.sh --only data,dump # только часть
bash deploy/fetch_data.sh --force          # перекачать архивы заново
bash deploy/fetch_data.sh --keep-zips      # не удалять скачанные архивы
python3 deploy/fetch_data.py --dest /tmp/x --status   # посмотреть другой корень
docker compose run --rm fetch --check      # проверить ссылки/размеры в облаке (без скачивания)
```

На сервере **без python3** на хосте то же самое делается в контейнере
(образ `python:3.11-slim`, только стандартная библиотека): `docker compose run --rm fetch`.

Если данные удалили, а `docker compose up` их не возвращает (контейнер `fetch` уже завершён
успешно и переиспользуется) — прогоните шаг принудительно:
`docker compose up --force-recreate fetch` или `docker compose run --rm fetch`.

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
| `docker compose up` пишет `dependency failed to start: ... fetch exited (N)` | скачивание упало: `docker compose logs fetch` (нет сети / нет доступа к файлу / мало места) |
| `[preflight] НЕ ХВАТАЕТ ФАЙЛОВ` | не выполнен `fetch_data.sh` (или упал на середине) |
| `OpenRouter` не отвечает / нет OCR-переранжирования | нет `.env` с `OPENROUTER_API_KEY` |
