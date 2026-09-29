# Развёртывание

Код и конфиги едут в git, а «тяжёлое» (данные каталога, веса, дамп индекса) скачивается
при первом `docker compose up`. На чистой машине:

```bash
git clone git@github.com:DmitryYartsev/LCT-26-Vinishko-V-Datasete.git
cd LCT-26-Vinishko-V-Datasete
cp .env.example .env     # вписать OPENROUTER_API_KEY
docker compose up
```

Время первого запуска ~10 минут: в основном скачивание SigLIP 2 (1.4 ГБ) и сборка образов.

## Файлы

| Файл | Что |
|---|---|
| `fetch_data.py` | сервис `fetch`: качает отсутствующие архивы (Google Drive) и SigLIP 2 (HuggingFace), только stdlib |
| `db_init.sh` | сервис `db-init`: в пустую БД восстанавливает `archives/pgvector.dump` (векторы индекса) |
| `run_eval.sh` | сервис `eval`: прогоняет `eval/participant_test.sh` организаторов против `ml` |
| `archives/` | сюда `fetch` кладёт `pgvector.dump` (gitignore) |

## Что скачивается

| Источник | Куда | Размер |
|---|---|---|
| Google Drive: `data` | `data/` — каталог для OCR-матчера, `start_photos/`, OCR-поля каталога | 247 МБ |
| Google Drive: `filtered` | `filtered/` — фото каталога `<slug>/NN.webp` + `catalog.csv` | 131 МБ |
| Google Drive: `models` | `models/` — `label_det_best.pt`, `yolo11n.pt` | 11 МБ |
| Google Drive: `dump` | `deploy/archives/pgvector.dump` | 30 МБ |
| HuggingFace `google/siglip2-base-patch16-256` | `models/siglip2-base-patch16-256/` | 1.4 ГБ |

Архивы после распаковки удаляются. Повторный `up` ничего не качает: `fetch` проверяет, что
нужные файлы уже на месте.

```bash
docker compose run --rm fetch --status          # что уже на диске
docker compose run --rm fetch --check           # доступность ссылок в облаке, без скачивания
docker compose run --rm fetch --only data,dump  # только часть
docker compose run --rm fetch --force           # перекачать заново
```

## HTTPS (нужен для голосового ввода)

Браузер даёт доступ к микрофону только по HTTPS или на `localhost`, поэтому по
`http://<ip>:3000` голосовой ввод сомелье не работает. Перед `web` стоит сервис `https`
(Caddy, конфиг `deploy/Caddyfile`), в `.env` задаётся адрес:

```bash
PUBLIC_HOST=192.168.1.50        # IP: самоподписанный сертификат, браузер один раз предупредит
PUBLIC_HOST=1-2-3-4.sslip.io    # домен на IP 1.2.3.4: настоящий Let's Encrypt, нужны порты 80/443
```

Открывать `https://<PUBLIC_HOST>`. Порты можно переопределить через `HTTP_PORT` / `HTTPS_PORT`.

## `db-init`

* в БД уже есть вина → ничего не делает;
* база пустая и есть дамп → `pg_restore` (≈2100 вин + векторы), `ml` не считает эмбеддинги;
* дампа нет → выходит успешно, `ml` построит индекс сам на старте (~40 минут на CPU).

## Обновление данных

1. Перезалить архив в Google Drive (`zip -r -0 lct-data.zip data` и т.п.), обновить `gid` и
   `expect` в `ARCHIVES` (`fetch_data.py`).
2. `docker compose run --rm fetch --force --only data,filtered`.
3. Новые фото каталога → пересобрать индекс: `docker compose exec ml python build_index.py --force`.

## Если что-то не так

| Симптом | Причина / что делать |
|---|---|
| `dependency failed to start: ... fetch exited (N)` | скачивание упало: `docker compose logs fetch` (сеть / доступ к файлу / место) |
| `файл закрыт: включите доступ «всем, у кого есть ссылка»` | у файла в Google Drive нет публичного доступа |
| `Google Drive вернул страницу вместо файла` | ссылка/`gid` устарели — проверьте `ARCHIVES` в `fetch_data.py` |
| `это не zip-архив` | файл скачался не полностью: `docker compose run --rm fetch --force` |
| `No space left on device` | нужно ~4 ГБ свободного места |
| `[preflight] НЕ ХВАТАЕТ ФАЙЛОВ` в логе `ml` | данные удалены после успешного `fetch`: `docker compose run --rm fetch` |
| `ml` долго не становится `healthy` | нет дампа → строится индекс (`docker compose logs -f ml`) |
| нет OCR-переранжирования / сомелье не отвечает | нет `OPENROUTER_API_KEY` в `.env`; после правки — `docker compose up -d` |
| `bad interpreter` / `$'\r'` в `*.sh` | скрипты с CRLF: `.gitattributes` держит `*.sh` в LF — переклонируйте репозиторий |
