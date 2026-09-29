# Эксплуатация

## Сервисы

| Сервис | Порт | Роль |
|---|---|---|
| `fetch` | — | одноразовый: докачивает `data/`, `filtered/`, `models/`, дамп индекса |
| `db` | не публикуется | Postgres + pgvector |
| `db-init` | — | одноразовый: пустая БД + есть дамп → `pg_restore` |
| `ml` | 8080 | распознавание |
| `sommelier` | 8090 | сомелье |
| `web` | 3000 | UI + BFF |
| `https` | 80 / 443 | Caddy перед `web` |
| `eval` | — | профиль `eval`, прогон скрипта организаторов |

Порядок: `fetch` → `db` → `db-init` → `ml` + `sommelier` → `web` → `https`. `ml` healthy только после прогрева.

Код `ML service/` и `sommelier service/` смонтирован в контейнеры — после правки Python достаточно
`docker compose restart ml` (или `sommelier`). Правка `web` или Dockerfile — `docker compose up -d --build`.

Тома: `pgdata` (БД и индекс — не терять), `hfcache`, `caddydata`.

## Данные и модели

Качает `fetch` (Google Drive + HuggingFace), в git не входят.

| Что | Куда | Содержимое |
|---|---|---|
| `data` | `data/` → `/app/ref` | `found_in_catalog_corrected.csv`, `catalog_ocr_fields.csv`, `start_photos/`, наборы для оценки |
| `filtered` | `filtered/` | фото каталога `<slug>/NN.webp` + `catalog.csv` (карточки) |
| `models` | `models/` | `yolo11n.pt`, `label_det_best.pt` |
| SigLIP2 | `models/siglip2-base-patch16-256/` | энкодер, 1.4 ГБ |
| `dump` | `deploy/archives/pgvector.dump` | готовые векторы индекса |

```bash
docker compose run --rm fetch --status          # что уже на диске
docker compose run --rm fetch --check           # доступность ссылок
docker compose run --rm fetch --only data,dump  # часть
docker compose run --rm fetch --force           # перекачать
```

Ссылки и ожидаемые размеры архивов — `ARCHIVES` в `deploy/fetch_data.py`.

## Деплой на сервер

Ресурсы: 4 CPU, 8 ГБ RAM, 30 ГБ диска.

```bash
cp .env.example .env         # OPENROUTER_API_KEY, PUBLIC_HOST
docker compose up -d --build
```

`PUBLIC_HOST` — адрес для HTTPS (микрофон в браузере работает только по HTTPS или на localhost):

- IP или `localhost` — сертификат внутреннего CA, браузер один раз предупредит;
- домен (например `1-2-3-4.sslip.io` для IP 1.2.3.4) — Let's Encrypt, нужны порты 80/443.

Наружу нужны только 80/443; 3000/8080/8090 — для отладки.

## Индекс

Пересборка нужна при смене энкодера, кропа индекса (`crop_*`, `label_*`, `label_align`) или фото каталога:

```bash
docker compose exec ml python build_index.py --force    # ~40 мин на CPU
```

Добавить вино: фото в `filtered/<slug>/`, строку в `filtered/catalog.csv`, для OCR — в
`data/found_in_catalog_corrected.csv`; затем пересборка.

Сохранить индекс в дамп:

```bash
docker compose exec -T db pg_dump -U vino -d vino -Fc -f /tmp/pgvector.dump
docker compose cp db:/tmp/pgvector.dump deploy/archives/pgvector.dump
```

Проверка: `docker compose exec db psql -U vino -d vino -c "select model, count(*) from wine_vectors group by 1;"` —
два ключа (модель и `…#label`) с одинаковым числом векторов.

## Типовые настройки

- **Сменить VLM**: `ocr.model` в `config/pipeline.yaml` → `docker compose restart ml`. Проверка —
  `ocr_model` в `/health`.
- **Выключить OCR**: `ocr.enabled: false`. Средняя зона гейта тогда всегда даёт «нет в каталоге».
- **Пороги гейта**: `thresh_score`, `thresh_score_lo`, `ocr.confirm_confidence`, `csv_match.agree_confidence`.
  Проверять на фото вин вне каталога (ложные карточки) и из каталога (потерянные карточки).

## Диагностика

```bash
docker compose ps
docker compose logs -f ml
curl -s localhost:8080/health
curl -s -F image=@photo.jpg localhost:8080/v1/search
```

| Симптом | Причина / что делать |
|---|---|
| `fetch exited (N)` | ошибка скачивания: `docker compose logs fetch` (сеть, доступ к файлу, место) |
| `Google Drive вернул страницу вместо файла` | устарел `gid` в `ARCHIVES` |
| `[preflight] НЕ ХВАТАЕТ ФАЙЛОВ` | данные удалены: `docker compose run --rm fetch` |
| `ml` долго не `healthy`, в логе `[warmup] эмбеддинг N фото` | нет дампа, индекс строится с нуля |
| всегда «нет в каталоге» / нет OCR | нет или неверный `OPENROUTER_API_KEY`; после правки `.env` — `docker compose up -d` |
| `400 bad image` | формат не читается (HEIC) |
| скан идёт десятки секунд | медленный ответ VLM: `ocr.timeout`, `ocr.retries`, другая модель |
| не работает голос | открыт не по HTTPS; `api_key: false` в `/health` сомелье |
| `bad interpreter` / `$'\r'` в `*.sh` | CRLF: `.gitattributes` держит `*.sh` в LF — переклонировать |
| `down -v` снёс индекс | восстановится из дампа при следующем `up` |
