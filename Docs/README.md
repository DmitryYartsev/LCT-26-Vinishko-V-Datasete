# Документация проекта «Своё Вино» (ЛЦТ-2026, кейс РСХБ)

Сервис распознаёт вино по фотографии этикетки и возвращает карточку каталога «Своё Вино»;
если вина в каталоге нет — показывает похожие по этикетке и аналоги других виноделен.
Дополнительно есть «цифровой сомелье»: диалог (текст/голос) → профиль предпочтений →
вердикт «подходит / частично / не под цель» для каждого скана.

## Навигация

| Раздел | О чём |
|---|---|
| [Service.md](Service.md) | Как устроены сервисы `ml`, `sommelier`, `web` и БД: эндпоинты, поля запросов/ответов, конфиги, тюнинг гейта «есть в каталоге» |
| [Evaluation.md](Evaluation.md) | Как считать метрики: датасеты (62, eval2, негативы), все скрипты прогонов, где лежат отчёты |
| [Architecture.md](Architecture.md) | Архитектура целиком: слои, поток данных, данные в БД, границы и принятые решения |
| [../deploy/README.md](../deploy/README.md) | Развёртывание на чистой машине: что скачивается, профили compose, разбор ошибок |
| [../ML service/README.md](../ML service/README.md) · [../ML evaluation/README.md](../ML evaluation/README.md) · [../web/README.md](../web/README.md) · [../sommelier service/README.md](../sommelier service/README.md) | README разделов кода |

## Как поднять сервис

Нужны Docker + Compose v2, ~4 ГБ свободного места и интернет (данные и модель скачиваются при
первом запуске). Единственный ручной шаг — ключ OpenRouter.

```bash
git clone -b ml-web git@github.com:DmitryYartsev/LCT-26-Vinishko-V-Datasete.git
cd LCT-26-Vinishko-V-Datasete
printf 'OPENROUTER_API_KEY=sk-or-...\n' > .env    # ключ для OCR-реранка и сомелье
docker compose up --build -d
curl -s localhost:8080/health                     # {"status":"ok", ...}
```

| Сервис | Адрес | Что там |
|---|---|---|
| `web` | http://localhost:3000 | UI: сканер, карточка вина, «нет совпадения», сканы за сессию, панель сомелье |
| `ml` | http://localhost:8080 | API распознавания: `/v1/search`, `/v1/eval/predict`, `/wine/{slug}`, `/ref/{slug}`, `/health` |
| `sommelier` | http://localhost:8090 | API сомелье: `/v1/chat`, `/v1/recommend`, `/v1/match`, `/v1/match_many`, `/v1/analogs`, `/v1/stt` |
| `db` | localhost:5432 | Postgres + pgvector (`wines`, `wine_vectors`, `models`, `wine_profiles`) |

Что происходит при первом запуске: `fetch` скачивает `data/`, `filtered/`, веса YOLO, дамп индекса
и модель SigLIP2 (1.4 ГБ), `db-init` восстанавливает дамп (векторы индекса), `ml` поднимается за
~10 секунд (пересчёт индекса не нужен), `sommelier` засеивает свои атрибуты, `web` отдаёт UI.

> **Несколько копий репозитория на одной машине**: у копий одинаковое имя compose-проекта, поэтому
> они будут останавливать контейнеры друг друга (симптом — `fetch exited with code 137`). Задайте
> второй копии своё имя в `.env`: `COMPOSE_PROJECT_NAME=vino-test2`. Подробности —
> [../deploy/README.md](../deploy/README.md#две-копии-репозитория-на-одной-машине).

Полезные команды:

```bash
docker compose logs -f fetch        # прогресс скачивания данных и модели
docker compose logs -f ml           # preflight, warm-up, старт
docker compose restart ml           # после правки .env или config/pipeline.yaml
docker compose down                 # остановить (том с БД сохраняется)
docker compose down -v              # остановить и снести том (индекс восстановится из дампа)
docker compose run --rm fetch --status   # что уже скачано, а чего не хватает
```
