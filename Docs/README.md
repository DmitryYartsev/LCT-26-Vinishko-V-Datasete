# Своё Вино — распознавание винных этикеток

Сканер винных этикеток: пользователь фотографирует бутылку, сервис находит вино в каталоге
(2 107 позиций) и отдаёт карточку, а если такого вина в каталоге нет — показывает похожие.
С телефона можно снять бутылку прямо в сервисе или выбрать фото из галереи, с компьютера —
загрузить файл: интерфейс адаптивный и одинаково удобен на обоих. Рядом работает цифровой
сомелье: превращает разговор (в том числе голосовой) в профиль вкуса и оценивает относительно
него каждую отсканированную бутылку. Голосовой ввод доступен и на телефоне, и на компьютере —
нужен только HTTPS (или `localhost`).

Распознавание двухступенчатое: сначала визуальный поиск по векторам фотокаталога, затем
переранжирование по тексту этикетки, который читает внешняя vision-модель.

## Быстрый старт

```bash
git clone <repo> && cd LCT-26-Vinishko-V-Datasete
cp .env.example .env          # впишите OPENROUTER_API_KEY — ключ https://openrouter.ai/keys
docker compose up -d --build  # первый запуск скачивает модели и данные (≈2 ГБ)
```

Когда `ml` в `docker compose ps` станет `healthy`, интерфейс доступен на
<http://localhost:3000>. С готовым дампом индекса сервис поднимается за минуту; если дампа нет,
векторы каталога считаются заново — это десятки минут (см. [reference/services.md](reference/services.md)).

Введение по шагам с проверками — [tutorials/quickstart.md](tutorials/quickstart.md).

## Что где лежит

| Папка | Что это |
|---|---|
| `ML service/` | сервис распознавания (FastAPI, порт 8080): энкодер, кропы, чтение этикетки, pgvector |
| `sommelier service/` | сервис сомелье (FastAPI, порт 8090): диалог, распознавание речи, подбор аналогов |
| `web/` | интерфейс (Nuxt): сканер, карточка вина, сессия, панель сомелье, BFF-роуты `/api/*` |
| `config/pipeline.yaml` | параметры пайплайна распознавания |
| `deploy/` | развёртывание: загрузка данных и моделей, инициализация БД, HTTPS, прогон оценки |
| `eval/` | скрипт прогона сервиса по набору фотографий: предсказания и время ответа |
| `data/`, `filtered/`, `models/` | данные, фотокаталог, веса — их скачивает контейнер `fetch` |
| `Docs/` | документация |

## Документация

### Начало работы

| Страница | О чём |
|---|---|
| [tutorials/quickstart.md](tutorials/quickstart.md) | Поднять стек с нуля и распознать первое фото. |
| [tutorials/first-evaluation.md](tutorials/first-evaluation.md) | Прогнать оценку на наборе фото и получить предсказания с латентностью. |

### Руководства

| Страница | О чём |
|---|---|
| [how-to/deploy-on-server.md](how-to/deploy-on-server.md) | Развернуть на сервере: `.env`, HTTPS, время старта, проверки. |
| [how-to/change-ocr-model.md](how-to/change-ocr-model.md) | Сменить vision-модель, читающую этикетки. |
| [how-to/tune-catalog-gate.md](how-to/tune-catalog-gate.md) | Настроить пороги решения «вино есть в каталоге». |
| [how-to/update-catalog-and-index.md](how-to/update-catalog-and-index.md) | Добавить вина и фото, пересобрать векторы, перенести индекс. |
| [how-to/troubleshoot.md](how-to/troubleshoot.md) | Сканер не отвечает, старт долгий, фото не читается. |

### Справочник

| Страница | О чём |
|---|---|
| [reference/services.md](reference/services.md) | Контейнеры, порты, тома, порядок запуска, ресурсы. |
| [reference/configuration.md](reference/configuration.md) | Параметры `config/pipeline.yaml` и переменные окружения. |
| [reference/ml-api.md](reference/ml-api.md) | Эндпоинты сервиса распознавания и поля ответа. |
| [reference/web-and-sommelier-api.md](reference/web-and-sommelier-api.md) | Роуты интерфейса и эндпоинты сомелье. |
| [reference/data-and-models.md](reference/data-and-models.md) | Данные, веса и схема БД. |
| [reference/metrics.md](reference/metrics.md) | Как считаются recall, precision, F1 и тождество вина. |
| [reference/scripts.md](reference/scripts.md) | Скрипты в `deploy/` и `eval/`. |

### Как устроено

| Страница | О чём |
|---|---|
| [explanation/architecture.md](explanation/architecture.md) | Из каких частей состоит система и как они взаимодействуют. |
| [explanation/pipeline.md](explanation/pipeline.md) | Обработка запроса по этапам: где какой шаг, кто отбирает кандидатов, на чём основаны решения. |
| [explanation/retrieval-and-crops.md](explanation/retrieval-and-crops.md) | Визуальный поиск: энкодер, две ветки, разные кропы для индекса и запроса. |
| [explanation/ocr-rerank.md](explanation/ocr-rerank.md) | Как чтение этикетки уточняет ответ и где предел подхода. |
| [explanation/catalog-gate.md](explanation/catalog-gate.md) | Как сервис решает, что вина нет в каталоге. |
| [explanation/evaluation.md](explanation/evaluation.md) | Как измеряется качество и что показывают замеры. |
| [explanation/tradeoffs.md](explanation/tradeoffs.md) | Компромиссы и известные ограничения. |
