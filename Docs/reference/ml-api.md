# Справочник: API сервиса распознавания (`ml`)

Базовый адрес локально: `http://localhost:8080`. Все запросы со изображением — `multipart/form-data`
с полем `image`. Поддерживаются JPEG, PNG, WebP; HEIC/HEIF (фото iPhone) нужно конвертировать.

## `POST /v1/search`

Основной эндпоинт для интерфейса: поиск по каталогу и подробный ответ.

```bash
curl -s -F "image=@photo.jpg" http://localhost:8080/v1/search | jq .
```

| Поле | Тип | Смысл |
|---|---|---|
| `in_catalog` | bool | показывать карточку (`true`) или сценарий «похожие вина» (`false`) |
| `gate_reason` | string | почему принято решение: `visual_hi`, `ocr_confirms`, `ocr_agrees`, `ocr_not_confirmed`, `low_score` |
| `final_slug` | string | итоговый ответ сервиса (после чтения этикетки) |
| `stage` | string | `retrieval` — ответ дал визуальный поиск, `ocr_rerank` — ответ переставила модель |
| `pre_ocr_slug` | string | что нашёл визуальный поиск до чтения этикетки |
| `csv_confidence` | float \| null | уверенность сопоставления полей этикетки с записью каталога |
| `csv_margin` | float \| null | отрыв от следующего кандидата по полям |
| `visual_similarity` | float \| null | сходство фото пользователя с эталонным фото каталога |
| `confidence.top1_score` | float | сходство первого кандидата визуального поиска |
| `confidence.margin` | float | отрыв первого кандидата от второго |
| `confidence.thresholds.score` | float | верхний порог решения (`thresh_score`) |
| `branch` | string | победившая ветка: `bottle` или `label` |
| `branches` | object | результаты обеих веток: `top1`, `score`, `results` |
| `top1` | object | первый кандидат с карточкой: `slug`, `score`, `card` |
| `results` | array | шорт-лист кандидатов (`slug`, `score`, `card`) |
| `ocr_input` | string | какой кроп ушёл в модель (`label_crop`, `label_crop+bottle_crop`, …) |
| `ocr_fields` | object \| null | что именно прочитала модель на этикетке |
| `top_candidates` | array \| null | кандидаты, из которых выбирала модель сопоставления |
| `elapsed_ms` | int | время обработки запроса, мс |
| `pipeline` | string | режим веток из конфига (`combined`) |

Пример (сокращённо):

```json
{
  "in_catalog": true,
  "gate_reason": "visual_hi",
  "final_slug": "denisov_rubin_klaret_krasnaya_strelka",
  "stage": "retrieval",
  "pre_ocr_slug": "denisov_pino_noir_klaret",
  "csv_confidence": 0.9185,
  "confidence": {"top1_score": 0.8965, "margin": 0.1595,
                 "thresholds": {"score": 0.81, "margin": 0.015}},
  "branch": "label",
  "elapsed_ms": 6420
}
```

## `POST /v1/eval/predict`

Плоский ответ для автоматической оценки: нужен только итоговый slug.

```bash
curl -s -F "image=@photo.jpg" http://localhost:8080/v1/eval/predict
# {"slug":"aristov-anima-millesimato-beloe-bryut","score":0.7323,"margin":0.0131,
#  "branch":"label","stage":"ocr_rerank","pre_ocr_slug":"...","csv_confidence":0.57, ...}
```

Поля: `slug`, `score`, `margin`, `branch`, `stage`, `pre_ocr_slug`, `csv_confidence`, `csv_margin`,
`visual_similarity`. Если включён `eval_abstain` и ответ ниже нижнего порога, возвращается
`{"slug": null}` — это значит «вина в каталоге нет».

## `GET /wine/{slug}`

Карточка вина из БД: `slug`, `name`, `winery`, `category`, `color`, `region`, `grape`,
`description`, `rating`. Неизвестный slug — `404`.

## `GET /ref/{slug}`

Эталонное фото вина из фотокаталога (первое по алфавиту). Параметр `h` (64…1600) отдаёт превью
по высоте в WebP — интерфейс использует его вместо оригинала. Нет фото — `404`.

## `GET /health`

```json
{"status": "ok", "model": "/app/models/siglip2-base-patch16-256", "crop": true, "wines": 2107,
 "pipeline": "combined", "label_branch": true, "ocr_rerank": true,
 "ocr_model": "google/gemini-2.5-flash", "models": [{"model": "...", "dim": 768, "n_vectors": 2108}]}
```

Сервис отвечает `status: ok` только после прогрева: индекс и (если включено) сопоставление полей
готовы. По этому эндпоинту работает healthcheck контейнера.

## Коды ответов

| Код | Когда |
|---|---|
| `200` | успешный ответ |
| `400` | изображение не пришло или не читается как картинка |
| `404` | неизвестный slug в `/wine/{slug}` или `/ref/{slug}` |

## Ориентиры по времени

| Сценарий | Время |
|---|---|
| Только визуальный поиск (`ocr.enabled: false`) | десятые доли секунды |
| Полный цикл с чтением этикетки | 5–20 с (основное время — внешний вызов модели) |
| Повторный проход по кропу бутылки (страховка) | +5–15 с, срабатывает примерно на пятом фото из двадцати |
