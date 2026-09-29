# API

Картинки — `multipart/form-data`, поле `image`. Форматы: JPEG, PNG, WebP (HEIC не читается).

## `ml` (:8080)

| Метод | Путь | Ответ |
|---|---|---|
| POST | `/v1/eval/predict` | `{slug, score, margin, branch, stage, pre_ocr_slug, csv_confidence, csv_margin, visual_similarity}` — для скрипта организаторов |
| POST | `/v1/search` | полный ответ для UI (ниже) |
| GET | `/wine/{slug}` | карточка: `slug, name, winery, category, color, region, grape, description, rating` |
| GET | `/ref/{slug}?h=` | эталонное фото; `h` (64–1600) — превью WebP по высоте |
| GET | `/health` | `{status, model, crop, wines, pipeline, label_branch, ocr_rerank, ocr_model, models}` |

`/v1/search`:

```json
{
  "in_catalog": true, "gate_reason": "visual_hi",
  "final_slug": "...", "stage": "retrieval", "pre_ocr_slug": "...",
  "confidence": {"top1_score": 0.8965, "margin": 0.1595, "thresholds": {"score": 0.81, "margin": 0.015}},
  "branch": "label", "branches": {"bottle": {...}, "label": {...}},
  "top1": {"slug": "...", "score": 0.8965, "card": {...}}, "results": [...],
  "ocr_input": "label_crop", "ocr_fields": {...}, "top_candidates": [...],
  "csv_confidence": 0.9185, "csv_margin": 0.4, "visual_similarity": null,
  "elapsed_ms": 6420, "pipeline": "combined"
}
```

Какое поле к какому этапу относится — [pipeline.md](pipeline.md#поля-ответа-v1search-по-этапам).
Коды: `400` — нет картинки или не читается, `404` — неизвестный slug / нет фото.

## `sommelier` (:8090)

Stateless: профиль и история приходят в каждом запросе.

| Метод | Путь | Вход → выход |
|---|---|---|
| POST | `/v1/chat` | `{messages, profile}` → `{reply, profile, picks, suggestions}` |
| POST | `/v1/recommend` | `{profile, k, exclude}` → `{profile, picks}` |
| POST | `/v1/match` | `{slug, profile}` → `{wine, score, reasons, checks, verdict, verdict_text}` |
| POST | `/v1/match_many` | `{slugs, profile}` → `{items}` |
| POST | `/v1/analogs` | `{slug, k}` → `{picks}` — аналоги других виноделен |
| POST | `/v1/stt` | multipart `audio` → `{text}` |
| GET | `/v1/wine/{slug}`, `/v1/vocab`, `/health` | атрибуты вина, словари, статус (`api_key` — задан ли ключ) |

Без `OPENROUTER_API_KEY` `/v1/chat` и `/v1/stt` отдают `502`, остальное работает.

## `web` BFF (:3000, `web/server/api`)

| Роут | Что делает |
|---|---|
| `POST /api/scan` | `image` (+ `profile`) → `ml /v1/search`; при `in_catalog: false` добавляет `similar` (match_many по top-5) и `analogs` |
| `POST /api/wine/{slug}` | карточка + вердикт по профилю + альтернативы/аналоги |
| `POST /api/wines` | карточки сканов сессии |
| `POST /api/chat`, `POST /api/stt` | прокси к сомелье |
| `GET /api/ref/{slug}` | прокси фото |
| `GET /api/stats` | счётчики |
