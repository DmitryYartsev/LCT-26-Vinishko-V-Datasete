# eda and image filtering — EDA и фильтрация изображений

Разведочный анализ каталога и фильтрация эталонных фото. **Не часть рантайма** —
запускается офлайн, готовит данные для сервиса. Построение npz-индекса — отдельный
шаг в `../ML service/build_index.py`.

## Файлы

| Файл | Что |
|---|---|
| `eda.ipynb` | EDA каталога (статистика, near-duplicates, покрытие фото, разрешения) |
| `process_images.py` | **главный скрипт**: дамп → `filtered/<slug>/*.webp` + `filtered/catalog.csv` |
| `scrape_images.py` | сбор доп. фото из веба (Yandex + Playwright) → `SCRAPED/` |

## process_images.py — что делает (фильтрация мусора)

1. снимает size-варианты Strapi (`thumbnail_/small_/…` + оригинал) → крупнейший;
2. контент-дедуп по sha1;
3. матчит фото с вином (translit «Название фото» / slug), собирает **все** фото на slug;
4. раскладывает в `filtered/<slug>/NN.webp`, пишет `filtered/catalog.csv` (карточки + список фото).

```bash
cd "eda and image filtering"
uv run python process_images.py                 # -> filtered/ + catalog.csv
```

Вход — `paths.UPLOADS` (распакованная медиатека Strapi). Пути — из корневого `paths.py`.
Дальше по `filtered/` строится индекс: `cd "../ML service" && uv run python build_index.py`.

## Скрейп доп. фото

```bash
cd "eda and image filtering"
uv run python scrape_images.py --all --per-wine 6 --sleep 3   # весь каталог
uv run python scrape_images.py --slugs slug-a,slug-b           # точечно
```
Движок — Yandex Images через Playwright (реальный браузер). Resume-safe. При капче
останавливается — добавь `--proxy http://user:pass@host:port`. Скрейп «как есть»,
чистка/верификация — отдельным шагом.
