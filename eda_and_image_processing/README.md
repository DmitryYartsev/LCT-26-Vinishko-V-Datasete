# eda_and_image_processing — EDA и подготовка изображений

Разведочный анализ каталога и подготовка эталонных фото для индекса. **Не часть рантайма** —
запускается офлайн, готовит данные для `service/`.

## Файлы

| Файл | Что |
|---|---|
| `eda.ipynb` | EDA каталога (статистика, near-duplicates, покрытие фото, разрешения) |
| `process_images.py` | **главный скрипт**: дамп → `filtered/<slug>/*.webp` + `filtered/catalog.csv` + npz-индекс |
| `scrape_images.py` | сбор доп. фото из веба (Yandex + Playwright) → `SCRAPED/` |

## process_images.py — что делает

1. снимает size-варианты Strapi (`thumbnail_/small_/…` + оригинал) → крупнейший;
2. контент-дедуп по sha1;
3. матчит фото с вином (translit «Название фото» / slug), собирает **все** фото на slug;
4. раскладывает в `filtered/<slug>/NN.webp`, пишет `filtered/catalog.csv` (карточки + список фото);
5. строит индекс `service/index/catalog[_crop].npz` (по вектору на каждое фото — мультивектор).

```bash
cd eda_and_image_processing
uv run python process_images.py                 # crop по CROP_ENABLED (деф 1)
uv run python process_images.py --both          # оба индекса (crop off + on)
uv run python process_images.py --no-index      # только filtered/ + catalog.csv
```

Вход — `paths.UPLOADS` (распакованная медиатека Strapi). Пути — из корневого `paths.py`.

## Скрейп доп. фото

```bash
uv run python scrape_images.py --all --per-wine 6 --sleep 3   # весь каталог
uv run python scrape_images.py --slugs slug-a,slug-b           # точечно
```
Движок — Yandex Images через Playwright (реальный браузер). Resume-safe. При капче
останавливается — добавь `--proxy http://user:pass@host:port`. Скрейп «как есть»,
чистка/верификация — отдельным шагом.
