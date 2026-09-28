# web — Nuxt (UI + BFF)

Mobile-first интерфейс сканера в стилистике «Своё Вино» (макет WineScanner) и тонкий BFF на Nitro
поверх `ml` и `sommelier`. Приложение — SPA (`ssr: false`): цель сомелье и сканы живут в браузере.

## Экраны

| Путь | Что |
|---|---|
| `/` | загрузка фото (камера / галерея / drag&drop / «попробовать на примере») → экран поиска |
| `/wine/:slug` | карточка: народный рейтинг, вердикт по цели с построчной проверкой, характеристики, к чему подать; ниже — «лучше под вашу цель» или аналоги других виноделен |
| `/no-match` | точного совпадения нет: похожие по этикетке (top-5 поиска) + аналоги из других виноделен |
| `/session` | сканы за сессию, ранжированные под цель |
| панель «Спросить сомелье» | чат (текст/голос), сохраняет цель; открывается с любого экрана |

## Структура

- `layouts/default.vue` — шапка, плашка цели, кнопка сомелье, панель чата
- `pages/` — экраны выше; `components/` — `WineTile`, `WineRow`, `WineImage`, `RatingPill`, `GoalBar`, `SommelierDrawer`, `UiIcon`
- `composables/useSommelier.ts` — переписка и цель (localStorage), `useScanner.ts` — скан и история (sessionStorage)
- `plugins/state.client.ts` — поднимает состояние из хранилищ до монтирования страниц
- `utils/wine.ts` — типы ответов, картинки, ранжирование сканов
- `assets/css/main.css` — токены (цвета, шрифты Playfair Display + Noto Sans), кнопки, чипы
- `server/api/` — BFF: `scan`, `wine/[slug]`, `wines`, `chat`, `stt`, `ref/[slug]`, `stats`
- `public/img/` — логотип, фоновая графика, `sample-label.jpg` для кнопки «на примере»

## Запуск

```bash
npm install
npm run dev          # http://localhost:3000, бэкенды: ML_URL / SOMMELIER_URL (по умолч. 127.0.0.1:8080 / :8090)
npm run build && node .output/server/index.mjs   # прод; адреса — NUXT_ML_URL / NUXT_SOMMELIER_URL
```

`NUXT_PUBLIC_DEBUG_INFO=1` (в compose — `DEBUG_INFO=1` в `.env`) показывает в UI техническую информацию:
время поиска, score/отрыв top-1 и «сходство» похожих вин. По умолчанию скрыто — пользователю это не нужно.

В Docker (`docker compose up web`) собирается прод-сборка, адреса бэкендов задаёт compose.
