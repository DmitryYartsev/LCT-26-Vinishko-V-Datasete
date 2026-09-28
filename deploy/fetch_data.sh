#!/usr/bin/env bash
# Скачивание данных и моделей для сервиса на новом сервере.
#
#   bash deploy/fetch_data.sh                 # data/ + filtered/ + models/ + дамп + SigLIP2
#   bash deploy/fetch_data.sh --no-encoder    # без SigLIP2 (если качаете его отдельно)
#   bash deploy/fetch_data.sh --only data,dump
#   bash deploy/fetch_data.sh --check         # только проверить ссылки и размеры, ничего не качать
#
# После успешного скачивания:
#   1) положить .env в корень репозитория (OPENROUTER_API_KEY=sk-or-...) — без него OCR-rerank
#      не работает, сервис поднимется в режиме только retrieval;
#   2) docker compose up -d  (сервис db-init сам восстановит дамп в пустую БД).
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO" || exit 1

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo '[fetch] нужен python3 (в нём только стандартная библиотека) — на Ubuntu/Debian: apt-get install -y python3'
  echo '[fetch] либо запустите скачивание в контейнере: docker compose --profile fetch run --rm fetch'
  exit 1
fi

echo "[fetch] репозиторий: $REPO"
python3 "$HERE/fetch_data.py" "$@"
rc=$?
if [ $rc -ne 0 ]; then
  echo '[fetch] ОШИБКА: данные скачаны не полностью (см. сообщение выше)'
  exit $rc
fi

for f in "data/found_in_catalog_corrected.csv" "data/start_photos" "filtered/catalog.csv" \
         "models/label_det_best.pt" "models/yolo11n.pt" "models/siglip2-base-patch16-256"; do
  if [ -e "$f" ]; then printf '  ✓ %s\n' "$f"; else printf '  ✗ %s ОТСУТСТВУЕТ\n' "$f"; fi
done
if [ -f "$REPO/.env" ]; then
  echo '  ✓ .env найден'
else
  echo '  ! .env нет: без OPENROUTER_API_KEY сервис поднимется только в режиме retrieval'
fi

echo
echo '[fetch] готово. Дальше:  docker compose up -d   (сервис db-init восстановит дамп и завершится)'
