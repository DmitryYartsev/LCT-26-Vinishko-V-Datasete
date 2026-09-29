#!/usr/bin/env bash
# Прогон скрипта-оценщика организаторов (eval/participant_test.sh) против сервиса ml
# внутри docker-сети. Запускается сервисом `eval` из docker-compose.yml:
#
#   docker compose run --rm eval                            # eval/queries + eval/queries.tsv
#   EVAL_DIR=/путь/к/набору docker compose run --rm eval    # другой набор той же структуры
#
# Набор — папка вида организаторского пакета: queries/ (фото) + queries.tsv (манифест).
# Результат — <набор>/predictions.jsonl (прошлый файл переименовывается в *.bak).
set -euo pipefail

DIR=/eval
IMAGES_DIR="${EVAL_IMAGES_DIR:-$DIR/queries}"
MANIFEST="${EVAL_MANIFEST:-$DIR/queries.tsv}"
OUTPUT="${EVAL_OUTPUT:-$DIR/predictions.jsonl}"
ENDPOINT="${EVAL_ENDPOINT:-http://ml:8080/v1/eval/predict}"
GRADER=/opt/participant_test.sh

[ -f "$MANIFEST" ] || { echo "[eval] нет манифеста $MANIFEST (EVAL_DIR указывает на набор с queries.tsv?)"; exit 1; }
[ -d "$IMAGES_DIR" ] || { echo "[eval] нет папки с фото $IMAGES_DIR"; exit 1; }

if [ -e "$OUTPUT" ]; then
  backup="$OUTPUT.$(date +%Y%m%d-%H%M%S).bak"
  mv "$OUTPUT" "$backup"
  echo "[eval] прошлые предсказания сохранены в $(basename "$backup")"
fi

total=$(awk 'NR > 1 && NF' "$MANIFEST" | wc -l)
echo "[eval] $total фото -> $ENDPOINT"
bash "$GRADER" --images-dir "$IMAGES_DIR" --manifest "$MANIFEST" \
  --endpoint "$ENDPOINT" --output "$OUTPUT"

jq -rs '
  (map(.latency_ms) | sort) as $lat
  | "[eval] готово: \(length) предсказаний, без slug: \(map(select(.predicted_slug == null)) | length)",
    "[eval] латентность, мс: среднее \(($lat | add / length) | floor), p95 \($lat[((length * 0.95) | ceil) - 1]), макс \($lat[-1])"
' "$OUTPUT"
echo "[eval] файл: $OUTPUT"
