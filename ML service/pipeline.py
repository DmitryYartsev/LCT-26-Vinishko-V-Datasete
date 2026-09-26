# -*- coding: utf-8 -*-
"""Обновлённый пайплайн предсказаний: image retrieval + OCR-rerank.

Все параметры — из YAML через OmegaConf (по умолчанию ``../config/pipeline.yaml``).

Запуск (одиночное фото):
    OPENROUTER_API_KEY=... python pipeline.py --config ../config/pipeline.yaml \
        --image /path/to/photo.webp

Запуск (оценка accuracy по eval-CSV):
    OPENROUTER_API_KEY=... python pipeline.py --config ../config/pipeline.yaml \
        --eval-csv data/real_data_with_slug_eval.csv --images-dir "data/Реальные фото"

Внутри: пре-OCR retrieval (SigLIP2 + YOLO-кроп + pgvector) -> OCR (gpt-4o-mini,
1 вызов) -> самописный мэтчинг по found_in_catalog_corrected.csv -> при
необходимости визуальная проверка фото каталога.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from pipeline_config import load_config, apply_retrieval_env

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WS_ROOT = REPO.parent

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))


def load_env_file(path: Path) -> None:
    """Простой парсер .env (KEY="value" или KEY=value) — НЕ перезаписывает существующие."""
    if not path.is_file():
        return
    for ln in path.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#') or '=' not in ln:
            continue
        k, v = ln.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def apply_host_paths(cfg) -> None:
    """Переопределяет Docker-пути конфига на пути рабочей станции.

    ``pipeline.py`` — standalone-прогон НА ХОСТЕ. Конфиг по умолчанию содержит
    контейнерные пути (/app/...), которые на хосте не существуют. Здесь они
    заменяются на абсолютные пути рабочей станции. Ключ pgvector-индекса
    (``index_model``) оставляем контейнерным — индекс строился в контейнере.
    """
    cfg.paths.models_dir = str(REPO / 'models')
    cfg.paths.reference_csv = str(WS_ROOT / 'data/found_in_catalog_reference/found_in_catalog_corrected.csv')
    cfg.paths.start_photos = str(WS_ROOT / 'data/found_in_catalog_reference/start_photos')
    cfg.paths.prompt_file = str(WS_ROOT / 'code/prompts_wine_match.txt')
    cfg.retrieval.model = str(REPO / 'models/siglip2-base-patch16-256')
    cfg.retrieval.crop_model = str(REPO / 'models/yolo11n.pt')
    cfg.retrieval.label_model = str(REPO / 'models/label_det_best.pt')
    cfg.retrieval.database_url = 'postgresql://vino:vino@127.0.0.1:5432/vino'
    cfg.retrieval.index_model = '/app/models/siglip2-base-patch16-256'


def build_rank(cfg):
    """Возвращает rank(img) -> (res, bottle, label) — копия логики app._rank."""
    import db
    from encoder import get_encoder
    from crop import maybe_crop, maybe_label_crop, USE_LABEL_BRANCH, LABEL_SUFFIX

    enc = get_encoder()
    conn = db.connect()
    # ключ индекса в pgvector может отличаться от пути загрузки энкодера
    # (напр. host-путь vs /app/models/... в контейнере)
    model = str(cfg.retrieval.index_model) if 'index_model' in cfg.retrieval \
        else enc.model_name
    pipeline = str(cfg.retrieval.pipeline)
    k = int(cfg.retrieval.top_k)

    def rank(img):
        bottle, _ = maybe_crop(img)
        want_a = pipeline in ('bottle', 'combined')
        want_b = USE_LABEL_BRANCH and pipeline in ('label', 'combined')
        label, found, _ = maybe_label_crop(bottle)
        if want_b and not found:
            label = bottle
        res_a = db.search(conn, enc.embed([bottle])[0], model, k=k) if want_a else None
        res_b = db.search(conn, enc.embed([label])[0], model + LABEL_SUFFIX, k=k) if want_b else None
        if res_a is None:
            res = res_b
        elif res_b is None:
            res = res_a
        elif res_a[0]['score'] >= res_b[0]['score']:
            res = res_a
        else:
            res = res_b
        return res, bottle, label

    return rank, enc, conn


def make_label_crop_fn():
    from crop import maybe_crop, maybe_label_crop

    def crop_fn(img):
        bottle, _ = maybe_crop(img)
        label, found, _ = maybe_label_crop(bottle)
        return label if found else bottle

    return crop_fn


def build_ocr_components(cfg, enc):
    import ocr_rerank as ocr

    extractor = ocr.OcrExtractor(cfg)
    matcher = ocr.CsvMatcher(cfg)
    crop_fn = make_label_crop_fn() if cfg.visual.enabled else None
    verifier = ocr.VisualVerifier(cfg, enc.embed, crop_fn)
    return extractor, matcher, verifier


def predict_one(cfg, img_path, rank, enc, extractor, matcher, verifier):
    img = Image.open(img_path).convert('RGB')
    res, bottle, label = rank(img)
    pre_ocr_slug = res[0]['slug'] if res else None
    pre_ocr_score = res[0]['score'] if res else 0.0
    import ocr_rerank
    out = ocr_rerank.rerank(cfg, img, label, pre_ocr_slug, pre_ocr_score,
                            extractor, matcher, verifier)
    out['top5_retrieval'] = [{'slug': r['slug'], 'score': r['score']} for r in (res or [])]
    return out


def run_eval(cfg, rank, enc, extractor, matcher, verifier, args):
    eval_csv = Path(args.eval_csv)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(eval_csv, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)  # пропускаем заголовок
        for r in reader:
            if len(r) > 2 and r[1].strip() and r[2].strip():
                rows.append((r[1].strip(), r[2].strip()))
    if args.limit:
        rows = rows[:args.limit]

    preds, correct_pre, correct_final = [], 0, 0
    t0 = time.time()
    for i, (img_name, true_slug) in enumerate(rows):
        img_path = images_dir / img_name
        if not img_path.exists():
            preds.append({'image': img_name, 'true_slug': true_slug,
                          'pre_ocr_slug': None, 'final_slug': None,
                          'stage': None, 'reason': 'image_missing',
                          'pre_ocr_score': None, 'csv_confidence': None,
                          'visual_similarity': None})
            continue
        r = predict_one(cfg, img_path, rank, enc, extractor, matcher, verifier)
        pre = r['pre_ocr_slug']
        fin = r['final_slug']
        correct_pre += int(pre == true_slug)
        correct_final += int(fin == true_slug)
        preds.append({'image': img_name, 'true_slug': true_slug,
                      'pre_ocr_slug': pre, 'final_slug': fin,
                      'stage': r['stage'], 'reason': r.get('reason'),
                      'pre_ocr_score': r['pre_ocr_score'],
                      'csv_confidence': r.get('csv_confidence'),
                      'visual_similarity': r.get('visual_similarity'),
                      'ocr_fields': r.get('ocr_fields')})
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(rows)} ({(time.time()-t0)/(i+1):.1f}s/фото)')

    n = len(preds)
    acc_pre = correct_pre / n if n else 0.0
    acc_final = correct_final / n if n else 0.0
    report = {'n': n, 'accuracy_retrieval': round(acc_pre, 4),
              'accuracy_final': round(acc_final, 4),
              'correct_retrieval': correct_pre, 'correct_final': correct_final,
              'seconds': round(time.time() - t0, 1)}
    print(f'\naccuracy retrieval (до OCR): {correct_pre}/{n} = {acc_pre:.4f}')
    print(f'accuracy final (с OCR-rerank): {correct_final}/{n} = {acc_final:.4f}')

    with open(out_dir / 'ocr_rerank_report.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': report, 'predictions': preds}, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'ocr_rerank_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'true_slug', 'pre_ocr_slug', 'final_slug', 'stage',
                    'reason', 'pre_ocr_score', 'csv_confidence', 'visual_similarity'])
        for p in preds:
            w.writerow([p['image'], p['true_slug'], p['pre_ocr_slug'], p['final_slug'],
                        p['stage'], p['reason'], p['pre_ocr_score'],
                        p.get('csv_confidence'), p.get('visual_similarity')])
    print('отчёт ->', out_dir)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--image', help='одиночное фото для предсказания')
    ap.add_argument('--eval-csv', help='CSV с фото+slug для оценки accuracy')
    ap.add_argument('--images-dir', help='папка с фото (для --eval-csv)')
    ap.add_argument('--out-dir', help='куда писать отчёты')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)

    # ключ OpenRouter удобно держать в code/.env
    load_env_file(WS_ROOT / 'code' / '.env')

    cfg = load_config(args.config)
    apply_host_paths(cfg)          # standalone-прогон на хосте
    apply_retrieval_env(cfg)

    rank, enc, conn = build_rank(cfg)
    extractor, matcher, verifier = build_ocr_components(cfg, enc)

    if args.image:
        r = predict_one(cfg, Path(args.image), rank, enc, extractor, matcher, verifier)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    if args.eval_csv:
        images_dir = args.images_dir or str(cfg.output.images_dir)
        run_eval(cfg, rank, enc, extractor, matcher, verifier, argparse.Namespace(
            eval_csv=args.eval_csv, images_dir=images_dir,
            out_dir=args.out_dir, limit=args.limit))
        return 0

    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
