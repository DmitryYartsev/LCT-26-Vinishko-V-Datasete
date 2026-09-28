# -*- coding: utf-8 -*-
"""Аудит детектора кропов (бутылка + этикетка): метрики, флаги, монтажи.

Проверяет утверждение «детектор кропа этикетки на части кейсов отрабатывает
некорректно» метриками по ВСЕМУ набору, а не разглядыванием одной картинки.

Для каждого фото eval-набора повторяется ровно тот препроцесс, что и в пайплайне
(``BottleCropper.detect/pick`` -> ``LabelCropper.detect``): геометрия боксов
пишется в ``crop_audit.csv``, поверх неё ставятся эвристические флаги
(см. пороги ниже), а для подозрительных кейсов складывается 3-панельный монтаж
(фото | кроп бутылки | кроп этикетки) — чтобы глазами подтвердить/опровергнуть.

Флаги дополнительно сшиваются с union-ошибками энкодеров
(``ML evaluation/reports/encoder_errors.json``): если кейсы с «плохим кропом»
ошибаются заметно чаще остальных, кроп — реальный ограничитель, а не шум.

    python3 crop_audit.py                                   # 62-набор (data/real_photo)
    python3 crop_audit.py --eval-csv ../data/eval.csv --images-dir ../data/eval
    python3 crop_audit.py --all-montages --dump             # монтажи/кропы по всем фото

Выход (по умолчанию ``../Reports/crop_audit/``):
    crop_audit.csv      метрики и флаги по каждому фото
    summary.json        сводка: частоты флагов, ошибки в группах, источники
    montages/<name>.jpg 3-панельные монтажи (флагированные; с --all-montages — все)
    bottles/, labels/   сырые кропы (только с --dump)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from paths import DATA, EVAL_REPORTS                                  # noqa: E402

IMG_EXT = {'.webp', '.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tif', '.tiff'}

# --- пороги эвристик (числа цитируются в Reports/15_Crop_audit.md) ---
BOTTLE_FULL_AREA = 0.90   # бокс бутылки ≥ 90% кадра -> детектор фактически «вернул кадр»
LABEL_NARROW_W = 0.60     # ширина кропа этикетки / ширина кропа бутылки
LABEL_SHORT_H = 0.20      # высота кропа этикетки / высота кропа бутылки
LABEL_BIG_AREA = 0.85     # кроп этикетки ≥ 85% площади бутылочного кропа
TOUCH_EPS = 0.015         # касание границы кропа бутылки, допуск (доля стороны)
MULTI_REL_CONF = 0.5      # 2-й бокс «различим» -> контротэкетка/шейный ярлык?

CSV_COLS = ['image', 'true_slug', 'w', 'h', 'bottle_found', 'bottle_n', 'bottle_conf',
            'bottle_conf_low', 'bottle_area', 'bottle_w', 'bottle_h', 'label_found',
            'label_n', 'label_conf', 'label_conf_low', 'label_w', 'label_h', 'label_area',
            'label_aspect', 'label_touch_l', 'label_touch_r', 'label_touch_t',
            'label_touch_b', 'label_touch_photo', 'label_photo_conf', 'anchor_inside',
            'flags', 'in_union_errors']

LOWCONF = 0.05            # retry-порог: «спасается» ли детекция понижением порога


def read_eval(csv_path: Path | None, images_dir: Path):
    """[(image_name, true_slug)] из eval-CSV; без CSV — все картинки папки (slug '')."""
    rows = []
    if csv_path and csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter=';'):
                img = (r.get('image') or '').strip()
                slug = (r.get('Slug') or '').strip()
                if img:
                    rows.append((img, slug))
    seen = {n for n, _ in rows}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMG_EXT and p.name not in seen:
            rows.append((p.name, ''))
    return rows


def load_font(size: int = 15):
    """TTF с кириллицей, иначе встроенный шрифт PIL."""
    for p in ('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
              '/System/Library/Fonts/Supplemental/Arial.ttf',
              '/Library/Fonts/Arial.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def _frac(box, w, h):
    x1, y1, x2, y2 = box
    return (x2 - x1) / w, (y2 - y1) / h, ((x2 - x1) * (y2 - y1)) / (w * h)


def _best_conf(model, img, classes=None) -> float:
    """Максимальный conf бокса при очень низком пороге (проверка «спасается ли детекция»)."""
    res = model.predict(img, verbose=False, conf=LOWCONF, classes=classes)
    best = 0.0
    for r in res:
        for b in r.boxes:
            best = max(best, float(b.conf[0]))
    return round(best, 3)


def _thumb(im: Image.Image, cell: int) -> Image.Image:
    t = im.copy()
    t.thumbnail((cell - 8, cell - 8))
    canvas = Image.new('RGB', (cell, cell), (238, 238, 238))
    canvas.paste(t, ((cell - t.width) // 2, (cell - t.height) // 2))
    return canvas


def analyse_one(path: Path, bc, lc):
    """Повторяет препроцесс пайплайна и собирает метрики/флаги по одному фото.

    -> (rec, original, bottle_crop, box_bottle, box_label, label_boxes)
    """
    img = Image.open(path).convert('RGB')
    W, H = img.size
    from crop import BOTTLE_CLASS, LABEL_MIN_CONF, filter_bottle_boxes
    # боксы — ровно как в пайплайне: без боксов меньше CROP_MIN_AREA и с тем же pick
    bb = filter_bottle_boxes(bc.detect(img), img.size)
    rec = {'image': path.name, 'w': W, 'h': H, 'bottle_found': bool(bb), 'bottle_n': len(bb),
           'bottle_conf': None, 'bottle_conf_low': None, 'bottle_area': None,
           'bottle_w': None, 'bottle_h': None, 'label_photo_conf': None,
           'anchor_inside': None}
    box_bottle, bottle = None, img
    if bb:
        x1, y1, x2, y2, c = bc.pick(bb, (W, H))
        box_bottle = (x1, y1, x2, y2)
        rec['bottle_conf'] = round(c, 3)
        rec['bottle_w'], rec['bottle_h'], rec['bottle_area'] = [round(v, 3) for v in
                                                                _frac(box_bottle, W, H)]
        bottle = img.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
    else:
        rec['bottle_conf_low'] = _best_conf(bc.model, img, [BOTTLE_CLASS])
    # якорь «где на фото этикетка»: 1) оценка CROP_FALLBACK=label; 2) проверка, что
    # выбранный бокс бутылки — та самая (соседняя у края кадра = частая ошибка)
    a_boxes = [b for b in lc.detect(img) if b[4] >= LABEL_MIN_CONF]
    rec['label_photo_conf'] = round(a_boxes[0][4], 3) if a_boxes else 0.0
    anchor = a_boxes[0][:4] if a_boxes else None
    if box_bottle is not None and anchor is not None:
        acx, acy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
        rec['anchor_inside'] = bool(box_bottle[0] <= acx <= box_bottle[2]
                                    and box_bottle[1] <= acy <= box_bottle[3])

    BW, BH = bottle.size
    lb = lc.detect(bottle)
    rec.update({'label_found': bool(lb), 'label_n': len(lb), 'label_conf': None,
                'label_conf_low': None, 'label_w': None, 'label_h': None, 'label_area': None,
                'label_aspect': None, 'label_touch_l': False, 'label_touch_r': False,
                'label_touch_t': False, 'label_touch_b': False, 'label_touch_photo': False})
    ox1, oy1 = (box_bottle or (0.0, 0.0, float(W), float(H)))[:2]   # кроп = всё фото?
    box_label = None
    if lb:
        lx1, ly1, lx2, ly2, lc_conf = lb[0]
        box_label = (lx1, ly1, lx2, ly2)
        lw, lh, la = _frac(box_label, BW, BH)
        touch = {'l': lx1 <= BW * TOUCH_EPS, 'r': lx2 >= BW * (1 - TOUCH_EPS),
                 't': ly1 <= BH * TOUCH_EPS, 'b': ly2 >= BH * (1 - TOUCH_EPS)}
        # бокс этикетки в координатах ИСХОДНОГО фото: касание его рамки = этикетку
        # физически отрезал кадр, кроп её уже не восстановит (текст потерян)
        gx1, gy1, gx2, gy2 = lx1 + ox1, ly1 + oy1, lx2 + ox1, ly2 + oy1
        touch_photo = (gx1 <= W * TOUCH_EPS or gy1 <= H * TOUCH_EPS
                       or gx2 >= W * (1 - TOUCH_EPS) or gy2 >= H * (1 - TOUCH_EPS))
        rec.update({'label_conf': round(lc_conf, 3), 'label_w': round(lw, 3),
                    'label_h': round(lh, 3), 'label_area': round(la, 3),
                    'label_aspect': round((lx2 - lx1) / max(1.0, ly2 - ly1), 3),
                    'label_touch_l': touch['l'], 'label_touch_r': touch['r'],
                    'label_touch_t': touch['t'], 'label_touch_b': touch['b'],
                    'label_touch_photo': touch_photo})
    else:
        rec['label_conf_low'] = _best_conf(lc.model, bottle)

    flags = []
    if not bb:
        flags.append('no_bottle')          # ветка этикетки работает по всему фото
    elif rec['bottle_area'] >= BOTTLE_FULL_AREA:
        flags.append('bottle_full')        # бокс бутылки ≈ кадр
    if rec['anchor_inside'] is False:
        flags.append('wrong_bottle')       # кроп не содержит главной этикетки фото
    if not lb:
        flags.append('no_label')           # ветка B = ветка A (кроп этикетки не найден)
    else:
        if rec['label_area'] >= LABEL_BIG_AREA:
            flags.append('label_big')      # «этикеткой» назван почти весь кроп бутылки
        if rec['label_w'] < LABEL_NARROW_W:
            flags.append('label_narrow')   # узкий фрагмент (шейный ярлык/блик)
        if rec['label_h'] < LABEL_SHORT_H:
            flags.append('label_short')    # низкая полоска вместо этикетки
        if rec['label_touch_photo']:
            flags.append('label_clip')     # этикетку режет рамка исходного фото
        if len(lb) > 1 and lb[1][4] >= MULTI_REL_CONF * lb[0][4]:
            flags.append('label_multi')    # лицевая + контрэтикетка / 2 бокса рядом
    rec['flags'] = flags
    return rec, img, bottle, box_bottle, box_label, lb


def label_padded(bottle: Image.Image, box, margin: float):
    """Кроп этикетки с паддингом — ровно как в ``LabelCropper.crop``."""
    BW, BH = bottle.size
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return bottle.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                        min(BW, int(x2 + mx)), min(BH, int(y2 + my))))


def build_montage(name: str, panels, flags, out_path: Path, cell: int = 380):
    """3 панели (фото | кроп бутылки | кроп этикетки) + строка флагов."""
    font = load_font(15)
    head = 48
    grid = Image.new('RGB', (cell * len(panels), cell + head), (255, 255, 255))
    for i, p in enumerate(panels):
        grid.paste(_thumb(p, cell), (i * cell, head))
    d = ImageDraw.Draw(grid)
    d.text((8, 4), name[:70], fill=(0, 0, 0), font=font)
    d.text((8, 24), ('FLAGS: ' + (', '.join(flags) if flags else 'ok'))[:160],
           fill=(170, 0, 0) if flags else (0, 120, 0), font=font)
    grid.save(out_path, quality=88)


def mark(im: Image.Image, box, color=(255, 0, 0), w: int = 4) -> Image.Image:
    """Копия изображения с обведённым боксом (для монтажа)."""
    out = im.copy()
    if box:
        ImageDraw.Draw(out).rectangle([box[0], box[1], box[2], box[3]], outline=color, width=w)
    return out


def summarize(recs: list, union_errors: set) -> dict:
    """Сводка: частоты флагов + доля ошибок в группах (сшивка с union-ошибками)."""
    n = len(recs)
    names = {r['image'] for r in recs}
    joined = bool(union_errors) and union_errors <= names and len(union_errors) < n
    flag_counts, per_flag_err = {}, {}
    for r in recs:
        for f in r['flags']:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    def err_rate(sub):
        return round(sum(1 for r in sub if r['image'] in union_errors) / len(sub), 3) if sub else None

    flagged = [r for r in recs if r['flags']]
    clean = [r for r in recs if not r['flags']]
    for f in sorted(flag_counts):
        sub = [r for r in recs if f in r['flags']]
        per_flag_err[f] = {'n': len(sub),
                           'union_errors': sum(1 for r in sub if r['image'] in union_errors),
                           'error_rate': err_rate(sub) if joined else None}
    return {
        'n': n,
        'n_flagged': len(flagged),
        'pct_flagged': round(100 * len(flagged) / n, 1) if n else None,
        'flag_counts': flag_counts,
        'flag_pct': {f: round(100 * c / n, 1) for f, c in flag_counts.items()},
        'bottle_found_pct': round(100 * sum(1 for r in recs if r['bottle_found']) / n, 1) if n else None,
        'label_found_pct': round(100 * sum(1 for r in recs if r['label_found']) / n, 1) if n else None,
        'joined_to_union': joined,
        'union_errors_n': len(union_errors),
        'error_rate_all': err_rate(recs) if joined else None,
        'error_rate_flagged': err_rate(flagged) if joined else None,
        'error_rate_clean': err_rate(clean) if joined else None,
        'per_flag': per_flag_err,
        'lowconf_recovery': {
            'no_bottle_n': sum(1 for r in recs if 'no_bottle' in r['flags']),
            'no_bottle_ge_0.10': sum(1 for r in recs if 'no_bottle' in r['flags']
                                     and (r['bottle_conf_low'] or 0) >= 0.10),
            'no_bottle_ge_0.15': sum(1 for r in recs if 'no_bottle' in r['flags']
                                     and (r['bottle_conf_low'] or 0) >= 0.15),
            'no_label_n': sum(1 for r in recs if 'no_label' in r['flags']),
            'no_label_ge_0.10': sum(1 for r in recs if 'no_label' in r['flags']
                                    and (r['label_conf_low'] or 0) >= 0.10),
            'no_label_ge_0.15': sum(1 for r in recs if 'no_label' in r['flags']
                                    and (r['label_conf_low'] or 0) >= 0.15),
        },
        'rescue_label_fallback': {
            'n': sum(1 for r in recs if r.get('rescue_label_fallback')),
            'union_errors': sum(1 for r in recs if r.get('rescue_label_fallback')
                                and r['image'] in union_errors),
            'error_rate': err_rate([r for r in recs if r.get('rescue_label_fallback')]) if joined else None,
        },
        'flags_by_image': {r['image']: r['flags'] for r in recs if r['flags']},
    }


def main(argv=None) -> int:
    import os
    from pipeline_config import load_config, apply_retrieval_env
    import pipeline as P

    ap = argparse.ArgumentParser(description='Аудит детектора кропов (бутылка + этикетка)')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--eval-csv', default=str(DATA / 'real_data_with_slug_eval.csv'))
    ap.add_argument('--images-dir', default=str(DATA / 'real_photo'))
    ap.add_argument('--out', default=str(REPO / 'Reports' / 'crop_audit'))
    ap.add_argument('--union-errors', default=str(EVAL_REPORTS / 'encoder_errors.json'))
    ap.add_argument('--all-montages', action='store_true',
                    help='монтажи по всем фото, не только флагированным')
    ap.add_argument('--dump', action='store_true', help='сохранить сырые кропы bottles/ и labels/')
    ap.add_argument('--query-policy', action='store_true',
                    help='аудит кропа ЗАПРОСА: политика query_crop_* (query_crop_min_conf и т.п.)')
    ap.add_argument('--cell', type=int, default=380, help='размер панели монтажа, px')
    # оверрайды политики кропа (env читается модулем crop при импорте, поэтому
    # выставка делается ниже — после apply_retrieval_env и до импорта crop)
    ap.add_argument('--crop-pick', choices=['big_center', 'big_center_border'])
    ap.add_argument('--crop-min-conf', type=float)
    ap.add_argument('--crop-min-area', type=float, dest='crop_min_area',
                    help='мин. площадь бокса бутылки, доля кадра (0 — не фильтровать)')
    ap.add_argument('--crop-fallback', choices=['orig', 'label'])
    ap.add_argument('--label-pick', choices=['conf', 'conf_size_center'])
    ap.add_argument('--label-min-w', type=float, help='отсечь боксы этикетки уже доли ширины')
    ap.add_argument('--label-min-conf', type=float)
    ap.add_argument('--border-penalty', type=float, help='штраф за срезанную рамкой сторону')
    ap.add_argument('--aspect-penalty', type=float, help='штраф за небутылочные пропорции бокса')
    ap.add_argument('--min-aspect', type=float, help='минимальное h/w «бутылочного» бокса')
    ap.add_argument('--center-w', type=float, help='вес удалённости бокса от центра кадра')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    apply_retrieval_env(cfg)                       # env для кропа (YOLO/этикетка)
    for k, v in (('CROP_PICK', args.crop_pick), ('CROP_MIN_CONF', args.crop_min_conf),
                 ('CROP_MIN_AREA', args.crop_min_area),
                 ('CROP_FALLBACK', args.crop_fallback), ('LABEL_PICK', args.label_pick),
                 ('LABEL_MIN_W_FRAC', args.label_min_w), ('LABEL_MIN_CONF', args.label_min_conf),
                 ('BORDER_PENALTY', args.border_penalty),
                 ('ASPECT_PENALTY', args.aspect_penalty),
                 ('BOTTLE_MIN_ASPECT', args.min_aspect), ('CROP_CENTER_W', args.center_w)):
        if v is not None:                          # CLI-оверрайд политики кропа
            os.environ[k] = str(v)
    # импорт crop — ПОСЛЕ env: модуль читает CROP_*/LABEL_* на импорте
    from crop import (get_cropper, get_label_cropper, LABEL_MARGIN, LABEL_MIN_CONF,
                      CROP_FALLBACK, CROP_MIN_CONF, CROP_MIN_AREA, CROP_PICK, LABEL_PICK,
                      LABEL_MIN_W_FRAC, BORDER_PENALTY, ASPECT_PENALTY, BOTTLE_MIN_ASPECT,
                      CENTER_W)
    if args.query_policy:                     # политика ЗАПРОСА (query_crop_*), как в сервисе
        from crop import use_query_policy, policy
        ov = use_query_policy()
        print('query-политика:', ', '.join(ov) if ov else 'нет оверрайдов (QUERY_* пусты)')
    else:
        from crop import policy
    pol = {k: policy(k) for k in ('CROP_PICK', 'CROP_MIN_CONF', 'CROP_MIN_AREA',
                                  'CROP_FALLBACK', 'LABEL_PICK', 'LABEL_MIN_W_FRAC')}
    bc, lc = get_cropper(), get_label_cropper()

    images_dir, csv_path = Path(args.images_dir), Path(args.eval_csv)
    rows = read_eval(csv_path, images_dir)
    slug_of = {n: s for n, s in rows}
    print(f'аудит кропа: {len(rows)} фото ({images_dir})')
    print(f"политика: CROP_PICK={pol['CROP_PICK']} CROP_MIN_CONF={pol['CROP_MIN_CONF']} "
          f"CROP_MIN_AREA={pol['CROP_MIN_AREA']} "
          f"CROP_FALLBACK={pol['CROP_FALLBACK']} LABEL_PICK={pol['LABEL_PICK']} "
          f"LABEL_MIN_W_FRAC={pol['LABEL_MIN_W_FRAC']} LABEL_MIN_CONF={LABEL_MIN_CONF}"
          f"{'  [query-политика]' if args.query_policy else '  [политика индекса]'}")
    print(f"          BORDER_PENALTY={BORDER_PENALTY} ASPECT_PENALTY={ASPECT_PENALTY} "
          f"BOTTLE_MIN_ASPECT={BOTTLE_MIN_ASPECT} CROP_CENTER_W={CENTER_W}")

    union = set()
    up = Path(args.union_errors)
    if up.exists():
        try:
            union = set(json.loads(up.read_text(encoding='utf-8'))['summary']['union_images'])
        except Exception as e:  # noqa: BLE001
            print(f'  union-ошибки не прочитаны ({e})')

    out = Path(args.out)
    for sub in ('montages', 'bottles', 'labels'):
        d = out / sub
        if d.exists():                                 # чистим, чтобы не читать stale-файлы
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    recs = []
    for name, _slug in rows:
        p = images_dir / name
        if not p.exists():
            print(f'  нет файла: {name}')
            continue
        rec, img, bottle, box_bottle, box_label, _lb = analyse_one(p, bc, lc)
        rec['true_slug'] = slug_of.get(name, '')
        rec['in_union_errors'] = name in union
        rec['rescue_label_fallback'] = bool(                                  # CROP_FALLBACK=label
            not rec['bottle_found'] and (rec['label_photo_conf'] or 0) >= LABEL_MIN_CONF)
        recs.append(rec)
        if rec['flags'] or args.all_montages:
            label_img = label_padded(bottle, box_label, LABEL_MARGIN) if box_label else bottle
            build_montage(name, [mark(img, box_bottle), mark(bottle, box_label), label_img],
                          rec['flags'], out / 'montages' / f'{Path(name).stem}.jpg',
                          cell=args.cell)
        if args.dump and rec['flags']:
            bottle.save(out / 'bottles' / f'{Path(name).stem}.jpg', quality=90)
            if box_label:
                label_padded(bottle, box_label, LABEL_MARGIN).save(
                    out / 'labels' / f'{Path(name).stem}.jpg', quality=90)

    with open(out / 'crop_audit.csv', 'w', newline='', encoding='utf-8') as f:
        cols = CSV_COLS + ['true_slug', 'in_union_errors']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            row = {k: r.get(k) for k in cols}
            row['flags'] = ','.join(r['flags'])
            row['in_union_errors'] = str(bool(r['in_union_errors'])).lower()
            w.writerow(row)

    s = summarize(recs, union)
    s.update({'images_dir': str(images_dir), 'eval_csv': str(csv_path), 'out': str(out),
              'union_errors_file': str(up) if up.exists() else None,
              'policy': {**pol, 'LABEL_MIN_CONF': LABEL_MIN_CONF,
                         'query_policy': bool(args.query_policy),
                         'index_policy': {'CROP_PICK': CROP_PICK, 'CROP_MIN_CONF': CROP_MIN_CONF,
                                          'CROP_MIN_AREA': CROP_MIN_AREA,
                                          'CROP_FALLBACK': CROP_FALLBACK,
                                          'LABEL_PICK': LABEL_PICK,
                                          'LABEL_MIN_W_FRAC': LABEL_MIN_W_FRAC}}})
    (out / 'summary.json').write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\nбутылка найдена: {s['bottle_found_pct']}% | этикетка найдена: {s['label_found_pct']}%")
    print(f"флаги: {s['n_flagged']}/{s['n']} фото ({s['pct_flagged']}%)")
    for fl, c in sorted(s['flag_counts'].items(), key=lambda kv: -kv[1]):
        pe = s['per_flag'][fl]
        er = f" | доля union-ошибок {pe['error_rate']}" if pe['error_rate'] is not None else ''
        print(f"  {fl:14s} {c:3d} ({s['flag_pct'][fl]:4.1f}%){er}")
    if s['joined_to_union']:
        print(f"union-ошибок: все {s['error_rate_all']} | "
              f"с флагом {s['error_rate_flagged']} | без флага {s['error_rate_clean']}")
    else:
        print('сшивка с union-ошибками не построена (файл отсутствует/не покрывает набор)')
    lo = s['lowconf_recovery']
    print(f"понижение порога (conf 0.05): no_bottle {lo['no_bottle_n']} -> "
          f">=0.10 {lo['no_bottle_ge_0.10']}, >=0.15 {lo['no_bottle_ge_0.15']} | "
          f"no_label {lo['no_label_n']} -> >=0.10 {lo['no_label_ge_0.10']}, "
          f">=0.15 {lo['no_label_ge_0.15']}")
    rf = s['rescue_label_fallback']
    print(f"CROP_FALLBACK=label спасёт {rf['n']} фото (union-ошибок среди них {rf['union_errors']})")
    print('выгружено ->', out, '(crop_audit.csv, summary.json, montages/)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

