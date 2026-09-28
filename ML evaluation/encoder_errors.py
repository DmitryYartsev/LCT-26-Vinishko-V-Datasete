# -*- coding: utf-8 -*-
"""Ошибки энкодеров: пересечение + выгрузка union-ошибок (images/crops) + CSV.

Использует КЭШ эмбеддингов A/B (``ML evaluation/reports/encab_*.npz``), поэтому
не делает ни одного сетевого вызова. Для каждого энкодера восстанавливает
per-image предсказание (argmax косинуса), считает пересечение ошибок и копирует
все изображения, где ошибся ХОТЯ БЫ ОДИН энкодер (union).

Структура вывода (по умолчанию ``Reports/encoder_errors_union/``):
    images/  <name>_query.<ext>  <name>_gt.<ext>  <name>_pred_<enc>.<ext>   (полные фото)
    crops/   то же самое, но после YOLO-кропа бутылки + кропа этикетки (как в пайплайне)
    errors.csv   image, <enc1>, <enc2>, ...   (true/false = верно ли предсказал)

    python3 encoder_errors.py --out ../Reports/encoder_errors_union
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from encoder_ab import EVAL_REPORTS, catalog_images, prep   # noqa: E402
from paths import DATA                                      # noqa: E402

EVAL_STEM = 'real_data_with_slug_eval'


def enc_label(tag: str) -> str:
    t = tag.lower()
    if 'gemini' in t:
        return 'gemini'
    if 'voyage' in t:
        return 'voyage'
    if 'siglip' in t:
        return 'siglip2'
    return tag


def discover_encoders():
    """[(label, cat_npz, q_npz)] из кэшей A/B для eval-набора 62."""
    out = []
    for q in sorted(EVAL_REPORTS.glob(f'encab_q_*_{EVAL_STEM}.npz')):
        tag = q.name[len('encab_q_'):-len(f'_{EVAL_STEM}.npz')]
        cat = EVAL_REPORTS / f'encab_cat_{tag}_filtered.npz'
        if cat.exists():
            out.append((enc_label(tag), cat, q))
    return out


def predict(ncat: Path, nq: Path) -> dict:
    """-> {image: {'true':slug, 'pred':slug, 'correct':bool, 'score':float}}."""
    cz = np.load(ncat, allow_pickle=True)
    qz = np.load(nq, allow_pickle=True)
    slugs = list(cz['slugs'])
    sims = cz['emb'] @ qz['emb'].T                      # [C, Q]
    names, trues = list(qz['names']), list(qz['trues'])
    best = np.argmax(sims, axis=0)
    res = {}
    for j, nm in enumerate(names):
        pred = slugs[int(best[j])]
        res[nm] = {'true': trues[j], 'pred': pred,
                   'correct': bool(pred == trues[j]),
                   'score': round(float(sims[best[j], j]), 4)}
    return res


def analyze(preds: dict, labels: list) -> dict:
    """Пересечение ошибок: сколько картинок ломает каждый набор энкодеров."""
    names = sorted(set().union(*[set(preds[e]) for e in labels]))
    correct_any = {nm: all(preds[e][nm]['correct'] for e in labels if nm in preds[e]) for nm in names}
    union = [nm for nm in names if not correct_any[nm]]
    combos = Counter()
    for nm in union:
        wrong = frozenset(e for e in labels if nm in preds[e] and not preds[e][nm]['correct'])
        combos[wrong] += 1
    pairwise = {}
    for a, b in combinations(labels, 2):
        both = sum(1 for nm in union if (not preds[a][nm]['correct']) and (not preds[b][nm]['correct']))
        only_a = sum(1 for nm in union if (not preds[a][nm]['correct']) and preds[b][nm]['correct'])
        only_b = sum(1 for nm in union if not (not preds[a][nm]['correct']) and (not preds[b][nm]['correct']))
        pairwise[f'{a} vs {b}'] = {'both_wrong': both, 'only_' + a: only_a, 'only_' + b: only_b}
    acc = {e: round(sum(1 for nm in names if preds[e][nm]['correct']) / max(len(names), 1), 4) for e in labels}
    return {'n': len(names), 'per_encoder_correct@1': acc,
            'union_errors': len(union), 'union_images': union,
            'wrong_sets': {'+'.join(sorted(k)) if k else 'none': v for k, v in combos.most_common()},
            'pairwise': pairwise}


def _ext(p: Path) -> str:
    return '.jpg' if p.suffix.lower() in ('.jpeg', '.jfif') else p.suffix.lower()


def dump_union(union, preds, labels, cat_map, out_dir, images_dir, src_exts):
    from PIL import Image
    for sub in ('images', 'crops'):
        d = out_dir / sub
        if d.exists():
            shutil.rmtree(d)                    # чистим, чтобы не смешивать старое/новое
        d.mkdir(parents=True, exist_ok=True)
    missing = 0
    for nm in union:
        base = Path(nm).stem
        rec0 = preds[labels[0]][nm]
        true_slug = rec0['true']
        jobs = []                                  # (out_name_stem, source_path)
        qsrc = Path(images_dir) / nm
        if qsrc.exists():
            jobs.append((f'{base}_query', qsrc))
        gt = cat_map.get(true_slug)
        if gt:
            jobs.append((f'{base}_gt_{true_slug}', gt))     # к gt приписан его true_slug
        else:
            missing += 1
        for e in labels:
            rec = preds[e][nm]
            flag = 'true' if rec['correct'] else 'false'
            p = cat_map.get(rec['pred'])
            if p:                                            # предсказанный слаг + корректность
                jobs.append((f'{base}_pred_{e}_{rec["pred"]}_{flag}', p))
            else:
                missing += 1
        for stem, src in jobs:
            ext = src_exts.get(src, _ext(src))
            shutil.copyfile(src, out_dir / 'images' / f'{stem}{ext}')     # images: как есть
            try:
                cropped = prep(Image.open(src).convert('RGB'))
                cropped.save(out_dir / 'crops' / f'{stem}.jpg', quality=92)  # crops: YOLO+этикетка
            except Exception as ex:  # noqa: BLE001
                print(f'  crop fail {src.name}: {ex}')
    return missing


def main(argv=None) -> int:
    from pipeline_config import load_config, apply_retrieval_env
    import pipeline as P
    ap = argparse.ArgumentParser(description='Ошибки энкодеров: пересечение + выгрузка')
    ap.add_argument('--config', default=str(REPO / 'config' / 'pipeline.yaml'))
    ap.add_argument('--images-dir', default=str(DATA / 'real_photo'))
    ap.add_argument('--out', default=str(REPO / 'Reports' / 'encoder_errors_union'))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    P.apply_host_paths(cfg)
    apply_retrieval_env(cfg)                       # env для crop (YOLO/этикетка)

    encs = discover_encoders()
    labels = [e[0] for e in encs]
    print('энкодеры из кэшей:', labels)
    preds = {lab: predict(cat, q) for lab, cat, q in encs}

    rep = analyze(preds, labels)
    print(json.dumps({k: v for k, v in rep.items() if k != 'union_images'}, ensure_ascii=False, indent=2))
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    (EVAL_REPORTS / 'encoder_errors.json').write_text(
        json.dumps({'summary': rep, 'per_image': {
            nm: {e: preds[e][nm] for e in labels} for nm in rep['union_images']}},
            ensure_ascii=False, indent=2), encoding='utf-8')

    # имена/расширения изображений из исходной папки (для сохранения формата images/)
    src_exts = {}
    cat_items = catalog_images('filtered')
    cat_map = {s: p for s, p in cat_items}
    for nm in rep['union_images']:
        q = Path(args.images_dir) / nm
        if q.exists():
            src_exts[q] = _ext(q)
    for p in cat_map.values():
        src_exts[p] = _ext(p)

    out_dir = Path(args.out)
    miss = dump_union(rep['union_images'], preds, labels, cat_map, out_dir, args.images_dir, src_exts)

    with open(out_dir / 'errors.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['image', 'true_slug']
        for e in labels:
            header += [f'{e}_predicted_slug', e]
        w.writerow(header)
        for nm in rep['union_images']:
            row = [nm, preds[labels[0]][nm]['true']]
            for e in labels:
                row += [preds[e][nm]['pred'], str(preds[e][nm]['correct']).lower()]
            w.writerow(row)

    print(f'union-ошибок: {len(rep["union_images"])} | файлов без каталожного фото: {miss}')
    print('выгружено ->', out_dir, '(images/, crops/, errors.csv)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

