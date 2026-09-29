# -*- coding: utf-8 -*-
"""Warm-up: сид каталога в Postgres + построение индекса в pgvector из filtered/.

Вызывается на старте сервиса (app.startup) и как CLI. Если векторы уже есть — пропускает
(если не --force). Размерность vector(D) берётся от текущей модели (retrieval.model).

  docker compose exec ml python build_index.py           # собрать, если пусто
  docker compose exec ml python build_index.py --force   # пересобрать
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
if __name__ == "__main__":           # CLI: env из config/pipeline.yaml ДО импорта encoder/crop (как в app.py)
    from pipeline_config import load_config, apply_retrieval_env
    apply_retrieval_env(load_config())
from paths import FILTERED, CATALOG_CSV
import db
import encoder as enc_mod
from encoder import get_encoder
from crop import maybe_crop, maybe_label_crop, USE_LABEL_BRANCH, LABEL_SUFFIX

IMG_EXT = {".webp", ".png", ".jpg", ".jpeg", ".jfif", ".heic", ".tif", ".tiff"}


def _pairs():
    out = []
    for d in sorted(FILTERED.iterdir()):
        if d.is_dir():
            for img in sorted(d.iterdir()):
                if img.suffix.lower() in IMG_EXT:
                    out.append((d.name, img))
    return out


def warmup(force: bool = False):
    enc = get_encoder()
    model = enc.model_name
    label_model = model + LABEL_SUFFIX          # вторая ветка: кроп этикетки
    dim = enc.embed([Image.new("RGB", (224, 224), "white")]).shape[1]
    conn = db.connect()
    # каталог
    cards = pd.read_csv(CATALOG_CSV, dtype=str, keep_default_na=False).to_dict("records")
    db.upsert_wines(conn, cards)
    if force:                        # только при явной пересборке: слаги, убранные из каталога
        db.prune_wines(conn, [c["slug"] for c in cards])
    need_a = force or not db.has_vectors(conn, model)
    need_b = USE_LABEL_BRANCH and (force or not db.has_vectors(conn, label_model))
    if not need_a and not need_b:
        print(f"[warmup] индексы '{model}' и '{label_model}' уже есть, пропускаю (dim={dim})")
        return
    pairs = _pairs()
    what = " + ".join(x for x, y in (("бутылка", need_a), ("этикетка", need_b)) if y)
    print(f"[warmup] эмбеддинг {len(pairs)} фото ({what}) моделью '{model}' (dim={dim})")
    vecs_a, vecs_b = [], []
    B = 32
    for i in range(0, len(pairs), B):
        chunk = pairs[i:i + B]
        crops_a = [maybe_crop(Image.open(p).convert("RGB"))[0] for _, p in chunk]
        # кроп этикетки поверх бутылочного; если этикетка не найдена — сам бутылочный кроп
        crops_b = [maybe_label_crop(c)[0] if need_b else None for c in crops_a]
        # энкодим одним батчем: [A-кропы..., B-кропы...]
        emb = enc.embed(crops_a + crops_b, batch_size=B)
        half = len(crops_a)
        if need_a:
            vecs_a.extend((chunk[j][0], emb[j]) for j in range(half))
        if need_b:
            vecs_b.extend((chunk[j][0], emb[half + j]) for j in range(half))
        done = min(i + B, len(pairs))
        if done % 256 == 0 or done == len(pairs):
            print(f"  {done}/{len(pairs)}")
    if need_a:
        db.replace_vectors(conn, model, vecs_a, dim, enc_mod.BACKEND)
        print(f"[warmup] ветка 'бутылка': {len(vecs_a)} векторов / {len(set(s for s, _ in vecs_a))} вин (model='{model}')")
    if need_b:
        db.replace_vectors(conn, label_model, vecs_b, dim, enc_mod.BACKEND)
        print(f"[warmup] ветка 'этикетка': {len(vecs_b)} векторов / {len(set(s for s, _ in vecs_b))} вин (model='{label_model}')")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    warmup(ap.parse_args().force)
