# -*- coding: utf-8 -*-
"""Warm-up: сид каталога в Postgres + построение индекса в pgvector из filtered/.

Вызывается на старте сервиса (app.startup) и как CLI. Если векторы уже есть — пропускает
(если не --force). Размерность vector(D) берётся от текущей модели (env SEARCH_MODEL).

  uv run python build_index.py            # собрать, если пусто
  uv run python build_index.py --force    # пересобрать
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from paths import FILTERED, CATALOG_CSV
import db
import encoder as enc_mod
from encoder import get_encoder
from crop import maybe_crop

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
    dim = enc.embed([Image.new("RGB", (224, 224), "white")]).shape[1]
    conn = db.connect()
    # каталог
    cards = pd.read_csv(CATALOG_CSV, dtype=str, keep_default_na=False).to_dict("records")
    db.upsert_wines(conn, cards)
    if db.has_vectors(conn, model) and not force:
        print(f"[warmup] индекс модели '{model}' уже есть, пропускаю (dim={dim})")
        return
    pairs = _pairs()
    print(f"[warmup] эмбеддинг {len(pairs)} фото моделью '{model}' (dim={dim})")
    vecs = []
    B = 32
    for i in range(0, len(pairs), B):
        imgs = [maybe_crop(Image.open(p).convert("RGB"))[0] for _, p in pairs[i:i + B]]
        embs = enc.embed(imgs)
        vecs.extend((pairs[i + j][0], embs[j]) for j in range(len(imgs)))
        print(f"  {min(i + B, len(pairs))}/{len(pairs)}")
    db.replace_vectors(conn, model, vecs, dim, enc_mod.BACKEND)
    print(f"[warmup] залито {len(vecs)} векторов / {len(set(s for s, _ in vecs))} вин (model='{model}')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    warmup(ap.parse_args().force)
