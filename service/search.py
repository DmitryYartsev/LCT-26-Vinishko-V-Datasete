# -*- coding: utf-8 -*-
"""Индекс каталога в памяти: загрузка векторов + карточек, косинусный поиск.

Спроектировано под будущий МУЛЬТИ-ВЕКТОР: если на один slug несколько векторов,
матч = max cosine по векторам этого вина (агрегация в search)."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CATALOG_CSV, index_file, meta_file
from crop import CROP_ENABLED

# карточки хранятся в filtered/catalog.csv (выход process_images.py)
CARD_FIELDS = ["slug", "name", "winery", "category", "color", "region", "grape", "description", "images"]


class CatalogIndex:
    def __init__(self, idx_path: Path = None, cards_path: Path = CATALOG_CSV):
        idx_path = idx_path or index_file(CROP_ENABLED)
        data = np.load(idx_path, allow_pickle=True)
        self.emb = data["emb"].astype(np.float32)          # [N, D], L2-норм
        self.slugs = [str(s) for s in data["slugs"]]       # [N], параллельно emb (мультивектор)
        mp = meta_file("crop" in idx_path.name)
        self.meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        # карточки
        self.cards = {}
        if Path(cards_path).exists():
            df = pd.read_csv(cards_path, dtype=str, keep_default_na=False)
            for _, r in df.iterrows():
                self.cards[r["slug"]] = {c: r.get(c, "") for c in CARD_FIELDS if c in df.columns}
        print(f"[index] {self.emb.shape[0]} векторов / {len(set(self.slugs))} вин, "
              f"{len(self.cards)} карточек, модель {self.meta.get('model')}")

    def card(self, slug: str) -> dict:
        return self.cards.get(slug, {"slug": slug})

    def search(self, qvec: np.ndarray, k: int = 5) -> list[dict]:
        """qvec: [D] L2-норм. Возвращает top-k по винам (агрегация max по slug)."""
        scores = self.emb @ qvec                            # косинус (векторы нормированы)
        # агрегируем по slug (на будущее — мультивектор): лучший вектор на вино
        best = {}
        for i, s in enumerate(self.slugs):
            v = float(scores[i])
            if s not in best or v > best[s]:
                best[s] = v
        ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
        return [{"slug": s, "score": round(v, 4), **({"card": self.card(s)})}
                for s, v in ranked]
