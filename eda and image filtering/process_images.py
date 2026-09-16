# -*- coding: utf-8 -*-
"""Фильтрация мусора: папка-дамп -> filtered/<slug>/*.webp + catalog.csv.

Готовит эталоны для индекса — сам npz-индекс строит `../ML service/build_index.py`.
Пайплайн (НЕ EDA — EDA лежит в eda.ipynb):
 1. снять size-варианты Strapi (thumbnail_/small_/medium_/large_ + оригинал) -> крупнейший;
 2. контент-дедуп по sha1;
 3. сопоставить изображения с вином (translit «Название фото» / slug), собрать ВСЕ фото на slug;
 4. разложить в filtered/<slug>/NN.<ext>, записать filtered/catalog.csv (карточки + n_images).

Запуск:
  uv run python process_images.py                  # -> filtered/ + catalog.csv
  uv run python process_images.py --uploads PATH    # свой каталог-дамп
"""
import re, sys, hashlib, shutil, argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import UPLOADS, FILTERED, CATALOG_CSV, CATALOG_SRC  # noqa

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# --- транслитерация Strapi/GOST-B (проверено на дампе) ---
TRANSLIT = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh','з':'z','и':'i',
    'й':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u',
    'ф':'f','х':'h','ц':'cz','ч':'ch','ш':'sh','щ':'shh','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'}
def translit(s): return ''.join(TRANSLIT.get(c, c) for c in str(s).lower())

SIZE_PREFIX = re.compile(r"^(thumbnail|small|medium|large)_", re.I)
IMG_EXT = {".webp", ".png", ".jpg", ".jpeg", ".jfif", ".heic", ".tif", ".tiff"}
SIZE_RANK = {"": 5, "large": 4, "medium": 3, "small": 2, "thumbnail": 1}
CARD_COLS = {"Название вина": "name", "Винодельня": "winery", "Категория": "category",
             "Цвет": "color", "Регион": "region", "Сорт винограда": "grape", "Описание": "description"}

def _parse(fname):
    stem = re.sub(r"\.[^.]+$", "", fname)
    m = SIZE_PREFIX.match(stem)
    return (m.group(1).lower() if m else ""), SIZE_PREFIX.sub("", stem)
def _strip_hash(core): return re.sub(r"_[0-9a-f]{6,}$", "", core.lower())
def _norm(s):
    s = translit(str(s).strip().lower()); s = re.sub(r"\.\w+$", "", s)
    return _strip_hash(re.sub(r"[^a-z0-9]+", "_", s).strip("_"))


def dedup_and_match(uploads: Path):
    """-> (slug -> [пути к фото]), карточки-DataFrame."""
    files = [p for p in uploads.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
    # 1. size-дедуп: лучший вариант на core
    groups = defaultdict(list)
    for p in files:
        v, core = _parse(p.name); groups[core].append((SIZE_RANK.get(v, 0), p))
    best = {c: max(v, key=lambda x: x[0])[1] for c, v in groups.items()}
    # 2. контент-дедуп
    by_hash = {}
    for core, path in best.items():
        h = hashlib.sha1(path.read_bytes()).hexdigest()
        by_hash.setdefault(h, (core, path))
    uniq = {core: path for core, path in by_hash.values()}
    print(f"файлов {len(files)} -> size-дедуп {len(best)} -> контент-дедуп {len(uniq)}")
    # 3. индекс base(без хеша) -> [core]
    base_index = defaultdict(list)
    for core in uniq: base_index[_strip_hash(core)].append(core)
    base_keys = list(base_index.keys())
    # каталог
    df = pd.read_csv(CATALOG_SRC, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    u = df.drop_duplicates(subset=["Slug"]).reset_index(drop=True)
    # 4. матч + сбор ВСЕХ фото на slug
    slug_photos, cards, methods = {}, [], defaultdict(int)
    for _, r in u.iterrows():
        slug = r["Slug"]; kp = _norm(r["Название фото"]); ks = re.sub("-", "_", slug.lower())
        base = how = None
        if kp and kp in base_index: base, how = kp, "photo_exact"
        elif ks in base_index: base, how = ks, "slug_exact"
        else:
            for k in (ks, kp):
                if len(k) >= 18:
                    cand = [b for b in base_keys if b.startswith(k) or k.startswith(b)]
                    if len(cand) == 1: base, how = cand[0], "prefix"; break
        if not base: methods["none"] += 1; continue
        methods[how] += 1
        photos = [uniq[c] for c in base_index[base]]
        slug_photos[slug] = photos
        cards.append({"slug": slug, "n_images": len(photos),
                      **{v: r.get(k, "") for k, v in CARD_COLS.items()}})
    print("методы:", dict(methods), f"| вин с фото: {len(slug_photos)}")
    return slug_photos, pd.DataFrame(cards)


def write_filtered(slug_photos, cards):
    if FILTERED.exists(): shutil.rmtree(FILTERED)
    FILTERED.mkdir(parents=True)
    rows = []
    for slug, photos in slug_photos.items():
        d = FILTERED / slug; d.mkdir()
        names = []
        for i, src in enumerate(photos):
            dst = d / f"{i:02d}{src.suffix.lower()}"
            shutil.copy2(src, dst); names.append(dst.name)
        rows.append((slug, ";".join(names)))
    img_map = dict(rows)
    cards["images"] = cards["slug"].map(img_map)
    cards.to_csv(CATALOG_CSV, index=False, encoding="utf-8")
    total = sum(len(v) for v in slug_photos.values())
    print(f"filtered/: {len(slug_photos)} вин, {total} фото -> {FILTERED}")
    print(f"catalog.csv -> {CATALOG_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", default=str(UPLOADS))
    args = ap.parse_args()

    slug_photos, cards = dedup_and_match(Path(args.uploads))
    write_filtered(slug_photos, cards)


if __name__ == "__main__":
    main()
