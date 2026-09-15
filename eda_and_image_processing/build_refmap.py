# -*- coding: utf-8 -*-
"""Строит карту slug -> файл эталонного фото из дампа Strapi. Считает покрытие."""
import re, sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("D:/_hack/lct_26")
OUT = ROOT / "solution/eda"

_raw = (OUT / "archive_filelist.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
files = []
for _l in _raw:
    _l = _l.strip().replace("\\", "/")
    if "/uploads/" in _l:
        files.append(_l.split("/uploads/")[-1])

def strip_hash(stem):
    return re.sub(r"_[0-9a-f]{6,}$", "", stem)

def fbase(f):
    return strip_hash(re.sub(r"\.\w+$", "", f).lower())

# индекс базовых имён -> файл (первый)
fb = {}
for f in files:
    fb.setdefault(fbase(f), f)
fb_items = list(fb.items())

def norm_photo(n):
    s = re.sub(r"\.\w+$", "", str(n).strip()).lower()
    s = re.sub(r"[^a-z0-9а-яё]+", "_", s)
    return strip_hash(s.strip("_"))

def su(slug):
    return re.sub(r"-", "_", slug.strip().lower())

df = pd.read_csv(ROOT / "strapi_output0709.csv", dtype=str, keep_default_na=False)
df.columns = [c.strip() for c in df.columns]
u = df.drop_duplicates(subset=["Slug"]).reset_index(drop=True)

rows = []
methods = {"photo_exact": 0, "slug_exact": 0, "prefix": 0, "none": 0}
for _, r in u.iterrows():
    slug = r["Slug"]
    kp = norm_photo(r["Название фото"])
    ks = su(slug)
    hit = None; how = None
    if kp and kp in fb:
        hit, how = fb[kp], "photo_exact"
    elif ks in fb:
        hit, how = fb[ks], "slug_exact"
    else:
        for k in (ks, kp):
            if len(k) >= 20:
                c = [f for b, f in fb_items if b.startswith(k[:40])]
                if c:
                    hit, how = c[0], "prefix"; break
    if hit:
        methods[how] += 1
        rows.append({"slug": slug, "file": hit, "method": how,
                     "name": r["Название вина"], "winery": r["Винодельня"],
                     "category": r["Категория"], "color": r["Цвет"], "region": r["Регион"]})
    else:
        methods["none"] += 1

m = pd.DataFrame(rows)
m.to_csv(OUT / "reference_map.csv", index=False, encoding="utf-8")
covered = len(m)
print("Методы сопоставления:", methods)
print(f"ИТОГО эталонных фото найдено: {covered} / {len(u)}  ({100*covered/len(u):.1f}%)")
print(f"Уникальных файлов использовано: {m['file'].nunique()}")
print(f"Позиций БЕЗ эталонного фото: {len(u)-covered}")
print("Сохранено:", OUT / "reference_map.csv")
