# -*- coding: utf-8 -*-
"""Дедуп медиатеки Strapi + надёжное сопоставление slug -> эталонное фото.

Strapi на каждую загрузку хранит size-варианты (thumbnail_/small_/medium_/large_ + оригинал)
с одинаковым хешем. Имя файла = translit(оригинальное имя, часто кириллица) + _<hash>.
Слаги в CSV транслитерированы той же схемой (ц->cz, й->j, ч->ch ...).

Пайплайн:
 1) снять size-варианты (оставить крупнейший), убрать дубли по контенту (sha1);
 2) сопоставить каждое вино с файлом по приоритету методов (translit/photo/slug exact),
    prefix — только если однозначно и без коллизий;
 3) разложить эталоны в reference/<slug>.<ext>.
"""
import re, sys, hashlib, shutil, json
from pathlib import Path
from collections import defaultdict
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("D:/_hack/lct_26")
SRC = ROOT / "solution/images/_uploads_raw"
REF = ROOT / "solution/images/reference"
OUT = ROOT / "solution/eda"

# --- транслитерация как в Strapi/GOST-B (проверено на дампе) ---
TRANSLIT = {
    'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh','з':'z','и':'i',
    'й':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t',
    'у':'u','ф':'f','х':'h','ц':'cz','ч':'ch','ш':'sh','щ':'shh','ъ':'','ы':'y','ь':'',
    'э':'e','ю':'yu','я':'ya',
}
def translit(s):
    return ''.join(TRANSLIT.get(ch, ch) for ch in s.lower())

SIZE_PREFIX = re.compile(r"^(thumbnail|small|medium|large)_", re.I)
IMG_EXT = {".webp", ".png", ".jpg", ".jpeg", ".jfif", ".heic", ".tif", ".tiff"}
SIZE_RANK = {"": 5, "large": 4, "medium": 3, "small": 2, "thumbnail": 1}

def parse(fname):
    stem = re.sub(r"\.[^.]+$", "", fname)
    m = SIZE_PREFIX.match(stem)
    return (m.group(1).lower() if m else ""), SIZE_PREFIX.sub("", stem)

def strip_hash(core):
    return re.sub(r"_[0-9a-f]{6,}$", "", core.lower())

def norm(s):
    """латиница/цифры -> _, кириллицу транслитерируем, снять хвостовой хеш."""
    s = translit(str(s).strip().lower())
    s = re.sub(r"\.\w+$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return strip_hash(s.strip("_"))

# ---------- 1. size-дедуп + контент-дедуп ----------
all_files = [p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
groups = defaultdict(list)
for p in all_files:
    variant, core = parse(p.name)
    groups[core].append((SIZE_RANK.get(variant, 0), p))
best_per_core = {c: max(v, key=lambda x: x[0])[1] for c, v in groups.items()}

by_hash, dup_content = {}, 0
for core, path in best_per_core.items():
    h = hashlib.sha1(path.read_bytes()).hexdigest()
    if h in by_hash: dup_content += 1
    else: by_hash[h] = (core, path)
uniq = {core: path for core, path in by_hash.values()}   # core -> path
print(f"Файлов-картинок: {len(all_files)} | после size-дедупа: {len(best_per_core)} | "
      f"контент-дублей убрано: {dup_content} | уникальных: {len(uniq)}")

# индекс base(без хеша) -> [core]
base_index = defaultdict(list)
for core in uniq:
    base_index[strip_hash(core)].append(core)
base_keys = list(base_index.keys())

# ---------- 2. сопоставление ----------
df = pd.read_csv(ROOT / "strapi_output0709.csv", dtype=str, keep_default_na=False)
df.columns = [c.strip() for c in df.columns]
u = df.drop_duplicates(subset=["Slug"]).reset_index(drop=True)

rows, methods = [], defaultdict(int)
for _, r in u.iterrows():
    slug = r["Slug"]
    kp = norm(r["Название фото"])          # translit имени фото
    ks = re.sub(r"-", "_", slug.strip().lower())
    core = how = None
    if kp and kp in base_index:
        core, how = base_index[kp][0], "photo_exact"
    elif ks in base_index:
        core, how = base_index[ks][0], "slug_exact"
    else:
        # prefix: файл-база — префикс slug ИЛИ наоборот, но только однозначно
        cand = set()
        for k in (kp, ks):
            if len(k) >= 18:
                for bk in base_keys:
                    if len(bk) >= 18 and (bk.startswith(k) or k.startswith(bk)):
                        cand.add(bk)
        if len(cand) == 1:
            core, how = base_index[next(iter(cand))][0], "prefix"
    if core:
        methods[how] += 1
        rows.append({"slug": slug, "core": core, "src": uniq[core].name, "method": how,
                     "name": r["Название вина"], "winery": r["Винодельня"],
                     "category": r["Категория"], "color": r["Цвет"], "region": r["Регион"]})
    else:
        methods["none"] += 1

m = pd.DataFrame(rows)

# ---------- 3. снять оставшиеся коллизии (один core на >1 slug) для prefix ----------
core_counts = m.groupby("core")["slug"].transform("nunique")
bad = (m["method"] == "prefix") & (core_counts > 1)
dropped = int(bad.sum())
m_clean = m[~bad].reset_index(drop=True)
print(f"Убрано коллизийных prefix-привязок: {dropped}")

# оставшиеся коллизии по exact — это реально общие фото у near-dup вин (данные такие)
rest_coll = m_clean.groupby("core")["slug"].nunique()
rest_coll = int((rest_coll > 1).sum())

print("Методы:", dict(methods))
print(f"ИТОГО вин с надёжным эталоном: {len(m_clean)} / {len(u)}  ({100*len(m_clean)/len(u):.1f}%)")
print(f"(exact-коллизий — общее фото у near-dup — осталось групп: {rest_coll})")

# ---------- 4. чистая раскладка reference/<slug>.<ext> ----------
if REF.exists(): shutil.rmtree(REF)
REF.mkdir(parents=True, exist_ok=True)
for _, r in m_clean.iterrows():
    src = uniq[r["core"]]
    shutil.copy2(src, REF / f"{r['slug']}{src.suffix.lower()}")
m_clean.to_csv(OUT / "reference_map_full.csv", index=False, encoding="utf-8")

miss = u[~u["Slug"].isin(set(m_clean["slug"]))]
miss[["Slug", "Название вина", "Название фото"]].to_csv(OUT / "unmatched_wines.csv", index=False, encoding="utf-8")
print(f"Эталонов разложено в reference/: {len(m_clean)} | без фото: {len(miss)}")

summary = {
    "raw_files": len(all_files), "after_size_dedup": len(best_per_core),
    "content_dups_removed": dup_content, "unique_images": len(uniq),
    "wines_total": len(u), "wines_with_photo": len(m_clean),
    "dropped_collision_prefix": dropped, "methods": dict(methods),
}
(OUT / "dedupe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("SUMMARY:", json.dumps(summary, ensure_ascii=False))
