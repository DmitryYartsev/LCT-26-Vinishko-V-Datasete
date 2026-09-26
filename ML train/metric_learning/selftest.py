# -*- coding: utf-8 -*-
"""Самотест ключевой логики без реальных данных (быстро, без обучения).

Проверяет то, что легко сломать незаметно:
  * ``metrics.ExactSlugAccuracy`` — точные совпадения slug, accuracy@1/@k, coverage;
  * triplet-лоссы — 0 на «идеальных» эмбеддингах, > 0 на плохих, edge-cases без пар;
  * ``cross_modal_triplet_loss`` — кропы против референсной базы;
  * ``PKBatchSampler`` — в каждом батче P классов × K картинок (позитивы гарантированы);
  * ``read_index_csv`` — автоопределение колонок и относительных путей;
  * игрушечный бэкбон + ``MetricModel.forward`` — размерность и L2-норма.

Запуск::

    python selftest.py

Полный интеграционный прогон (обучение/тест/чекпоинт) — в README:
``make_dummy_data.py`` + ``train.py --backbone tiny --clearml offline``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))                       # repo root (paths.py)

import losses                                                    # noqa: E402
from dataset import PKBatchSampler, read_index_csv               # noqa: E402
from metrics import ExactSlugAccuracy                            # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, extra: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' | ' + extra) if extra else ''}")


# ------------------------------------------------------------------ метрика
def test_metric() -> None:
    print("\n== метрика (exact slug accuracy) ==")
    metric = ExactSlugAccuracy(k=5)
    gal_slugs = ["w0", "w1", "w2", "w3"]
    gal = torch.eye(4)                       # 4 ортогональных эталона
    q = gal + 0.01 * torch.randn(4, 4)       # запросы = сами эталоны + шум
    res = metric(q, list(gal_slugs), gal, gal_slugs)
    check("accuracy@1 == 1.0 на идеальных эмбеддингах", res["acc@1"] == 1.0, str(res["acc@1"]))
    check("coverage == 1.0", res["coverage"] == 1.0)
    check("n_evaluated == n_queries == 4", res["n_evaluated"] == res["n_queries"] == 4)

    gal_perm = gal[[1, 2, 3, 0]]             # эталоны приписаны не своим slug'ам
    res_bad = metric(q, list(gal_slugs), gal_perm, gal_slugs)
    check("accuracy@1 == 0.0 при неверной галерее", res_bad["acc@1"] == 0.0, str(res_bad["acc@1"]))

    res_cov = metric(torch.cat([gal, gal[:1]]), gal_slugs + ["unknown"], gal, gal_slugs)
    check("coverage учитывает непокрытые slug'и", res_cov["coverage"] == 0.8,
          str(res_cov["coverage"]))
    check("n_evaluated == только покрытые", res_cov["n_evaluated"] == 4)

    res_all = ExactSlugAccuracy(k=5, only_covered=False)(
        torch.cat([gal, gal[:1]]), gal_slugs + ["unknown"], gal, gal_slugs)
    check("only_covered=False даёт acc@1 = 4/5", abs(res_all["acc@1"] - 0.8) < 1e-9,
          str(res_all["acc@1"]))
    check("предсказания сохранены (metric.last_predictions)",
          len(metric.last_predictions) == 5
          and metric.last_predictions[0]["true_slug"] == "w0"
          and metric.last_predictions[0]["pred_slug"] == "w0",
          str(metric.last_predictions[:1]))


# ------------------------------------------------------------------- лоссы
def test_triplet() -> None:
    print("\n== triplet loss ==")
    # хорошие: внутри класса почти одинаково, между классами — ортогонально
    good = torch.tensor([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0], [0.05, 0.99]])
    labels = torch.tensor([0, 0, 1, 1])
    loss_good, st_good = losses.batch_hard_triplet_loss(good, labels, margin=0.3)
    check("batch-hard: 0 на разделённых классах", float(loss_good) == 0.0,
          f"loss={float(loss_good):.4f}")
    check("frac_valid == 1", st_good["frac_valid"] == 1.0)

    # вырожденный случай: все точки совпадают -> d_pos == d_neg == 0 -> loss == margin
    bad = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    loss_bad, _ = losses.batch_hard_triplet_loss(bad, labels, margin=0.3)
    check("batch-hard: loss == margin при d_pos == d_neg",
          abs(float(loss_bad) - 0.3) < 1e-6, f"loss={float(loss_bad):.4f}")

    # без пар в батче -> 0 (и backward не падает)
    loss_none, st_none = losses.batch_hard_triplet_loss(good, torch.tensor([0, 1, 2, 3]))
    check("нет позитивов -> loss 0 и frac_valid 0",
          float(loss_none) == 0.0 and st_none["frac_valid"] == 0.0)
    loss_one, _ = losses.triplet_loss(good[:1], labels[:1])
    check("батч из 1 объекта не роняет лосс", float(loss_one) == 0.0)

    # все майнинги и дистанции считаются
    for mining in losses.MININGS:
        for dist in losses.DISTANCES:
            l, s = losses.triplet_loss(good, labels, margin=0.3, distance=dist, mining=mining)
            check(f"triplet({dist}, {mining}) считается", bool(torch.isfinite(l)) and "loss" in s)

    # cross-modal: кропы против референсной базы
    ref = torch.tensor([[1.0, 0.0], [0.0, 1.0]])              # эталоны классов 0 и 1
    crops = torch.tensor([[0.99, 0.05], [0.04, 0.99]])        # кропы классов 0 и 1
    l_ref, s_ref = losses.cross_modal_triplet_loss(crops, [0, 1], ref, [0, 1], margin=0.3)
    check("cross-modal: 0 при точном попадании в эталон", float(l_ref) == 0.0,
          f"loss={float(l_ref):.4f}, n_valid={s_ref['n_valid']}")
    l_bad, _ = losses.cross_modal_triplet_loss(crops, [1, 0], ref, [0, 1], margin=0.3)
    check("cross-modal: > 0 при перепутанных метках", float(l_bad) > 0.0,
          f"loss={float(l_bad):.4f}")
    l_unknown, _ = losses.cross_modal_triplet_loss(torch.tensor([[1.0, 0.0]]), [-1],
                                                   ref, [0, 1])
    check("cross-modal: неизвестный класс пропускается", float(l_unknown) == 0.0)


# ------------------------------------------------------------------ семплер
def test_sampler() -> None:
    print("\n== PK-семплер ==")
    labels = [0] * 5 + [1] * 3 + [2] * 1 + [3] * 7
    sp = PKBatchSampler(labels, p=3, k=4, seed=0)
    batches = list(iter(sp))
    check("len(sampler) == num_batches == 2", len(sp) == len(batches) == 2, str(len(sp)))
    check("в батче P*k объектов", all(len(b) == 12 for b in batches))
    ok = True
    for b in batches:
        counts: dict[int, int] = {}
        for i in b:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        ok &= all(c >= 2 for c in counts.values()) and len(counts) == 3
    check("в батче ≥2 объекта каждого класса (позитивы всегда есть)", ok)


# ------------------------------------------------------------- чтение CSV
def test_csv() -> None:
    print("\n== чтение CSV ==")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "images").mkdir()
        (tmp / "images" / "a.png").write_bytes(b"")
        (tmp / "reference.csv").write_text("photo;Slug\nimages/a.png;wine-a\n", encoding="utf-8")
        recs = read_index_csv(tmp / "reference.csv")
        check("авто-разделитель ';' и колонки photo/Slug",
              len(recs) == 1 and recs[0]["slug"] == "wine-a", str(recs[:1]))
        check("относительный путь разрешён от папки CSV",
              recs[0]["image_path"] == tmp / "images" / "a.png", str(recs[0]["image_path"]))

        (tmp / "broken.csv").write_text("image_path,slug\nmissing.png,wine-x\n", encoding="utf-8")
        check("несуществующий файл пропускается", read_index_csv(tmp / "broken.csv") == [])
        (tmp / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            read_index_csv(tmp / "bad.csv")
            check("ошибка на CSV без колонок path/slug", False)
        except ValueError:
            check("ошибка на CSV без колонок path/slug", True)


# ------------------------------------------------------------------- модель
def test_model() -> None:
    print("\n== модель (tiny-бэкбон) ==")
    import config
    import modeling
    spec, _ = config.make_backbone(type("A", (), {"backbone": "tiny"})())
    model = modeling.MetricModel(spec, n_classes=3, embed_dim=16, head="linear")
    z = model(torch.randn(4, 3, spec.image_size, spec.image_size))
    check("форма эмбеддингов [4, 16]", tuple(z.shape) == (4, 16), str(tuple(z.shape)))
    check("L2-норма == 1", torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5),
          str([round(v, 5) for v in z.norm(dim=-1).tolist()]))
    z_nn = model(torch.randn(2, 3, spec.image_size, spec.image_size), normalize=False)
    check("normalize=False даёт ненормированные",
          not torch.allclose(z_nn.norm(dim=-1), torch.ones(2), atol=1e-3))
    frozen = modeling.MetricModel(spec, n_classes=3, freeze_backbone=True)
    check("freeze_backbone замораживает бэкбон",
          not any(p.requires_grad for p in frozen.backbone.parameters()))
    check("freeze_backbone оставляет голову обучаемой",
          all(p.requires_grad for p in frozen.head.parameters()))


def main() -> int:
    test_metric()
    test_triplet()
    test_sampler()
    test_csv()
    test_model()
    print(f"\n=== итог: ok {len(PASS)}, fail {len(FAIL)} ===")
    for name in FAIL:
        print(f"  FAIL: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())


