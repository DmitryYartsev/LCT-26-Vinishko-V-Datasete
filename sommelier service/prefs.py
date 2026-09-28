# -*- coding: utf-8 -*-
"""Профиль предпочтений (stateless: живёт на фронте) + детерминированный скоринг вина.

Профиль — компактный JSON, который LLM обновляет на каждом ходе диалога. Скоринг НЕ зовёт
LLM: соответствие карточки профилю = взвешенная доля выполненных критериев (только тех,
что пользователь задал) + штраф за «не хочу». Поэтому /v1/match быстрый и воспроизводимый.
"""
import os

COLORS =["красное", "белое", "розовое", "оранжевое"]
# брют/экстра брют по сахару = «сухое»; игристость — отдельный флаг sparkling
SWEETNESS = ["сухое", "полусухое", "полусладкое", "сладкое"]

# сырые теги блюд с сайта -> укрупнённые группы (LLM выбирает только из групп)
DISH_GROUPS = {
    "Сыры": ["Сыры"],
    "Рыба и морепродукты": ["Рыба и морепродукты", "Морепродукты", "Блюда из рыбы", "Устрицы"],
    "Мясо и гриль": ["Мясо и стейки", "BBQ"],
    "Птица": ["Блюда из птицы"],
    "Закуски": ["Легкие закуски", "Лёгкие закуски", "Закуски", "Брускетты", "Мясное ассорти",
                "Паштеты", "Несладкая выпечка"],
    "Салаты и овощи": ["Салаты", "Овощи гриль", "Запеченные овощи", "Свежие овощи"],
    "Паста": ["Паста"],
    "Пицца и фастфуд": ["Пицца", "Фастфуд"],
    "Десерты и фрукты": ["Выпечка и десерты", "Десерты", "Мороженое", "Фруктово-ягодные десерты",
                         "Шоколад", "Фрукты"],
    "Азиатская и острая кухня": ["Азиатская кухня", "Острое"],
    "Кавказская кухня": ["Кавказская кухня"],
    "Кухни мира": ["Кухни народов мира", "Средиземноморская кухня", "Русская кухня"],
}
RAW2GROUP = {raw: g for g, raws in DISH_GROUPS.items() for raw in raws}

# веса критериев (сумма по заданным пользователем = знаменатель)
W = {"colors": 3, "sweetness": 3, "sparkling": 2, "dishes": 3, "regions": 1,
     "grapes": 1.5, "notes": 1.5, "alcohol_max": 1}
EXCLUDE_PENALTY = 0.3   # множитель, если вино попало в «не хочу»
# народный рейтинг ниже порога — «хорошее ли вино само по себе» (медиана на сайте ~5, 10-й перцентиль ~4.2)
RATING_MIN = float(os.environ.get("RATING_MIN", "4.0"))


def dish_groups(raw_dishes) -> list[str]:
    return sorted({RAW2GROUP[d] for d in raw_dishes or [] if d in RAW2GROUP})


def _pick(vals, allowed) -> list[str]:
    """Оставить только значения из словаря (регистр не важен), без дублей, в порядке словаря."""
    if not isinstance(vals, list):
        return []
    low = {str(v).strip().lower() for v in vals}
    return [a for a in allowed if a.lower() in low]


def _strs(vals, limit=6) -> list[str]:
    if not isinstance(vals, list):
        return []
    out = [str(v).strip() for v in vals if str(v).strip()]
    return list(dict.fromkeys(out))[:limit]


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize(p, regions: list[str]) -> dict:
    """Привести профиль (от LLM или с фронта) к схеме; неизвестные значения отбрасываются."""
    p = p if isinstance(p, dict) else {}
    sp = p.get("sparkling")
    return {
        "colors": _pick(p.get("colors"), COLORS),
        "sweetness": _pick(p.get("sweetness"), SWEETNESS),
        "sparkling": sp if isinstance(sp, bool) else None,
        "dishes": _pick(p.get("dishes"), list(DISH_GROUPS)),
        "regions": _pick(p.get("regions"), regions),
        "grapes": _strs(p.get("grapes")),
        "notes": _strs(p.get("notes"), 5),
        "exclude_colors": _pick(p.get("exclude_colors"), COLORS),
        "exclude_sweetness": _pick(p.get("exclude_sweetness"), SWEETNESS),
        "exclude_grapes": _strs(p.get("exclude_grapes")),
        "alcohol_max": _num(p.get("alcohol_max")),
        "occasion": (str(p["occasion"]).strip() or None) if p.get("occasion") else None,
        "summary": (str(p["summary"]).strip() or None) if p.get("summary") else None,
    }


def is_empty(p: dict) -> bool:
    return not any(p.get(k) not in (None, [], "") for k in W) and not any(
        p.get(k) for k in ("exclude_colors", "exclude_sweetness", "exclude_grapes"))


def _grape_hit(wanted: list[str], grapes: list[str]) -> list[str]:
    g = [x.lower() for x in grapes or []]
    return [w for w in wanted if any(w.lower() in x or x in w.lower() for x in g)]


def sugar_label(w: dict) -> str | None:
    """«Экстра брют» / «Полусладкое»: остаток категории сайта после цвета, иначе — сладость."""
    _, _, rest = (w.get("category") or "").strip().partition(" ")
    rest = rest.strip() or w.get("sweetness") or ""
    return rest.capitalize() or None


def tagline(w: dict) -> str:
    """«Красное · Сухое» — короткая подпись для списков."""
    return " · ".join(x for x in ((w.get("color") or "").capitalize(), sugar_label(w)) if x)


def score(w: dict, p: dict, note_hits: dict[str, set] | None = None) -> dict:
    """Соответствие вина профилю.

    -> {"score": 0..100 | None, "reasons": [{"ok", "text"}],
        "checks": [{"key", "label", "value", "ok"}], "verdict": ok|part|bad|None, "verdict_text"}

    Критерий учитывается, только если пользователь его задал И у вина есть данные.
    Народный рейтинг в score не входит, но участвует в вердикте (хорошее ли вино само по себе).
    note_hits: {заметка: множество slug, где она нашлась в описании (FTS в БД)}.
    """
    total = earned = 0.0
    reasons, checks = [], []

    def crit(key, frac, ok_text, bad_text, label, value):
        nonlocal total, earned
        total += W[key]
        earned += W[key] * frac
        ok = frac >= 0.5
        reasons.append({"ok": ok, "text": ok_text if ok else bad_text})
        checks.append({"key": key, "label": label, "value": value, "ok": ok})

    if p["colors"] and w.get("color"):
        crit("colors", float(w["color"] in p["colors"]),
             f"{w['color'].capitalize()} — как вы хотели", f"{w['color'].capitalize()}, а вы хотели {' / '.join(p['colors'])}",
             "Цвет", w["color"].capitalize())
    if p["sweetness"] and w.get("sweetness"):
        lvl = SWEETNESS.index(w["sweetness"])
        dist = min(abs(lvl - SWEETNESS.index(s)) for s in p["sweetness"])
        crit("sweetness", {0: 1.0, 1: 0.4}.get(dist, 0.0),
             f"{w['sweetness'].capitalize()} — по вкусу", f"{w['sweetness'].capitalize()}, а не {' / '.join(p['sweetness'])}",
             "Сладость", sugar_label(w))
    if p["sparkling"] is not None and w.get("sparkling") is not None:
        crit("sparkling", float(w["sparkling"] == p["sparkling"]),
             "Игристое" if w["sparkling"] else "Тихое, без пузырьков",
             "Игристое, а вы хотели тихое" if w["sparkling"] else "Тихое, а вы хотели игристое",
             "Игристость", "Игристое" if w["sparkling"] else "Тихое")
    if p["dishes"] and w.get("dish_groups"):
        hit = [d for d in p["dishes"] if d in w["dish_groups"]]
        n = len(p["dishes"])
        crit("dishes", len(hit) / n,
             "Подходит к: " + ", ".join(hit).lower() if hit else "",
             "Сомелье не советует к: " + ", ".join(d for d in p["dishes"] if d not in hit).lower(),
             "Сочетание: " + ", ".join(p["dishes"]).lower(),
             ("Подходит" if len(hit) == n else f"К {len(hit)} из {n}") if hit else "Не рекомендуется")
    if p["regions"] and w.get("region"):
        crit("regions", float(w["region"] in p["regions"]),
             f"Регион: {w['region']}", f"Регион {w['region']}, а не {' / '.join(p['regions'])}",
             "Регион", w["region"])
    if p["grapes"] and w.get("grapes"):
        hit = _grape_hit(p["grapes"], w["grapes"])
        crit("grapes", float(bool(hit)), "Сорт: " + ", ".join(hit), "Другой сорт: " + ", ".join(w["grapes"][:3]),
             "Сорт", ", ".join(w["grapes"][:3]))
    if p["notes"] and note_hits is not None:
        hit = [n for n in p["notes"] if w["slug"] in note_hits.get(n, ())]
        crit("notes", len(hit) / len(p["notes"]),
             "В описании: " + ", ".join(hit), "В описании нет: " + ", ".join(n for n in p["notes"] if n not in hit),
             "Ноты: " + ", ".join(p["notes"]), ", ".join(hit) if hit else "Не упоминаются")
    if p["alcohol_max"] is not None and w.get("alcohol") is not None:
        crit("alcohol_max", float(w["alcohol"] <= p["alcohol_max"]),
             f"Крепость {w['alcohol']:g}%", f"Крепость {w['alcohol']:g}% — выше желаемой",
             f"Крепость до {p['alcohol_max']:g}%", f"{w['alcohol']:g}%")

    bad = []
    if w.get("color") in p["exclude_colors"]:
        bad.append(("Вы не хотели", w["color"]))
    if w.get("sweetness") in p["exclude_sweetness"]:
        bad.append(("Вы не хотели", w["sweetness"]))
    if _grape_hit(p["exclude_grapes"], w.get("grapes")):
        bad.append(("Исключённый сорт", ", ".join(_grape_hit(p["exclude_grapes"], w.get("grapes")))))
    reasons += [{"ok": False, "text": f"{label}: {value}"} for label, value in bad]
    checks += [{"key": "exclude", "label": label, "value": value.capitalize(), "ok": False} for label, value in bad]

    if total == 0 and not bad:
        return {"score": None, "reasons": [], "checks": [], "verdict": None, "verdict_text": None}
    s = earned / total if total else 1.0
    if bad:
        s *= EXCLUDE_PENALTY
    s = round(100 * s)
    reasons.sort(key=lambda r: not r["ok"])   # сначала плюсы
    rating = w.get("rating")
    if rating is not None:
        checks.append({"key": "rating", "label": "Народный рейтинг", "value": f"{rating:.2f}",
                       "ok": rating >= RATING_MIN})
    verdict, text = _verdict(checks, s, rating)
    return {"score": s, "reasons": [r for r in reasons if r["text"]], "checks": checks,
            "verdict": verdict, "verdict_text": text}


def _verdict(checks: list[dict], s: int, rating: float | None) -> tuple[str, str]:
    """ok — всё совпало и рейтинг в порядке; part — частично; bad — лучше взять другое."""
    fails = [c for c in checks if not c["ok"] and c["key"] != "rating"]
    rating_ok = rating is None or rating >= RATING_MIN
    if not fails and rating_ok:
        return "ok", ("Совпадает со всеми пожеланиями, и у вина высокий народный рейтинг. Можно брать."
                      if rating is not None else "Совпадает со всеми вашими пожеланиями. Можно брать.")
    if not fails:
        return "part", f"По вкусу подходит, но народный рейтинг низкий: {rating:.2f} из 5. Ниже — варианты надёжнее."
    v = "part" if s >= 70 else "bad"
    style = [c["value"].lower() for c in fails if c["key"] in ("colors", "sweetness", "sparkling") and c["value"]]
    if rating is not None and rating_ok and style:
        return v, (f"Само по себе вино хорошее ({rating:.2f}), но оно {' и '.join(style)}. "
                   + ("Под вашу цель подойдёт с оговорками." if v == "part" else "Под вашу цель сомелье советует другие — они ниже."))
    taste = [c for c in checks if c["key"] != "rating"]
    return v, (f"Совпадает {len(taste) - len(fails)} из {len(taste)} пожеланий, не подходит: "
               f"{', '.join(c['label'].lower() for c in fails)}.")
