# -*- coding: utf-8 -*-
"""Семантика «то же вино» для метрик (уточнение организаторов).

Правило: **год/крепость (хвостовое число слага) на тождество не влияет** — если в каталоге нет
винтажа с таким годом, но вино во всём остальном то же, это ТО ЖЕ вино. А сорт, цвет, сахар/тип
и линейка — влияют: `...-beloe-suhoe-12` и `...-beloe-polusladkoe-12` это РАЗНЫЕ вина, даже если
этикетка похожа.

В слагах каталога «остальное» зашито словами (`...-krasnoe-suhoe-...`, `...-beloe-bryut-...`),
а изменчивое (год/крепость) — хвостовым числом, поэтому:

    same_wine('...-bryut-115', '...-bryut-125')  -> True    (то же вино, другой винтаж)
    same_wine('...-suhoe-12',  '...-polusladkoe-12') -> False (другой сахар — другое вино)
    same_wine('...-krasnoe-suhoe-14', '...-beloe-suhoe-14') -> False (другой цвет)

    python3 wine_identity.py 'slug-a' 'slug-b'      # проверить пару
    python3 wine_identity.py --selftest             # быстрые проверки правила
"""
from __future__ import annotations

import re
import sys

TRAILING_NUMBER = re.compile(r'-\d+(?:[.,]\d+)?$')


def wine_base(slug: str) -> str:
    """Слаг без хвостовых чисел (год/крепость): '...-suhoe-12' -> '...-suhoe'."""
    s = (slug or '').strip().lower()
    prev = None
    while prev != s:
        prev = s
        s = TRAILING_NUMBER.sub('', s)
    return s


def same_wine(a: str, b: str) -> bool:
    """То же ли это вино по правилу организаторов (год/крепость не важны)."""
    if not a or not b:
        return False
    if a == b:
        return True
    return wine_base(a) == wine_base(b)


def _selftest() -> int:
    cases = [
        # (a, b, ожидаем)
        ('abrau-dyurso-brut-dor-blanc-de-noirs-pino-nuar-beloe-bryut-115',
         'abrau-dyurso-brut-dor-blanc-de-noirs-pino-nuar-beloe-bryut-125', True),
        ('fanagoriya-formula-q-saperavi-krasnoe-suhoe-135',
         'fanagoriya-formula-q-saperavi-krasnoe-suhoe-14', True),
        ('derbent-vino-desono-risling-ekstra-bryut-beloe-105-125',
         'derbent-vino-desono-risling-ekstra-bryut-beloe-105', True),
        ('valeriy-zaharin-bastardo-kefesiya-avtohtonnoe-vino-kryma-bastardo-magarachskiy-krasnoe-suhoe-115',
         'valeriy-zaharin-bastardo-kefesiya-avtohtonnoe-vino-kryma-bastardo-magarachskiy-krasnoe-polusuhoe-12',
         False),
        ('abelyan-villa-krasnoe-suhoe-13', 'abelyan-villa-beloe-suhoe-13', False),
        ('zhemchuzhnaya-9-aligote-czitron', 'zhemchuzhnaya-9-czitron-shardone', False),
        ('kuban-vino-aristov-8-roze-07-merlo-rozovoe-suhoe-8',
         'kuban-vino-aristov-8-roze-merlo-rozovoe-suhoe-8', False),   # разные линейки
    ]
    bad = 0
    for a, b, want in cases:
        got = same_wine(a, b)
        mark = 'ok ' if got == want else 'FAIL'
        bad += got != want
        print(f'  [{mark}] same_wine({a[-40:]!r}, {b[-40:]!r}) = {got} (ожидалось {want})')
    print('selftest:', 'ок' if not bad else f'{bad} ошибок')
    return 1 if bad else 0


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--selftest':
        sys.exit(_selftest())
    if len(sys.argv) == 3:
        a, b = sys.argv[1], sys.argv[2]
        print(f'base(a) = {wine_base(a)}')
        print(f'base(b) = {wine_base(b)}')
        print('то же вино:', same_wine(a, b))
        sys.exit(0)
    print(__doc__)
