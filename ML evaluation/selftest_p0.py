# -*- coding: utf-8 -*-
"""Самопроверки P0 (парсинг ответа VLM) — без сети/VLM.

    python3 selftest_p0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))

import ocr_rerank as O          # noqa: E402

OK = FAIL = 0


def check(name, cond):
    global OK, FAIL
    if cond:
        OK += 1
        print(f'  ok   {name}')
    else:
        FAIL += 1
        print(f'  FAIL {name}')


def test_ocr_parse():
    raw = ('{"year":"2013","winery":"Абрау","grape":"Рислинг","color":"белое",'
           '"sugar":"брют","wine_type":"белое брют","additional_text":"Victоr",'
           '"raw_text":"ABRAU DURSO 2013 BRUT","field_confidence":{"year":0.9,"color":0.8}}')
    f = O.OcrExtractor._parse(raw)
    check('parse year', f['year'] == '2013')
    check('parse raw_text', f['raw_text'] == 'ABRAU DURSO 2013 BRUT')
    check('parse field_confidence str', '0.9' in f['field_confidence'])
    check('parse all str', all(isinstance(v, str) for v in f.values()))
    check('empty -> all empty', all(v == '' for k, v in O.OcrExtractor._parse('{}').items()
                                    if k not in ('raw_text', 'field_confidence')))


if __name__ == '__main__':
    test_ocr_parse()
    print(f'\nитог: ok {OK}, fail {FAIL}')
    sys.exit(1 if FAIL else 0)
