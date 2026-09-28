# -*- coding: utf-8 -*-
"""Самопроверки P0 (recall-харнесс, текстовый путь, fusion) — без сети/VLM.

    python3 selftest_p0.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ML service'))

import rerank_fusion as RF      # noqa: E402
import text_retrieval as TR     # noqa: E402
import ocr_rerank as O          # noqa: E402
import pipeline as P            # noqa: E402
from pipeline_config import load_config   # noqa: E402

OK = FAIL = 0


def check(name, cond):
    global OK, FAIL
    if cond:
        OK += 1
        print(f'  ok   {name}')
    else:
        FAIL += 1
        print(f'  FAIL {name}')


def test_build_pool():
    check('build_pool order/dedup',
          RF.build_pool(['a', 'b', 'c'], ['a', 'c', 'd'], image_k=3, text_k=3) == ['a', 'b', 'c', 'd'])
    check('build_pool respects image_k',
          RF.build_pool(['a', 'b', 'c'], ['d'], image_k=2, text_k=1) == ['a', 'b', 'd'])
    check('build_pool image-first', RF.build_pool(['z'], ['y'], 1, 1) == ['z', 'y'])
    check('build_pool empty', RF.build_pool([], []) == [])


def test_features():
    entry = {'slug': 'formula-q-kaberne', 'winery_tok': ['fanagoria'], 'grape_tok': ['cabernet'],
             'color_tok': ['red'], 'sweet_tok': ['dry'], 'title_tok': ['formula'],
             'slug_tok': ['formula', 'kaberne'], 'year': '2013'}
    q = {'winery': ['fanagoria'], 'grape': ['cabernet'], 'color': ['red'],
         'wine_type': ['dry'], 'year': '2013', 'additional_text': ['formula']}
    x = RF.candidate_features(q, entry, image_cos=0.8, in_image_pool=1, image_top1='other',
                              csv_conf=0.7, csv_conf_top1=0.5, e5_cos=0.6, visual_sim=0.9)
    idx = {n: i for i, n in enumerate(RF.FEATURE_NAMES)}
    check('features shape', x.shape == (RF.N_FEATURES,))
    check('year_match == 1', abs(float(x[idx['year_match']]) - 1.0) < 1e-6)
    check('color_match == 1', abs(float(x[idx['color_match']]) - 1.0) < 1e-6)
    check('is_image_top1 == 0', float(x[idx['is_image_top1']]) == 0.0)
    check('csv_margin == 0.2', abs(float(x[idx['csv_margin']]) - 0.2) < 1e-6)
    # год не совпал -> 0
    q2 = dict(q, year='2014')
    x2 = RF.candidate_features(q2, entry, 0.1, 0, 'z', 0.2, 0.2, 0.1, 0.5)
    check('year mismatch == 0', float(x2[idx['year_match']]) == 0.0)
    # пустой год у запроса -> нейтрально 0.5
    q3 = dict(q, year='')
    x3 = RF.candidate_features(q3, entry, 0.1, 0, 'z', 0.2, 0.2, 0.1, 0.5)
    check('year missing -> 0.5', abs(float(x3[idx['year_match']]) - 0.5) < 1e-6)


def test_ranker():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, RF.N_FEATURES))
    y = (X[:, 4] + X[:, 10] > 0).astype(float)      # csv_conf + year_match
    r = RF.LogisticRanker().fit(X, y, iters=3000)
    acc = float(((r.score(X) > 0.5).astype(float) == y).mean())
    check('logreg learns separable (>0.95)', acc > 0.95)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'w.json'
        r.save(p)
        r2 = RF.LogisticRanker.load(p)
        check('ranker save/load', np.allclose(r.score(X[:5]), r2.score(X[:5])))
    check('load_or_default fallback', RF.load_or_default('/nope/x.json') is not None)


def test_text():
    check('query_text', TR.query_text({'year': '2013', 'winery': 'Fanagoria',
                                       'grape': 'Cabernet'}) == '2013 Fanagoria Cabernet')
    check('query_text empty', TR.query_text({}) == '')
    e = {'title': 'Formula Q', 'winery': 'Fanagoria', 'grape': 'Cabernet',
         'category': 'сухое', 'color': 'красное', 'slug': 'x'}
    check('catalog_text has fields', 'Fanagoria' in TR.catalog_text(e)
          and 'красное' in TR.catalog_text(e))


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


def test_fusion_flag():
    cfg = load_config(str(REPO / 'config' / 'pipeline.yaml'))
    P.apply_host_paths(cfg)
    cfg.fusion.enabled = False
    check('build_fusion off -> None', P.build_fusion(cfg, None) is None)


if __name__ == '__main__':
    test_build_pool()
    test_features()
    test_ranker()
    test_text()
    test_ocr_parse()
    test_fusion_flag()
    print(f'\nитог: ok {OK}, fail {FAIL}')
    sys.exit(1 if FAIL else 0)
