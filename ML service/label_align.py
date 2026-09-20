# -*- coding: utf-8 -*-
"""Постобработка этикетки после YOLO-детектора: выравнивание по углам (rectify).

YOLO-детектор этикеток (`crop.py`: `LabelCropper`) отдаёт только axis-aligned
bbox. На реальном UGC-фото этикетка наклонена и снята под углом, поэтому bbox
содержит фон по краям, а сама этикетка внутри «скошена» перспективой — это
размывает эмбеддинг и подмешивает фон в вектор. Модуль повторяет приём
сканеров документов (Adobe Scan / CamScanner):

  1. внутри bbox ищет границы этикетки -> 4 угла (Canny/Оцу + контуры +
     ``approxPolyDP``; если 4-угольник не собрался — ``minAreaRect``, т.е.
     только доворот без растягивания);
  2. упорядочивает углы (tl, tr, br, bl);
  3. перспективной гомографией «распрямляет» четырёхугольник в прямоугольник
     (``cv2.warpPerspective``) — это и есть «выравнивание по углам»
     с растягиванием, как в Adobe Scan;
  4. опционально доворачивает результат в портрет (``LABEL_AUTO_ORIENT=1``).

Модуль не падает на «плохом» входе: если углы не нашлись (этикетка слилась с
фоном) — мягкий fallback на прямоугольный кроп bbox, как было раньше.

ENV:
  LABEL_ALIGN=0               1 — включить выравнивание по углам
  LABEL_ALIGN_MARGIN=0.06     паддинг вокруг bbox перед поиском углов (доля стороны)
  LABEL_ALIGN_PAD=0.02        паддинг fallback-кропа (углы не найдены)
  LABEL_ALIGN_MIN_AREA=0.15   мин. площадь 4-угольника от площади окна поиска
  LABEL_ALIGN_MAX_SIDE=0.35   макс. дисбаланс противоположных сторон (доля)
  LABEL_ALIGN_WORK=800        длинная сторона рабочей копии для поиска углов, px
  LABEL_AUTO_ORIENT=0         1 — повернуть результат в портрет (h >= w)

Использование как библиотеки::

    from label_align import align_label, maybe_align_label, LabelAligner

    aligned = align_label(img, box=(x1, y1, x2, y2))   # bbox от YOLO
    out, ok = maybe_align_label(img, box)              # с учётом LABEL_ALIGN
    quad = LabelAligner().find_quad(img, box)          # только углы, без варпа

CLI::

    # самопроверка на синтетической перспективной этикетке:
    python label_align.py
    # реальное фото (bbox в пикселях исходника x1,y1,x2,y2):
    python label_align.py --image photo.jpg --box 120,300,460,900 --out ./_scratch
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------- настройки
ALIGN_ENABLED = os.environ.get("LABEL_ALIGN", "0") == "1"
ALIGN_MARGIN = float(os.environ.get("LABEL_ALIGN_MARGIN", "0.06"))
ALIGN_PAD = float(os.environ.get("LABEL_ALIGN_PAD", "0.02"))
ALIGN_MIN_AREA = float(os.environ.get("LABEL_ALIGN_MIN_AREA", "0.15"))
ALIGN_MAX_SIDE = float(os.environ.get("LABEL_ALIGN_MAX_SIDE", "0.35"))
ALIGN_WORK = int(os.environ.get("LABEL_ALIGN_WORK", "800"))
ALIGN_AUTO_ORIENT = os.environ.get("LABEL_AUTO_ORIENT", "0") == "1"

ALIGN_MIN_SIDE = 32        # мин. сторона результата, px (защита от мусорных углов)
QUAD_MAX_AREA = 0.97       # 4-угольник ~всё окно поиска = граница не найдена
ASPC_MIN, ASPC_MAX = 0.15, 6.0   # допустимое соотношение сторон этикетки

# Ленивый импорт OpenCV: тянется транзитивно через ultralytics, но модуль должен
# давать понятное сообщение, если его нет.
_HAS_CV2 = None


def _require_cv2():
    """Ленивый импорт OpenCV с понятным сообщением об ошибке."""
    global _HAS_CV2
    if _HAS_CV2 is None:
        try:
            import cv2  # noqa: F401
            _HAS_CV2 = True
        except ImportError as e:
            _HAS_CV2 = False
            raise ImportError(
                "opencv-python не установлен (нужен для выравнивания этикетки):\n"
                '  cd "ML service" && pip install opencv-python\n'
                f"(исходная ошибка: {e})"
            ) from e
    import cv2
    return cv2


# --------------------------------------------------------------- геометрия
def poly_area(quad: np.ndarray) -> float:
    """Площадь четырёхугольника (shoelace) в его единицах."""
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def order_corners(pts: Sequence) -> np.ndarray:
    """Упорядочить 4 точки как (tl, tr, br, bl) — формат для getPerspectiveTransform.

    Сортировка по сумме/разности координат (учебниковый приём) устойчива и к
    повороту на 90°, и к лёгкому скосу перспективы.
    """
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if p.shape[0] != 4:
        raise ValueError(f"нужно 4 угла, получено {p.shape[0]}")
    s = p.sum(axis=1)          # tl — min(x+y), br — max(x+y)
    d = p[:, 0] - p[:, 1]      # tr — max(x-y), bl — min(x-y)
    return np.stack([p[int(np.argmin(s))], p[int(np.argmax(d))],
                     p[int(np.argmax(s))], p[int(np.argmin(d))]])


def _is_convex(q: np.ndarray) -> bool:
    """Все векторные произведения одного знака -> выпуклый (без самопересечений)."""
    nxt = np.roll(q, -1, axis=0)
    nxt2 = np.roll(q, -2, axis=0)
    v1, v2 = nxt - q, nxt2 - nxt
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    eps = 1e-6
    return bool(np.all(cross > eps) or np.all(cross < -eps))


def _quad_sides(q: np.ndarray) -> Tuple[float, float, float, float]:
    """(верх, право, низ, лево) для углов в порядке tl,tr,br,bl."""
    tl, tr, br, bl = q
    return (float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - tr)),
            float(np.linalg.norm(bl - br)), float(np.linalg.norm(tl - bl)))


def _touches_borders(q: np.ndarray, w: int, h: int, tol: float = 3.0) -> int:
    """Сколько сторон окна поиска касается 4-угольник (0..4)."""
    n = 0
    n += int(q[:, 0].min() <= tol)
    n += int(q[:, 1].min() <= tol)
    n += int(q[:, 0].max() >= w - 1 - tol)
    n += int(q[:, 1].max() >= h - 1 - tol)
    return n


def check_quad(quad: np.ndarray, w: int, h: int) -> bool:
    """Похож ли четырёхугольник на этикетку внутри окна w x h.

    Отсеиваем: невыпуклые, слишком мелкие (< ``LABEL_ALIGN_MIN_AREA``), совпадающие
    с рамкой окна (=граница не найдена), с разъехавшимися сторонами и с
    «неэтикеточным» соотношением сторон.
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not _is_convex(q):
        return False
    ratio = poly_area(q) / float(max(1, w * h))
    if ratio < ALIGN_MIN_AREA or ratio > QUAD_MAX_AREA:
        return False
    if _touches_borders(q, w, h) >= 3:
        return False
    top, right, bottom, left = _quad_sides(q)
    sides = (top, right, bottom, left)
    if min(sides) < 8:
        return False
    for a, b in ((top, bottom), (left, right)):        # параллельность «на глаз»
        if max(a, b) > 0 and min(a, b) / max(a, b) < 1.0 - ALIGN_MAX_SIDE:
            return False
    aspect = max(top, bottom) / max(1e-6, max(left, right))
    return ASPC_MIN <= aspect <= ASPC_MAX


# ---------------------------------------------------- поиск углов этикетки
def _label_masks(gray: np.ndarray):
    """Две взаимодополняющие маски границ этикетки (края + бинаризация Оцу)."""
    cv2 = _require_cv2()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 1) градиентные края: рамка/текст/переход «этикетка-фон».
    v = float(np.median(blur))
    lo, hi = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lo, max(1, hi))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((7, 7), np.uint8), iterations=2)
    # 2) бинаризация Оцу: этикетка обычно светлее фона/бутылки.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8), iterations=2)
    # края — как есть; бинаризацию берём в обеих полярностях (тёмная этикетка тоже бывает)
    return [edges, otsu, cv2.bitwise_not(otsu)]


def _find_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """4 угла этикетки в координатах ``gray`` (float32 [4,2]) или None.

    Сначала ищем настоящий 4-угольник (approxPolyDP); если не вышло —
    отдаём повёрнутый прямоугольник ``minAreaRect`` (доворот без растягивания).
    """
    cv2 = _require_cv2()
    h, w = gray.shape[:2]
    min_area_px = ALIGN_MIN_AREA * w * h
    quads, rects = [], []
    for mask in _label_masks(gray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            c_area = float(cv2.contourArea(c))
            if c_area < min_area_px:
                continue
            peri = float(cv2.arcLength(c, True))
            found = None
            for eps in (0.01, 0.02, 0.03, 0.045):
                ap = cv2.approxPolyDP(c, eps * peri, True)
                if len(ap) == 4 and check_quad(ap.reshape(4, 2), w, h):
                    found = ap.reshape(4, 2).astype(np.float32)
                    break
            if found is not None:
                quads.append((c_area, found))
                continue
            box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
            if check_quad(box, w, h):
                rects.append((c_area, box))
    if quads:
        return max(quads, key=lambda t: t[0])[1]
    if rects:
        return max(rects, key=lambda t: t[0])[1]
    return None


def _clamp_box(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """bbox -> целые координаты внутри изображения (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
            min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))))


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


# ------------------------------------------------------------- выравнивание
class LabelAligner:
    """Выравнивание этикетки по углам (rectify) в духе Adobe Scan.

    Parameters
    ----------
    margin : паддинг вокруг bbox перед поиском углов (доля стороны), чтобы
        границы этикетки не обрезались самим bbox.
    pad : паддинг fallback-кропа, когда углы найти не удалось.
    min_area / max_side : пороги ``check_quad`` (см. ENV в docstring модуля).
    work : длинная сторона рабочей копии для поиска углов (скорость/стабильность).
    auto_orient : доворачивать результат в портрет (h >= w).
    """

    def __init__(self, margin: float = ALIGN_MARGIN, pad: float = ALIGN_PAD,
                 min_area: float = ALIGN_MIN_AREA, max_side: float = ALIGN_MAX_SIDE,
                 work: int = ALIGN_WORK, auto_orient: bool = ALIGN_AUTO_ORIENT):
        self.margin = float(margin)
        self.pad = float(pad)
        self.min_area = float(min_area)
        self.max_side = float(max_side)
        self.work = int(work)
        self.auto_orient = bool(auto_orient)

    # --- окно поиска ---
    def search_box(self, img: Image.Image, box: Optional[Sequence[float]] = None):
        """Окно поиска углов в координатах ``img``: bbox + margin (или всё изображение)."""
        w, h = img.size
        if box is None:
            return (0, 0, w, h)
        x1, y1, x2, y2 = _clamp_box(box, w, h)
        if x2 <= x1 or y2 <= y1:
            return None
        mx, my = (x2 - x1) * self.margin, (y2 - y1) * self.margin
        return (max(0, int(x1 - mx)), max(0, int(y1 - my)),
                min(w, int(x2 + mx)), min(h, int(y2 + my)))

    def find_quad(self, img: Image.Image,
                  box: Optional[Sequence[float]] = None) -> Optional[np.ndarray]:
        """4 угла этикетки в координатах исходного ``img`` (float32 [4,2]) или None."""
        img = img.convert("RGB")
        win = self.search_box(img, box)
        if win is None:
            return None
        x0, y0, _, _ = win
        region = img.crop(win)
        rw, rh = region.size
        if min(rw, rh) < 16:
            return None
        gray = _to_gray(region)
        scale = min(1.0, self.work / float(max(rw, rh)))
        cv2 = _require_cv2()
        if scale < 1.0:
            gray = cv2.resize(gray, (max(1, int(round(rw * scale))),
                                     max(1, int(round(rh * scale)))),
                              interpolation=cv2.INTER_AREA)
        quad = _find_quad(gray)
        if quad is None:
            return None
        quad = (quad / scale) + np.array([x0, y0], dtype=np.float32)   # -> координаты img
        return order_corners(quad)

    # --- варп ---
    def align_quad(self, img: Image.Image, quad: np.ndarray,
                   auto_orient: Optional[bool] = None) -> Image.Image:
        """«Распрямить» четырёхугольник в прямоугольник (перспективная гомография).

        Размер результата берётся из средних длин противоположных сторон — это и
        есть лёгкое «растягивание» под прямой угол, как в Adobe Scan.
        """
        cv2 = _require_cv2()
        img = img.convert("RGB")
        q = order_corners(quad)
        top, right, bottom, left = _quad_sides(q)
        tw = max(ALIGN_MIN_SIDE, int(round(max(top, bottom))))
        th = max(ALIGN_MIN_SIDE, int(round(max(left, right))))
        src = np.asarray(q, dtype=np.float32)
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
                       dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        bgr = np.asarray(img)[:, :, ::-1]
        warp = cv2.warpPerspective(bgr, m, (tw, th), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        out = Image.fromarray(warp[:, :, ::-1])
        orient = self.auto_orient if auto_orient is None else auto_orient
        if orient and out.width > out.height:
            out = out.transpose(Image.ROTATE_270)      # 90° по часовой -> портрет
        return out

    def _fallback(self, img: Image.Image, box: Optional[Sequence[float]]) -> Image.Image:
        """Углы не найдены -> прямоугольный кроп bbox с паддингом ``pad``."""
        img = img.convert("RGB")
        w, h = img.size
        if box is None:
            return img
        x1, y1, x2, y2 = _clamp_box(box, w, h)
        mx, my = (x2 - x1) * self.pad, (y2 - y1) * self.pad
        return img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                         min(w, int(x2 + mx)), min(h, int(y2 + my))))

    def align(self, img: Image.Image,
              box: Optional[Sequence[float]] = None) -> Image.Image:
        """Полный пайплайн: bbox от YOLO -> выровненная этикетка (PIL, RGB).

        Никогда не бросает исключение на «плохом» входе: нет углов -> fallback.
        """
        try:
            quad = self.find_quad(img, box)
            if quad is not None:
                return self.align_quad(img, quad)
        except ImportError:
            raise
        except Exception as e:                          # noqa: BLE001 — не ронять пайплайн
            print(f"[label_align] выравнивание не удалось ({e}); fallback", flush=True)
        return self._fallback(img, box)


_ALIGNER = None


def get_aligner() -> LabelAligner:
    """Singleton-выравниватель с настройками из ENV."""
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = LabelAligner()
    return _ALIGNER


def align_label(img: Image.Image, box: Optional[Sequence[float]] = None,
                **kwargs) -> Image.Image:
    """Разовый вызов выравнивания (создаёт ``LabelAligner`` с переопределениями)."""
    if kwargs:
        return LabelAligner(**kwargs).align(img, box)
    return get_aligner().align(img, box)


def maybe_align_label(img: Image.Image, box: Optional[Sequence[float]] = None):
    """Выравнивание с учётом ``LABEL_ALIGN``. -> (PIL, aligned: bool)."""
    if not ALIGN_ENABLED:
        return img, False
    return get_aligner().align(img, box), True


# ------------------------------------------------------- самопроверка / CLI
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция Пирсона двух серых картинок одного размера."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom > 1e-9 else 0.0


def _sim(pil_img: Image.Image, ref_gray: np.ndarray) -> float:
    """Совпадение выровненной этикетки с эталоном (пиксельная корреляция)."""
    g = np.asarray(pil_img.convert("L").resize(
        (ref_gray.shape[1], ref_gray.shape[0]), Image.BILINEAR))
    return _pearson(g, ref_gray)


def _synth_scene(size: Tuple[int, int] = (1000, 1400), seed: int = 0):
    """Синтетика: асимметричная этикетка под перспективой на текстурном фоне.

    -> (scene_pil, label_rgb: np.ndarray, box: (x1,y1,x2,y2), quad: float32[4,2])
    """
    cv2 = _require_cv2()
    rng = np.random.default_rng(seed)
    W, H = size
    lw, lh = 300, 430
    label = np.full((lh, lw, 3), 242, np.uint8)              # светлая этикетка (BGR)
    label[10:22, 10:lw - 10] = (30, 30, 170)                 # красная полоса сверху
    label[lh - 30:lh - 16, 10:lw - 10] = (35, 35, 35)        # тёмная полоса снизу
    cv2.putText(label, "VINO", (34, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                (25, 25, 25), 6, cv2.LINE_AA)
    cv2.putText(label, "2026", (44, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (70, 70, 70), 4, cv2.LINE_AA)
    cv2.rectangle(label, (24, 300), (lw - 24, 400), (90, 60, 20), 5)
    # фон: шум + сильный блюр (обои/полка)
    bg = rng.integers(55, 130, (H, W, 3), np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 9)
    # перспективная проекция этикетки на фон
    quad = np.float32([[340, 300], [700, 240], [760, 900], [300, 1000]])
    src = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
    hm = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(label, hm, (W, H))
    mask = cv2.warpPerspective(np.full((lh, lw), 255, np.uint8), hm, (W, H))
    scene = bg.copy()
    scene[mask > 0] = warped[mask > 0]
    noise = rng.normal(0, 4, scene.shape).astype(np.int16)   # шум сенсора
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    box = (float(quad[:, 0].min()), float(quad[:, 1].min()),
           float(quad[:, 0].max()), float(quad[:, 1].max()))
    return (Image.fromarray(scene[:, :, ::-1]), label[:, :, ::-1].copy(), box, quad)


def _selftest() -> int:
    """Прогон на синтетике: углы находятся, варп распрямляет, результат ближе к
    эталону, чем сырой bbox-кроп; на «пустом» входе — мягкий fallback."""
    checks = []

    def check(cond, name):
        checks.append(bool(cond))
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}")

    # 1) порядок углов — на перемешанных точках
    pts = [[700, 240], [300, 1000], [340, 300], [760, 900]]     # tl,tr,br,bl вперемешку
    oc = order_corners(pts)
    check(np.allclose(oc[0], [340, 300]) and np.allclose(oc[1], [700, 240])
          and np.allclose(oc[2], [760, 900]) and np.allclose(oc[3], [300, 1000]),
          "order_corners -> tl,tr,br,bl")

    # 2) фильтр четырёхугольников
    check(not check_quad([[0, 0], [199, 0], [199, 199], [0, 199]], 200, 200),
          "check_quad: рамка окна (граница не найдена) отвергнута")
    check(check_quad([[20, 20], [80, 30], [90, 150], [10, 140]], 200, 200),
          "check_quad: наклонная этикетка принята")

    # 3) углы на синтетической сцене
    scene, ref_rgb, box, quad_true = _synth_scene(seed=1)
    aligner = LabelAligner()
    found = aligner.find_quad(scene, box)
    check(found is not None, "find_quad нашёл 4 угла на синтетике")
    if found is not None:
        diag = float(np.hypot(box[2] - box[0], box[3] - box[1])) or 1.0
        err = float(np.mean([min(np.linalg.norm(p - t) for t in quad_true)
                             for p in found])) / diag
        check(err < 0.08, f"углы близки к истинным (err={err:.3f} диаг. bbox)")

    # 4) выровненная этикетка ближе к эталону, чем сырой bbox-кроп
    ref_gray = np.asarray(Image.fromarray(ref_rgb).convert("L"))
    aligned = aligner.align(scene, box)
    raw = scene.crop(_clamp_box(box, *scene.size)).resize(
        (ref_gray.shape[1], ref_gray.shape[0]), Image.BILINEAR)
    sim_aligned, sim_raw = _sim(aligned, ref_gray), _pearson(np.asarray(raw.convert("L")), ref_gray)
    print(f"  корреляция с эталоном: align={sim_aligned:.3f}  raw bbox={sim_raw:.3f}")
    check(sim_aligned > sim_raw + 0.05, "выравнивание лучше сырого bbox-кропа")
    check(sim_aligned > 0.9, "выровненная этикетка почти совпала с эталоном")
    ar_out = aligned.width / aligned.height
    ar_ref = ref_gray.shape[1] / ref_gray.shape[0]
    check(abs(ar_out / ar_ref - 1) < 0.25,
          f"aspect близок к эталону ({ar_out:.2f} vs {ar_ref:.2f})")

    # 5) авто-ориентация и fallback на однородном фоне
    portrait = aligner.align_quad(scene, quad_true, auto_orient=True)
    check(portrait.height >= portrait.width, "auto_orient -> портрет (h >= w)")
    plain = Image.new("RGB", (400, 400), (128, 128, 128))
    out = aligner.align(plain, (100, 100, 300, 320))
    check(190 <= out.width <= 215 and 215 <= out.height <= 235,
          f"нет этикетки -> fallback-кроп bbox {out.size}")

    print(f"\nСамопроверка label_align: {sum(checks)}/{len(checks)} ok")
    return 0 if all(checks) else 1


def _cli(argv=None) -> int:
    """CLI: без аргументов — самопроверка; с ``--image`` — выравнивание фото."""
    ap = argparse.ArgumentParser(
        description="Выравнивание этикетки по углам (Adobe-Scan-style).")
    ap.add_argument("--image", help="фото; без него запускается самопроверка")
    ap.add_argument("--box", help="bbox от YOLO: x1,y1,x2,y2 (пиксели исходника)")
    ap.add_argument("--out", default="./_scratch", help="куда сохранить результат")
    ap.add_argument("--auto-orient", action="store_true",
                    help="довернуть результат в портрет (h >= w)")
    args = ap.parse_args(argv)
    if not args.image:
        return _selftest()

    box = None
    if args.box:
        try:
            box = [float(v) for v in args.box.replace(";", ",").split(",")]
        except ValueError:
            ap.error("--box должен быть x1,y1,x2,y2")
        if len(box) != 4:
            ap.error("--box должен быть x1,y1,x2,y2")

    img = Image.open(args.image).convert("RGB")
    aligner = LabelAligner(auto_orient=args.auto_orient)
    quad = aligner.find_quad(img, box)
    out = aligner.align(img, box)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    out.save(dst / f"{stem}_aligned.jpg", quality=92)
    print(f"img {img.size} -> aligned {out.size}; "
          f"box={box}; углы: {'найдены' if quad is not None else 'не найдены (fallback)'}")
    if quad is not None:                                     # отладочный кадр с контуром
        cv2 = _require_cv2()
        arr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        cv2.polylines(arr, [np.round(quad).astype(np.int32)], True, (0, 0, 255), 4)
        Image.fromarray(arr[:, :, ::-1]).save(dst / f"{stem}_quad.jpg", quality=92)
    print(f"saved: {dst / (stem + '_aligned.jpg')}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
