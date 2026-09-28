# -*- coding: utf-8 -*-
"""Кроп бутылки по COCO-YOLO (класс `bottle`, без дообучения).

На полочном фото бутылок несколько — берём наиболее КРУПНУЮ и ЦЕНТРАЛЬНУЮ
(пользователь целится в неё). Если бутылка не найдена — возвращаем оригинал.

ENV:
  CROP_ENABLED=1        включить кроп (0 — выключить)
  CROP_MODEL=yolo11n.pt  веса YOLO (скачиваются автоматически, ~6МБ)
  CROP_MARGIN=0.06       паддинг вокруг бокса (доля от размера бутылки)
  CROP_MIN_CONF=0.25     мин. уверенность детекции

Постобработка этикетки (выравнивание по 4 углам, как в Adobe Scan) — отдельный
модуль ``label_align.py``, включается `LABEL_ALIGN=1` (см. docstring модуля).
"""
import os
from pathlib import Path
from PIL import Image

from label_align import ALIGN_ENABLED, get_aligner      # постобработка: выравнивание этикетки

# веса YOLO хранятся в одном месте (models/), чтобы не расползались по cwd
_MODELS = Path(__file__).resolve().parents[1] / "models"
CROP_ENABLED = os.environ.get("CROP_ENABLED", "1") == "1"
CROP_MODEL = os.environ.get("CROP_MODEL", str(_MODELS / "yolo11n.pt"))
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", "0.06"))
CROP_MIN_CONF = float(os.environ.get("CROP_MIN_CONF", "0.25"))
# Минимальная площадь бокса бутылки (доля кадра): ниже — детекция не считается
# (низкий порог conf 0.10 иногда вытаскивает крошечные боксы: 6–9% кадра, аудит
# Reports/15_Crop_audit.md). 0 — не фильтровать.
CROP_MIN_AREA = float(os.environ.get("CROP_MIN_AREA", "0.0"))
CENTER_W = float(os.environ.get("CROP_CENTER_W", "0.6"))          # вес удалённости от центра кадра
TOUCH_EPS = float(os.environ.get("CROP_TOUCH_EPS", "0.015"))      # что считать «срезано рамкой»
# Выбор бокса бутылки. `big_center` — историческая формула (площадь − 0.6·удалённость
# от центра кадра). `big_center_border` (аудит Reports/15_Crop_audit.md) дополнительно
# штрафует бокс, срезанный рамкой кадра (соседняя бутылка у края бывает крупнее
# целевой и «выигрывает» площадь) и бокс небутылочных пропорций.
CROP_PICK = os.environ.get("CROP_PICK", "big_center")           # big_center | big_center_border
BORDER_PENALTY = float(os.environ.get("BORDER_PENALTY", "0.15"))  # × число срезанных сторон
ASPECT_PENALTY = float(os.environ.get("ASPECT_PENALTY", "0.10"))
BOTTLE_MIN_ASPECT = float(os.environ.get("BOTTLE_MIN_ASPECT", "1.3"))   # h/w бутылки
# Что делать, если бутылка не найдена: `orig` — вернуть фото как есть (историческое),
# `label` — поискать этикетку прямо на полном фото и вернуть её кроп (на 24% фото
# real_photo бутылочный детектор молчит, а этикетка находится).
CROP_FALLBACK = os.environ.get("CROP_FALLBACK", "orig")          # orig | label
BOTTLE_CLASS = 39  # COCO id класса "bottle"

# ---------------------------------------------------------------- политика: индекс vs запрос
# Индекс (`build_index.py`) собирается политикой `CROP_*`/`LABEL_*`. Запрос может
# использовать ДРУГУЮ политику: аудит (Reports/15_Crop_audit.md) показал, что на UGC-фото
# полки выгодно снижать порог бутылочного детектора и отбрасывать крошечные боксы
# (`CROP_MIN_CONF=0.10`, `CROP_MIN_AREA=0.12`, fallback на этикетку по полному фото),
# а на студийных фото каталога та же политика кроп этикетки портит.
# Переключение — только явное и только в query-тракте: `use_query_policy()`
# (app/pipeline/recall_at_k), индекс его не включает, эталонные кропы — `index_policy()`.
_POLICY_ENV = {
    'CROP_PICK': 'QUERY_CROP_PICK',
    'CROP_MIN_CONF': 'QUERY_CROP_MIN_CONF',
    'CROP_MIN_AREA': 'QUERY_CROP_MIN_AREA',
    'CROP_FALLBACK': 'QUERY_CROP_FALLBACK',
    'LABEL_PICK': 'QUERY_LABEL_PICK',
    'LABEL_MIN_W_FRAC': 'QUERY_LABEL_MIN_W_FRAC',
}
_OVERRIDE: dict = {}


def policy(name: str) -> str:
    """Текущее значение политики: query-оверрайд, если включён, иначе константа модуля."""
    return _OVERRIDE.get(name, str(globals()[name]))


def policy_f(name: str) -> float:
    return float(policy(name))


def use_query_policy() -> list:
    """Включить query-политику (env ``QUERY_*``, их выставляет `pipeline_config`).

    Идемпотентно; вызывается в query-тракте ПЕРЕД кропом запроса. Пустое значение
    env = оверрайда нет (политика индекса). -> применённые оверрайды (для логов).
    """
    for key, env in _POLICY_ENV.items():
        val = os.environ.get(env)
        if val not in (None, ''):
            _OVERRIDE[key] = val
    return [f'{k}={v}' for k, v in _OVERRIDE.items()]


def reset_policy() -> None:
    """Вернуть индексную политику (кропы индекса/эталонов)."""
    _OVERRIDE.clear()


class _IndexPolicy:
    """Контекст `with index_policy():` — кропы эталонов под индексной политикой."""

    def __enter__(self):
        self._saved = dict(_OVERRIDE)
        _OVERRIDE.clear()
        return self

    def __exit__(self, *exc):
        _OVERRIDE.clear()
        _OVERRIDE.update(self._saved)
        return False


def index_policy() -> _IndexPolicy:
    return _IndexPolicy()


class BottleCropper:
    def __init__(self, model_name: str = CROP_MODEL):
        from ultralytics import YOLO
        _MODELS.mkdir(exist_ok=True)          # чтобы веса скачивались сюда, а не в cwd
        self.model = YOLO(model_name)
        self.enabled = True

    def detect(self, img: Image.Image):
        """Все боксы класса `bottle`: [(x1, y1, x2, y2, conf)] в координатах img.

        Отдельный метод (а не код внутри `crop`) — чтобы аудит/дебаг
        (`ML evaluation/crop_audit.py`) видел РОВНО те боксы, что и пайплайн.
        """
        res = self.model.predict(img, verbose=False, conf=policy_f('CROP_MIN_CONF'),
                                 classes=[BOTTLE_CLASS])
        boxes = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes

    @staticmethod
    def pick(boxes, size):
        """Выбор «крупной + центральной» бутылки (формула score, что и в `crop`).

        При ``CROP_PICK=big_center_border`` штрафуется бокс, срезанный рамкой кадра
        (+`BORDER_PENALTY` за каждую срезанную сторону) и бокс «небутылочных»
        пропорций (h/w < `BOTTLE_MIN_ASPECT`, +`ASPECT_PENALTY`). Без этого
        соседняя бутылка у края кадра выигрывает по площади у целевой.

        Вариант «выбирать бокс, содержащий главную этикетку фото» проверен и
        отклонён: правила «главной этикетки» на полочном фото нет (net −1/−2,
        см. Reports/15_Crop_audit.md), диагностика — флаг `wrong_bottle` в
        `ML evaluation/crop_audit.py`.
        """
        W, H = size
        cx_img, cy_img = W / 2, H / 2
        diag = (W ** 2 + H ** 2) ** 0.5

        def score(bx):
            x1, y1, x2, y2, conf = bx
            area = ((x2 - x1) * (y2 - y1)) / (W * H)                  # доля площади
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            center_dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5 / diag
            s = area - CENTER_W * center_dist                         # крупная + центральная
            if policy('CROP_PICK') == "big_center_border":
                nt = sum((x1 <= W * TOUCH_EPS, y1 <= H * TOUCH_EPS,
                          x2 >= W * (1 - TOUCH_EPS), y2 >= H * (1 - TOUCH_EPS)))
                s -= BORDER_PENALTY * nt
                if (y2 - y1) / max(1.0, x2 - x1) < BOTTLE_MIN_ASPECT:
                    s -= ASPECT_PENALTY
            return s
        return max(boxes, key=score)

    def crop(self, img: Image.Image):
        """-> (PIL, detected: bool). Если бутылки нет — вернёт оригинал, False."""
        img = img.convert("RGB")
        W, H = img.size
        boxes = filter_bottle_boxes(self.detect(img), img.size)
        if not boxes:
            if policy('CROP_FALLBACK') == "label":
                fb = _label_fallback_crop(img)     # кроп этикетки по полному фото
                if fb is not None:
                    return fb, True
            return img, False
        x1, y1, x2, y2, _ = self.pick(boxes, (W, H))
        # паддинг
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        box = (max(0, int(x1 - mx)), max(0, int(y1 - my)),
               min(W, int(x2 + mx)), min(H, int(y2 + my)))
        return img.crop(box), True


_CROPPER = None


def filter_bottle_boxes(boxes, size):
    """Отбросить боксы бутылки меньше `CROP_MIN_AREA` (доля кадра; 0 — не фильтровать).

    Один источник правды для пайплайна (`BottleCropper.crop`) и аудита
    (`ML evaluation/crop_audit.py`): иначе низкий `CROP_MIN_CONF` вытаскивает
    крошечные ложные боксы, которые выигрывают у целевой бутылки.
    """
    if policy_f('CROP_MIN_AREA') <= 0:
        return list(boxes)
    W, H = size
    min_area = policy_f('CROP_MIN_AREA') * W * H
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= min_area]


def get_cropper():
    global _CROPPER
    if _CROPPER is None:
        _CROPPER = BottleCropper()
    return _CROPPER


def maybe_crop(img: Image.Image):
    """Кроп с учётом CROP_ENABLED. -> (PIL, detected|None)."""
    if not CROP_ENABLED:
        return img, None
    return get_cropper().crop(img)

# ---------------------------------------------------------------- label crop
# Кроп этикетки: YOLO-детектор этикеток (дообученный, data/test_labels_detector).
# Запускается ПОВЕРХ бутылочного кропа. Используется второй веткой поиска:
#   ветка A: эмбеддинг бутылочного кропа (индекс model)
#   ветка B: эмбеддинг кропа этикетки   (индекс model#label)
# итог — ветка с максимальным top-1 score (см. norm_exp/eval_pipelines.py: A vs B).
USE_LABEL_BRANCH = os.environ.get("USE_LABEL_BRANCH", "1") == "1"
LABEL_SUFFIX = "#label"      # суффикс модели-ветки этикетки в pgvector (model + LABEL_SUFFIX)
LABEL_MODEL = os.environ.get("LABEL_MODEL", str(_MODELS / "label_det_best.pt"))
LABEL_MIN_CONF = float(os.environ.get("LABEL_MIN_CONF", "0.2"))   # F1=0.99 @ 0.21
LABEL_MARGIN = float(os.environ.get("LABEL_MARGIN", "0.02"))      # паддинг кропа этикетки
# Выбор бокса этикетки среди найденных. `conf` — историческое «максимальный conf»;
# `conf_size_center` — conf × (площадь + центральность): лицевая этикетка крупнее и
# центрее контрэтикетки/шейного ярлыка, хотя те могут иметь более высокий conf.
# `LABEL_MIN_W_FRAC` (доля ширины кропа) отсекает узкие фрагменты: 0 — не отсекать.
LABEL_PICK = os.environ.get("LABEL_PICK", "conf")                 # conf | conf_size_center
LABEL_MIN_W_FRAC = float(os.environ.get("LABEL_MIN_W_FRAC", "0.0"))
# Стратегия OCR-входа (VLM) — см. `ocr_label_crop`. `align` = историческое поведение.
OCR_LABEL_CROP = os.environ.get("OCR_LABEL_CROP", "align")        # align|bbox|bottle|auto
LABEL_CROP_MIN_CONF = float(os.environ.get("LABEL_CROP_MIN_CONF", "0.45"))
LABEL_CROP_MIN_AREA = float(os.environ.get("LABEL_CROP_MIN_AREA", "0.02"))   # доля кропа бутылки
LABEL_CROP_MAX_ASPECT = float(os.environ.get("LABEL_CROP_MAX_ASPECT", "3.0"))  # h/w бокса


class LabelCropper:
    """Детектор этикетки на бутылочном кропе -> кроп этикетки (fallback: сам кроп бутылки)."""

    def __init__(self, model_name: str = LABEL_MODEL):
        from ultralytics import YOLO
        self.model = YOLO(model_name)

    def detect(self, img: Image.Image):
        """Все боксы этикеток по убыванию conf: [(x1, y1, x2, y2, conf)].

        Нужен аудиту (`ML evaluation/crop_audit.py`): по второму боксу видно, не
        нашёл ли детектор контротэкетку/шейный ярлык вместо лицевой этикетки.
        """
        res = self.model.predict(img, verbose=False, conf=LABEL_MIN_CONF, imgsz=640)
        boxes = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes

    def best_box(self, img: Image.Image):
        """Лучший бокс этикетки -> ([x1,y1,x2,y2], conf); без боксов — ``(None, -1.0)``.

        ``LABEL_PICK=conf`` (историческое) — максимальный conf; ``conf_size_center``
        — conf × (площадь + центральность) (см. константы модуля).
        ``LABEL_MIN_W_FRAC`` отсекает боксы уже порога по ширине: если после этого
        боксов нет, детекции нет и вызывающий код откатывается на кроп бутылки.
        """
        boxes = self.detect(img)
        min_w = policy_f('LABEL_MIN_W_FRAC')
        if min_w > 0:
            boxes = [b for b in boxes if (b[2] - b[0]) >= min_w * img.size[0]]
        if not boxes:
            return None, -1.0
        if policy('LABEL_PICK') == "conf_size_center":
            W, H = img.size
            diag = (W ** 2 + H ** 2) ** 0.5
            max_area = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) or 1.0

            def score(b):
                x1, y1, x2, y2, c = b
                area_rel = ((x2 - x1) * (y2 - y1)) / max_area
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                center = 1 - (((cx - W / 2) ** 2 + (cy - H / 2) ** 2) ** 0.5) / diag
                return c * (0.4 + 0.4 * area_rel + 0.2 * center)
            x1, y1, x2, y2, c = max(boxes, key=score)
            return [x1, y1, x2, y2], c
        x1, y1, x2, y2, c = boxes[0]                  # detect() уже сортирует по conf
        return [x1, y1, x2, y2], c

    def crop(self, bottle_crop: Image.Image, align=None):
        """-> (PIL, label_found: bool, conf). Без этикетки — вернёт вход, False.

        При ``LABEL_ALIGN=1`` bbox детектора проходит постобработку в
        ``label_align``: этикетка выравнивается по своим 4 углам (rectify, как в
        Adobe Scan). Важно: флаг влияет и на индекс (build_index), и на запрос —
        препроцесс обязан совпадать.

        ``align`` переопределяет ``LABEL_ALIGN`` для ОДНОГО вызова: так OCR-тракт
        получает выпрямленный кроп этикетки, не меняя препроцесс retrieval-индекса.
        """
        img = bottle_crop.convert("RGB")
        W, H = img.size
        box, conf = self.best_box(img)
        if box is None:
            return img, False, 0.0
        use_align = ALIGN_ENABLED if align is None else bool(align)
        if use_align:
            try:
                return get_aligner().align(img, box), True, conf
            except Exception as e:            # noqa: BLE001 — не ронять пайплайн
                print(f"[label_align] {e}; fallback на bbox-кроп", flush=True)
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * LABEL_MARGIN, (y2 - y1) * LABEL_MARGIN
        return (img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                          min(W, int(x2 + mx)), min(H, int(y2 + my)))),
                True, conf)


_LCROPPER = None
def get_label_cropper():
    global _LCROPPER
    if _LCROPPER is None:
        _LCROPPER = LabelCropper()
    return _LCROPPER


def _label_fallback_crop(img: Image.Image):
    """Кроп этикетки прямо на полном фото — fallback при ``CROP_FALLBACK=label``.

    Если бутылочный детектор молчит, «кроп бутылки» = всё фото, и кроп этикетки
    считается по нему: в кадр попадают соседние бутылки/фон, а нужная этикетка
    может оказаться вообще за пределами бокса (Reports/15_Crop_audit.md). Здесь
    этикетка ищется сразу на фото. -> PIL | None
    """
    box, _conf = get_label_cropper().best_box(img)
    if box is None:
        return None
    W, H = img.size
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * LABEL_MARGIN, (y2 - y1) * LABEL_MARGIN
    return img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                     min(W, int(x2 + mx)), min(H, int(y2 + my))))


def maybe_label_crop(bottle_crop: Image.Image):
    """Кроп этикетки с учётом USE_LABEL_BRANCH. -> (PIL, detected|None)."""
    if not USE_LABEL_BRANCH:
        return bottle_crop, None
    return get_label_cropper().crop(bottle_crop)


def ocr_label_crop(bottle_crop: Image.Image, mode: str | None = None):
    """Кроп для OCR-модели (VLM): стратегия выбирается ``OCR_LABEL_CROP``.

    Отличие от ``maybe_label_crop``: выравнивание по 4 углам включается принудительно,
    даже если ``LABEL_ALIGN=0`` для retrieval-индекса. Инвариант «препроцесс запроса ==
    препроцесс индекса» не нарушается: этот кроп идёт ТОЛЬКО в VLM, а эмбеддинг
    (ветка этикетки + визуальная проверка) по-прежнему берёт ``maybe_label_crop``.

    Стратегии (``mode`` переопределяет env):

    * ``align`` (по умолчанию, историческое) — bbox детектора этикетки + rectify;
    * ``bbox`` — bbox детектора без выравнивания;
    * ``bottle`` — страховка: детектор этикетки не используется вовсе, VLM читает
      кроп бутылки (помогает, когда детектор этикетки обрезает надписи — отчёты 19/20:
      87.09, 87.54, 94.02, 95.63);
    * ``auto`` — bbox+rectify, но если бокс детектора выглядит ненадёжным
      (уверенность < ``LABEL_CROP_MIN_CONF``, площадь < ``LABEL_CROP_MIN_AREA`` доли
      кропа бутылки или аспект вне ``[1/LABEL_CROP_MAX_ASPECT, LABEL_CROP_MAX_ASPECT]``),
      отдаём кроп бутылки.

    Без ветки этикетки или без детекции -> возвращает вход (бутылочный кроп).
    -> (PIL, found: bool)
    """
    mode = (mode or OCR_LABEL_CROP).lower()
    if not USE_LABEL_BRANCH or mode == 'bottle':
        return bottle_crop.convert("RGB"), False
    if mode == 'auto':
        box, conf = get_label_cropper().best_box(bottle_crop)
        if box is None or not _label_box_reliable(box, conf, bottle_crop.size):
            return bottle_crop.convert("RGB"), False
    crop, found, _ = get_label_cropper().crop(bottle_crop, align=(mode != 'bbox'))
    return crop, found


def _label_box_reliable(box, conf: float, size) -> bool:
    """Похож ли бокс этикетки на надёжный (для ``OCR_LABEL_CROP=auto``).

    Детектор этикеток на UGC-фото иногда отдаёт узкую полосу/кусок этикетки — тогда
    надписи (бренд/линейка) в кроп не попадают и VLM их не читает.
    """
    W, H = size
    x1, y1, x2, y2 = box
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    area = (w * h) / max(W * H, 1)
    aspect = h / max(w, 1.0)
    return (conf >= LABEL_CROP_MIN_CONF and area >= LABEL_CROP_MIN_AREA
            and (1.0 / LABEL_CROP_MAX_ASPECT) <= aspect <= LABEL_CROP_MAX_ASPECT)


if __name__ == "__main__":
    import sys, glob
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8")
    # монтаж: оригинал | кроп для 3 публичных query + 2 эталонов
    ROOT = Path("D:/_hack/lct_26")
    qs = sorted((ROOT / "eval_extracted/queries").glob("*.jpg")) + \
         sorted((ROOT / "eval_extracted/queries").glob("*.webp"))
    refs = glob.glob(str(ROOT / "solution/images/reference/massandra-muskatel-belyy*.webp"))[:2]
    files = qs + [Path(r) for r in refs]
    cr = BottleCropper()
    cell = 260
    grid = Image.new("RGB", (cell * 2, cell * len(files)), (255, 255, 255))
    for i, f in enumerate(files):
        orig = Image.open(f).convert("RGB")
        crop, det = cr.crop(orig)
        for j, im in enumerate([orig, crop]):
            t = im.copy(); t.thumbnail((cell, cell))
            c = Image.new("RGB", (cell, cell), (235, 235, 235))
            c.paste(t, ((cell - t.width) // 2, (cell - t.height) // 2))
            grid.paste(c, (j * cell, i * cell))
        print(f"{f.name}: detected={det}  {orig.size} -> {crop.size}")
    out = ROOT / "solution/images/_crop_test.jpg"
    grid.save(out, quality=88)
    print("saved", out)
