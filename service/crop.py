# -*- coding: utf-8 -*-
"""Кроп бутылки по COCO-YOLO (класс `bottle`, без дообучения).

На полочном фото бутылок несколько — берём наиболее КРУПНУЮ и ЦЕНТРАЛЬНУЮ
(пользователь целится в неё). Если бутылка не найдена — возвращаем оригинал.

ENV:
  CROP_ENABLED=1        включить кроп (0 — выключить)
  CROP_MODEL=yolo11n.pt  веса YOLO (скачиваются автоматически, ~6МБ)
  CROP_MARGIN=0.06       паддинг вокруг бокса (доля от размера бутылки)
  CROP_MIN_CONF=0.25     мин. уверенность детекции
"""
import os
from PIL import Image

CROP_ENABLED = os.environ.get("CROP_ENABLED", "1") == "1"
CROP_MODEL = os.environ.get("CROP_MODEL", "yolo11n.pt")
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", "0.06"))
CROP_MIN_CONF = float(os.environ.get("CROP_MIN_CONF", "0.25"))
BOTTLE_CLASS = 39  # COCO id класса "bottle"


class BottleCropper:
    def __init__(self, model_name: str = CROP_MODEL):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.enabled = True

    def crop(self, img: Image.Image):
        """-> (PIL, detected: bool). Если бутылки нет — вернёт оригинал, False."""
        img = img.convert("RGB")
        W, H = img.size
        res = self.model.predict(img, verbose=False, conf=CROP_MIN_CONF, classes=[BOTTLE_CLASS])
        boxes = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        if not boxes:
            return img, False
        cx_img, cy_img = W / 2, H / 2
        diag = (W ** 2 + H ** 2) ** 0.5

        def score(bx):
            x1, y1, x2, y2, conf = bx
            area = ((x2 - x1) * (y2 - y1)) / (W * H)                  # доля площади
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            center_dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5 / diag
            return area - 0.6 * center_dist                          # крупная + центральная
        x1, y1, x2, y2, _ = max(boxes, key=score)
        # паддинг
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        box = (max(0, int(x1 - mx)), max(0, int(y1 - my)),
               min(W, int(x2 + mx)), min(H, int(y2 + my)))
        return img.crop(box), True


_CROPPER = None
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
