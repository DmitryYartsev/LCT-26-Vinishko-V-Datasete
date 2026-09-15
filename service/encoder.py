# -*- coding: utf-8 -*-
"""SigLIP 2 image-энкодер: картинка -> L2-нормированный вектор.

Общий модуль: используется и для построения индекса каталога (baseline retrieval),
и как верификатор при чистке скрейпа. Модель выбирается через env SIGLIP_MODEL.

  CPU-дефолт : google/siglip2-base-patch16-256
  GPU-качество: google/siglip2-so400m-patch16-384   (export SIGLIP_MODEL=...)
"""
import os
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-256")


class SiglipEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[encoder] loading {model_name} on {self.device} ({self.dtype})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        # размерность эмбеддинга
        self.dim = int(getattr(self.model.config, "projection_dim", 0)
                       or self.model.config.vision_config.hidden_size)

    @staticmethod
    def _load(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return Image.open(img).convert("RGB")

    @torch.no_grad()
    def embed(self, images, batch_size: int = 32) -> np.ndarray:
        """images: список путей/PIL. -> np.float32 [N, dim], L2-нормировано."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        out = []
        for i in range(0, len(images), batch_size):
            batch = [self._load(x) for x in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):     # transformers 5.x -> output object
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


_ENC = None
def get_encoder() -> SiglipEncoder:
    global _ENC
    if _ENC is None:
        _ENC = SiglipEncoder()
    return _ENC


if __name__ == "__main__":
    # быстрый self-test: эмбеддинг одного эталона
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import FILTERED
    sys.stdout.reconfigure(encoding="utf-8")
    enc = get_encoder()
    sample = [str(p) for p in FILTERED.glob("*/*")][:2]
    v = enc.embed(sample)
    print("dim:", enc.dim, "| shape:", v.shape, "| |v|:", np.linalg.norm(v, axis=1))
