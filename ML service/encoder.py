# -*- coding: utf-8 -*-
"""Image-энкодер: картинка -> L2-нормированный вектор.

Модель и бэкенд выбираются через env (свап без правки кода):

  SEARCH_MODEL    любая HF image-модель (SigLIP/CLIP/ViT). Деф: google/siglip2-base-patch16-256
  SEARCH_BACKEND  local | remote
  SEARCH_API_URL  URL инференс-эндпоинта (для remote; HF Inference API или свой сервис)
  SEARCH_API_TOKEN  токен для remote

local поддерживает и SigLIP-стиль (get_image_features), и ViT-стиль (pooler_output / mean-pool).
remote шлёт байты картинки POST-ом и ждёт вектор (или токен-эмбеддинги — усредняются).
"""
import os, io
import numpy as np
from PIL import Image

MODEL = os.environ.get("SEARCH_MODEL") or os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-256")
BACKEND = os.environ.get("SEARCH_BACKEND", "local").lower()
API_URL = os.environ.get("SEARCH_API_URL", "")
API_TOKEN = os.environ.get("SEARCH_API_TOKEN", "")


def _to_pil(x):
    return x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


def _l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


class LocalEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.model_name = MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[encoder] local {MODEL} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL, torch_dtype=self.dtype).to(self.device).eval()

    def _feats(self, inputs):
        m = self.model
        if hasattr(m, "get_image_features"):
            f = m.get_image_features(**inputs)
            return f if isinstance(f, self.torch.Tensor) else f.pooler_output
        out = m(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)          # ViT: усреднить токены

    def embed(self, images, batch_size: int = 32) -> np.ndarray:
        if not isinstance(images, (list, tuple)):
            images = [images]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = [_to_pil(x) for x in images[i:i + batch_size]]
                inp = self.processor(images=batch, return_tensors="pt").to(self.device)
                out.append(self._feats(inp).float().cpu().numpy())
        return _l2(np.concatenate(out, axis=0)).astype(np.float32)


class RemoteEncoder:
    """POST байтов картинки -> вектор. Ответ: список чисел или токен-эмбеддинги (усредняются)."""
    def __init__(self):
        import requests
        self.requests = requests
        self.model_name = MODEL + " (remote)"
        self.url = API_URL or f"https://api-inference.huggingface.co/models/{MODEL}"
        self.headers = {"Content-Type": "application/octet-stream"}
        if API_TOKEN:
            self.headers["Authorization"] = f"Bearer {API_TOKEN}"

    def _one(self, x):
        b = io.BytesIO(); _to_pil(x).save(b, "JPEG")
        r = self.requests.post(self.url, data=b.getvalue(), headers=self.headers, timeout=60)
        r.raise_for_status()
        v = np.array(r.json(), dtype=np.float32)
        while v.ndim > 1:                                  # [seq, hid] / [1, seq, hid] -> [hid]
            v = v.mean(axis=0)
        return v

    def embed(self, images) -> np.ndarray:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return _l2(np.stack([self._one(x) for x in images])).astype(np.float32)


_ENC = None
def get_encoder():
    global _ENC
    if _ENC is None:
        _ENC = RemoteEncoder() if BACKEND == "remote" else LocalEncoder()
    return _ENC
