# -*- coding: utf-8 -*-
"""Image-энкодер: картинка -> L2-нормированный вектор.

Энкодер выбирается через env (свап без правки кода):

  SEARCH_ENCODER  siglip | dinov2 | auto   (деф: auto)
  SEARCH_MODEL    HF-id или локальный путь (деф: google/siglip2-base-patch16-256)
  SEARCH_BACKEND  local | remote
  SEARCH_API_URL / SEARCH_API_TOKEN — для remote

Интерфейс одинаковый: get_encoder() -> объект с .model_name и
.embed(images, batch_size=32) -> L2-нормированный np.float32 [N, dim].

auto: "dinov2" в SEARCH_MODEL -> DINOv2, иначе SigLIP/CLIP-стиль.
SEARCH_ENCODER=dinov2 включает DINOv2 даже без SEARCH_MODEL
(дефолт facebook/dinov2-base).

DINOv2: CLS-токен last_hidden_state[:, 0] + L2 (у DINO именно CLS
учили быть глобальным дескриптором; mean-pool размывает детали этикетки).
dim: dinov2-base 768 (= siglip2-base), large 1024. Вектора разных моделей
сосуществуют в pgvector под разными `model`, индекс SigLIP не трогаем.
"""
import os, io
import numpy as np
from PIL import Image

SIGLIP_DEFAULT = "google/siglip2-base-patch16-256"
DINOV2_DEFAULT = "facebook/dinov2-base"

ENCODER = os.environ.get("SEARCH_ENCODER", "auto").lower()   # siglip|dinov2|auto
MODEL = os.environ.get("SEARCH_MODEL") or os.environ.get("SIGLIP_MODEL", SIGLIP_DEFAULT)
BACKEND = os.environ.get("SEARCH_BACKEND", "local").lower()
API_URL = os.environ.get("SEARCH_API_URL", "")
API_TOKEN = os.environ.get("SEARCH_API_TOKEN", "")


def _resolve_encoder_kind() -> str:
    if ENCODER == "dinov2":
        return "dinov2"
    if ENCODER == "siglip":
        return "siglip"
    if "dinov2" in MODEL.lower():
        return "dinov2"
    return "siglip"


KIND = _resolve_encoder_kind()
if KIND == "dinov2" and ENCODER == "dinov2" and MODEL == SIGLIP_DEFAULT:
    MODEL = DINOV2_DEFAULT


def _to_pil(x):
    return x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


def _l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


class LocalEncoder:
    """SigLIP/CLIP-стиль: get_image_features либо pooler_output."""

    kind = "siglip"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.model_name = MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[encoder] local siglip {MODEL} on {self.device}", flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=False)
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

    kind = "remote"
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

    def embed(self, images, batch_size: int = 32) -> np.ndarray:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return _l2(np.stack([self._one(x) for x in images])).astype(np.float32)


class DinoV2Encoder:
    """DINOv2: CLS-токен last_hidden_state[:, 0] + L2."""

    kind = "dinov2"

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.model_name = MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[encoder] local dinov2 {MODEL} on {self.device}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(MODEL, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(MODEL, torch_dtype=self.dtype).to(self.device).eval()

    def embed(self, images, batch_size: int = 32) -> np.ndarray:
        if not isinstance(images, (list, tuple)):
            images = [images]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = [_to_pil(x) for x in images[i:i + batch_size]]
                inp = self.processor(images=batch, return_tensors="pt").to(self.device)
                h = self.model(**inp).last_hidden_state[:, 0, :]
                out.append(h.float().cpu().numpy())
        return _l2(np.concatenate(out, axis=0)).astype(np.float32)


_ENC = None
def get_encoder():
    global _ENC
    if _ENC is None:
        if BACKEND == "remote":
            _ENC = RemoteEncoder()
        elif KIND == "dinov2":
            _ENC = DinoV2Encoder()
        else:
            _ENC = LocalEncoder()
    return _ENC
