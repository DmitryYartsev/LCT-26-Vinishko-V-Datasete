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
import os, io, base64
import numpy as np
from PIL import Image

SIGLIP_DEFAULT = "google/siglip2-base-patch16-256"
DINOV2_DEFAULT = "facebook/dinov2-base"
OR_EMBED_DEFAULT = "voyageai/voyage-multimodal-3.5"

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
    if ENCODER == "openrouter":
        return "openrouter"
    m = MODEL.lower()
    if any(k in m for k in ("voyage", "gemini-embedding", "nemotron-embed")):
        return "openrouter"
    if "dinov2" in m:
        return "dinov2"
    return "siglip"


KIND = _resolve_encoder_kind()
if KIND == "dinov2" and ENCODER == "dinov2" and MODEL == SIGLIP_DEFAULT:
    MODEL = DINOV2_DEFAULT
if KIND == "openrouter" and MODEL == SIGLIP_DEFAULT:
    MODEL = OR_EMBED_DEFAULT


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


class OpenRouterEncoder:
    """Мультимодальный эмбеддер через OpenRouter `/embeddings` (текст+картинка).

    Модели: ``voyageai/voyage-multimodal-3.5`` (dim 1024), ``google/gemini-embedding-2``
    (dim 3072), ``nvidia/llama-nemotron-embed-vl-1b-v2:free`` (dim 2048, бесплатно).
    Картинка отправляется как base64 data-URL в ``input[].content[]``; батчинг —
    несколько ``content``-объектов в одном запросе (проверено до 40 шт.).

    ENV: ``OPENROUTER_API_KEY``, ``OR_EMBED_BATCH`` (деф 32), ``OR_EMBED_MAX_SIDE``
    (деф 512 — ресайз перед отправкой, влияет на токены/цену), ``OR_EMBED_TIMEOUT``,
    ``OR_EMBED_RETRIES``.
    """

    kind = "openrouter"

    def __init__(self):
        import requests
        self.requests = requests
        self.model_name = MODEL
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.url = os.environ.get("SEARCH_API_URL") or "https://openrouter.ai/api/v1/embeddings"
        self.batch = int(os.environ.get("OR_EMBED_BATCH", "32"))
        self.max_side = int(os.environ.get("OR_EMBED_MAX_SIDE", "512"))
        self.timeout = int(os.environ.get("OR_EMBED_TIMEOUT", "180"))
        self.retries = int(os.environ.get("OR_EMBED_RETRIES", "3"))
        if not self.api_key:
            print('[encoder] ВНИМАНИЕ: OPENROUTER_API_KEY не задан — embeddings не сработают')
        print(f"[encoder] openrouter {MODEL} (dim ~1024, batch={self.batch}, max_side={self.max_side})", flush=True)

    def _data_url(self, x) -> str:
        im = _to_pil(x)
        if self.max_side:
            im = im.copy()
            im.thumbnail((self.max_side, self.max_side))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    def _post(self, chunk):
        body = {"model": MODEL,
                "input": [{"content": [{"type": "image_url",
                                        "image_url": {"url": self._data_url(x)}}]}
                          for x in chunk]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(1, self.retries + 1):
            r = self.requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 529):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                raise RuntimeError(f"openrouter embeddings error: {str(data['error'])[:200]}")
            return [d["embedding"] for d in data["data"]]
        raise last or RuntimeError("openrouter embeddings failed")

    def embed(self, images, batch_size: int = None) -> np.ndarray:
        if not isinstance(images, (list, tuple)):
            images = [images]
        bs = int(batch_size or self.batch)
        out = []
        for i in range(0, len(images), bs):
            out.extend(self._post(images[i:i + bs]))
        return _l2(np.asarray(out, dtype=np.float32))


_ENC = None
def get_encoder():
    global _ENC
    if _ENC is None:
        if BACKEND == "remote":
            _ENC = RemoteEncoder()
        elif KIND == "openrouter":
            _ENC = OpenRouterEncoder()
        elif KIND == "dinov2":
            _ENC = DinoV2Encoder()
        else:
            _ENC = LocalEncoder()
    return _ENC
