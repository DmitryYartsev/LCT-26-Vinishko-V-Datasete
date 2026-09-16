FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models_cache PYTHONUNBUFFERED=1
WORKDIR /app/ml
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
# torch/torchvision CPU-сборкой (по умолчанию dev на CPU; GPU-образ — отдельно при желании)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements.txt
# код монтируется volume'ом из compose (dev-режим); paths.py -> /app/paths.py, код -> /app/ml
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
