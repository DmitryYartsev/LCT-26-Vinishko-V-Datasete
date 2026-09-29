FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR "/app/sommelier service"
# ffmpeg — перекодировать голос из браузера (webm/opus, mp4/aac) в wav для STT
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# код монтируется volume'ом из compose (dev-режим); paths.py -> /app/paths.py, код -> "/app/sommelier service" (как в репо, чтобы paths.py находил данные)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
