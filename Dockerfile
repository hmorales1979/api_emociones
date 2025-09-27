FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libs nativas necesarias para opencv/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# código y modelo
COPY server.py /app/
COPY modelo_emociones.h5 /app/

ENV PORT=8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
