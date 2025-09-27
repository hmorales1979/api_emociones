# Imagen base
FROM python:3.11-slim

# Configuración de entorno
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Dependencias nativas mínimas para OpenCV/Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# --- Copiar el cascade a /app/cascades ---
RUN mkdir -p /app/cascades
COPY haarcascade_frontalface_default.xml /app/cascades/

# Copiar código y modelo
COPY server.py /app/
COPY modelo_emociones.h5 /app/

# Puerto (opcional, Render lo detecta por logs)
EXPOSE 8000

# Ejecutar la API
ENV PORT=8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

