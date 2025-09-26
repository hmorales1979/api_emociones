# imagen base liviana
FROM python:3.10-slim

# evita prompts interactivos y configura TZ (opcional)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# dependencias del sistema para opencv/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# crea directorio de la app
WORKDIR /app

# copia requirements e instala
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copia el código y el modelo
COPY server.py /app/
COPY modelo_emociones.h5 /app/

# Render expone el puerto via $PORT
ENV PORT=8000

# comando de arranque
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
