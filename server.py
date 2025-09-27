# server.py
# API de predicción de emociones (FastAPI)
# Entradas: imagen (multipart o base64)
# Salida: JSON con emoción top, probabilidades y gráfica base64
#
# Mejora de preproceso:
# - Detección de cara con HaarCascade
# - Recorte con margen
# - CLAHE + corrección de gamma
# - Escala de grises 48x48, normalizado (1,48,48,1)
# - (meta) bbox, si hubo fallback y blur flag

import io
import os
import base64
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import tensorflow as tf

# -------------------------
# Configuración
# -------------------------
NOMBRES_CLASES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)

# Ruta del cascade DENTRO del contenedor (Dockerfile lo copia a /app/cascades)
FACE_CASCADE_PATH = "cascades/haarcascade_frontalface_default.xml"
if not os.path.exists(FACE_CASCADE_PATH):
    # fallback por si lo copiaste a /app por error
    if os.path.exists("haarcascade_frontalface_default.xml"):
        FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# cargar modelo una sola vez
modelo = tf.keras.models.load_model("modelo_emociones.h5")

app = FastAPI(title="API Emociones", version="1.1")

# habilitar CORS (ajusta dominios si lo deseas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ingesistems.sisweb.site", "http://localhost:5500", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utilidades de preproceso
# -------------------------
def _b64_to_bgr(data_url: str) -> np.ndarray:
    """Convierte data:image/...;base64,AAAA a BGR (OpenCV)."""
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def _bytes_to_bgr(raw: bytes) -> np.ndarray:
    """Bytes de imagen a BGR (OpenCV)."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def _auto_gamma(gray: np.ndarray) -> np.ndarray:
    """Corrige gamma según brillo medio (suave)."""
    mean = float(gray.mean())
    if mean < 60:
        gamma = 0.7   # subir sombras
    elif mean > 190:
        gamma = 1.3   # bajar altas luces
    else:
        return gray
    table = np.array([((i/255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)

def _is_blurry(gray: np.ndarray, thr: float = 80.0) -> bool:
    """Heurística de desenfoque (varianza del Laplaciano)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thr

def preprocess_for_model(bgr: np.ndarray):
    """
    Devuelve:
      x (1,48,48,1) float32 [0,1]
      meta dict: {bbox, blurry, fallback}
    """
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    used_fallback = False
    if len(faces) == 0:
        # si no hay detección, center crop cuadrado
        h, w = gray_full.shape
        side = min(h, w)
        x = (w - side) // 2
        y = (h - side) // 2
        face_gray = gray_full[y:y+side, x:x+side]
        bbox = (x, y, side, side)
        used_fallback = True
    else:
        # tomar la cara más grande, con margen
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        m = int(0.25 * max(w, h))
        x0 = max(0, x - m)
        y0 = max(0, y - m)
        x1 = min(gray_full.shape[1], x + w + m)
        y1 = min(gray_full.shape[0], y + h + m)
        face_gray = gray_full[y0:y1, x0:x1]
        bbox = (x0, y0, x1 - x0, y1 - y0)

    # Mejora de contraste/iluminación
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_gray = clahe.apply(face_gray)
    face_gray = _auto_gamma(face_gray)

    # Blur check
    blurry = _is_blurry(face_gray)

    # Resize 48x48, normalizar y dar forma para el modelo
    face48 = cv2.resize(face_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face48 = face48.astype("float32") / 255.0
    face48 = np.expand_dims(face48, axis=(0, -1))  # (1,48,48,1)

    meta = {"bbox": bbox, "blurry": blurry, "fallback": used_fallback}
    return face48, meta

def _grafica_base64(probs: List[float], etiquetas: List[str]) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(etiquetas, probs)
    ax.set_title("Probabilidades por emoción")
    ax.set_ylabel("Probabilidad")
    ax.set_ylim(0, 1)
    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _predecir_numpy(arr: np.ndarray) -> Dict:
    pred = modelo.predict(arr, verbose=0)[0]
    idx_top = int(np.argmax(pred))
    emocion_top = NOMBRES_CLASES[idx_top]
    probs = {NOMBRES_CLASES[i]: float(pred[i]) for i in range(len(NOMBRES_CLASES))}
    graf_b64 = _grafica_base64([probs[c] for c in NOMBRES_CLASES], NOMBRES_CLASES)
    return {
        "emocion_top": emocion_top,
        "probabilidades": probs,
        "grafico_base64": graf_b64
    }

# -------------------------
# Esquemas
# -------------------------
class BodyBase64(BaseModel):
    imagen_base64: str  # data:image/png;base64,AAAA... o solo el base64

# -------------------------
# Endpoints
# -------------------------
@app.get("/salud")
def salud():
    return {"ok": True, "ts": int(time.time())}

@app.post("/predecir-archivo")
async def predecir_archivo(archivo: UploadFile = File(...)):
    raw = await archivo.read()
    bgr = _bytes_to_bgr(raw)
    x, meta = preprocess_for_model(bgr)
    out = _predecir_numpy(x)
    return {"status": "ok", "meta": meta, **out}

@app.post("/predecir-base64")
def predecir_base64(body: BodyBase64):
    bgr = _b64_to_bgr(body.imagen_base64)
    x, meta = preprocess_for_model(bgr)
    out = _predecir_numpy(x)
    return {"status": "ok", "meta": meta, **out}
