# server.py
# API de predicción de emociones (FastAPI) con preprocesamiento de rostro
# - HaarCascade (recorte + margen)
# - CLAHE + gamma
# - Escala de grises 48x48 -> (1,48,48,1)
# - Respuestas JSON con tipos nativos (sin numpy.*) para evitar 500

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

# Ruta del cascade dentro del contenedor
FACE_CASCADE_PATH = "cascades/haarcascade_frontalface_default.xml"
if not os.path.exists(FACE_CASCADE_PATH):
    if os.path.exists("haarcascade_frontalface_default.xml"):
        FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Cargar modelo una vez
modelo = tf.keras.models.load_model("modelo_emociones.h5")

app = FastAPI(title="API Emociones", version="1.2")

# CORS (ajusta dominios si quieres)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ingesistems.sisweb.site", "http://localhost:5500", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utilidades
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
    """Corrige gamma suave según brillo medio."""
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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < thr

def preprocess_for_model(bgr: np.ndarray):
    """
    Devuelve:
      x: (1,48,48,1) float32 [0,1]
      meta: dict con tipos nativos (bbox:list[int], blurry:bool, fallback:bool)
    """
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    used_fallback = False
    if len(faces) == 0:
        # center-crop cuadrado
        h, w = gray_full.shape
        side = min(int(h), int(w))
        x = int((w - side) // 2)
        y = int((h - side) // 2)
        face_gray = gray_full[y:y+side, x:x+side]
        bbox = [int(x), int(y), int(side), int(side)]  # LISTA de int
        used_fallback = True
    else:
        # cara más grande + margen
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        m = int(0.25 * max(int(w), int(h)))
        x0 = max(0, int(x) - m)
        y0 = max(0, int(y) - m)
        x1 = min(int(gray_full.shape[1]), int(x) + int(w) + m)
        y1 = min(int(gray_full.shape[0]), int(y) + int(h) + m)
        face_gray = gray_full[y0:y1, x0:x1]
        bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]  # LISTA de int

    # CLAHE + gamma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_gray = clahe.apply(face_gray)
    face_gray = _auto_gamma(face_gray)

    blurry = bool(_is_blurry(face_gray))

    # 48x48, normalizado
    face48 = cv2.resize(face_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face48 = face48.astype("float32") / 255.0
    face48 = np.expand_dims(face48, axis=(0, -1))  # (1,48,48,1)

    meta = {"bbox": bbox, "blurry": bool(blurry), "fallback": bool(used_fallback)}
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
    # Asegurar tipos nativos en la respuesta
    pred_list = [float(x) for x in pred.tolist()]
    idx_top = int(np.argmax(pred))
    emocion_top = str(NOMBRES_CLASES[idx_top])
    probs = {str(NOMBRES_CLASES[i]): float(pred_list[i]) for i in range(len(NOMBRES_CLASES))}
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
    # Convertir meta a tipos nativos por si acaso (ya viene nativo, pero reforzamos)
    meta_py = {
        "bbox": [int(v) for v in meta["bbox"]],
        "blurry": bool(meta["blurry"]),
        "fallback": bool(meta["fallback"])
    }
    return {"status": "ok", "meta": meta_py, **out}

@app.post("/predecir-base64")
def predecir_base64(body: BodyBase64):
    bgr = _b64_to_bgr(body.imagen_base64)
    x, meta = preprocess_for_model(bgr)
    out = _predecir_numpy(x)
    meta_py = {
        "bbox": [int(v) for v in meta["bbox"]],
        "blurry": bool(meta["blurry"]),
        "fallback": bool(meta["fallback"])
    }
    return {"status": "ok", "meta": meta_py, **out}
