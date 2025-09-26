# server.py
# api de predicción de emociones (fastapi)
# entradas: imagen (multipart o base64)
# salida: json con emoción top, probabilidades y gráfica base64

import io
import base64
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import tensorflow as tf

# -------------------------
# configuración
# -------------------------
NOMBRES_CLASES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)

# cargar modelo una sola vez
modelo = tf.keras.models.load_model("modelo_emociones.h5")

app = FastAPI(title="API Emociones", version="1.0")

# habilitar CORS para tu dominio (ajusta si usas otro)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ingesistems.sisweb.site", "http://localhost:5500", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# utilidades
# -------------------------
def _preprocesar(im: Image.Image) -> np.ndarray:
    """convierte a gris 48x48 y normaliza -> (1,48,48,1)"""
    im = im.convert("L")  # escala de grises
    im = im.resize(IMG_SIZE)
    arr = np.array(im).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

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
# esquemas
# -------------------------
class BodyBase64(BaseModel):
    imagen_base64: str  # data:image/png;base64,AAAA...  o solo el base64

# -------------------------
# endpoints
# -------------------------
@app.get("/salud")
def salud():
    return {"ok": True, "ts": int(time.time())}

@app.post("/predecir-archivo")
async def predecir_archivo(archivo: UploadFile = File(...)):
    contenido = await archivo.read()
    im = Image.open(io.BytesIO(contenido)).convert("RGB")
    arr = _preprocesar(im)
    out = _predecir_numpy(arr)
    return {"status": "ok", **out}

@app.post("/predecir-base64")
def predecir_base64(body: BodyBase64):
    b64 = body.imagen_base64
    if "," in b64:  # si viene con prefijo data:image/...
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = _preprocesar(im)
    out = _predecir_numpy(arr)
    return {"status": "ok", **out}
