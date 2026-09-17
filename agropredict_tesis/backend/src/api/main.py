"""
src/api/main.py — API REST que expone al frontend:
  - la Matriz de Consistencia (para mostrarla en el dashboard)
  - los resultados ya calculados por backend/main.py (tablas + reporte)
  - un endpoint /predict que usa el modelo final guardado

Correr (desde la carpeta backend/):
    uvicorn src.api.main:app --reload --port 8000

Requiere haber corrido antes `python main.py` al menos una vez, para que
existan outputs/models/random_forest_final.pkl y outputs/tables/*.csv.
"""
import json
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MATRIZ_CONSISTENCIA, OUT_TAB, OUT_MOD  # noqa: E402

app = FastAPI(
    title="AgroPredict API",
    description="API de predicción de precios agrícolas — Piura, 2025 (tesis)",
    version="1.0.0",
)

# En producción, reemplaza "*" por el dominio real de tu frontend (Firebase Hosting, etc.)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://agropredict-tesis.web.app,https://agropredict-tesis.firebaseapp.com,http://localhost:5500,http://127.0.0.1:5500"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache = {}


def _load_model():
    if "model" not in _model_cache:
        path = os.path.join(OUT_MOD, "random_forest_final.pkl")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=503,
                detail="Modelo no encontrado. Corre 'python main.py' primero desde backend/.",
            )
        bundle = joblib.load(path)
        _model_cache["model"] = bundle["model"]
        _model_cache["features"] = bundle["features"]
    return _model_cache["model"], _model_cache["features"]


def _read_csv_safe(filename):
    path = os.path.join(OUT_TAB, filename)
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _read_json_safe(filename):
    path = os.path.join(OUT_TAB, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# 1. Matriz de consistencia — para pintar el PG/PE/OG/OE/HG/HE en el frontend
# ------------------------------------------------------------------
@app.get("/api/consistency-matrix")
def get_consistency_matrix():
    return MATRIZ_CONSISTENCIA


# ------------------------------------------------------------------
# 2. Resultados ya calculados (tablas del capítulo 4)
# ------------------------------------------------------------------
@app.get("/api/results/model-comparison")
def get_model_comparison():
    return _read_csv_safe("tabla_4.1_comparacion_modelos.csv")


@app.get("/api/results/ablation")
def get_ablation():
    return _read_csv_safe("tabla_4.4_wilcoxon_hipotesis.csv")


@app.get("/api/results/hypothesis-report")
def get_hypothesis_report():
    reporte = _read_json_safe("reporte_contrastacion_hipotesis.json")
    if reporte is None:
        raise HTTPException(status_code=404, detail="Aún no se generó el reporte. Corre main.py.")
    return reporte


@app.get("/api/results/anomalies-sample")
def get_anomalies_sample():
    return _read_csv_safe("muestra_50_anomalias_revision_manual.csv")


# ------------------------------------------------------------------
# 3. Predicción — para el formulario del frontend
# ------------------------------------------------------------------
class PrediccionInput(BaseModel):
    ANO: int
    MES: int
    SIEMBRA: float
    COSECHA: float
    PRODUCCION: float
    PROVINCIA_enc: int
    DISTRITO_enc: int
    CULTIVO_enc: int
    PRECIO_LAG1: float
    VARIACION_PRECIO: float = 0.0
    MEDIA_MOVIL_3: float


@app.post("/api/predict")
def predict(payload: PrediccionInput):
    import numpy as np

    model, features = _load_model()
    data = payload.dict()
    data["MES_SEN"] = float(np.sin(2 * np.pi * data["MES"] / 12))
    data["MES_COS"] = float(np.cos(2 * np.pi * data["MES"] / 12))
    m = data["MES"]
    data["ESTACION"] = 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3
    data["RENDIMIENTO_TON_HA"] = data["PRODUCCION"] / data["COSECHA"] if data["COSECHA"] > 0 else 0

    row = pd.DataFrame([{f: data.get(f, 0) for f in features}])
    pred = float(model.predict(row)[0])
    return {"precio_predicho_S_por_kg": round(pred, 3)}


@app.get("/api/health")
def health():
    return {"status": "ok"}
