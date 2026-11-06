"""
API REST de AgroPredict (FASE 9: despliegue).

    cd backend
    uvicorn src.api.main:app --reload --port 8000

Requiere haber ejecutado antes `python main.py` (genera outputs/models/modelo_final.pkl
y las tablas). Entradas del formulario: nombres de provincia y distrito y el
código de cultivo; la API traduce con el catálogo guardado junto al modelo,
así que el frontend ya no necesita mapas de codificación escritos a mano.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MATRIZ_CONSISTENCIA, OUT_MOD, OUT_TAB  # noqa: E402

app = FastAPI(title="AgroPredict API", version="2.0.0",
              description="Predicción de precios agrícolas — Piura, 2025 (tesis UNSCH)")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://agropredict-tesis.web.app,https://agropredict-tesis.firebaseapp.com,"
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080",
).split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_cache = {}


def _bundle():
    if "b" not in _cache:
        ruta = os.path.join(OUT_MOD, "modelo_final.pkl")
        if not os.path.exists(ruta):
            raise HTTPException(503, "Modelo no encontrado. Ejecuta 'python main.py' desde backend/.")
        _cache["b"] = joblib.load(ruta)
    return _cache["b"]


def _csv(nombre):
    ruta = os.path.join(OUT_TAB, nombre)
    return pd.read_csv(ruta).replace({np.nan: None}).to_dict(orient="records") if os.path.exists(ruta) else []


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/consistency-matrix")
def matriz():
    return MATRIZ_CONSISTENCIA


@app.get("/api/results/model-comparison")
def comparacion():
    return _csv("tabla_resultados_prueba.csv")


@app.get("/api/results/validation")
def validacion():
    return _csv("tabla_comparacion_validacion.csv")


@app.get("/api/results/ablation")
def ablation():
    # Obs. 2: la ablación vigente es la de LightGBM (misma familia que el modelo final). Random Forest queda
    # solo como respaldo si aún no se corrió la versión nueva.
    cols = ["Dimension", "R2_sin_dim", "R2_completo", "Diferencia_R2", "p_valor", "p_holm", "Significativa_estadistica"]
    filas = _csv("tabla_ablation_wilcoxon_lgbm.csv") or _csv("tabla_ablation_wilcoxon.csv")
    return [{c: r.get(c) for c in cols} for r in filas]


@app.get("/api/results/significance")
def significancia():
    return _csv("tabla_significancia_vs_baseline.csv")


@app.get("/api/results/gap")
def brecha():
    return _csv("tabla_mae_por_brecha.csv")


@app.get("/api/results/hypothesis-report")
def reporte():
    ruta = os.path.join(OUT_TAB, "reporte_contrastacion_hipotesis.json")
    if not os.path.exists(ruta):
        raise HTTPException(404, "Aún no se generó el reporte. Ejecuta main.py.")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/results/anomalies-sample")
def anomalias():
    return _csv("muestra_50_anomalias_revision_manual.csv")


@app.get("/api/catalog")
def catalogo():
    ruta = os.path.join(OUT_TAB, "catalogo_codificacion.json")
    if not os.path.exists(ruta):
        raise HTTPException(404, "Catálogo no generado. Ejecuta main.py.")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


class Entrada(BaseModel):
    ANO: int = Field(ge=2000, le=2100)
    MES: int = Field(ge=1, le=12)
    # Obs. 8: SIEMBRA/COSECHA/PRODUCCION del MISMO mes no están disponibles al emitir un pronóstico. En la
    # evaluación, quitarlas o rezagarlas no empeora el modelo (tabla_variante_x2_rezagada.csv), así que son
    # OPCIONALES: si faltan, el pipeline las imputa con la mediana de entrenamiento.
    SIEMBRA: float | None = Field(default=None, ge=0)
    COSECHA: float | None = Field(default=None, ge=0)
    PRODUCCION: float | None = Field(default=None, ge=0)
    PROVINCIA: str
    DISTRITO: str
    COD_CULTIVO: int
    PRECIO_LAG1: float = Field(gt=0, description="último precio observado (S/ por kg)")
    MEDIA_MOVIL_3: float | None = Field(default=None, description="media de los últimos 3 precios; si falta se usa PRECIO_LAG1")
    PRECIO_LAG2: float | None = Field(default=None, description="penúltimo precio observado (opcional)")
    MESES_DESDE_PRECIO_PREVIO: float = 1.0


@app.post("/api/predict")
def predecir(p: Entrada):
    b = _bundle()
    cat = b["catalogo"]
    prov, dist, cod = p.PROVINCIA.strip().upper(), p.DISTRITO.strip().upper(), str(p.COD_CULTIVO)
    for nombre, valor, clave in (("Provincia", prov, "PROVINCIA"), ("Distrito", dist, "DISTRITO"), ("Cultivo", cod, "COD_CULTIVO")):
        if valor not in cat[clave]:
            raise HTTPException(422, f"{nombre} '{valor}' no existe en los datos de entrenamiento.")
    if p.COSECHA is not None and p.SIEMBRA is not None and p.COSECHA > p.SIEMBRA:
        raise HTTPException(422, "COSECHA no puede superar a SIEMBRA.")
    mes = p.MES
    fila = {
        "ANO": p.ANO, "MES_NUM": mes,
        "MES_SEN": float(np.sin(2 * np.pi * mes / 12)), "MES_COS": float(np.cos(2 * np.pi * mes / 12)),
        "ESTACION": 0 if mes in (12, 1, 2) else 1 if mes in (3, 4, 5) else 2 if mes in (6, 7, 8) else 3,
        "SIEMBRA": p.SIEMBRA if p.SIEMBRA is not None else np.nan,
        "COSECHA": p.COSECHA if p.COSECHA is not None else np.nan,
        "PRODUCCION": p.PRODUCCION if p.PRODUCCION is not None else np.nan,
        "RENDIMIENTO_TON_HA": (p.PRODUCCION / p.COSECHA if p.COSECHA > 0 else 0.0)
                              if (p.COSECHA is not None and p.PRODUCCION is not None) else np.nan,
        "PROVINCIA_enc": cat["PROVINCIA"][prov], "DISTRITO_enc": cat["DISTRITO"][dist],
        "CULTIVO_enc": cat["COD_CULTIVO"][cod],
        "PRECIO_LAG1": p.PRECIO_LAG1,
        "MEDIA_MOVIL_3": p.MEDIA_MOVIL_3 if p.MEDIA_MOVIL_3 is not None else p.PRECIO_LAG1,
        "VARIACION_PRECIO_PREVIA": ((p.PRECIO_LAG1 - p.PRECIO_LAG2) / p.PRECIO_LAG2 * 100) if p.PRECIO_LAG2 else np.nan,
        "MESES_DESDE_PRECIO_PREVIO": p.MESES_DESDE_PRECIO_PREVIO, "TIENE_HISTORIAL": 1,
    }
    X = pd.DataFrame([{f: fila.get(f, np.nan) for f in b["features"]}])
    precio = float(b["model"].predict(X)[0])
    q = b["q90_abs"]
    return {
        "precio_predicho_S_por_kg": round(precio, 3),
        "intervalo_90": [round(max(precio - q, 0.0), 3), round(precio + q, 3)],
        "modelo": b["nombre"],
        "nota_intervalo": "Intervalo empírico: cubre ~90% de los errores absolutos observados en validación.",
    }
