"""
API REST de AgroPredict (FASE 9: despliegue).

    cd backend
    uvicorn src.api.main:app --reload --port 8000

Requiere haber ejecutado antes `python main.py --horizonte 3`.
"""
import json
import os
import sys

# Carpeta de salida vigente (h = 3). Debe fijarse ANTES de importar src.config.
os.environ.setdefault("AGRO_OUT_SUBDIR", "horizonte_3m")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MATRIZ_CONSISTENCIA, OUT_MOD, OUT_TAB  # noqa: E402

app = FastAPI(title="AgroPredict API", version="2.1.0",
              description="Predicción de precios agrícolas — Piura, 2025 (tesis UNSCH)")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://agropredict-tesis.web.app,https://agropredict-tesis.firebaseapp.com,"
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080",
).split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_cache = {}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

def _bundle():
    if "b" not in _cache:
        ruta = os.path.join(OUT_MOD, "modelo_final.pkl")
        if not os.path.exists(ruta):
            raise HTTPException(503, "Modelo no encontrado. Ejecuta 'python main.py --horizonte 3' desde backend/.")
        _cache["b"] = joblib.load(ruta)
    return _cache["b"]


def _csv(nombre):
    ruta = os.path.join(OUT_TAB, nombre)
    return pd.read_csv(ruta).replace({np.nan: None}).to_dict(orient="records") if os.path.exists(ruta) else []


@app.get("/api/health")
def health():
    return {"status": "ok", "salida": os.getenv("AGRO_OUT_SUBDIR", ""), "tablas": OUT_TAB, "modelo": OUT_MOD,
            "modelo_existe": os.path.exists(os.path.join(OUT_MOD, "modelo_final.pkl"))}


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
    filas = _csv("tabla_ablation_wilcoxon_lgbm.csv") or _csv("tabla_ablation_wilcoxon.csv")
    cols = ["Dimension", "R2_sin_dim", "R2_completo", "Diferencia_MAE", "Diferencia_R2",
            "p_mae_holm", "p_holm", "p_valor", "Significativa_estadistica"]
    return [{c: r[c] for c in cols if c in r} for r in filas]


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
        raise HTTPException(404, "Aún no se generó el reporte. Ejecuta main.py o regenerar_reporte.py.")
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
    # SIEMBRA/COSECHA/PRODUCCION son opcionales: si faltan, el pipeline las imputa con la mediana de entrenamiento.
    SIEMBRA: float | None = Field(default=None, ge=0)
    COSECHA: float | None = Field(default=None, ge=0)
    PRODUCCION: float | None = Field(default=None, ge=0)
    PROVINCIA: str
    DISTRITO: str
    COD_CULTIVO: int
    PRECIO_LAG1: float = Field(gt=0, description="último precio observado (S/ por kg)")
    MEDIA_MOVIL_3: float | None = Field(default=None, description="media de los últimos 3 precios; si falta se usa PRECIO_LAG1")
    PRECIO_LAG2: float | None = Field(default=None, description="penúltimo precio observado (opcional)")
    VARIACION_PRECIO_PREVIA: float | None = Field(default=None, description="variación % del último precio frente al anterior (opcional)")
    # Con h = 3 el último precio disponible tiene al menos 3 meses de antigüedad (≥ 3 por construcción).
    MESES_DESDE_PRECIO_PREVIO: float = Field(default=3.0, ge=3)


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

    if p.VARIACION_PRECIO_PREVIA is not None:
        variacion = p.VARIACION_PRECIO_PREVIA
    elif p.PRECIO_LAG2:
        variacion = (p.PRECIO_LAG1 - p.PRECIO_LAG2) / p.PRECIO_LAG2 * 100
    else:
        variacion = np.nan

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
        "VARIACION_PRECIO_PREVIA": variacion,
        "MESES_DESDE_PRECIO_PREVIO": p.MESES_DESDE_PRECIO_PREVIO, "TIENE_HISTORIAL": 1,
    }

    faltan = [f for f in b["features"] if f not in fila]
    if faltan:
        raise HTTPException(500, f"El modelo espera variables que la API no construye: {faltan}")

    X = pd.DataFrame([{f: fila[f] for f in b["features"]}])
    precio = float(b["model"].predict(X)[0])
    q = b["q90_abs"]
    return {
        "precio_predicho_S_por_kg": round(precio, 3),
        "intervalo_90": [round(max(precio - q, 0.0), 3), round(precio + q, 3)],
        "modelo": b["nombre"],
        "nota_intervalo": "Intervalo empírico: cubre ~90% de los errores absolutos observados en validación.",
    }