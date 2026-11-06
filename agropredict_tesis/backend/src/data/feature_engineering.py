"""
feature_engineering.py — FASE 3 (preparación): ingeniería de características
por dimensión (X1–X4) SIN fuga de información.

Regla de oro: toda variable de X4 usa únicamente precios de períodos
ESTRICTAMENTE ANTERIORES del mismo cultivo-provincia-distrito. La versión
anterior incluía VARIACION_PRECIO = (precio_t - precio_{t-1}) / precio_{t-1},
que contiene la variable objetivo (precio_t): con ella el modelo "ve" la
respuesta. Aquí se reemplaza por la variación PREVIA (t-1 vs t-2).

Los valores faltantes (primer registro de cada serie) NO se imputan aquí:
los trata el Pipeline de scikit-learn (fit solo con entrenamiento).
"""
import json
import os

import numpy as np
import pandas as pd

from src.config import TARGET, MATRIZ_CONSISTENCIA, OUT_TAB

SERIE = ["CULTIVO", "PROVINCIA", "DISTRITO"]


def _estacion(m):
    # Hemisferio sur (Perú): verano dic-feb, otoño mar-may, invierno jun-ago, primavera sep-nov
    return 0 if m in (12, 1, 2) else 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8) else 3


def build_features(df: pd.DataFrame):
    df = df.copy()

    # ---------------- X1 Temporal ----------------
    df["MES_SEN"] = np.sin(2 * np.pi * df["MES_NUM"] / 12)
    df["MES_COS"] = np.cos(2 * np.pi * df["MES_NUM"] / 12)
    df["ESTACION"] = df["MES_NUM"].map(_estacion)

    # ---------------- X2 Productiva ----------------
    df["RENDIMIENTO_TON_HA"] = np.where(df["COSECHA"] > 0, df["PRODUCCION"] / df["COSECHA"], 0.0)

    # ---------------- X3 Geográfica (códigos ordinales estables) ----------------
    catalogo = {}
    for col, base in (("PROVINCIA", "PROVINCIA"), ("DISTRITO", "DISTRITO"), ("CULTIVO", "COD_CULTIVO")):
        llaves = sorted(df[base].astype(str).unique())
        mapa = {k: i for i, k in enumerate(llaves)}
        df[col + "_enc"] = df[base].astype(str).map(mapa).astype(int)
        catalogo[base] = mapa
    uni = df.drop_duplicates("COD_CULTIVO")
    catalogo["NOMBRE_CULTIVO"] = dict(zip(uni["COD_CULTIVO"].astype(str), uni["CULTIVO"]))

    # ---------------- X4 Económica histórica (solo pasado) ----------------
    ser = (df.groupby(SERIE + ["MES_IDX"], as_index=False)[TARGET].mean()
             .sort_values(SERIE + ["MES_IDX"]))
    g = ser.groupby(SERIE)
    ser["PRECIO_LAG1"] = g[TARGET].shift(1)
    lag2 = g[TARGET].shift(2)
    ser["VARIACION_PRECIO_PREVIA"] = (ser["PRECIO_LAG1"] - lag2) / lag2.replace(0, np.nan) * 100
    ser["MEDIA_MOVIL_3"] = g[TARGET].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    ser["MESES_DESDE_PRECIO_PREVIO"] = ser["MES_IDX"] - g["MES_IDX"].shift(1)
    ser["TIENE_HISTORIAL"] = ser["PRECIO_LAG1"].notna().astype(int)
    cols = ["PRECIO_LAG1", "VARIACION_PRECIO_PREVIA", "MEDIA_MOVIL_3",
            "MESES_DESDE_PRECIO_PREVIO", "TIENE_HISTORIAL"]
    df = df.merge(ser[SERIE + ["MES_IDX"] + cols], on=SERIE + ["MES_IDX"], how="left")
    df = df.sort_values(["PERIODO", "CULTIVO", "PROVINCIA", "DISTRITO"]).reset_index(drop=True)

    dimensiones = {d["clave"]: list(d["variables"]) for d in MATRIZ_CONSISTENCIA["dimensiones"]}
    features = [f for v in dimensiones.values() for f in v]

    # Auditoría anti-fuga: ninguna variable debe reconstruir el objetivo
    X = df[features].astype(float)
    corr_max = X.corrwith(df[TARGET]).abs().max()
    print(f"[Fase 3] {len(features)} variables (X1={len(dimensiones['X1_Temporal'])}, "
          f"X2={len(dimensiones['X2_Productiva'])}, X3={len(dimensiones['X3_Geografica'])}, "
          f"X4={len(dimensiones['X4_Economica_Historica'])}); "
          f"sin historial previo: {(df['TIENE_HISTORIAL'] == 0).mean() * 100:.1f}% "
          f"| |corr| máx. con el objetivo = {corr_max:.3f}")

    with open(os.path.join(OUT_TAB, "catalogo_codificacion.json"), "w", encoding="utf-8") as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=1)
    return df, features, dimensiones, catalogo
