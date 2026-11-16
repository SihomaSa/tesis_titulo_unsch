"""
feature_engineering.py — FASE 3: ingeniería de características por dimensión (X1–X4) sin fuga.

  horizonte=None : comportamiento anterior ("siguiente observación disponible").
  horizonte=h    : v2. El objetivo es el precio del mes T y TODAS las variables dependientes de la
                   serie (X2, X4) se calculan con información de meses <= T-h (origen del pronóstico).
                   Si X2_MODO="inmediata", X2 usa la observación inmediatamente anterior sin restricción de T-h.
Los faltantes los trata el Pipeline de scikit-learn (fit solo con entrenamiento).
"""
import json
import os

import numpy as np
import pandas as pd

from src.config import (TARGET, MATRIZ_CONSISTENCIA, OUT_TAB, X2_MODO, X2_REZAGO_EXTRA_MESES, USAR_ANO_V2)

SERIE = ["CULTIVO", "PROVINCIA", "DISTRITO"]
X2C = ["SIEMBRA", "COSECHA", "PRODUCCION", "RENDIMIENTO_TON_HA"]


def _estacion(m):
    return 0 if m in (12, 1, 2) else 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8) else 3


# ---------------------------------------------------------------- X4 / X2: versión anterior
def _x4_x2_anterior(df):
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

    x2 = (df.groupby(SERIE + ["MES_IDX"], as_index=False)[X2C].mean().sort_values(SERIE + ["MES_IDX"]))
    gx = x2.groupby(SERIE)
    for c in X2C:
        x2[c + "_PREV"] = gx[c].shift(1)
    df = df.merge(x2[SERIE + ["MES_IDX"] + [c + "_PREV" for c in X2C]], on=SERIE + ["MES_IDX"], how="left")
    return df


# ---------------------------------------------------------------- X4 / X2: horizonte fijo (v2)
def _x4_x2_horizonte(df, h):
    base = (df.groupby(SERIE + ["MES_IDX"], as_index=False)
              .agg(PREC=(TARGET, "mean"), **{c: (c, "mean") for c in X2C})
              .sort_values(SERIE + ["MES_IDX"]).reset_index(drop=True))
    base["MES_IDX"] = base["MES_IDX"].astype("int64")
    g = base.groupby(SERIE)["PREC"]
    base["MM3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())   # incluye el último precio
    prev = g.shift(1)
    base["VAR"] = (base["PREC"] - prev) / prev.replace(0, np.nan) * 100

    izq = df[SERIE + ["MES_IDX"]].drop_duplicates().copy()
    izq["MES_IDX"] = izq["MES_IDX"].astype("int64")

    # --- X4: último precio observado con mes <= origen (= mes objetivo - h)
    izq["ORIGEN"] = izq["MES_IDX"] - h
    der = base[SERIE + ["MES_IDX", "PREC", "MM3", "VAR"]].rename(columns={"MES_IDX": "MES_ULT_PRECIO"})
    m = pd.merge_asof(izq.sort_values("ORIGEN"), der.sort_values("MES_ULT_PRECIO"),
                      left_on="ORIGEN", right_on="MES_ULT_PRECIO", by=SERIE, direction="backward")
    m["PRECIO_LAG1"] = m["PREC"]
    m["MEDIA_MOVIL_3"] = m["MM3"]
    m["VARIACION_PRECIO_PREVIA"] = m["VAR"]
    m["MESES_DESDE_PRECIO_PREVIO"] = m["ORIGEN"] - m["MES_ULT_PRECIO"]      # edad del precio EN EL ORIGEN
    m["TIENE_HISTORIAL"] = m["PREC"].notna().astype(int)
    cols = ["PRECIO_LAG1", "VARIACION_PRECIO_PREVIA", "MEDIA_MOVIL_3", "MESES_DESDE_PRECIO_PREVIO",
            "TIENE_HISTORIAL", "MES_ULT_PRECIO"]
    df = df.merge(m[SERIE + ["MES_IDX"] + cols], on=SERIE + ["MES_IDX"], how="left")

    # --- X2: se conserva la versión contemporánea (*_ACT) solo como cota superior
    for c in X2C:
        df[c + "_ACT"] = df[c]

    if X2_MODO == "rezagada":
        # Variante principal: último dato con mes <= T - h
        izq2 = izq.copy()
        izq2["ORIGEN_X2"] = izq2["MES_IDX"] - h - X2_REZAGO_EXTRA_MESES
        der2 = base[SERIE + ["MES_IDX"] + X2C].rename(
            columns={"MES_IDX": "MES_ULT_X2", **{c: c + "_LAG" for c in X2C}})
        m2 = pd.merge_asof(izq2.sort_values("ORIGEN_X2"), der2.sort_values("MES_ULT_X2"),
                           left_on="ORIGEN_X2", right_on="MES_ULT_X2", by=SERIE, direction="backward")
        df = df.merge(m2[SERIE + ["MES_IDX", "MES_ULT_X2"] + [c + "_LAG" for c in X2C]],
                      on=SERIE + ["MES_IDX"], how="left")
        for c in X2C:
            df[c] = df[c + "_LAG"]        # las columnas X2 pasan a contener solo información del origen

    elif X2_MODO == "inmediata":
        # Variante de sensibilidad: observación inmediatamente anterior sin restricción de T-h
        # (permite T-1 o T-2, lo que en producción sería fuga, pero es prueba de estrés)
        izq2 = izq.copy()
        izq2["ORIGEN_X2"] = izq2["MES_IDX"] - 1  # Solo restamos 1 mes para evitar usar el mes objetivo exacto
        der2 = base[SERIE + ["MES_IDX"] + X2C].rename(
            columns={"MES_IDX": "MES_ULT_X2", **{c: c + "_LAG" for c in X2C}})
        m2 = pd.merge_asof(izq2.sort_values("ORIGEN_X2"), der2.sort_values("MES_ULT_X2"),
                           left_on="ORIGEN_X2", right_on="MES_ULT_X2", by=SERIE, direction="backward")
        df = df.merge(m2[SERIE + ["MES_IDX", "MES_ULT_X2"] + [c + "_LAG" for c in X2C]],
                      on=SERIE + ["MES_IDX"], how="left")
        for c in X2C:
            df[c] = df[c + "_LAG"]

    elif X2_MODO == "sin_x2":
        # Variante extrema: eliminar X2 (se maneja en la selección de features, pero aquí la vaciamos)
        for c in X2C:
            df[c] = np.nan

    df["HORIZONTE"] = h
    return df


def build_features(df: pd.DataFrame, guardar_catalogo: bool = True, horizonte=None):
    df = df.copy()
    df["MES_IDX"] = df["MES_IDX"].astype("int64")

    # ---------------- X1 Temporal (mes objetivo: se conoce de antemano) ----------------
    df["MES_SEN"] = np.sin(2 * np.pi * df["MES_NUM"] / 12)
    df["MES_COS"] = np.cos(2 * np.pi * df["MES_NUM"] / 12)
    df["ESTACION"] = df["MES_NUM"].map(_estacion)

    # ---------------- X2 Productiva (contemporánea; en v2 se reemplaza por la del origen) ----------------
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

    # ---------------- X4 (+ X2 rezagada en v2) ----------------
    df = _x4_x2_anterior(df) if horizonte is None else _x4_x2_horizonte(df, int(horizonte))
    df = df.sort_values(["PERIODO", "CULTIVO", "PROVINCIA", "DISTRITO"]).reset_index(drop=True)

    dimensiones = {d["clave"]: list(d["variables"]) for d in MATRIZ_CONSISTENCIA["dimensiones"]}
    if horizonte is not None and not USAR_ANO_V2:
        dimensiones["X1_Temporal"] = [v for v in dimensiones["X1_Temporal"] if v != "ANO"]
    features = [f for v in dimensiones.values() for f in v]

    # Auditoría anti-fuga
    X = df[features].astype(float)
    corr_max = X.corrwith(df[TARGET]).abs().max()
    if horizonte is not None:
        h = int(horizonte)
        ok = df["MES_ULT_PRECIO"].notna()
        assert ((df.loc[ok, "MES_IDX"] - df.loc[ok, "MES_ULT_PRECIO"]) >= h).all(), \
            "FUGA: hay un precio previo posterior al origen del pronóstico"
        if X2_MODO == "rezagada":
            ok2 = df["MES_ULT_X2"].notna()
            assert ((df.loc[ok2, "MES_IDX"] - df.loc[ok2, "MES_ULT_X2"]) >= h + X2_REZAGO_EXTRA_MESES).all(), \
                "FUGA: X2 usa información posterior al origen"
        elif X2_MODO == "inmediata":
            ok2 = df["MES_ULT_X2"].notna()
            assert ((df.loc[ok2, "MES_IDX"] - df.loc[ok2, "MES_ULT_X2"]) >= 1).all(), \
                "FUGA: X2 de la variante inmediata usó el mismo mes objetivo"
        print(f"[Fase 3] Horizonte h={h} mes(es): X2 modo='{X2_MODO}'; prueba anti-fuga superada.")
    print(f"[Fase 3] {len(features)} variables; sin historial previo: {(df['TIENE_HISTORIAL'] == 0).mean() * 100:.1f}% "
          f"| |corr| máx. con el objetivo = {corr_max:.3f}")

    if guardar_catalogo:
        with open(os.path.join(OUT_TAB, "catalogo_codificacion.json"), "w", encoding="utf-8") as f:
            json.dump(catalogo, f, ensure_ascii=False, indent=1)
    return df, features, dimensiones, catalogo