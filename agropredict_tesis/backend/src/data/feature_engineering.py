"""
feature_engineering.py — Ingeniería de características por dimensión
(sección 3.7.2.1.C). Cada bloque está etiquetado con el problema/objetivo/
hipótesis específica al que responde (ver config.MATRIZ_CONSISTENCIA).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import TARGET, USE_UBIGEO_EN_X3


def build_features(df: pd.DataFrame):
    """Genera X1-X4 y devuelve (df, FEATURES, DIMENSIONES).

    FEATURES: lista plana de todas las columnas predictoras.
    DIMENSIONES: dict {nombre_dimension: [columnas]} usado por el
    ablation study para responder a PE-a..d / OE-a..d / H1-H4.
    """
    df = df.copy()
    GRUPO = ["CULTIVO", "PROVINCIA"]

    # --- X1 Temporal (PE-a / OE-a / H1) ---
    df["MES_SEN"] = np.sin(2 * np.pi * df["MES"] / 12)
    df["MES_COS"] = np.cos(2 * np.pi * df["MES"] / 12)
    df["ESTACION"] = df["MES"].map(
        lambda m: 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3
    )
    dim_x1 = ["ANO", "MES", "MES_SEN", "MES_COS", "ESTACION"]

    # --- X2 Productiva (PE-b / OE-b / H2) ---
    df["RENDIMIENTO_TON_HA"] = np.where(df["COSECHA"] > 0, df["PRODUCCION"] / df["COSECHA"], 0)
    dim_x2 = ["SIEMBRA", "COSECHA", "PRODUCCION", "RENDIMIENTO_TON_HA"]

    # --- X3 Geográfica (PE-c / OE-c / H3) ---
    cols_x3 = ["PROVINCIA", "DISTRITO", "CULTIVO"]
    if USE_UBIGEO_EN_X3 and "UBIGEO" in df.columns:
        cols_x3.append("UBIGEO")
    for col in cols_x3:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    dim_x3 = [c + "_enc" for c in cols_x3]

    # --- X4 Económica histórica (PE-d / OE-d / H4) ---
    group = df.groupby(GRUPO)[TARGET]
    df["PRECIO_LAG1"] = group.shift(1)
    df["VARIACION_PRECIO"] = ((df[TARGET] - df["PRECIO_LAG1"]) / df["PRECIO_LAG1"].replace(0, np.nan)) * 100
    df["MEDIA_MOVIL_3"] = group.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    median_precio = df[TARGET].median()
    df["PRECIO_LAG1"] = df["PRECIO_LAG1"].fillna(median_precio)
    df["VARIACION_PRECIO"] = df["VARIACION_PRECIO"].fillna(0)
    df["MEDIA_MOVIL_3"] = df["MEDIA_MOVIL_3"].fillna(median_precio)
    dim_x4 = ["PRECIO_LAG1", "VARIACION_PRECIO", "MEDIA_MOVIL_3"]

    dimensiones = {
        "X1_Temporal": dim_x1,
        "X2_Productiva": dim_x2,
        "X3_Geografica": dim_x3,
        "X4_Economica_Historica": dim_x4,
    }
    features = dim_x1 + dim_x2 + dim_x3 + dim_x4

    print(f"[feature_engineering] Features totales: {len(features)} "
          f"(X1={len(dim_x1)}, X2={len(dim_x2)}, X3={len(dim_x3)}, X4={len(dim_x4)})")

    return df, features, dimensiones
