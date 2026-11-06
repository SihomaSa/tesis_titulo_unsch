"""
robustez.py — Obs. 1 (HG-b), 2, 8 y 9 del informe del jurado.

  2) Ablación con LightGBM directo (misma familia que el modelo final).
  9) X3 separada en TERRITORIO (provincia, distrito) y PRODUCTO (cultivo).
  8) Variante con X2 REZAGADA (valores de la observación previa de la serie)
     para descartar fuga temporal por disponibilidad de variables.
  1) Contribución de cada dimensión EN PRUEBA: modelo completo vs sin la dimensión,
     con pérdidas pareadas por mes + HAC + Holm (contribución predictiva, no causalidad).

Nada de esto toca el modelo final ni la selección: el modelo se elige en validación;
aquí solo se contrastan hipótesis y se comprueba robustez.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.analysis import significancia as sg
from src.analysis.ablation_study import run_ablation
from src.config import ALPHA, OUT_TAB
from src.models.pipelines import build_pipeline

X2_PREV = ["SIEMBRA_PREV", "COSECHA_PREV", "PRODUCCION_PREV", "RENDIMIENTO_TON_HA_PREV"]


def _params(p_lgbm, rapido):
    prm = {f"model__{k}": v for k, v in p_lgbm.items()}
    if rapido and "model__n_estimators" in prm:
        prm["model__n_estimators"] = min(int(prm["model__n_estimators"]), 100)
    return prm


def dimensiones_x3_separada(dimensiones):
    d = {k: list(v) for k, v in dimensiones.items() if k != "X3_Geografica"}
    d["X3a_Territorio"] = ["PROVINCIA_enc", "DISTRITO_enc"]
    d["X3b_Producto"] = ["CULTIVO_enc"]
    orden = ["X1_Temporal", "X2_Productiva", "X3a_Territorio", "X3b_Producto", "X4_Economica_Historica"]
    return {k: d[k] for k in orden}


def ablaciones(X, y, idx_dev, features, dimensiones, p_lgbm, rapido=False):
    """Obs. 2 (LightGBM) y Obs. 9 (X3 separada, con LightGBM)."""
    print("\n[Obs.2] Ablación con LightGBM (misma familia que el modelo final)...")
    a1 = run_ablation(X, y, idx_dev, features, dimensiones, p_lgbm, rapido, clave="lgbm", sufijo="_lgbm")
    print("\n[Obs.9] Ablación con X3 separada en territorio y producto (LightGBM)...")
    a2 = run_ablation(X, y, idx_dev, features, dimensiones_x3_separada(dimensiones), p_lgbm, rapido,
                      clave="lgbm", sufijo="_lgbm_x3_separada")
    return a1, a2


def _ajusta_predice(df, feats, idx_dev, idx_te, prm):
    Xv = df[feats].astype(float)
    pipe = build_pipeline("lgbm", feats, **prm)
    pipe.fit(Xv.iloc[idx_dev], df["PRECIO_CHACRA"].astype(float).iloc[idx_dev])
    return pipe.predict(Xv.iloc[idx_te])


def contribucion_en_prueba(df, y, idx_dev, idx_te, features, dimensiones, p_lgbm, rapido=False, etiqueta="lgbm"):
    """Obs. 1 (HG-b): ΔMAE en PRUEBA al quitar cada dimensión, pareado por mes (HAC) + Holm."""
    prm = _params(p_lgbm, rapido)
    yt = y.iloc[idx_te].to_numpy()
    per = df["PERIODO"].iloc[idx_te].to_numpy()
    p_full = _ajusta_predice(df, features, idx_dev, idx_te, prm)
    filas = []
    for k, cols in dimensiones.items():
        feats_sin = [f for f in features if f not in cols]
        p_sin = _ajusta_predice(df, feats_sin, idx_dev, idx_te, prm)
        d = sg.perdida_mensual(per, yt, p_sin, p_full)     # >0 => el modelo completo es mejor
        h = sg.hac_media(d)
        lo, hi, _ = sg.bootstrap_bloques_media(d)
        filas.append({"Dimension": k, "MAE_completo": mean_absolute_error(yt, p_full),
                      "MAE_sin_dim": mean_absolute_error(yt, p_sin),
                      "R2_completo": r2_score(yt, p_full), "R2_sin_dim": r2_score(yt, p_sin),
                      "Dif_MAE_mensual": h["media"], "IC95_inf": lo, "IC95_sup": hi,
                      "p_unilateral_HAC": h["p_unilateral"], "n_meses": h["n_meses"]})
    t = pd.DataFrame(filas)
    t["p_holm"] = sg.holm(t["p_unilateral_HAC"].to_numpy())
    t["Contribuye"] = np.where(t["p_holm"] < ALPHA, "Sí", "No")
    t.to_csv(os.path.join(OUT_TAB, f"tabla_contribucion_dimensiones_prueba_{etiqueta}.csv"), index=False)
    print(t[["Dimension", "Dif_MAE_mensual", "IC95_inf", "IC95_sup", "p_holm", "Contribuye"]].to_string(index=False))
    return t


def variante_x2_rezagada(df, y, idx_dev, idx_te, features, dimensiones, p_lgbm, rapido=False):
    """Obs. 8: reemplaza X2 (mismo mes) por X2 de la observación previa y compara en PRUEBA."""
    faltan = [c for c in X2_PREV if c not in df.columns]
    if faltan:
        print(f"[Obs.8] Faltan columnas {faltan}: usa el feature_engineering.py actualizado.")
        return None
    prm = _params(p_lgbm, rapido)
    yt = y.iloc[idx_te].to_numpy()
    per = df["PERIODO"].iloc[idx_te].to_numpy()
    x2 = dimensiones["X2_Productiva"]
    esc = {"Original (X2 del mismo mes)": features,
           "X2 rezagada (observación previa)": [f for f in features if f not in x2] + X2_PREV,
           "Sin X2": [f for f in features if f not in x2]}
    preds = {n: _ajusta_predice(df, fs, idx_dev, idx_te, prm) for n, fs in esc.items()}
    base = preds["Original (X2 del mismo mes)"]
    filas = []
    for n, p in preds.items():
        fila = {"Escenario": n, "n_variables": len(esc[n]), "R2": r2_score(yt, p), "MAE": mean_absolute_error(yt, p)}
        if n != "Original (X2 del mismo mes)":
            d = sg.perdida_mensual(per, yt, p, base)   # >0 => el original es mejor que la variante
            h = sg.hac_media(d)
            fila.update({"Dif_MAE_vs_original": mean_absolute_error(yt, p) - mean_absolute_error(yt, base),
                         "p_HAC_original_mejor": h["p_unilateral"]})
        filas.append(fila)
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_variante_x2_rezagada.csv"), index=False)
    print(t.to_string(index=False))
    return t
