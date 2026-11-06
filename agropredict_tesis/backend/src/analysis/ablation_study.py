"""
ablation_study.py — Contraste de las hipótesis específicas H1–H4.

Se evalúan, con TimeSeriesSplit sobre el tramo de desarrollo (train+val; la
PRUEBA no se toca), tres familias de modelos Random Forest con los
hiperparámetros ya optimizados:

  A) Completo vs. SIN cada dimensión (leave-one-dimension-out)  -> ΔR² y Wilcoxon
  B) SOLO cada dimensión                                          -> poder predictivo propio
  C) X4 (historia de precios) vs. X4 + cada dimensión            -> aporte INCREMENTAL

Wilcoxon de rangos con signo pareado, unilateral (H1: quitar la dimensión
empeora el R²), con corrección de Holm para pruebas múltiples. B y C explican
por qué una dimensión puede ser relevante por sí sola y aun así no mejorar
el modelo cuando ya se conoce el precio histórico (redundancia informativa).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from src import tracking
from src.config import ALPHA, N_SPLITS_ABLATION, OUT_TAB
from src.models.pipelines import build_pipeline


def _holm(pvals):
    p = np.asarray(pvals, float)
    orden = np.argsort(p)
    aj = np.empty_like(p)
    m = len(p)
    mx = 0.0
    for rank, i in enumerate(orden):
        mx = max(mx, (m - rank) * p[i])
        aj[i] = min(1.0, mx)
    return aj


def _wilcoxon_mayor(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def run_ablation(X, y, idx_dev, features, dimensiones, rf_params, rapido=False):
    n_splits = 4 if rapido else N_SPLITS_ABLATION
    Xd, yd = X.iloc[idx_dev], y.iloc[idx_dev]
    tss = TimeSeriesSplit(n_splits=n_splits)
    params = {f"model__{k}": v for k, v in rf_params.items()}
    params["model__n_estimators"] = min(int(params.get("model__n_estimators", 150)), 60 if rapido else 150)

    def r2_folds(feats):
        out = []
        for a, b in tss.split(Xd):
            pipe = build_pipeline("rf", feats, **params)
            pipe.fit(Xd.iloc[a][feats], yd.iloc[a])
            out.append(r2_score(yd.iloc[b], pipe.predict(Xd.iloc[b][feats])))
        return np.array(out)

    claves = list(dimensiones)
    with tracking.run("03_ablation_hipotesis"):
        completo = r2_folds(features)
        sin = {k: r2_folds([f for f in features if f not in dimensiones[k]]) for k in claves}
        solo = {k: r2_folds(dimensiones[k]) for k in claves}
        x4 = "X4_Economica_Historica"
        inc = {k: r2_folds(dimensiones[x4] + dimensiones[k]) for k in claves if k != x4}

        filas = []
        p_sin = [_wilcoxon_mayor(completo, sin[k]) for k in claves]
        p_sin_h = _holm(p_sin)
        p_inc = [_wilcoxon_mayor(inc[k], solo[x4]) for k in inc]
        p_inc_h = _holm(p_inc)
        for i, k in enumerate(claves):
            fila = {
                "Dimension": k, "R2_completo": completo.mean(), "R2_sin_dim": sin[k].mean(),
                "Diferencia_R2": completo.mean() - sin[k].mean(),
                "p_valor": p_sin[i], "p_holm": p_sin_h[i],
                "Significativa_estadistica": "Sí" if p_sin_h[i] < ALPHA else "No",
                "R2_solo_dim": solo[k].mean(),
            }
            if k in inc:
                j = list(inc).index(k)
                fila.update({"R2_X4_mas_dim": inc[k].mean(), "Aporte_incremental_R2": inc[k].mean() - solo[x4].mean(),
                             "p_incremental": p_inc[j], "p_incremental_holm": p_inc_h[j]})
            else:
                fila.update({"R2_X4_mas_dim": solo[x4].mean(), "Aporte_incremental_R2": np.nan,
                             "p_incremental": np.nan, "p_incremental_holm": np.nan})
            filas.append(fila)
            print(f"[Hipótesis] Sin {k}: R²={sin[k].mean():.4f} | Δ={fila['Diferencia_R2']:+.4f} | "
                  f"p={p_sin[i]:.4f} | p_Holm={p_sin_h[i]:.4f} | solo-dim R²={solo[k].mean():.4f}")
        tabla = pd.DataFrame(filas)
        tabla.to_csv(os.path.join(OUT_TAB, "tabla_ablation_wilcoxon.csv"), index=False)
        tracking.metrics({"r2_completo_cv": completo.mean()})
        tracking.artifact(os.path.join(OUT_TAB, "tabla_ablation_wilcoxon.csv"))
    return {"tabla": tabla, "completo": completo, "n_splits": n_splits, "folds": {
        "completo": completo.tolist(), **{f"sin_{k}": v.tolist() for k, v in sin.items()}}}
