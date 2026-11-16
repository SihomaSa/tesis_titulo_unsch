"""
ablation_study.py — Contraste de las hipótesis específicas H1–H4.

Se evalúan, con MesesExpansivo (v2) o TimeSeriesSplit (anterior), tres familias
de modelos con los hiperparámetros ya optimizados:

  A) Completo vs. SIN cada dimensión (leave-one-dimension-out)  -> ΔR², ΔMAE y Wilcoxon
  B) SOLO cada dimensión                                          -> poder predictivo propio
  C) X4 (historia de precios) vs. X4 + cada dimensión            -> aporte INCREMENTAL
"""
import os
import sys


def _forzar_utf8():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_forzar_utf8()

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src import tracking
from src.config import ALPHA, N_SPLITS_ABLATION, OUT_TAB
from src.models.pipelines import build_pipeline
from src.data.splits import MesesExpansivo

X4 = "X4_Economica_Historica"


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


def run_ablation(X, y, idx_dev, features, dimensiones, params_modelo, rapido=False,
                 clave="rf", sufijo="", nombre_modelo=None, mes_idx=None, gap=0):
    n_splits = 4 if rapido else N_SPLITS_ABLATION
    Xd, yd = X.iloc[idx_dev], y.iloc[idx_dev]
    
    # FIX: Elegir el esquema de CV correcto sin sobrescribirlo después
    if mes_idx is None:
        tss = TimeSeriesSplit(n_splits=n_splits)
    else:
        tss = MesesExpansivo(np.asarray(mes_idx)[idx_dev], n_splits=n_splits, gap=gap)
    
    params = {f"model__{k}": v for k, v in params_modelo.items()}
    if clave == "rf":
        params["model__n_estimators"] = min(int(params.get("model__n_estimators", 150)), 60 if rapido else 150)
    elif rapido and "model__n_estimators" in params:
        params["model__n_estimators"] = min(int(params["model__n_estimators"]), 100)

    def folds(feats):
        r2, mae = [], []
        for a, b in tss.split(Xd):
            Xa = Xd.iloc[a][feats]
            if not Xa.notna().any().any():      # guarda para folds vacíos (h=12)
                p = np.full(len(b), float(yd.iloc[a].median()))
            else:
                pipe = build_pipeline(clave, feats, **params)
                pipe.fit(Xa, yd.iloc[a])
                p = pipe.predict(Xd.iloc[b][feats])
            r2.append(r2_score(yd.iloc[b], p))
            mae.append(mean_absolute_error(yd.iloc[b], p))
        return np.array(r2), np.array(mae)

    claves = list(dimensiones)
    archivo = f"tabla_ablation_wilcoxon{sufijo}.csv"
    with tracking.run(f"03_ablation_hipotesis{sufijo}"):
        comp_r2, comp_mae = folds(features)
        sin = {k: folds([f for f in features if f not in dimensiones[k]]) for k in claves}
        solo = {k: folds(dimensiones[k]) for k in claves}
        inc = {k: folds(dimensiones[X4] + dimensiones[k]) for k in claves if k != X4}

        p_sin = [_wilcoxon_mayor(comp_r2, sin[k][0]) for k in claves]
        p_sin_h = _holm(p_sin)
        p_mae = [_wilcoxon_mayor(sin[k][1], comp_mae) for k in claves]
        p_mae_h = _holm(p_mae)
        p_inc = [_wilcoxon_mayor(inc[k][0], solo[X4][0]) for k in inc]
        p_inc_h = _holm(p_inc)

        filas = []
        for i, k in enumerate(claves):
            fila = {
                "Dimension": k, "R2_completo": comp_r2.mean(), "R2_sin_dim": sin[k][0].mean(),
                "Diferencia_R2": comp_r2.mean() - sin[k][0].mean(),
                "p_valor": p_sin[i], "p_holm": p_sin_h[i],
                "Significativa_estadistica": "Sí" if p_sin_h[i] < ALPHA else "No",
                "R2_solo_dim": solo[k][0].mean(),
            }
            if k in inc:
                j = list(inc).index(k)
                fila.update({"R2_X4_mas_dim": inc[k][0].mean(),
                             "Aporte_incremental_R2": inc[k][0].mean() - solo[X4][0].mean(),
                             "p_incremental": p_inc[j], "p_incremental_holm": p_inc_h[j]})
            else:
                fila.update({"R2_X4_mas_dim": solo[X4][0].mean(), "Aporte_incremental_R2": np.nan,
                             "p_incremental": np.nan, "p_incremental_holm": np.nan})
            fila.update({"MAE_completo": comp_mae.mean(), "MAE_sin_dim": sin[k][1].mean(),
                         "Diferencia_MAE": sin[k][1].mean() - comp_mae.mean(),
                         "p_mae": p_mae[i], "p_mae_holm": p_mae_h[i],
                         "Significativa_MAE": "Sí" if p_mae_h[i] < ALPHA else "No"})
            filas.append(fila)

            print(
                f"[Hipotesis{sufijo}] Sin {k}: R2={sin[k][0].mean():.4f} | "
                f"DeltaR2={fila['Diferencia_R2']:+.4f} | p_Holm={p_sin_h[i]:.4f} | "
                f"DeltaMAE={fila['Diferencia_MAE']:+.4f} (p_Holm={p_mae_h[i]:.4f}) | "
                f"solo-dim R2={solo[k][0].mean():.4f}"
            )

        tabla = pd.DataFrame(filas)
        tabla.to_csv(os.path.join(OUT_TAB, archivo), index=False)
        tracking.metrics({f"r2_completo_cv{sufijo}": comp_r2.mean()})
        tracking.artifact(os.path.join(OUT_TAB, archivo))
    return {"tabla": tabla, "completo": comp_r2, "n_splits": n_splits, "clave": clave, "archivo": archivo,
            "folds": {"completo": comp_r2.tolist(), **{f"sin_{k}": v[0].tolist() for k, v in sin.items()}},
            "folds_mae": {"completo": comp_mae.tolist(),                       # <- NUEVO
                          **{f"sin_{k}": v[1].tolist() for k, v in sin.items()}}}  # <- NUEVO