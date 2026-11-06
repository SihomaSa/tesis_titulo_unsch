"""
interpretabilidad.py — FASE 8: SHAP (Lundberg & Lee, 2017) agregado por
dimensión + importancia por permutación AGRUPADA por dimensión. La segunda
es robusta a la correlación entre variables (a diferencia de la ganancia de
los árboles, que se reparte arbitrariamente entre variables correlacionadas).
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import OUT_TAB, RANDOM_STATE
from src.models.pipelines import importancias_modelo, pipeline_arbol


def permutacion_agrupada(modelo, X, y, dimensiones, n_rep=5):
    rng = np.random.RandomState(RANDOM_STATE)
    base_r2 = r2_score(y, modelo.predict(X))
    base_mae = mean_absolute_error(y, modelo.predict(X))
    filas = []
    for dim, cols in dimensiones.items():
        dr2, dmae = [], []
        for _ in range(n_rep):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            p = modelo.predict(Xp)
            dr2.append(base_r2 - r2_score(y, p))
            dmae.append(mean_absolute_error(y, p) - base_mae)
        filas.append({"Dimension": dim, "caida_R2_permutacion": np.mean(dr2), "aumento_MAE_permutacion": np.mean(dmae)})
    return pd.DataFrame(filas)


def shap_por_dimension(modelo, X, dimensiones, features, max_n=1500):
    try:
        import sys
        sys.modules.setdefault("cv2", None)      # evita el ruido de OpenCV/NumPy 2 al importar shap
        try:
            import shap
        except ImportError:
            sys.modules.pop("cv2", None)
            import shap
        if hasattr(modelo, "base_"):               # modelo sobre persistencia: se explica el cambio relativo
            m = modelo._mascara(X)
            X = X[m]
        pipe = pipeline_arbol(modelo)
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_") and not hasattr(est, "booster_"):
            return None
        Xs = X.sample(min(max_n, len(X)), random_state=RANDOM_STATE)
        Xi = pd.DataFrame(pipe.named_steps["imp"].transform(Xs), columns=features)
        sv = shap.TreeExplainer(est).shap_values(Xi)
        media = np.abs(np.asarray(sv)).mean(axis=0)
        s = pd.Series(media, index=features)
        por_dim = {d: float(s[c].sum() / s.sum()) for d, c in dimensiones.items()}
        return {"por_variable": (s / s.sum()).sort_values(ascending=False), "por_dimension": por_dim,
                "valores": sv, "X": Xi}
    except Exception as e:  # SHAP es opcional: no debe tumbar el pipeline
        print(f"[Fase 8] SHAP no disponible ({e}); se usa solo permutación agrupada.")
        return None


def interpretar(res, X, y, idx_va, idx_te, features, dimensiones):
    modelo_tr = res["modelos_train"][res["clave_sel"]]
    perm = permutacion_agrupada(modelo_tr, X.iloc[idx_va], y.iloc[idx_va], dimensiones)
    imp = pd.Series(importancias_modelo(res["modelo_final"], features), index=features).sort_values(ascending=False)
    imp_dim = {d: float(imp[c].sum()) for d, c in dimensiones.items()}
    shp = shap_por_dimension(res["modelo_final"], X.iloc[idx_te], dimensiones, features)
    perm.to_csv(os.path.join(OUT_TAB, "tabla_permutacion_por_dimension.csv"), index=False)
    top = (shp["por_variable"] if shp else imp).head(10)
    top.rename("importancia").to_csv(os.path.join(OUT_TAB, "tabla_importancia_top10.csv"))
    print("[Fase 8] Importancia por dimensión (ganancia): " + " | ".join(f"{d[:2]}={v*100:.1f}%" for d, v in imp_dim.items()))
    if shp:
        print("[Fase 8] SHAP por dimensión: " + " | ".join(f"{d[:2]}={v*100:.1f}%" for d, v in shp["por_dimension"].items()))
    return {"importancia": imp, "imp_dim": imp_dim, "shap": shp, "permutacion": perm}
