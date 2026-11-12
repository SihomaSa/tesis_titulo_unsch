"""
interpretabilidad.py — FASE 8 (v2): SHAP por dimensión y permutación agrupada.

  * SHAP del modelo final (sobre persistencia): explica log(precio/último precio).
  * SHAP del modelo directo (LightGBM): explica el precio en S/ por kg.
  Ninguno mide efecto causal ni 'capacidad predictiva' del precio: cada uno se rotula con su escala.
  % por dimensión: (i) suma de mean|SHAP| por variable y (ii) SHAP agrupado, mean|Σ SHAP| de la dimensión
  (el agrupado es el correcto cuando hay variables correlacionadas dentro de la dimensión).
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import OUT_TAB, RANDOM_STATE
from src.models.pipelines import importancias_modelo, pipeline_arbol


def _predecir_con_ancla(modelo, Xp, X_orig):
    """Predice con las columnas permutadas en Xp, manteniendo FIJO el ancla de persistencia
    (PRECIO_LAG1 y la máscara de historial de X_orig): el ancla NO se permuta junto con X4."""
    if not hasattr(modelo, "base_"):
        return np.asarray(modelo.predict(Xp), float)
    out = np.asarray(modelo.fallback_.predict(Xp), float)
    m = modelo._mascara(X_orig)
    if m.any():
        lag = X_orig["PRECIO_LAG1"].to_numpy()
        out[m] = lag[m] * np.exp(modelo.base_.predict(Xp[m]))
    return out


def _ic(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def permutacion_agrupada(modelo, X, y, dimensiones, n_rep=30, conjunto="validación"):
    rng = np.random.RandomState(RANDOM_STATE)
    base_pred = _predecir_con_ancla(modelo, X, X)
    base_r2, base_mae = r2_score(y, base_pred), mean_absolute_error(y, base_pred)
    filas = []
    for dim, cols in dimensiones.items():
        dr2, dmae = [], []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]    # permuta el grupo completo, en bloque
            p = _predecir_con_ancla(modelo, Xp, X)
            dr2.append(base_r2 - r2_score(y, p))
            dmae.append(mean_absolute_error(y, p) - base_mae)
        dr2, dmae = np.array(dr2), np.array(dmae)
        filas.append({"Dimension": dim,
                      "caida_R2_permutacion": dr2.mean(), "caida_R2_IC95_inf": _ic(dr2)[0], "caida_R2_IC95_sup": _ic(dr2)[1],
                      "aumento_MAE_permutacion": dmae.mean(), "aumento_MAE_IC95_inf": _ic(dmae)[0], "aumento_MAE_IC95_sup": _ic(dmae)[1],
                      "n_repeticiones": n_rep, "conjunto": conjunto, "n_registros": len(X),
                      "ancla_persistencia_fija": hasattr(modelo, "base_"), "ancla_se_permuta_con_X4": False,
                      "nota_IC": "Percentiles 2,5–97,5 de las repeticiones (variabilidad de la permutación, no de la muestra)"})
    return pd.DataFrame(filas)


def _resumen_shap(sv, features, dimensiones, B=1000, seed=RANDOM_STATE):
    sv = np.asarray(sv, float)
    pos = {f: i for i, f in enumerate(features)}
    cols = {d: [pos[c] for c in cs] for d, cs in dimensiones.items()}

    def calc(s):
        a = np.array([np.abs(s[:, c]).mean(axis=0).sum() for c in cols.values()])
        b = np.array([np.abs(s[:, c].sum(axis=1)).mean() for c in cols.values()])
        return a / max(a.sum(), 1e-12), b / max(b.sum(), 1e-12)

    a0, b0 = calc(sv)
    rng, n = np.random.default_rng(seed), len(sv)
    A, Bm = np.empty((B, len(cols))), np.empty((B, len(cols)))
    for i in range(B):
        A[i], Bm[i] = calc(sv[rng.integers(0, n, n)])
    filas = []
    for j, d in enumerate(cols):
        filas.append({"Dimension": d, "n_variables": len(cols[d]),
                      "Pct_suma_mean_abs": a0[j] * 100, "Pct_suma_IC95_inf": _ic(A[:, j] * 100)[0], "Pct_suma_IC95_sup": _ic(A[:, j] * 100)[1],
                      "Pct_agrupado": b0[j] * 100, "Pct_agrupado_IC95_inf": _ic(Bm[:, j] * 100)[0], "Pct_agrupado_IC95_sup": _ic(Bm[:, j] * 100)[1]})
    return pd.DataFrame(filas)


def shap_por_dimension(modelo, X, dimensiones, features, max_n=1500, conjunto="prueba"):
    try:
        import sys
        sys.modules.setdefault("cv2", None)
        try:
            import shap
        except ImportError:
            sys.modules.pop("cv2", None)
            import shap
        residual = hasattr(modelo, "base_")
        if residual:
            X = X[modelo._mascara(X)]
        pipe = pipeline_arbol(modelo)
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_") and not hasattr(est, "booster_"):
            return None
        Xs = X.sample(min(max_n, len(X)), random_state=RANDOM_STATE)
        Xi = pd.DataFrame(pipe.named_steps["imp"].transform(Xs), columns=features)
        sv = shap.TreeExplainer(est).shap_values(Xi)
        s = pd.Series(np.abs(np.asarray(sv)).mean(axis=0), index=features)
        res = _resumen_shap(sv, features, dimensiones)
        escala = ("log(precio / último precio): corrección respecto de la persistencia" if residual else "precio (S/ por kg)")
        return {"por_variable": (s / s.sum()).sort_values(ascending=False),
                "por_dimension": {d: float(s[c].sum() / s.sum()) for d, c in dimensiones.items()},
                "por_dimension_agrupado": {r.Dimension: float(r.Pct_agrupado / 100) for r in res.itertuples()},
                "resumen": res, "valores": sv, "X": Xi,
                "n_subconjunto": int(len(X)), "n_muestra": int(len(Xs)), "escala": escala,
                "explainer": "shap.TreeExplainer (LightGBM)", "conjunto": conjunto, "semilla": RANDOM_STATE}
    except Exception as e:
        print(f"[Fase 8] SHAP no disponible ({e}); se usa solo permutación agrupada.")
        return None


def interpretar(res, X, y, idx_va, idx_te, features, dimensiones):
    modelo_tr = res["modelos_train"][res["clave_sel"]]
    perm = permutacion_agrupada(modelo_tr, X.iloc[idx_va], y.iloc[idx_va], dimensiones)
    imp = pd.Series(importancias_modelo(res["modelo_final"], features), index=features).sort_values(ascending=False)
    imp_dim = {d: float(imp[c].sum()) for d, c in dimensiones.items()}
    shp = shap_por_dimension(res["modelo_final"], X.iloc[idx_te], dimensiones, features)
    directo = res.get("modelos_finales", {}).get("lgbm")
    shp_dir = shap_por_dimension(directo, X.iloc[idx_te], dimensiones, features) if directo is not None else None

    perm.to_csv(os.path.join(OUT_TAB, "tabla_permutacion_por_dimension.csv"), index=False)
    top = (shp["por_variable"] if shp else imp).head(10)
    top.rename("importancia").to_csv(os.path.join(OUT_TAB, "tabla_importancia_top10.csv"))
    tablas = []
    for nombre, s in (("LightGBM sobre persistencia (modelo final)", shp), ("LightGBM directo", shp_dir)):
        if s:
            t = s["resumen"].copy()
            t.insert(0, "Modelo", nombre)
            for k in ("escala", "explainer", "conjunto", "n_subconjunto", "n_muestra", "semilla"):
                t[k] = s[k]
            tablas.append(t)
    if tablas:
        pd.concat(tablas).to_csv(os.path.join(OUT_TAB, "tabla_shap_por_dimension_ic.csv"), index=False)

    print("[Fase 8] Importancia por dimensión (ganancia): " + " | ".join(f"{d[:2]}={v*100:.1f}%" for d, v in imp_dim.items()))
    for nombre, s in (("final", shp), ("directo", shp_dir)):
        if s:
            print(f"[Fase 8] SHAP {nombre} [{s['escala']}]: " + " | ".join(f"{d[:2]}={v*100:.1f}%" for d, v in s["por_dimension"].items()))
    return {"importancia": imp, "imp_dim": imp_dim, "shap": shp, "shap_directo": shp_dir, "permutacion": perm}