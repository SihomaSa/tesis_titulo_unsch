"""
cierre_observaciones.py — Tres análisis que cierran puntos abiertos del informe del jurado.

  A) Robustez de la significancia SOLO con registros que tienen precio previo real
     (la "persistencia" de los 46 registros sin historial es una mediana de respaldo,
     no un baseline válido).                                           -> Obs. 3
  B) Variante X2 rezagada / sin X2 sobre el MODELO FINAL (LightGBM sobre persistencia),
     que es el que se despliega.                                       -> Obs. 8
  C) ΔMAE de cada dimensión por fold de la ablación y relación con el % de registros con
     historial en cada fold (contrasta la hipótesis CV vs prueba para X3). -> Obs. 2 y 9

Ubicación: backend/src/analysis/cierre_observaciones.py
No modifica el modelo final ni la selección. Solo escribe tablas en outputs/tables/.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.analysis import significancia as sg
from src.analysis.robustez import X2_PREV, _params, params_residual
from src.config import N_SPLITS_ABLATION, OUT_TAB
from src.models.pipelines import build_pipeline


# ---------------------------------------------------------------------------
# A) Robustez de la significancia con historial real
# ---------------------------------------------------------------------------
def robustez_con_historial(periodo, y, pred_modelo, refs: dict, nombre_modelo, tiene_hist):
    """Repite pérdida absoluta/cuadrática, L-HAC, bloques y prueba de signos en el subconjunto
    con historial. Escribe tabla_significancia_robustez_con_historial.csv."""
    m = np.asarray(tiene_hist).astype(bool)
    per, yy, pm = np.asarray(periodo)[m], np.asarray(y)[m], np.asarray(pred_modelo)[m]
    refs_m = {k: np.asarray(v)[m] for k, v in refs.items()}

    filas = []
    for ref, pr in refs_m.items():
        for tipo in ("abs", "sq"):
            d = sg.perdida_mensual(per, yy, pr, pm, tipo)
            T = len(d)
            ganados = int((d > 0).sum())
            p_signo = float(stats.binomtest(ganados, T, 0.5, alternative="greater").pvalue)
            for L in (1, 2, 3):
                h = sg.hac_media(d, L=L)
                for b in (2, 3, 4):
                    lo, hi, _ = sg.bootstrap_bloques_media(d, bloque=b)
                    filas.append({"Subconjunto": f"Solo con historial (n={int(m.sum())})",
                                  "Modelo": nombre_modelo, "Referencia": ref,
                                  "Perdida": "absoluta" if tipo == "abs" else "cuadrática",
                                  "L_HAC": L, "bloque_bootstrap": b, "n_meses": T,
                                  "meses_modelo_mejor": ganados, "p_signos_exacto": p_signo,
                                  "Dif_media_mensual": h["media"], "p_unilateral_HAC": h["p_unilateral"],
                                  "IC95_inf": lo, "IC95_sup": hi, "IC_excluye_0": "Sí" if lo > 0 else "No"})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_significancia_robustez_con_historial.csv"), index=False)
    res = (t.groupby(["Referencia", "Perdida"])
             .agg(meses_mejor=("meses_modelo_mejor", "first"), n=("n_meses", "first"),
                  p_signos=("p_signos_exacto", "first"),
                  p_HAC_max=("p_unilateral_HAC", "max"), p_HAC_min=("p_unilateral_HAC", "min"),
                  IC_excluye_0_en=("IC_excluye_0", lambda s: f"{(s == 'Sí').sum()}/{len(s)}")).reset_index())
    print(res.round(4).to_string(index=False))
    return t


# ---------------------------------------------------------------------------
# B) X2 rezagada / sin X2 sobre el MODELO FINAL (residual)
# ---------------------------------------------------------------------------
def variante_x2_modelo_final(df, y, idx_dev, idx_te, features, dimensiones, p_res, rapido=False):
    """Mismo diseño que robustez.variante_x2_rezagada, pero con LightGBM sobre persistencia y sus
    hiperparámetros ajustados (p_res = res['mejores_params']['lgbm_res'])."""
    faltan = [c for c in X2_PREV if c not in df.columns]
    if faltan:
        print(f"[Obs.8] Faltan columnas {faltan}.")
        return None
    prm = params_residual(p_res, rapido)
    yt = y.iloc[idx_te].to_numpy()
    per = df["PERIODO"].iloc[idx_te].to_numpy()
    x2 = dimensiones["X2_Productiva"]
    esc = {"Original (X2 del mismo mes)": features,
           "X2 rezagada (observación previa)": [f for f in features if f not in x2] + X2_PREV,
           "Sin X2": [f for f in features if f not in x2]}

    def ajusta(feats):
        Xv = df[feats].astype(float)
        pipe = build_pipeline("lgbm_res", feats, **prm)
        pipe.fit(Xv.iloc[idx_dev], df["PRECIO_CHACRA"].astype(float).iloc[idx_dev])
        return pipe.predict(Xv.iloc[idx_te])

    preds = {n: ajusta(fs) for n, fs in esc.items()}
    base = preds["Original (X2 del mismo mes)"]
    filas = []
    for n, p in preds.items():
        fila = {"Modelo": "LightGBM sobre persistencia (modelo final)", "Escenario": n,
                "n_variables": len(esc[n]), "R2": r2_score(yt, p), "MAE": mean_absolute_error(yt, p)}
        if n != "Original (X2 del mismo mes)":
            d = sg.perdida_mensual(per, yt, p, base)          # > 0 => el original es mejor que la variante
            h = sg.hac_media(d)
            lo, hi, _ = sg.bootstrap_bloques_media(d)
            fila.update({"Dif_MAE_vs_original": mean_absolute_error(yt, p) - mean_absolute_error(yt, base),
                         "Dif_mensual_original_mejor": h["media"], "IC95_inf": lo, "IC95_sup": hi,
                         "p_HAC_original_mejor": h["p_unilateral"]})
        filas.append(fila)
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_variante_x2_modelo_final.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


# ---------------------------------------------------------------------------
# C) ΔMAE por fold de la ablación (LightGBM directo) y % con historial
# ---------------------------------------------------------------------------
def ablacion_por_fold(df, y, idx_dev, features, dimensiones, p_lgbm, rapido=False):
    """Para cada fold de la MISMA validación temporal de run_ablation, calcula el ΔMAE
    (sin dimensión − completo; > 0 => la dimensión ayuda) y el % de registros de validación con
    historial. Resume con Spearman entre ΔMAE y % con historial, y con el promedio sin el fold 1.

    Pasa dimensiones_x3_separada(dimensiones) para ver territorio y producto por separado."""
    n_splits = 4 if rapido else N_SPLITS_ABLATION
    prm = _params(p_lgbm, rapido)
    Xd = df[features].astype(float).iloc[idx_dev]
    yd = y.iloc[idx_dev]
    th = df["TIENE_HISTORIAL"].iloc[idx_dev].to_numpy()
    tss = TimeSeriesSplit(n_splits=n_splits)

    def mae_folds(feats):
        out = []
        for a, b in tss.split(Xd):
            pipe = build_pipeline("lgbm", feats, **prm)
            pipe.fit(Xd.iloc[a][feats], yd.iloc[a])
            out.append(mean_absolute_error(yd.iloc[b], pipe.predict(Xd.iloc[b][feats])))
        return np.array(out)

    pct_hist = np.array([th[b].mean() * 100 for _, b in tss.split(Xd)])
    comp = mae_folds(features)
    filas, resumen = [], []
    for k, cols in dimensiones.items():
        sin = mae_folds([f for f in features if f not in cols])
        dif = sin - comp
        for i, (c, s, d, p) in enumerate(zip(comp, sin, dif, pct_hist), start=1):
            filas.append({"Dimension": k, "Fold": i, "pct_valid_con_historial": p,
                          "MAE_completo": c, "MAE_sin_dim": s, "Dif_MAE": d})
        rho, p_rho = (stats.spearmanr(pct_hist, dif) if len(dif) > 3 else (np.nan, np.nan))
        resumen.append({"Dimension": k, "Dif_MAE_media_todos_folds": dif.mean(),
                        "Dif_MAE_media_sin_fold1": dif[1:].mean(),
                        "folds_donde_ayuda": int((dif > 0).sum()), "n_folds": len(dif),
                        "Spearman_dif_vs_pct_historial": rho, "p_Spearman": p_rho})
    pd.DataFrame(filas).to_csv(os.path.join(OUT_TAB, "tabla_ablacion_por_fold.csv"), index=False)
    r = pd.DataFrame(resumen)
    r.to_csv(os.path.join(OUT_TAB, "tabla_ablacion_por_fold_resumen.csv"), index=False)
    print(r.round(4).to_string(index=False))
    return pd.DataFrame(filas), r
