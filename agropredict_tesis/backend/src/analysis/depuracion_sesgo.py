"""
depuracion_sesgo.py — Obs. 10 del informe del jurado.

  (a) tabla_motivos_exclusion.csv     por qué se descartó cada registro (mismo orden del embudo)
  (b) tabla_sesgo_depuracion.csv      cultivo / provincia / año: % antes vs % después y tasa de retención
      tabla_sesgo_chi2.csv            ¿la exclusión depende de la categoría? (chi², V de Cramér)
  (c) tabla_sensibilidad_depuracion.csv  modelo y baseline con la regla estricta vs relajada
      (COSECHA=0 y PRODUCCION>0 se CONSERVA: puede ser cosecha de otro mes/campaña, no un error)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config import OUT_TAB, PROVINCIAS_PIURA, TARGET
from src.data.data_loader import load_and_clean
from src.data.feature_engineering import build_features
from src.data.splits import split_cronologico
from src.models.evaluacion import baseline_persistencia, metricas
from src.models.pipelines import build_pipeline


def diagnostico(path):
    raw = pd.read_csv(path, low_memory=False)
    raw["PROVINCIA"] = raw["PROVINCIA"].astype(str).str.strip().str.upper()
    raw = raw[raw["PROVINCIA"].isin(PROVINCIAS_PIURA)].copy()
    p = raw[TARGET]
    x2_falta = raw[["COSECHA", "PRODUCCION", "SIEMBRA"]].isna().any(axis=1)
    incons = (raw["COSECHA"] == 0) & (raw["PRODUCCION"] > 0)
    motivo = np.select([p.isna(), p <= 0, x2_falta, incons],
                       ["Sin precio informado", "Precio = 0 (sin precio registrado)",
                        "Faltan SIEMBRA/COSECHA/PRODUCCION", "COSECHA=0 y PRODUCCION>0"], default="Retenido")
    raw["MOTIVO"] = motivo
    ret = raw["MOTIVO"] == "Retenido"
    dup = raw[ret].duplicated(keep="first")
    raw.loc[dup[dup].index, "MOTIVO"] = "Duplicado exacto"
    raw["RETENIDO"] = (raw["MOTIVO"] == "Retenido").astype(int)

    m = raw["MOTIVO"].value_counts().rename_axis("Motivo").reset_index(name="n")
    m["Pct"] = m["n"] / m["n"].sum() * 100
    m.to_csv(os.path.join(OUT_TAB, "tabla_motivos_exclusion.csv"), index=False)

    filas, chi = [], []
    for var in ("CULTIVO", "PROVINCIA", "ANO"):
        g = raw.groupby(var)["RETENIDO"].agg(n_inicial="size", n_final="sum").reset_index().rename(columns={var: "Categoria"})
        g["Variable"] = var
        g["pct_inicial"] = g["n_inicial"] / g["n_inicial"].sum() * 100
        g["pct_final"] = g["n_final"] / g["n_final"].sum() * 100
        g["dif_pp"] = g["pct_final"] - g["pct_inicial"]
        g["tasa_retencion_pct"] = g["n_final"] / g["n_inicial"] * 100
        filas.append(g)
        tab = pd.crosstab(raw[var], raw["RETENIDO"])
        c2, pv, _, _ = chi2_contingency(tab)
        v = np.sqrt(c2 / (tab.values.sum() * (min(tab.shape) - 1)))
        chi.append({"Variable": var, "chi2": c2, "p": pv, "V_Cramer": v, "categorias": tab.shape[0]})
    s = pd.concat(filas)[["Variable", "Categoria", "n_inicial", "n_final", "pct_inicial", "pct_final", "dif_pp", "tasa_retencion_pct"]]
    s.sort_values(["Variable", "dif_pp"], ascending=[True, False]).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_depuracion.csv"), index=False)
    pd.DataFrame(chi).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_chi2.csv"), index=False)
    print(m.to_string(index=False))
    print(pd.DataFrame(chi).round(4).to_string(index=False))
    return m, s, pd.DataFrame(chi)


def sensibilidad(path):
    """Mismo modelo (LightGBM sobre persistencia, parámetros por defecto) con regla estricta vs relajada."""
    filas = []
    for etiqueta, relajar in (("Regla estricta (la del estudio)", False), ("Se conserva COSECHA=0 y PRODUCCION>0", True)):
        df, _ = load_and_clean(path, relajar_cosecha=relajar, guardar=False)
        df, feats, _, _ = build_features(df, guardar_catalogo=False)
        X, y = df[feats].astype(float), df[TARGET].astype(float)
        itr, iva, ite, _ = split_cronologico(df)
        dev = np.concatenate([itr, iva])
        mod = build_pipeline("lgbm_res", feats).fit(X.iloc[dev], y.iloc[dev])
        pm = mod.predict(X.iloc[ite])
        pb = baseline_persistencia(X.iloc[ite], float(y.iloc[dev].median()))
        mm, mb = metricas(y.iloc[ite], pm), metricas(y.iloc[ite], pb)
        filas.append({"Escenario": etiqueta, "n_registros": len(df), "n_prueba": len(ite),
                      "R2_modelo": mm["R2"], "MAE_modelo": mm["MAE"], "R2_persistencia": mb["R2"],
                      "MAE_persistencia": mb["MAE"], "Reduccion_MAE_pct": (mb["MAE"] - mm["MAE"]) / mb["MAE"] * 100})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_sensibilidad_depuracion.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t
