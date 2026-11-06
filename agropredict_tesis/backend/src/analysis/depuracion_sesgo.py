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
    # Obs. 10: distinguir registros inválidos de exclusiones estructurales (períodos sin cosecha / sin mercado).
    tipo = {"Sin precio informado": "Dato faltante del objetivo",
            "Precio = 0 (sin precio registrado)": "Estructural: no hubo precio de chacra registrado (el objetivo no existe)",
            "Faltan SIEMBRA/COSECHA/PRODUCCION": "Registro inválido: variables del registro agrícola incompletas",
            "COSECHA=0 y PRODUCCION>0": "Regla de consistencia: posible cosecha de otra campaña/mes (discutible, ver sensibilidad)",
            "Duplicado exacto": "Registro inválido: duplicado",
            "Retenido": "Retenido (muestra de modelado)"}
    m["Tipo"] = m["Motivo"].map(tipo)
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
    # Efecto ESPECÍFICO de la regla de cosecha: entre los registros que SÍ tienen precio > 0 y X2 completa,
    # ¿qué cultivos/provincias/años pierde la regla COSECHA=0 y PRODUCCION>0?  (aísla el filtro discutible
    # del 69 % de 'sin precio', que es estructural y no una decisión del investigador)
    base = raw[raw["MOTIVO"].isin(["Retenido", "Duplicado exacto", "COSECHA=0 y PRODUCCION>0"])].copy()
    chi_r = []
    for var in ("CULTIVO", "PROVINCIA", "ANO"):
        tab = pd.crosstab(base[var], base["RETENIDO"])
        c2, pv, _, _ = chi2_contingency(tab)
        v = np.sqrt(c2 / (tab.values.sum() * (min(tab.shape) - 1)))
        chi_r.append({"Variable": var, "chi2": c2, "p": pv, "V_Cramer": v, "categorias": tab.shape[0],
                      "n_base": int(len(base)), "alcance": "Solo regla de cosecha (entre registros con precio>0 y X2 completa)"})
    pd.DataFrame(chi_r).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_chi2_regla_cosecha.csv"), index=False)
    print("[Obs.10] Sesgo atribuible SOLO a la regla de cosecha:")
    print(pd.DataFrame(chi_r).round(4).to_string(index=False))
    s.sort_values(["Variable", "dif_pp"], ascending=[True, False]).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_depuracion.csv"), index=False)
    pd.DataFrame(chi).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_chi2.csv"), index=False)
    print(m.to_string(index=False))
    print(pd.DataFrame(chi).round(4).to_string(index=False))
    return m, s, pd.DataFrame(chi)


def sensibilidad(path, params_res=None, rapido=False):
    """Mismo modelo (LightGBM sobre persistencia) con regla estricta vs relajada.

    `params_res` = mejores_params['lgbm_res'] (ya ajustados en entrenamiento). Con parámetros por defecto el
    modelo queda PEOR que la persistencia en ambos escenarios, así que esa comparación no dice nada sobre el
    modelo del estudio. Aquí se usan los hiperparámetros ajustados (sin re-ajustar por escenario: limitación
    a declarar) y se reporta además la versión por defecto como referencia.
    """
    from src.analysis.robustez import params_residual
    configs = [("ajustados", params_residual(params_res, rapido) if params_res else {}), ("por defecto", {})]
    filas = []
    for etiqueta, relajar in (("Regla estricta (la del estudio)", False), ("Se conserva COSECHA=0 y PRODUCCION>0", True)):
        df, _ = load_and_clean(path, relajar_cosecha=relajar, guardar=False)
        df, feats, _, _ = build_features(df, guardar_catalogo=False)
        X, y = df[feats].astype(float), df[TARGET].astype(float)
        itr, iva, ite, _ = split_cronologico(df)
        dev = np.concatenate([itr, iva])
        pb = baseline_persistencia(X.iloc[ite], float(y.iloc[dev].median()))
        mb = metricas(y.iloc[ite], pb)
        for nom_cfg, prm in configs:
            if nom_cfg == "ajustados" and not prm:
                continue
            mod = build_pipeline("lgbm_res", feats, **prm).fit(X.iloc[dev], y.iloc[dev])
            mm = metricas(y.iloc[ite], mod.predict(X.iloc[ite]))
            filas.append({"Escenario": etiqueta, "Hiperparametros": nom_cfg, "n_registros": len(df), "n_prueba": len(ite),
                          "R2_modelo": mm["R2"], "MAE_modelo": mm["MAE"], "R2_persistencia": mb["R2"],
                          "MAE_persistencia": mb["MAE"], "Reduccion_MAE_pct": (mb["MAE"] - mm["MAE"]) / mb["MAE"] * 100})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_sensibilidad_depuracion.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t
