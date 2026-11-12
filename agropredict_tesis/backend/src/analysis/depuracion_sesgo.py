"""
depuracion_sesgo.py — Obs. 10 del informe del jurado.

  (a) tabla_motivos_exclusion.csv     por qué se descartó cada registro (mismo orden del embudo)
  (b) tabla_sesgo_depuracion.csv      cultivo / provincia / año: % antes vs % después y tasa de retención
      tabla_sesgo_chi2.csv            ¿la exclusión depende de la categoría? (chi², V de Cramér)
      tabla_sesgo_chi2_regla_cosecha.csv  lo mismo, aislando solo la regla de cosecha
      tabla_6a_provincias.csv / tabla_6b_anios.csv / tabla_6c_cultivos_top10.csv   tablas listas para la tesis
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


def _chi2(tab):
    """chi², p, V de Cramér y % de celdas con frecuencia esperada < 5 (validez de la prueba)."""
    c2, pv, _, esp = chi2_contingency(tab)
    v = np.sqrt(c2 / (tab.values.sum() * (min(tab.shape) - 1)))
    return c2, pv, v, float((esp < 5).mean() * 100)


def tablas_6abc(s, top=10):
    """Tablas 6a (provincias), 6b (años) y 6c (cultivos: los `top` que más pierden y más ganan peso).

    Devuelve los rangos de diferencia (pp) que se citan en el párrafo de la sección 4.1.2.
    """
    cols = ["Categoria", "pct_inicial", "pct_final", "dif_pp", "tasa_retencion_pct"]
    t6a = s[s["Variable"] == "PROVINCIA"][cols].sort_values("dif_pp", ascending=False)
    t6b = s[s["Variable"] == "ANO"][cols].sort_values("Categoria")
    cult = s[s["Variable"] == "CULTIVO"][cols].sort_values("dif_pp")
    t6c = pd.concat([cult.head(top).assign(Grupo="Pierden peso"),
                     cult.tail(top).iloc[::-1].assign(Grupo="Ganan peso")])
    for nombre, t in (("tabla_6a_provincias", t6a), ("tabla_6b_anios", t6b), ("tabla_6c_cultivos_top10", t6c)):
        t.round(1).to_csv(os.path.join(OUT_TAB, nombre + ".csv"), index=False)
    rangos = {k: (float(t["dif_pp"].min()), float(t["dif_pp"].max()))
              for k, t in (("provincias", t6a), ("anios", t6b))}
    rangos["cultivos_ganan"] = cult.tail(top)["Categoria"].astype(str).tolist()[::-1]
    rangos["cultivos_pierden"] = cult.head(top)["Categoria"].astype(str).tolist()
    print("[Obs.10] Rangos para el párrafo (pp):")
    print(f"  Provincias: {rangos['provincias'][0]:.1f} a {rangos['provincias'][1]:.1f}")
    print(f"  Años:       {rangos['anios'][0]:.1f} a {rangos['anios'][1]:.1f}")
    print(f"  Cultivos que ganan peso:   {', '.join(rangos['cultivos_ganan'][:5])}")
    print(f"  Cultivos que pierden peso: {', '.join(rangos['cultivos_pierden'][:5])}")
    return rangos


def diagnostico(path, n_esperado=None):
    """`n_esperado`: tamaño de la muestra de modelado (p. ej. 14393) para verificar que el embudo coincide."""
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

    # Control: si no coincide con la muestra de modelado, las tablas describen otra población.
    if n_esperado is not None:
        assert int(raw["RETENIDO"].sum()) == n_esperado, (
            f"Retenidos en el diagnóstico = {int(raw['RETENIDO'].sum()):,} vs muestra de modelado = {n_esperado:,}")

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
        c2, pv, v, bajas = _chi2(pd.crosstab(raw[var], raw["RETENIDO"]))
        chi.append({"Variable": var, "chi2": c2, "p": pv, "V_Cramer": v, "categorias": g.shape[0],
                    "pct_celdas_esp_menor_5": bajas})
    s = pd.concat(filas)[["Variable", "Categoria", "n_inicial", "n_final", "pct_inicial", "pct_final", "dif_pp", "tasa_retencion_pct"]]
    # Efecto ESPECÍFICO de la regla de cosecha: entre los registros que SÍ tienen precio > 0 y X2 completa,
    # ¿qué cultivos/provincias/años pierde la regla COSECHA=0 y PRODUCCION>0?  (aísla el filtro discutible
    # del 69 % de 'sin precio', que es estructural y no una decisión del investigador).
    # Se excluyen los duplicados: no son efecto de la regla y, si se dejaran, contarían como "no retenidos".
    base = raw[raw["MOTIVO"].isin(["Retenido", "COSECHA=0 y PRODUCCION>0"])].copy()
    chi_r = []
    for var in ("CULTIVO", "PROVINCIA", "ANO"):
        tab = pd.crosstab(base[var], base["RETENIDO"])
        c2, pv, v, bajas = _chi2(tab)
        chi_r.append({"Variable": var, "chi2": c2, "p": pv, "V_Cramer": v, "categorias": tab.shape[0],
                      "pct_celdas_esp_menor_5": bajas, "n_base": int(len(base)),
                      "alcance": "Solo regla de cosecha (entre registros con precio>0, X2 completa y sin duplicados)"})
    pd.DataFrame(chi_r).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_chi2_regla_cosecha.csv"), index=False)
    print("[Obs.10] Sesgo atribuible SOLO a la regla de cosecha:")
    print(pd.DataFrame(chi_r).round(4).to_string(index=False))
    s.sort_values(["Variable", "dif_pp"], ascending=[True, False]).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_depuracion.csv"), index=False)
    pd.DataFrame(chi).to_csv(os.path.join(OUT_TAB, "tabla_sesgo_chi2.csv"), index=False)
    print(m.to_string(index=False))
    print(pd.DataFrame(chi).round(4).to_string(index=False))
    tablas_6abc(s)
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