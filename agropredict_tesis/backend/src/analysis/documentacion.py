"""
documentacion.py — Obs. 4, 11 y 12 del informe del jurado (compatible con v2: horizonte fijo + brecha).

  tabla_particiones.csv           fechas, tamaños, rezagos y regla de actualización
  tabla_folds.csv                 fechas y tamaños de cada fold (tuning y ablación), con brecha real
  tabla_mae_por_brecha.csv        error según la antigüedad del último precio EN EL ORIGEN (v2) o la brecha (anterior)
  tabla_shap_detalle*.csv         mean|SHAP| por variable, % por dimensión (suma y agrupado) y escala
  tabla_importancia_comparada.csv cada medida rotulada con su modelo y muestra
Sin horizonte (horizonte=None) todo se comporta como en lo ya entregado.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import N_SPLITS_ABLATION, N_SPLITS_TUNING, OUT_TAB
from src.data.splits import MesesExpansivo


def _regla_rezagos(h):
    if h is None:
        return ("Sí: pronóstico de un paso adelante. PRECIO_LAG1/MEDIA_MOVIL_3 se calculan sobre los precios "
                "OBSERVADOS de la serie, también durante la prueba (no se usan predicciones previas).")
    return (f"Sí: pronóstico a h={h} mes(es). Último precio, media móvil, variación y variables productivas usan solo "
            f"observaciones con mes <= (mes objetivo − {h}), también durante la prueba; no se usan predicciones previas.")


def tabla_particiones(df, idx_tr, idx_va, idx_te, horizonte=None):
    h = int(horizonte) if horizonte else None
    filas = []
    for nombre, idx in (("Entrenamiento", idx_tr), ("Validación", idx_va), ("Prueba", idx_te)):
        d = df.iloc[idx]
        con = d["TIENE_HISTORIAL"] == 1
        edad = d.loc[con, "MESES_DESDE_PRECIO_PREVIO"]              # v2: edad EN EL ORIGEN; anterior: distancia al objetivo
        dist = edad + (h or 0)                                       # distancia último precio -> mes objetivo
        filas.append({
            "Conjunto": nombre, "Desde": int(d["PERIODO"].min()), "Hasta": int(d["PERIODO"].max()),
            "n_registros": len(d), "pct_del_total": len(d) / len(df) * 100,
            "meses_distintos": int(d["PERIODO"].nunique()),
            "series_cultivo_distrito": int(d.drop_duplicates(["CULTIVO", "PROVINCIA", "DISTRITO"]).shape[0]),
            "n_con_historial_previo": int(d["TIENE_HISTORIAL"].sum()),
            "pct_con_historial_previo": float(d["TIENE_HISTORIAL"].mean() * 100),
            # se conserva el nombre de columna: ahora es SIEMPRE la distancia del último precio al mes objetivo
            "brecha_mediana_meses_al_precio_previo": float(dist.median()),
            "edad_mediana_precio_en_origen_meses": float(edad.median()) if h else np.nan,
            "horizonte_meses": h if h else np.nan,
            "rezagos_actualizados_con_precio_observado": _regla_rezagos(h),
        })
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_particiones.csv"), index=False)
    print(t.drop(columns="rezagos_actualizados_con_precio_observado").to_string(index=False))
    return t


def tabla_folds(df, idx_tr, idx_dev, rapido=False, horizonte=None):
    """Mismos folds que usan entrenamiento.py (tuning sobre idx_tr) y run_ablation (sobre idx_dev).
    v2: MesesExpansivo (meses completos, brecha h-1). Anterior: TimeSeriesSplit.
    OJO: en v2 hay que pasar idx_dev = info_split['idx_dev'], no np.concatenate([idx_tr, idx_va])."""
    gap = max(int(horizonte) - 1, 0) if horizonte else 0
    esquema = f"MesesExpansivo (meses completos, brecha={gap} m)" if horizonte else "TimeSeriesSplit (por registros)"
    per_all, mes_all = df["PERIODO"].to_numpy(), df["MES_IDX"].to_numpy()
    hist_all = df["TIENE_HISTORIAL"].to_numpy()
    filas = []
    for etapa, idx, k in (("Tuning (solo entrenamiento)", idx_tr, 3 if rapido else N_SPLITS_TUNING),
                          ("Ablación (entrenamiento+validación)", idx_dev, 4 if rapido else N_SPLITS_ABLATION)):
        idx = np.asarray(idx)
        per, mes, hist = per_all[idx], mes_all[idx], hist_all[idx]
        cv = MesesExpansivo(mes, n_splits=k, gap=gap) if horizonte else TimeSeriesSplit(n_splits=k)
        for i, (a, b) in enumerate(cv.split(idx), 1):
            filas.append({"Etapa": etapa, "Fold": i, "esquema_cv": esquema,
                          "n_entrenamiento": len(a), "entrena_desde": int(per[a].min()), "entrena_hasta": int(per[a].max()),
                          "n_validacion": len(b), "valida_desde": int(per[b].min()), "valida_hasta": int(per[b].max()),
                          "meses_entre_fin_entrenamiento_y_validacion": int(mes[b].min() - mes[a].max()),
                          "comparte_mes_limite": bool(per[a].max() >= per[b].min()),
                          "pct_valid_con_historial": float(hist[b].mean() * 100)})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_folds.csv"), index=False)
    print(t.to_string(index=False))
    if t["comparte_mes_limite"].any():
        print("[Obs.11] ATENCIÓN: hay folds que comparten el mes límite entre entrenamiento y validación.")
    return t


def mae_por_brecha(df_te, y_te, pred, pred_persistencia, horizonte=None):
    """Obs. 4. Con horizonte h, MESES_DESDE_PRECIO_PREVIO es la ANTIGÜEDAD del último precio en el origen (>= 0);
    la distancia al objetivo es antigüedad + h. Los grupos cubren TODA la prueba (se verifica que sumen 100 %).
    'Sin historial' no es comparable (la persistencia es la mediana de desarrollo): su reducción queda en NaN."""
    h = int(horizonte) if horizonte else None
    y = np.asarray(y_te, float)
    edad = df_te["MESES_DESDE_PRECIO_PREVIO"].to_numpy(float)
    hist = df_te["TIENE_HISTORIAL"].to_numpy() == 1
    if h is None:                                                    # comportamiento anterior
        grupos = [("Sin historial previo", ~hist),
                  ("Precio previo hace 1 mes", hist & (edad == 1)),
                  ("Hace 2–3 meses", hist & (edad >= 2) & (edad <= 3)),
                  ("Hace 4–12 meses", hist & (edad >= 4) & (edad <= 12)),
                  ("Hace más de 12 meses", hist & (edad > 12))]
    else:
        grupos = [("Sin historial previo", ~hist),
                  (f"Último precio en el origen (distancia {h} m)", hist & (edad == 0)),
                  (f"Último precio con 1–3 m de antigüedad (distancia {h+1}–{h+3} m)", hist & (edad >= 1) & (edad <= 3)),
                  (f"Con 4–12 m de antigüedad (distancia {h+4}–{h+12} m)", hist & (edad >= 4) & (edad <= 12)),
                  (f"Con más de 12 m de antigüedad (distancia > {h+12} m)", hist & (edad > 12))]
    filas = []
    for nombre, m in grupos:
        if m.sum() < 2:
            print(f"[Obs.4] Grupo omitido por n<2: {nombre} (n={int(m.sum())})")
            continue
        mm, mb = mean_absolute_error(y[m], pred[m]), mean_absolute_error(y[m], pred_persistencia[m])
        comparable = nombre != "Sin historial previo"
        filas.append({"Grupo": nombre, "N": int(m.sum()), "Pct": m.mean() * 100,
                      "MAE_modelo": mm, "MAE_persistencia": mb if comparable else np.nan,
                      "Reduccion_MAE_pct": (mb - mm) / mb * 100 if (mb and comparable) else np.nan,
                      "R2_modelo": r2_score(y[m], pred[m]),
                      "Nota": "" if comparable else "Persistencia = mediana de desarrollo (sin último precio); no comparable."})
    t = pd.DataFrame(filas)
    tot = t["N"].sum()
    if tot != len(y):
        print(f"[Obs.4] ATENCIÓN: los grupos cubren {tot} de {len(y)} registros ({tot / len(y) * 100:.1f} %).")
    t.to_csv(os.path.join(OUT_TAB, "tabla_mae_por_brecha.csv"), index=False)
    print(t.drop(columns="Nota").to_string(index=False))
    return t


def shap_detalle(interp, features, dimensiones, clave="shap", sufijo=""):
    """clave='shap' (modelo final, log-ratio) o 'shap_directo' (S/ por kg, sufijo='_directo')."""
    shp = interp.get(clave)
    if not shp:
        print(f"[Obs.12] SHAP '{clave}' no disponible: instala `shap` y vuelve a correr.")
        return None
    sv = np.abs(np.asarray(shp["valores"])).mean(axis=0)
    dim_de = {f: d for d, fs in dimensiones.items() for f in fs}
    t = pd.DataFrame({"Variable": features, "Dimension": [dim_de[f] for f in features], "mean_abs_SHAP": sv})
    t["Pct_total"] = t["mean_abs_SHAP"] / t["mean_abs_SHAP"].sum() * 100
    t = t.sort_values("mean_abs_SHAP", ascending=False)
    t["Pct_acumulado"] = t["Pct_total"].cumsum()
    res = shp.get("resumen")
    if res is not None:                                              # SHAP agrupado e IC bootstrap (interpretabilidad v2)
        r = res.set_index("Dimension")
        for c in ("Pct_agrupado", "Pct_agrupado_IC95_inf", "Pct_agrupado_IC95_sup"):
            t[c] = t["Dimension"].map(r[c])
    t["n_registros_muestra"] = shp.get("n_muestra", len(shp["X"]))
    t["n_subconjunto_con_historial"] = shp.get("n_subconjunto", np.nan)
    t["escala_shap"] = shp.get("escala", "")
    t["explainer"] = shp.get("explainer", "")
    t["semilla"] = shp.get("semilla", "")
    t["formula"] = "Pct_total = mean|SHAP_j| / Σ mean|SHAP|; Pct_dimensión = suma por dimensión. Pct_agrupado = mean|Σ SHAP de la dimensión| (normalizado)"
    t["nota"] = np.where(t["Variable"] == "TIENE_HISTORIAL", "Constante (=1) en el subconjunto explicado: 0 por construcción", "")
    t.to_csv(os.path.join(OUT_TAB, f"tabla_shap_detalle{sufijo}.csv"), index=False)
    print(f"[Obs.12] % por dimensión ({clave}) = Σ mean|SHAP| / Σ total "
          f"(muestra {t['n_registros_muestra'].iloc[0]} de {t['n_subconjunto_con_historial'].iloc[0]}; escala {shp.get('escala')}):")
    print(t.groupby("Dimension")["Pct_total"].sum().round(1).to_string())
    return t


def importancia_comparada(interp, abl, dimensiones, contrib_final=None, esquema_ablacion="TimeSeriesSplit",
                          nombre_modelo_ablacion="LightGBM (directo)", nombre_modelo_final="LightGBM sobre persistencia"):
    """Cada columna declara con QUÉ modelo y en QUÉ muestra se calculó.
    contrib_final: DataFrame de tabla_contribucion_dimensiones_prueba_lgbm_res (aporte en prueba sobre el MODELO FINAL,
    el comparable con ganancia/SHAP/permutación). X3_Geografica queda NaN allí porque el modelo final separa X3a/X3b."""
    ab = abl["tabla"].set_index("Dimension")
    perm = interp["permutacion"].set_index("Dimension")
    cf = None if contrib_final is None else contrib_final.set_index("Dimension")
    shp = interp.get("shap")
    filas = []
    for d in dimensiones:
        filas.append({
            "Dimension": d,
            "Ganancia_pct": interp["imp_dim"][d] * 100,
            "SHAP_pct": shp["por_dimension"][d] * 100 if shp else np.nan,
            "SHAP_agrupado_pct": shp["por_dimension_agrupado"][d] * 100 if shp and shp.get("por_dimension_agrupado") else np.nan,
            "Permutacion_caida_R2": perm.loc[d, "caida_R2_permutacion"] if d in perm.index else np.nan,
            "Permutacion_aumento_MAE": perm.loc[d, "aumento_MAE_permutacion"] if d in perm.index and "aumento_MAE_permutacion" in perm.columns else np.nan,
            "Aporte_MAE_prueba_modelo_final": cf.loc[d, "Dif_MAE_mensual"] if cf is not None and d in cf.index else np.nan,
            "p_holm_prueba_modelo_final": cf.loc[d, "p_holm"] if cf is not None and d in cf.index else np.nan,
            "Ablacion_dif_R2": ab.loc[d, "Diferencia_R2"] if d in ab.index else np.nan,
            "Aporte_incremental_R2": ab.loc[d, "Aporte_incremental_R2"] if d in ab.index and "Aporte_incremental_R2" in ab.columns else np.nan,
            "Modelo_ganancia_SHAP": nombre_modelo_final + " (reentrenado con idx_dev)",
            "Muestra_SHAP": "Prueba, registros con historial (hasta 1 500)",
            "Modelo_permutacion": nombre_modelo_final + " (entrenado solo con entrenamiento), ancla de persistencia fija",
            "Muestra_permutacion": "Validación",
            "Modelo_ablacion": nombre_modelo_ablacion + " (NO es el modelo final)",
            "Muestra_ablacion": f"Validación temporal ({esquema_ablacion}) sobre idx_dev",
            "Nota_X4": "Aporte_incremental_R2 = NaN en X4: es la dimensión base del cálculo incremental." if d.startswith("X4") else "",
            "Concepto": "Importancia predictiva (ganancia, SHAP, permutación) vs aporte incremental (ablación, incremental, aporte en prueba). "
                        "Ninguna mide efecto causal: el estudio es observacional."})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_importancia_comparada.csv"), index=False)
    print(t[["Dimension", "Ganancia_pct", "SHAP_pct", "SHAP_agrupado_pct", "Permutacion_caida_R2",
             "Aporte_MAE_prueba_modelo_final", "Ablacion_dif_R2"]].round(4).to_string(index=False))
    return t