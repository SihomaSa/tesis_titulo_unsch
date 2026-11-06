"""
documentacion.py — Obs. 4, 11 y 12 del informe del jurado.

  Obs. 11  tabla_particiones.csv           fechas, tamaños, rezagos y regla de actualización por partición
           tabla_folds.csv                 fechas y tamaños de cada fold (tuning y ablación)
  Obs. 4   tabla_mae_por_brecha.csv        error según meses desde el precio previo (horizonte real)
  Obs. 12  tabla_shap_detalle.csv          mean|SHAP| por variable, % y su escala/subconjunto
           tabla_importancia_comparada.csv ganancia / SHAP / permutación / ablación, cada una con su modelo y muestra
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import N_SPLITS_ABLATION, N_SPLITS_TUNING, OUT_TAB

REGLA_REZAGOS = ("Sí: pronóstico de un paso adelante. PRECIO_LAG1/MEDIA_MOVIL_3 se calculan sobre los precios "
                 "OBSERVADOS de la serie, también durante la prueba (no se usan predicciones previas).")


def tabla_particiones(df, idx_tr, idx_va, idx_te):
    filas = []
    for nombre, idx in (("Entrenamiento", idx_tr), ("Validación", idx_va), ("Prueba", idx_te)):
        d = df.iloc[idx]
        filas.append({
            "Conjunto": nombre, "Desde": int(d["PERIODO"].min()), "Hasta": int(d["PERIODO"].max()),
            "n_registros": len(d), "pct_del_total": len(d) / len(df) * 100,
            "meses_distintos": int(d["PERIODO"].nunique()),
            "series_cultivo_distrito": int(d.drop_duplicates(["CULTIVO", "PROVINCIA", "DISTRITO"]).shape[0]),
            "n_con_historial_previo": int(d["TIENE_HISTORIAL"].sum()),
            "pct_con_historial_previo": float(d["TIENE_HISTORIAL"].mean() * 100),
            "brecha_mediana_meses_al_precio_previo": float(d.loc[d["TIENE_HISTORIAL"] == 1, "MESES_DESDE_PRECIO_PREVIO"].median()),
            "rezagos_actualizados_con_precio_observado": REGLA_REZAGOS if nombre == "Prueba" else "Sí (precios observados anteriores)",
        })
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_particiones.csv"), index=False)
    print(t.drop(columns="rezagos_actualizados_con_precio_observado").to_string(index=False))
    return t


def tabla_folds(df, idx_tr, idx_dev, rapido=False):
    """Obs. 11: detalle de los folds de validación temporal (tuning sobre train; ablación sobre train+val).
    TimeSeriesSplit es expansivo: entrena con todo lo anterior y valida en el bloque siguiente."""
    filas = []
    for etapa, idx, k in (("Tuning (solo entrenamiento)", idx_tr, 3 if rapido else N_SPLITS_TUNING),
                          ("Ablación (entrenamiento+validación)", idx_dev, 4 if rapido else N_SPLITS_ABLATION)):
        per = df["PERIODO"].to_numpy()[idx]
        for i, (a, b) in enumerate(TimeSeriesSplit(n_splits=k).split(idx), 1):
            filas.append({"Etapa": etapa, "Fold": i,
                          "n_entrenamiento": len(a), "entrena_desde": int(per[a].min()), "entrena_hasta": int(per[a].max()),
                          "n_validacion": len(b), "valida_desde": int(per[b].min()), "valida_hasta": int(per[b].max()),
                          "pct_valid_con_historial": float(df["TIENE_HISTORIAL"].to_numpy()[idx][b].mean() * 100)})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_folds.csv"), index=False)
    print(t.to_string(index=False))
    return t


def mae_por_brecha(df_te, y_te, pred, pred_persistencia):
    """Obs. 4: el horizonte efectivo es 'la siguiente observación de la serie', que dista g meses del precio previo.
    'Sin historial' NO es comparable: la persistencia ahí es la mediana de desarrollo (no hay último precio),
    así que su 'reducción' se deja en NaN para que nadie la cite."""
    y = np.asarray(y_te, float)
    gap = df_te["MESES_DESDE_PRECIO_PREVIO"].to_numpy()
    hist = df_te["TIENE_HISTORIAL"].to_numpy() == 1
    grupos = [("Sin historial previo", ~hist),
              ("Precio previo hace 1 mes", hist & (gap == 1)),
              ("Hace 2–3 meses", hist & (gap >= 2) & (gap <= 3)),
              ("Hace 4–12 meses", hist & (gap >= 4) & (gap <= 12)),
              ("Hace más de 12 meses", hist & (gap > 12))]
    filas = []
    for nombre, m in grupos:
        if m.sum() < 2:
            continue
        mm, mb = mean_absolute_error(y[m], pred[m]), mean_absolute_error(y[m], pred_persistencia[m])
        comparable = nombre != "Sin historial previo"
        filas.append({"Grupo": nombre, "N": int(m.sum()), "Pct": m.mean() * 100,
                      "MAE_modelo": mm, "MAE_persistencia": mb if comparable else np.nan,
                      "Reduccion_MAE_pct": (mb - mm) / mb * 100 if (mb and comparable) else np.nan,
                      "R2_modelo": r2_score(y[m], pred[m]),
                      "Nota": "" if comparable else "Persistencia = mediana de desarrollo (sin último precio); no comparable."})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_mae_por_brecha.csv"), index=False)
    print(t.drop(columns="Nota").to_string(index=False))
    return t


def shap_detalle(interp, features, dimensiones):
    shp = interp.get("shap")
    if not shp:
        print("[Obs.12] SHAP no disponible: instala `shap` y vuelve a correr.")
        return None
    sv = np.abs(np.asarray(shp["valores"])).mean(axis=0)
    dim_de = {f: d for d, fs in dimensiones.items() for f in fs}
    t = pd.DataFrame({"Variable": features, "Dimension": [dim_de[f] for f in features], "mean_abs_SHAP": sv})
    t["Pct_total"] = t["mean_abs_SHAP"] / t["mean_abs_SHAP"].sum() * 100
    t = t.sort_values("mean_abs_SHAP", ascending=False)
    t["Pct_acumulado"] = t["Pct_total"].cumsum()
    t["n_registros_muestra"] = shp.get("n_muestra", len(shp["X"]))
    t["n_subconjunto_con_historial"] = shp.get("n_subconjunto", np.nan)
    t["escala_shap"] = shp.get("escala", "")
    t["formula"] = "Pct_dimensión = Σ mean|SHAP_j| (j∈dimensión) / Σ mean|SHAP_j| (todas); muestra del conjunto de PRUEBA"
    t["nota"] = np.where(t["Variable"] == "TIENE_HISTORIAL", "Constante (=1) en el subconjunto explicado: 0 por construcción", "")
    t.to_csv(os.path.join(OUT_TAB, "tabla_shap_detalle.csv"), index=False)
    print("[Obs.12] % por dimensión = Σ mean|SHAP| de sus variables / Σ total "
          f"(muestra {t['n_registros_muestra'].iloc[0]} de {t['n_subconjunto_con_historial'].iloc[0]}; escala {shp.get('escala')}):")
    print(t.groupby("Dimension")["Pct_total"].sum().round(1).to_string())
    return t


def importancia_comparada(interp, abl, dimensiones, nombre_modelo_ablacion="LightGBM (directo)", nombre_modelo_final="LightGBM sobre persistencia"):
    """Cada columna declara con QUÉ modelo y en QUÉ muestra se calculó: no se mezclan en silencio."""
    ab = abl["tabla"].set_index("Dimension")
    perm = interp["permutacion"].set_index("Dimension")
    filas = []
    for d in dimensiones:
        filas.append({
            "Dimension": d,
            "Ganancia_pct": interp["imp_dim"][d] * 100,
            "SHAP_pct": interp["shap"]["por_dimension"][d] * 100 if interp.get("shap") else np.nan,
            "Permutacion_caida_R2": perm.loc[d, "caida_R2_permutacion"] if d in perm.index else np.nan,
            "Ablacion_dif_R2": ab.loc[d, "Diferencia_R2"] if d in ab.index else np.nan,
            "Aporte_incremental_R2": ab.loc[d, "Aporte_incremental_R2"] if d in ab.index else np.nan,
            "Modelo_ganancia_SHAP": nombre_modelo_final + " (reentrenado con entrenamiento+validación)",
            "Muestra_SHAP": "Prueba, registros con historial (hasta 1 500)",
            "Modelo_permutacion": nombre_modelo_final + " (entrenado solo con entrenamiento), ancla de persistencia fija",
            "Muestra_permutacion": "Validación",
            "Modelo_ablacion": nombre_modelo_ablacion,
            "Muestra_ablacion": f"Validación temporal (TimeSeriesSplit, {abl.get('n_splits', '?')} folds) sobre entrenamiento+validación",
            "Concepto": "Importancia predictiva (ganancia, SHAP, permutación) vs aporte incremental (ablación, incremental). "
                        "Ninguna mide efecto causal: el estudio es observacional."})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_importancia_comparada.csv"), index=False)
    print(t[["Dimension", "Ganancia_pct", "SHAP_pct", "Permutacion_caida_R2", "Ablacion_dif_R2", "Aporte_incremental_R2"]].round(4).to_string(index=False))
    return t
