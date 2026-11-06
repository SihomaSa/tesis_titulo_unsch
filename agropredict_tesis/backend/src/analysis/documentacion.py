"""
documentacion.py — Obs. 4, 11 y 12 del informe del jurado (tablas que faltaban).

  Obs. 11  tabla_particiones.csv           fechas, tamaños y rezagos por partición
  Obs. 4   tabla_mae_por_brecha.csv        error según meses desde el precio previo (horizonte real)
  Obs. 12  tabla_shap_detalle.csv          mean|SHAP| por variable y su % (fórmula explícita)
           tabla_importancia_comparada.csv ganancia / SHAP / permutación / ablación por dimensión
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import OUT_TAB


def tabla_particiones(df, idx_tr, idx_va, idx_te):
    filas = []
    for nombre, idx in (("Entrenamiento", idx_tr), ("Validación", idx_va), ("Prueba", idx_te)):
        d = df.iloc[idx]
        filas.append({
            "Conjunto": nombre, "Desde": int(d["PERIODO"].min()), "Hasta": int(d["PERIODO"].max()),
            "n_registros": len(d), "pct_del_total": len(d) / len(df) * 100,
            "meses_distintos": int(d["PERIODO"].nunique()),
            "series_cultivo_distrito": int(d.drop_duplicates(["CULTIVO", "PROVINCIA", "DISTRITO"]).shape[0]),
            "pct_con_historial_previo": float(d["TIENE_HISTORIAL"].mean() * 100),
            "brecha_mediana_meses_al_precio_previo": float(d.loc[d["TIENE_HISTORIAL"] == 1, "MESES_DESDE_PRECIO_PREVIO"].median()),
        })
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_particiones.csv"), index=False)
    print(t.to_string(index=False))
    return t


def mae_por_brecha(df_te, y_te, pred, pred_persistencia):
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
        filas.append({"Grupo": nombre, "N": int(m.sum()), "Pct": m.mean() * 100,
                      "MAE_modelo": mm, "MAE_persistencia": mb, "Reduccion_MAE_pct": (mb - mm) / mb * 100 if mb else np.nan,
                      "R2_modelo": r2_score(y[m], pred[m])})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_mae_por_brecha.csv"), index=False)
    print(t.to_string(index=False))
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
    t["n_registros_muestra"] = len(shp["X"])
    t.to_csv(os.path.join(OUT_TAB, "tabla_shap_detalle.csv"), index=False)
    print("[Obs.12] % por dimensión = suma de mean|SHAP| de sus variables / suma total (n muestra = "
          f"{len(shp['X'])}):")
    print(t.groupby("Dimension")["Pct_total"].sum().round(1).to_string())
    return t


def importancia_comparada(interp, abl, dimensiones):
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
            "Concepto": "Importancia predictiva (ganancia, SHAP, permutación) vs aporte incremental (ablación, incremental). "
                        "Ninguna mide efecto causal."})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_importancia_comparada.csv"), index=False)
    print(t.drop(columns="Concepto").round(4).to_string(index=False))
    return t
