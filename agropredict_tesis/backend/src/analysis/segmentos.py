"""segmentos.py — Desempeño del modelo final en PRUEBA por provincia y por
segmento de precio (cultivos comunes vs. alto valor)."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import OUT_TAB, UMBRAL_PRECIO_ALTO


def por_provincia(df_te, y_te, pred):
    filas = []
    for prov, idx in df_te.reset_index(drop=True).groupby("PROVINCIA").groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < 10:
            continue
        yt, yp = y_te.to_numpy()[idx], pred[idx]
        filas.append({"Provincia": prov, "N_test": len(idx), "R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp)})
    t = pd.DataFrame(filas).sort_values("R2", ascending=False)
    t.to_csv(os.path.join(OUT_TAB, "tabla_rendimiento_provincial.csv"), index=False)
    return t


def por_segmento_precio(y_te, pred):
    y = y_te.to_numpy()
    bajo = y <= UMBRAL_PRECIO_ALTO
    filas = []
    for nombre, m in ((f"Precio ≤ S/ {UMBRAL_PRECIO_ALTO}/kg", bajo), (f"Precio > S/ {UMBRAL_PRECIO_ALTO}/kg", ~bajo)):
        if m.sum() > 2:
            filas.append({"Segmento": nombre, "N_test": int(m.sum()), "Pct_test": m.mean() * 100,
                          "R2": r2_score(y[m], pred[m]), "MAE": mean_absolute_error(y[m], pred[m])})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(OUT_TAB, "tabla_rendimiento_segmento_precio.csv"), index=False)
    return t
