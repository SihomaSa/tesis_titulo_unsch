"""
rendimiento_por_frecuencia_cultivo.py — Desempeño del modelo final
desagregado por frecuencia del cultivo (común vs. raro) y por rango de
precio. Nace de una revisión manual: al comparar el scatter real-vs-predicho
completo contra una evaluación restringida a precios bajos, se detectó que
el modelo predice mucho mejor los cultivos frecuentes y de bajo precio que
los de alto valor y baja frecuencia (ajo, cacao, café, mango, maní, paprika,
piquillo). Antes esta tabla/figura no existía — el hallazgo se documentó
primero en una conversación de soporte, aquí se deja el código que lo
reproduce con datos reales.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import r2_score, mean_absolute_error

from src.config import OUT_TAB, OUT_FIG, COLORES

UMBRAL_PRECIO_ALTO = 4.5  # S/·kg — divide "cultivos comunes" de "alto valor"


def rendimiento_por_frecuencia(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                                modelo, split_idx: int) -> dict:
    with mlflow.start_run(run_name="06_rendimiento_por_frecuencia_cultivo"):
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        df_test = df.iloc[split_idx:]
        y_pred = modelo.predict(X_test)

        mask_bajo = y_test.values <= UMBRAL_PRECIO_ALTO
        mask_alto = ~mask_bajo

        r2_bajo = r2_score(y_test[mask_bajo], y_pred[mask_bajo])
        mae_bajo = mean_absolute_error(y_test[mask_bajo], y_pred[mask_bajo])
        r2_alto = r2_score(y_test[mask_alto], y_pred[mask_alto])
        mae_alto = mean_absolute_error(y_test[mask_alto], y_pred[mask_alto])

        # Frecuencia de cada cultivo en TODO el dataset (no solo el hold-out),
        # para no confundir "poco frecuente en el test" con "poco frecuente
        # en general".
        freq_cultivo = df["CULTIVO"].value_counts(normalize=True)
        cultivos_alto_valor = sorted(df_test.loc[mask_alto, "CULTIVO"].unique())
        freq_pct = {c: round(freq_cultivo.get(c, 0) * 100, 2) for c in cultivos_alto_valor}

        resumen = {
            "umbral_precio_alto_S_kg": UMBRAL_PRECIO_ALTO,
            "n_test_bajo": int(mask_bajo.sum()),
            "pct_test_bajo": float(mask_bajo.mean() * 100),
            "r2_bajo": float(r2_bajo),
            "mae_bajo": float(mae_bajo),
            "n_test_alto": int(mask_alto.sum()),
            "pct_test_alto": float(mask_alto.mean() * 100),
            "r2_alto": float(r2_alto),
            "mae_alto": float(mae_alto),
            "cultivos_alto_valor": cultivos_alto_valor,
            "frecuencia_pct_en_dataset_completo": freq_pct,
        }
        for k, v in resumen.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        tabla = pd.DataFrame([
            {"Segmento": f"Cultivos comunes (≤S/.{UMBRAL_PRECIO_ALTO}/kg)",
             "N_test": resumen["n_test_bajo"], "%_del_test": round(resumen["pct_test_bajo"], 1),
             "R2": round(r2_bajo, 4), "MAE_S_kg": round(mae_bajo, 4)},
            {"Segmento": f"Cultivos de alto valor (>S/.{UMBRAL_PRECIO_ALTO}/kg)",
             "N_test": resumen["n_test_alto"], "%_del_test": round(resumen["pct_test_alto"], 1),
             "R2": round(r2_alto, 4), "MAE_S_kg": round(mae_alto, 4)},
        ])
        tabla_path = os.path.join(OUT_TAB, "tabla_11_rendimiento_por_frecuencia_cultivo.csv")
        tabla.to_csv(tabla_path, index=False)
        mlflow.log_artifact(tabla_path)

        fig_path = _plot_rendimiento_por_frecuencia(y_test.values, y_pred, mask_alto)
        mlflow.log_artifact(fig_path)

        print(f"[frecuencia_cultivo] Comunes (n={resumen['n_test_bajo']}): R²={r2_bajo:.4f} | "
              f"Alto valor (n={resumen['n_test_alto']}): R²={r2_alto:.4f}")
        print(f"[frecuencia_cultivo] Cultivos de alto valor: {', '.join(cultivos_alto_valor)}")

    return resumen


def _plot_rendimiento_por_frecuencia(y_test, y_pred, mask_alto):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test[~mask_alto], y_pred[~mask_alto], alpha=0.4, s=20,
               color=COLORES["principal"], label="Cultivos comunes (≤S/.4.5/kg)")
    ax.scatter(y_test[mask_alto], y_pred[mask_alto], alpha=0.6, s=35,
               color="#C0392B", marker="^", label="Cultivos de alto valor (>S/.4.5/kg)")
    lims = [0, max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Ideal", linewidth=1)
    ax.set_xlabel("Precio Real (S/·kg)")
    ax.set_ylabel("Precio Predicho (S/·kg)")
    ax.set_title("Real vs. Predicho por segmento de cultivo")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_25_real_vs_predicho_por_segmento.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path
