"""
rendimiento_provincial.py — Rendimiento del modelo final por provincia
(Tabla 9 del capítulo 4, objetivo específico c / dimensión geográfica).

Antes esta tabla no tenía script: los números vivían escritos a mano en el
texto de la tesis. Aquí se recalculan sobre datos reales, con el MISMO
modelo final (LightGBM o RF, el que se haya seleccionado) y el MISMO split
80/20 cronológico corregido, filtrando por provincia en el conjunto de
prueba.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import r2_score, mean_absolute_error

from src.config import OUT_TAB, OUT_FIG, COLORES


def _plot_r2_por_provincia(tabla: pd.DataFrame):
    """Figura 19 de la tesis: 'R² por Provincia - Región Piura'. Antes esta
    figura no tenía código: la Tabla 9 se escribía a mano en el texto."""
    tabla_ord = tabla.sort_values("R2_holdout_provincial", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(tabla_ord["Provincia"], tabla_ord["R2_holdout_provincial"],
                    color=COLORES["principal"])
    ax.axvline(x=0.85, color="orange", linestyle="--", label="Umbral oficial 0.85")
    for b, v in zip(bars, tabla_ord["R2_holdout_provincial"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("R² (hold-out provincial)")
    ax.set_title("R² por Provincia — Región Piura")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_19_r2_por_provincia.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def rendimiento_por_provincia(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                               modelo, split_idx: int) -> pd.DataFrame:
    """Evalúa el modelo final (ya entrenado) sobre el hold-out, desagregado
    por provincia. No re-entrena nada: usa las predicciones del modelo final
    para que la tabla sea coherente con el R² global reportado en 4.3.2.
    """
    with mlflow.start_run(run_name="05_rendimiento_por_provincia"):
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        df_test = df.iloc[split_idx:]
        y_pred = modelo.predict(X_test)

        filas = []
        for prov, idx in df_test.groupby("PROVINCIA").groups.items():
            pos = [X_test.index.get_loc(i) for i in idx]
            if len(pos) < 5:  # muestra insuficiente para un R² fiable
                continue
            yt, yp = y_test.iloc[pos], y_pred[pos]
            filas.append({
                "Provincia": prov,
                "N_registros_test": len(pos),
                "R2_holdout_provincial": r2_score(yt, yp),
                "MAE_S_kg": mean_absolute_error(yt, yp),
            })
        tabla = pd.DataFrame(filas).sort_values("R2_holdout_provincial", ascending=False)
        tabla_path = os.path.join(OUT_TAB, "tabla_9_rendimiento_provincial.csv")
        tabla.to_csv(tabla_path, index=False)
        mlflow.log_artifact(tabla_path)
        fig_path = _plot_r2_por_provincia(tabla)
        mlflow.log_artifact(fig_path)
        print("[rendimiento_provincial] " + " | ".join(
            f"{r.Provincia}=R²{r.R2_holdout_provincial:.3f}(n={r.N_registros_test})"
            for r in tabla.itertuples()))
    return tabla
