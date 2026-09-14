"""
ablation_study.py — Ablation por dimensión + prueba de Wilcoxon.
Responde DIRECTAMENTE a los Problemas/Objetivos/Hipótesis específicas
(PE-a..d / OE-a..d / H1-H4). Sigue la estrategia descrita en 3.7.1:
"cuatro variantes del modelo Random Forest (excluyendo una dimensión
cada vez)" + Wilcoxon pareado de 5 repeticiones.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.config import RANDOM_STATE, OUT_FIG, OUT_TAB, COLORES, UMBRAL_R2_OFICIAL


def _evaluar(X, y, tss, feats):
    r2s = []
    for tr_idx, te_idx in tss.split(X):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X.iloc[tr_idx][feats], y.iloc[tr_idx])
        r2s.append(r2_score(y.iloc[te_idx], rf.predict(X.iloc[te_idx][feats])))
    return np.array(r2s)


def run_ablation(X: pd.DataFrame, y: pd.Series, features: list,
                  dimensiones: dict, tss):
    """Ejecuta el ablation study y devuelve la lista de resultados
    (uno por dimensión / hipótesis específica), con figura y tabla
    exportadas (tabla_4.4, tal como aparece en tu capítulo 4)."""
    with mlflow.start_run(run_name="03_ablation_study_hipotesis"):
        r2_completo = _evaluar(X, y, tss, features)
        mlflow.log_metric("r2_completo_mean", np.mean(r2_completo))

        resultados = []
        for dim_name, feats in dimensiones.items():
            feats_sin = [f for f in features if f not in feats]
            r2_sin = _evaluar(X, y, tss, feats_sin)
            diff = np.mean(r2_completo) - np.mean(r2_sin)
            try:
                _, p_valor = wilcoxon(r2_completo, r2_sin)
            except ValueError:
                p_valor = 1.0
            significativa = p_valor < 0.05
            resultados.append({
                "Dimension": dim_name,
                "R2_sin_dim": float(np.mean(r2_sin)),
                "R2_completo": float(np.mean(r2_completo)),
                "Diferencia_R2": float(diff),
                "p_valor": float(p_valor),
                "Significativa_estadistica": "Sí" if significativa else "No",
            })
            mlflow.log_metric(f"{dim_name}_r2_sin", np.mean(r2_sin))
            mlflow.log_metric(f"{dim_name}_diff_r2", diff)
            mlflow.log_metric(f"{dim_name}_p_valor", p_valor)
            print(f"[ablation] Sin {dim_name}: R²={np.mean(r2_sin):.4f} | "
                  f"Δ={diff:.4f} | p={p_valor:.4f} | "
                  f"{'significativa' if significativa else 'no significativa'}")

        tabla = pd.DataFrame(resultados)
        tabla_path = os.path.join(OUT_TAB, "tabla_4.4_wilcoxon_hipotesis.csv")
        tabla.to_csv(tabla_path, index=False)
        mlflow.log_artifact(tabla_path)

        fig_path = _plot_ablation(tabla, r2_completo)
        mlflow.log_artifact(fig_path)
        print(f"[ablation] ✅ {fig_path}")

        fig_path_v2 = _plot_ablation_vertical(tabla, r2_completo)
        mlflow.log_artifact(fig_path_v2)

    return resultados


def _plot_ablation(tabla, r2_completo):
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = [COLORES["x1_temporal"], COLORES["x2_productiva"],
               COLORES["x3_geografica"], COLORES["x4_economica"]]
    ax.barh(tabla["Dimension"], tabla["R2_sin_dim"], color=colores)
    ax.axvline(x=np.mean(r2_completo), color="black", linestyle="--",
               label=f"Modelo completo R²={np.mean(r2_completo):.4f}")
    ax.set_xlabel("R² al excluir la dimensión")
    ax.set_title("Ablation Study: influencia de cada dimensión (H1–H4)")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_ablation_study.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def _plot_ablation_vertical(tabla, r2_completo):
    """Vista alterna: barras verticales con 'Completo' explícito como
    primera barra (en vez de línea punteada) y anotación Δ solo en X4, el
    estilo que ya usabas en una versión anterior de la figura."""
    labels = ["Completo\n(X1+X2+X3+X4)"] + [
        f"Sin {d.split('_')[0]}\n({d.split('_', 1)[1]})" for d in tabla["Dimension"]
    ]
    r2_vals = [np.mean(r2_completo)] + list(tabla["R2_sin_dim"])
    colores = [COLORES["principal"]] + [
        COLORES["x1_temporal"], COLORES["x2_productiva"],
        COLORES["x3_geografica"], COLORES["x4_economica"]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, r2_vals, color=colores)
    for b, v in zip(bars, r2_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", fontweight="bold")
    ax.axhline(y=UMBRAL_R2_OFICIAL, color="red", linestyle="--", label=f"Umbral {UMBRAL_R2_OFICIAL}")
    # Δ solo sobre la dimensión de mayor caída (X4), como en tu figura de referencia
    idx_min = int(np.argmin(r2_vals[1:])) + 1
    delta = r2_vals[idx_min] - r2_vals[0]
    ax.annotate(f"Δ = {delta:.4f}", xy=(idx_min, r2_vals[idx_min]),
                xytext=(idx_min - 0.9, r2_vals[idx_min] + 0.04),
                ha="center", color=colores[idx_min], fontweight="bold", fontsize=11,
                arrowprops=dict(arrowstyle="->", color=colores[idx_min]))
    ax.set_ylabel("R² Score")
    ax.set_ylim(min(r2_vals) - 0.15, max(r2_vals) * 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.35))
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_ablation_study_vertical.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path
