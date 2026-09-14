"""
final_model.py — Modelo final Random Forest, validación hold-out 80/20
cronológico. Responde al PROBLEMA GENERAL / OBJETIVO GENERAL / HIPÓTESIS
GENERAL: es el número (R²) que decide si HG se acepta o se rechaza.
"""
import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

from src.config import RANDOM_STATE, OUT_FIG, OUT_MOD, COLORES, \
    UMBRAL_R2_CAP3, UMBRAL_R2_CAP4
from src.models.compare_models import safe_mape


def _metrics_de(y_test, y_pred):
    return {
        "r2_final": r2_score(y_test, y_pred),
        "mae_final": mean_absolute_error(y_test, y_pred),
        "rmse_final": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mape_final": safe_mape(y_test.values, y_pred),
    }


def _reporta(nombre, metrics):
    print(f"[final_model] {nombre}: R²={metrics['r2_final']:.4f} | MAE={metrics['mae_final']:.4f} "
          f"| RMSE={metrics['rmse_final']:.4f} | MAPE={metrics['mape_final']:.1f}%")
    print(f"[final_model]   vs umbral 0.75 (cap.4): "
          f"{'✅ CUMPLE' if metrics['r2_final'] >= UMBRAL_R2_CAP4 else '❌ NO CUMPLE'}")
    print(f"[final_model]   vs umbral 0.85 (cap.3): "
          f"{'✅ CUMPLE' if metrics['r2_final'] >= UMBRAL_R2_CAP3 else '❌ NO CUMPLE'}")


def train_final_model(X: pd.DataFrame, y: pd.Series, features: list, dimensiones: dict):
    """Entrena el modelo final sobre un split 80/20 cronológico REAL (ver
    corrección en data_loader.py), calcula métricas, importancia de
    variables por dimensión, guarda el modelo (.pkl + MLflow) y genera las
    figuras de importancia y real-vs-predicho.

    Se evalúan DOS modelos sobre el MISMO hold-out (mismo train_idx/test_idx,
    nada de comparar hold-out de uno contra TSS de otro):
      - Random Forest (el que ya tenías, sirve de referencia/baseline)
      - LightGBM regularizado (modelo adicional, con L1/L2 y profundidad
        limitada para intentar generalizar mejor que RF en este hold-out)

    El "modelo final" que se guarda y reporta como resultado de la Hipótesis
    General es el que tenga MEJOR R² en este hold-out real — no se elige
    a priori.

    Devuelve un dict con todo lo que necesita la API / el reporte final.
    """
    with mlflow.start_run(run_name="02_modelo_final_holdout"):
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"[final_model] Split hold-out: train={len(X_train):,} filas | test={len(X_test):,} filas")

        # --- Modelo 1: Random Forest (baseline, el que ya tenías) ---
        rf_final = RandomForestRegressor(n_estimators=200, max_depth=12,
                                          random_state=RANDOM_STATE, n_jobs=-1)
        rf_final.fit(X_train, y_train)
        pred_rf = rf_final.predict(X_test)
        metrics_rf = _metrics_de(y_test, pred_rf)
        _reporta("Random Forest", metrics_rf)
        for k, v in metrics_rf.items():
            mlflow.log_metric(f"rf_{k}", v)

        # --- Modelo 2 (ADICIONAL): LightGBM regularizado ---
        # n_estimators bajo + max_depth/num_leaves limitados + reg_alpha/
        # reg_lambda + min_child_samples alto: todo apunta a reducir
        # sobreajuste en un dataset con solo 14k filas y una variable
        # (PRECIO_LAG1/MEDIA_MOVIL_3) que domina la importancia.
        lgbm_final = LGBMRegressor(
            n_estimators=300, max_depth=6, num_leaves=31,
            learning_rate=0.03, min_child_samples=30,
            reg_alpha=0.5, reg_lambda=0.5, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
        )
        lgbm_final.fit(X_train, y_train)
        pred_lgbm = lgbm_final.predict(X_test)
        metrics_lgbm = _metrics_de(y_test, pred_lgbm)
        _reporta("LightGBM (regularizado, modelo adicional)", metrics_lgbm)
        for k, v in metrics_lgbm.items():
            mlflow.log_metric(f"lgbm_{k}", v)

        # --- Selección del modelo final por R² real de hold-out ---
        if metrics_lgbm["r2_final"] >= metrics_rf["r2_final"]:
            modelo_final, nombre_final, y_pred, metrics = lgbm_final, "LightGBM", pred_lgbm, metrics_lgbm
        else:
            modelo_final, nombre_final, y_pred, metrics = rf_final, "Random Forest", pred_rf, metrics_rf
        print(f"[final_model] 🏆 Modelo seleccionado por mejor R² en hold-out real: {nombre_final}")

        mlflow.log_param("modelo_seleccionado", nombre_final)
        mlflow.log_metrics(metrics)
        if nombre_final == "LightGBM":
            mlflow.lightgbm.log_model(modelo_final, "modelo_final_seleccionado")
        else:
            mlflow.sklearn.log_model(modelo_final, "modelo_final_seleccionado")

        model_path = os.path.join(OUT_MOD, "modelo_final.pkl")
        joblib.dump({"model": modelo_final, "features": features, "nombre": nombre_final}, model_path)
        mlflow.log_artifact(model_path)

        # Importancia de variables: del modelo seleccionado.
        # OJO: RandomForest.feature_importances_ ya viene normalizado (Gini,
        # suma 1). LightGBM.feature_importances_ por defecto es "split"
        # (conteo de veces que se usó la variable) y NO suma 1 — mezclarlo
        # sin normalizar con el eje "Importancia (Gini)" de la figura sería
        # engañoso. Se usa importance_type="gain" (más comparable a Gini) y
        # se normaliza a proporción en ambos casos.
        if nombre_final == "LightGBM":
            raw_imp = modelo_final.booster_.feature_importance(importance_type="gain")
        else:
            raw_imp = modelo_final.feature_importances_
        raw_imp = np.asarray(raw_imp, dtype=float)
        raw_imp = raw_imp / raw_imp.sum() if raw_imp.sum() > 0 else raw_imp
        importancias = pd.Series(raw_imp, index=features).sort_values(ascending=False)
        imp_por_dim = {
            dim: float(importancias[importancias.index.isin(feats)].sum())
            for dim, feats in dimensiones.items()
        }
        for dim, val in imp_por_dim.items():
            mlflow.log_metric(f"importancia_{dim}", val)

        fig_imp = _plot_importancia(importancias, dimensiones)
        mlflow.log_artifact(fig_imp)
        fig_imp_dim = _plot_importancia_por_dimension(imp_por_dim)
        mlflow.log_artifact(fig_imp_dim)
        fig_scatter = _plot_real_vs_predicho(y_test, y_pred, metrics["r2_final"], nombre_final)
        mlflow.log_artifact(fig_scatter)
        fig_comp = _plot_comparacion_holdout(metrics_rf, metrics_lgbm)
        mlflow.log_artifact(fig_comp)

    return {
        "model": modelo_final,
        "modelo_seleccionado": nombre_final,
        "metrics": metrics,
        "metrics_por_modelo": {"Random Forest": metrics_rf, "LightGBM": metrics_lgbm},
        "importancias": importancias,
        "importancia_por_dimension": imp_por_dim,
        "model_path": model_path,
        "split_idx": split_idx,
    }


def _plot_comparacion_holdout(metrics_rf, metrics_lgbm):
    """Nueva figura: compara RF vs LightGBM en el MISMO hold-out real
    (a diferencia de la Figura 22, que compara los 4 modelos en TSS)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    modelos = ["Random Forest", "LightGBM"]
    for ax, metric, label in zip(
        axes, ["r2_final", "mae_final", "rmse_final"], ["R² (hold-out)", "MAE (S/·kg)", "RMSE (S/·kg)"]
    ):
        vals = [metrics_rf[metric], metrics_lgbm[metric]]
        ax.bar(modelos, vals, color=[COLORES["x2_productiva"], COLORES["lgbm"]])
        if metric == "r2_final":
            ax.axhline(y=UMBRAL_R2_CAP4, color="red", linestyle="--", label="Umbral 0.75 (cap.4)")
            ax.axhline(y=UMBRAL_R2_CAP3, color="orange", linestyle="--", label="Umbral 0.85 (cap.3)")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
    plt.suptitle("RF vs LightGBM — MISMO hold-out cronológico real (no TSS)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_23_comparacion_holdout_real.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def _color_de(var, dimensiones):
    key_map = {"X1_Temporal": "x1_temporal", "X2_Productiva": "x2_productiva",
               "X3_Geografica": "x3_geografica", "X4_Economica_Historica": "x4_economica"}
    for dim, feats in dimensiones.items():
        if var in feats:
            return COLORES[key_map[dim]]
    return "#999999"


def _plot_importancia(importancias, dimensiones):
    fig, ax = plt.subplots(figsize=(10, 8))
    top10 = importancias.head(10)
    colors_imp = [_color_de(v, dimensiones) for v in top10.index]
    ax.barh(top10.index[::-1], top10.values[::-1], color=colors_imp[::-1])
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Top 10 variables más importantes")
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_importancia_variables.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def _plot_importancia_por_dimension(imp_por_dim: dict):
    """Figura 6: 'Importancia por dimensión (X1-X4)' — panel doble (barra +
    pastel), igual al estilo que ya usas en tu tesis. Antes esta figura no
    tenía código: el número (ej. 82.86% para X4) se escribía a mano."""
    # Orden descendente, como en tu figura de referencia (X4, X2, X3, X1)
    orden = sorted(imp_por_dim.items(), key=lambda kv: kv[1], reverse=True)
    dims = [k.replace("_", " ") for k, _ in orden]
    vals = [v * 100 for _, v in orden]

    def _color_de(dim_key):
        dl = dim_key.lower().replace(" ", "_")
        for k, v in COLORES.items():
            if dl.startswith(k):
                return v
        return COLORES["principal"]
    colores = [_color_de(k) for k, _ in orden]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    bars = axes[0].bar(dims, vals, color=colores)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Importancia (%)")
    axes[0].set_title("Importancia por Dimensión (LightGBM)")
    axes[0].set_ylim(0, max(vals) * 1.15)
    axes[0].tick_params(axis="x", labelsize=8.5)

    axes[1].pie(vals, colors=colores, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
                startangle=90, pctdistance=0.75, textprops={"fontsize": 9, "color": "white", "fontweight": "bold"})
    axes[1].legend(dims, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    axes[1].set_title("Distribución de Importancia")

    plt.suptitle("FIGURA 6: IMPORTANCIA POR DIMENSIÓN (X1-X4)", fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_06_importancia_por_dimension.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def _plot_real_vs_predicho(y_test, y_pred, r2_final, nombre_modelo="Random Forest"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color=COLORES["principal"])
    lims = [0, max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Ideal")
    ax.set_xlabel("Precio Real (S/·kg)")
    ax.set_ylabel("Precio Predicho (S/·kg)")
    ax.set_title(f"Real vs Predicho — {nombre_modelo} (R²={r2_final:.4f})")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_real_vs_predicho.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path
