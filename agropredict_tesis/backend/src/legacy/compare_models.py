"""
compare_models.py — Comparación RF / XGBoost / LightGBM / Regresión Lineal
mediante TimeSeriesSplit. Responde al OBJETIVO GENERAL (determinar la
eficacia de las técnicas de ML). Genera la Figura 22 ("Dashboard
Comparación de 4 Modelos") que ya citas en tu capítulo 4.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import RANDOM_STATE, N_SPLITS_TSS, OUT_FIG, OUT_TAB, COLORES, \
    UMBRAL_R2_CAP3, UMBRAL_R2_CAP4


def safe_mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compare_models(X: pd.DataFrame, y: pd.Series):
    """Entrena y compara 4 modelos con TimeSeriesSplit (5 folds).
    Devuelve (resultados_dict, tss) y registra todo en MLflow + guarda
    la figura de comparación en outputs/figures/.
    """
    tss = TimeSeriesSplit(n_splits=N_SPLITS_TSS)
    resultados = {m: {"r2": [], "mae": [], "rmse": [], "mape": []}
                  for m in ["rf", "xgb", "lgbm", "lr"]}

    with mlflow.start_run(run_name="01_comparacion_modelos"):
        mlflow.log_params({
            "n_splits_tss": N_SPLITS_TSS,
            "n_features": X.shape[1],
            "n_registros": len(X),
            "random_state": RANDOM_STATE,
        })

        for fold, (train_idx, test_idx) in enumerate(tss.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                        random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)

            xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                random_state=RANDOM_STATE, verbosity=0)
            xgb.fit(X_train, y_train)
            pred_xgb = xgb.predict(X_test)

            lgbm = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, verbose=-1)
            lgbm.fit(X_train, y_train)
            pred_lgbm = lgbm.predict(X_test)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            lr = LinearRegression()
            lr.fit(X_train_s, y_train)
            pred_lr = lr.predict(X_test_s)

            for name, pred in [("rf", pred_rf), ("xgb", pred_xgb),
                                ("lgbm", pred_lgbm), ("lr", pred_lr)]:
                r2 = r2_score(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mape = safe_mape(y_test.values, pred)
                resultados[name]["r2"].append(r2)
                resultados[name]["mae"].append(mae)
                resultados[name]["rmse"].append(rmse)
                resultados[name]["mape"].append(mape)
                mlflow.log_metric(f"{name}_r2_fold{fold}", r2)

            print(f"[compare_models] Fold {fold}: "
                  f"RF={resultados['rf']['r2'][-1]:.4f} | "
                  f"XGB={resultados['xgb']['r2'][-1]:.4f} | "
                  f"LGBM={resultados['lgbm']['r2'][-1]:.4f} | "
                  f"LR={resultados['lr']['r2'][-1]:.4f}")

        for name in ["rf", "xgb", "lgbm", "lr"]:
            mlflow.log_metric(f"{name}_r2_mean", np.mean(resultados[name]["r2"]))
            mlflow.log_metric(f"{name}_r2_std", np.std(resultados[name]["r2"]))

        tabla = pd.DataFrame({
            "Modelo": ["Random Forest", "XGBoost", "LightGBM", "Regresión Lineal"],
            "R2_TSS_media": [np.mean(resultados[m]["r2"]) for m in ["rf", "xgb", "lgbm", "lr"]],
            "R2_TSS_std": [np.std(resultados[m]["r2"]) for m in ["rf", "xgb", "lgbm", "lr"]],
            "MAE_TSS": [np.mean(resultados[m]["mae"]) for m in ["rf", "xgb", "lgbm", "lr"]],
            "RMSE_TSS": [np.mean(resultados[m]["rmse"]) for m in ["rf", "xgb", "lgbm", "lr"]],
        })
        tabla_path = os.path.join(OUT_TAB, "tabla_4.1_comparacion_modelos.csv")
        tabla.to_csv(tabla_path, index=False)
        mlflow.log_artifact(tabla_path)

        fig_path = _plot_comparacion(resultados)
        mlflow.log_artifact(fig_path)
        print(f"[compare_models] ✅ {fig_path}")

        fig_fold = _plot_evolucion_fold(resultados)
        mlflow.log_artifact(fig_fold)
        print(f"[compare_models] ✅ {fig_fold}")

    return resultados, tss


def _plot_evolucion_fold(resultados):
    """Figura 3: 'Evolución del R² por fold (TimeSeriesSplit)'. Antes esta
    figura no tenía código en el pipeline actual — los valores por fold ya
    se calculaban e imprimían en consola, pero nunca se graficaban. Incluye
    los 4 modelos (tu imagen de referencia solo tenía 3: RF, XGBoost y
    Regresión Lineal; se agrega LightGBM por consistencia con el resto del
    capítulo)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    modelos = {"rf": ("Random Forest", COLORES["x2_productiva"], "o", -12),
               "xgb": ("XGBoost", COLORES["secundario"], "s", 8),
               "lgbm": ("LightGBM", COLORES["lgbm"], "D", 20),
               "lr": ("Regresión Lineal", COLORES["x4_economica"], "^", 8)}
    folds = list(range(1, len(resultados["rf"]["r2"]) + 1))
    for key, (nombre, color, marker, y_off) in modelos.items():
        vals = resultados[key]["r2"]
        ax.plot(folds, vals, marker=marker, color=color, label=nombre, linewidth=2)
        for f, v in zip(folds, vals):
            ax.annotate(f"{v:.3f}", (f, v), textcoords="offset points",
                        xytext=(0, y_off), ha="center", fontsize=7.5, color=color)
    ax.axhline(y=UMBRAL_R2_CAP3, color="red", linestyle="--", label="Umbral oficial 0.85")
    ax.set_xlabel("Fold cronológico")
    ax.set_ylabel("R² Score")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_title("FIGURA 3: EVOLUCIÓN DEL R² POR FOLD (TimeSeriesSplit)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_03_evolucion_r2_por_fold.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def _plot_comparacion(resultados):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    modelos = ["Random Forest", "XGBoost", "LightGBM", "Regresión Lineal"]
    colores_modelos = [COLORES["x2_productiva"], COLORES["x4_economica"],
                        COLORES["lgbm"], COLORES["x1_temporal"]]

    for ax, metric, label in zip(axes, ["r2", "mae", "rmse"],
                                  ["R²", "MAE (S/·kg)", "RMSE (S/·kg)"]):
        means = [np.mean(resultados[m][metric]) for m in ["rf", "xgb", "lgbm", "lr"]]
        stds = [np.std(resultados[m][metric]) for m in ["rf", "xgb", "lgbm", "lr"]]
        ax.bar(modelos, means, yerr=stds if metric == "r2" else None,
               capsize=5, color=colores_modelos)
        if metric == "r2":
            ax.axhline(y=UMBRAL_R2_CAP4, color="red", linestyle="--", label="Umbral 0.75 (cap.4)")
            ax.axhline(y=UMBRAL_R2_CAP3, color="orange", linestyle="--", label="Umbral 0.85 (cap.3)")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"Comparación de {label} (TSS)")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_22_dashboard_comparacion_4_modelos.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path
