"""
anomaly_detection.py — Isolation Forest (instrumento descrito en 3.6.1 y
módulo del sistema web que menciona 3.7.2.3.A).
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, OUT_TAB, OUT_FIG, COLORES


def _plot_anomalias(df: pd.DataFrame, pred, precio_normal: float, precio_anomalo: float):
    """Figura 20 de la tesis: 'Detección de Anomalías (Isolation Forest)'.
    Antes esta figura no tenía código: solo se reportaba el conteo (720,
    5%). Se grafica la distribución de PRECIO_CHACRA separando normales de
    anómalos, para visualizar la coherencia agronómica del instrumento."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    normales = df.loc[pred == 1, "PRECIO_CHACRA"]
    anomalos = df.loc[pred == -1, "PRECIO_CHACRA"]
    axes[0].hist(normales, bins=50, alpha=0.6, label=f"Normal (n={len(normales):,})",
                 color=COLORES["principal"])
    axes[0].hist(anomalos, bins=50, alpha=0.7, label=f"Anómalo (n={len(anomalos):,})",
                 color="#C0392B")
    axes[0].set_xlabel("Precio en chacra (S/·kg)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de precios: normal vs. anómalo")
    axes[0].legend()

    axes[1].bar(["Normal", "Anómalo"], [precio_normal, precio_anomalo],
                color=[COLORES["principal"], "#C0392B"])
    axes[1].set_ylabel("Precio promedio (S/·kg)")
    axes[1].set_title(f"Δ = S/. {precio_anomalo - precio_normal:.3f}/kg")
    for i, v in enumerate([precio_normal, precio_anomalo]):
        axes[1].text(i, v + 0.05, f"S/. {v:.3f}", ha="center")

    plt.suptitle("Detección de Anomalías (Isolation Forest)")
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_20_deteccion_anomalias.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def _plot_anomalias_dispersión(X: pd.DataFrame, pred, precio_normal: float, precio_anomalo: float):
    """Vista de dispersión 2D (PCA) de las anomalías — estilo alterno al de
    la Figura 20 (histograma). Antes esta vista tampoco tenía código."""
    from sklearn.decomposition import PCA
    Xs = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(8, 7))
    normal_mask = pred == 1
    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1], s=15, alpha=0.5,
               color=COLORES["principal"], label=f"Normal ({(normal_mask.mean()*100):.0f}%)")
    ax.scatter(coords[~normal_mask, 0], coords[~normal_mask, 1], s=35, alpha=0.8,
               color="#C0392B", marker="x", label=f"Anómalo ({(~normal_mask).mean()*100:.0f}%)")
    n_anom = int((~normal_mask).sum())
    ax.text(0.02, 0.98, f"Anomalías detectadas: {n_anom} ({(~normal_mask).mean()*100:.2f}%)\n"
                          f"Precio normal: S/. {precio_normal:.3f}\n"
                          f"Precio anómalo: S/. {precio_anomalo:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#C0392B"))
    ax.set_xlabel("Componente principal 1 (normalizada)")
    ax.set_ylabel("Componente principal 2 (normalizada)")
    ax.set_title("Detección de Anomalías — Vista de Dispersión (PCA)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path = os.path.join(OUT_FIG, "figura_20b_anomalias_dispersion_pca.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def detect_anomalies(df: pd.DataFrame, X: pd.DataFrame, contamination: float = 0.05):
    """Ejecuta Isolation Forest y exporta una muestra de 50 anomalías para
    revisión manual (la coherencia agronómica del 68%/32% que reportas en
    4.4 es un juicio humano — este módulo solo prepara la muestra)."""
    with mlflow.start_run(run_name="04_isolation_forest"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso = IsolationForest(contamination=contamination, n_estimators=100,
                               random_state=RANDOM_STATE, n_jobs=-1)
        pred = iso.fit_predict(X_scaled)

        n_anomalias = int((pred == -1).sum())
        pct_anomalias = n_anomalias / len(X) * 100
        mlflow.log_metric("n_anomalias", n_anomalias)
        mlflow.log_metric("pct_anomalias", pct_anomalias)
        print(f"[anomaly_detection] Anomalías: {n_anomalias} ({pct_anomalias:.2f}%)")

        muestra = df.loc[pred == -1].sample(min(50, max(n_anomalias, 1)),
                                             random_state=RANDOM_STATE)
        muestra_path = os.path.join(OUT_TAB, "muestra_50_anomalias_revision_manual.csv")
        muestra.to_csv(muestra_path, index=False)
        mlflow.log_artifact(muestra_path)

        # Precio promedio normal vs. anómalo (Tabla 7 del capítulo 4):
        # coherencia agronómica del instrumento — se calcula aquí, con datos
        # reales, en vez de dejarlo como número fijo en el texto.
        precio_normal = float(df.loc[pred == 1, "PRECIO_CHACRA"].mean())
        precio_anomalo = float(df.loc[pred == -1, "PRECIO_CHACRA"].mean())
        mlflow.log_metric("precio_promedio_normal", precio_normal)
        mlflow.log_metric("precio_promedio_anomalo", precio_anomalo)
        print(f"[anomaly_detection] Precio promedio normal={precio_normal:.3f} S/kg | "
              f"anómalo={precio_anomalo:.3f} S/kg | Δ={precio_anomalo - precio_normal:.3f}")

        fig_path_anom = _plot_anomalias(df, pred, precio_normal, precio_anomalo)
        mlflow.log_artifact(fig_path_anom)
        fig_path_pca = _plot_anomalias_dispersión(X, pred, precio_normal, precio_anomalo)
        mlflow.log_artifact(fig_path_pca)

    return {"n_anomalias": n_anomalias, "pct_anomalias": pct_anomalias,
            "muestra_path": muestra_path,
            "precio_promedio_normal": precio_normal,
            "precio_promedio_anomalo": precio_anomalo}
