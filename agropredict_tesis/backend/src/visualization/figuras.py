"""figuras.py — Todas las figuras de la tesis (PNG a 300 dpi, sin título
incrustado: el título va como leyenda de la figura en Word, como exige la
observación del jurado sobre la Figura 3)."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.config import COLORES, OUT_FIG, UMBRAL_R2_OFICIAL

DIM_COL = {"X1_Temporal": COLORES["X1"], "X2_Productiva": COLORES["X2"],
           "X3_Geografica": COLORES["X3"], "X4_Economica_Historica": COLORES["X4"]}


def _save(nombre):
    ruta = os.path.join(OUT_FIG, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    return ruta


def fig_flujo_metodologico():
    """Reemplaza el 'flujo propio' observado por el jurado: 9 fases CRISP-DM + KDD."""
    fases = [("1", "Comprensión\ndel problema"), ("2", "Comprensión\nde los datos"), ("3", "Preparación\nde los datos"),
             ("4", "Construcción\ndel pipeline"), ("5", "Modelado"), ("6", "Ajuste de\nhiperparámetros"),
             ("7", "Evaluación"), ("8", "Interpretabilidad"), ("9", "Validación y\ndespliegue")]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 9.4); ax.set_ylim(0, 2.2); ax.axis("off")
    for i, (n, t) in enumerate(fases):
        x = i * 1.04 + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.75), 0.92, 1.0, boxstyle="round,pad=0.02", fc="#E8F3EE", ec=COLORES["principal"], lw=1.4))
        ax.text(x + 0.46, 1.58, f"Fase {n}", ha="center", fontsize=8, fontweight="bold", color=COLORES["principal"])
        ax.text(x + 0.46, 1.15, t, ha="center", va="center", fontsize=8)
        if i < len(fases) - 1:
            ax.annotate("", xy=(x + 1.03, 1.25), xytext=(x + 0.93, 1.25), arrowprops=dict(arrowstyle="->", color="#555"))
    ax.text(4.7, 0.35, "Retroalimentación iterativa (CRISP-DM): los resultados de las fases 7–9 pueden devolver el proceso a las fases 1–3",
            ha="center", fontsize=8, style="italic", color="#555")
    return _save("figura_flujo_metodologico_9_fases.png")


def fig_dimensiones(dimensiones):
    """Figura original de las variables predictoras por dimensión (sustituye a la Figura 3 tomada de otra fuente)."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.3, 2.35), 1.4, 0.5, boxstyle="round,pad=0.03", fc="#F4F6F6", ec="#333"))
    ax.text(2.0, 2.6, "Precio en chacra\n(S/ por kg)", ha="center", va="center", fontsize=10, fontweight="bold")
    for i, (k, vars_) in enumerate(dimensiones.items()):
        x = i * 1.0 + 0.05
        c = DIM_COL[k]
        ax.add_patch(FancyBboxPatch((x, 0.1), 0.9, 1.75, boxstyle="round,pad=0.02", fc="white", ec=c, lw=2))
        ax.text(x + 0.45, 1.68, k.split("_")[0], ha="center", fontsize=12, fontweight="bold", color=c)
        ax.text(x + 0.45, 1.5, {"X1_Temporal": "Temporal", "X2_Productiva": "Productiva", "X3_Geografica": "Geográfica", "X4_Economica_Historica": "Económica histórica"}[k], ha="center", fontsize=8.5, color=c)
        ax.text(x + 0.45, 0.85, "\n".join(vars_), ha="center", va="center", fontsize=7.2)
        ax.annotate("", xy=(2.0, 2.33), xytext=(x + 0.45, 1.87), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    return _save("figura_variables_por_dimension.png")


def fig_comparacion(tabla_val, tabla_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    nombres = list(tabla_test["Modelo"])
    cols = [COLORES["gris"] if "baseline" in n else COLORES["principal"] for n in nombres]
    for ax, met, lab in zip(axes, ["R2", "MAE", "RMSE"], ["R²", "MAE (S/ por kg)", "RMSE (S/ por kg)"]):
        v = tabla_test[met].to_numpy()
        b = ax.barh(nombres[::-1], v[::-1], color=cols[::-1])
        for r, val in zip(b, v[::-1]):
            ax.text(val, r.get_y() + r.get_height() / 2, f" {val:.3f}", va="center", fontsize=8)
        if met == "R2":
            ax.axvline(UMBRAL_R2_OFICIAL, color="red", ls="--", lw=1, label=f"Umbral {UMBRAL_R2_OFICIAL}")
            ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel(lab + " — conjunto de prueba")
    return _save("figura_comparacion_modelos_prueba.png")


def fig_r2_folds(r2_folds, nombres):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, vals in r2_folds.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=nombres[k])
    ax.set_xlabel("Fold temporal (entrenamiento)"); ax.set_ylabel("R²"); ax.grid(alpha=.3); ax.legend()
    return _save("figura_r2_por_fold.png")


def fig_real_vs_predicho(y, pred, nombre, r2):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y, pred, s=14, alpha=.45, color=COLORES["principal"])
    lim = [0, max(np.max(y), np.max(pred)) * 1.03]
    ax.plot(lim, lim, "r--", label="Ideal")
    ax.set_xlabel("Precio real (S/ por kg)"); ax.set_ylabel("Precio predicho (S/ por kg)")
    ax.set_title(f"{nombre} — prueba (R² = {r2:.3f})", fontsize=10); ax.legend()
    return _save("figura_real_vs_predicho.png")


def fig_importancia(imp_dim, shap_dim, perm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    claves = list(imp_dim)
    etiquetas = [k.split("_")[0] for k in claves]
    colores = [DIM_COL[k] for k in claves]
    for ax, datos, tit in ((axes[0], [imp_dim[k] * 100 for k in claves], "Ganancia del modelo (%)"),
                           (axes[1], [(shap_dim or {}).get(k, np.nan) * 100 for k in claves], "SHAP medio absoluto (%)"),
                           (axes[2], [float(perm.set_index("Dimension").loc[k, "caida_R2_permutacion"]) for k in claves],
                            "Caída de R² al permutar la dimensión")):
        b = ax.bar(etiquetas, datos, color=colores)
        for r, v in zip(b, datos):
            if v == v:
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}" if "%" in tit else f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(tit, fontsize=10)
    return _save("figura_importancia_por_dimension.png")


def fig_ablation(tabla):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tabla)); w = 0.27
    etiquetas = [d.split("_")[0] for d in tabla["Dimension"]]
    ax.bar(x - w, tabla["R2_sin_dim"], w, label="Modelo SIN la dimensión", color=[DIM_COL[d] for d in tabla["Dimension"]], alpha=.55)
    ax.bar(x, tabla["R2_solo_dim"], w, label="SOLO la dimensión", color=[DIM_COL[d] for d in tabla["Dimension"]])
    ax.bar(x + w, tabla["R2_X4_mas_dim"], w, label="X4 + la dimensión", color="#BFC9CA", edgecolor="k", lw=.5)
    ax.axhline(tabla["R2_completo"].iloc[0], color="k", ls="--", lw=1, label="Modelo completo")
    ax.set_xticks(x); ax.set_xticklabels(etiquetas); ax.set_ylabel("R² medio (validación temporal)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    return _save("figura_ablation_por_dimension.png")


def fig_provincias(tabla):
    t = tabla.sort_values("R2")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    b = ax.barh(t["Provincia"], t["R2"], color=COLORES["principal"])
    for r, v in zip(b, t["R2"]):
        ax.text(v, r.get_y() + r.get_height() / 2, f" {v:.3f}", va="center", fontsize=9)
    ax.axvline(UMBRAL_R2_OFICIAL, color="orange", ls="--", label=f"Umbral {UMBRAL_R2_OFICIAL}")
    ax.set_xlabel("R² en prueba"); ax.legend(fontsize=8)
    return _save("figura_r2_por_provincia.png")


def fig_anomalias(df, pred, target):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(df.loc[pred == 1, target], bins=50, alpha=.65, color=COLORES["principal"], label="Normal")
    ax.hist(df.loc[pred == -1, target], bins=50, alpha=.75, color="#C0392B", label="Anómalo")
    ax.set_yscale("log"); ax.set_xlabel("Precio en chacra (S/ por kg)"); ax.set_ylabel("Frecuencia (log)"); ax.legend()
    return _save("figura_anomalias_isolation_forest.png")


def fig_shap_resumen(shp):
    try:
        import shap
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shp["valores"], shp["X"], show=False, max_display=12)
        return _save("figura_shap_resumen.png")
    except Exception as e:
        print(f"[figuras] SHAP summary omitido: {e}")
        plt.close("all")
        return None
