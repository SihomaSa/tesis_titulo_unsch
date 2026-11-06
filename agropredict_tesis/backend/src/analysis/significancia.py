"""
significancia.py — Obs. 1 y 3 del informe del jurado.

  * Diferencias de pérdida PAREADAS por mes de prueba (modelo vs. baseline).
    Se agrega por mes porque los errores de una misma serie/mes están
    correlacionados; tratar 2 148 filas como independientes inflaría la
    significancia.
  * Prueba t de la media con error estándar HAC (Newey-West, núcleo de Bartlett),
    robusta a autocorrelación temporal.  H1 unilateral: media(d) > 0 => el
    modelo reduce el error.
  * IC 95 % por bootstrap de BLOQUES móviles de meses (dependencia temporal).
  * IC 95 % del R² de prueba por bootstrap de bloques (HG-a: eficacia vs umbral).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from src.config import ALPHA, OUT_TAB, RANDOM_STATE, UMBRAL_R2_OFICIAL


def perdida_mensual(periodo, y, pred_a, pred_b, tipo="abs"):
    """d_t = pérdida media del baseline (a) - pérdida media del modelo (b), por mes.
    d_t > 0  =>  el modelo b es mejor ese mes."""
    y = np.asarray(y, float)
    ea, eb = np.abs(y - np.asarray(pred_a, float)), np.abs(y - np.asarray(pred_b, float))
    if tipo == "sq":
        ea, eb = ea ** 2, eb ** 2
    t = pd.DataFrame({"p": np.asarray(periodo), "a": ea, "b": eb}).groupby("p").mean().sort_index()
    return (t["a"] - t["b"]).to_numpy()


def hac_media(d, L=None):
    """Media, error estándar Newey-West, estadístico t y p unilateral (H1: media > 0)."""
    d = np.asarray(d, float)
    T = len(d)
    if L is None:
        L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    L = int(max(1, min(L, T - 2)))
    e = d - d.mean()
    s = float(e @ e) / T
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1.0)) * float(e[k:] @ e[:-k]) / T
    se = np.sqrt(max(s, 1e-18) / T)
    t = d.mean() / se
    return {"media": float(d.mean()), "se_HAC": float(se), "t": float(t),
            "p_unilateral": float(1 - stats.t.cdf(t, T - 1)), "rezagos_HAC": L, "n_meses": T}


def bootstrap_bloques_media(d, B=5000, bloque=None, seed=RANDOM_STATE):
    d = np.asarray(d, float)
    n = len(d)
    L = bloque or max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / L))
    medias = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n - L + 1, k)
        medias[i] = np.concatenate([d[s:s + L] for s in st])[:n].mean()
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return float(lo), float(hi), L


def holm(pvals):
    p = np.asarray(pvals, float)
    orden = np.argsort(p)
    aj = np.empty_like(p)
    m, mx = len(p), 0.0
    for rank, i in enumerate(orden):
        mx = max(mx, (m - rank) * p[i])
        aj[i] = min(1.0, mx)
    return aj


def comparar_pronosticos(periodo, y, pred_modelo, preds_ref: dict, nombre_modelo, tipo="abs"):
    """Modelo vs cada referencia (baselines u otros modelos). Devuelve DataFrame."""
    filas = []
    for nombre, pr in preds_ref.items():
        d = perdida_mensual(periodo, y, pr, pred_modelo, tipo)
        h = hac_media(d)
        lo, hi, L = bootstrap_bloques_media(d)
        mae_ref = float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pr, float))))
        mae_mod = float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred_modelo, float))))
        filas.append({
            "Modelo": nombre_modelo, "Referencia": nombre,
            "MAE_referencia": mae_ref, "MAE_modelo": mae_mod,
            "Reduccion_MAE_pct": (mae_ref - mae_mod) / mae_ref * 100,
            "Dif_media_mensual": h["media"], "IC95_inf": lo, "IC95_sup": hi,
            "t_HAC": h["t"], "p_unilateral_HAC": h["p_unilateral"],
            "n_meses": h["n_meses"], "rezagos_HAC": h["rezagos_HAC"], "bloque_bootstrap": L})
    t = pd.DataFrame(filas)
    t["p_holm"] = holm(t["p_unilateral_HAC"].to_numpy())
    t["Significativa"] = np.where(t["p_holm"] < ALPHA, "Sí", "No")
    return t


def ic_r2_bloques(periodo, y, pred, B=2000, seed=RANDOM_STATE):
    """IC 95 % del R² de prueba remuestreando bloques de MESES completos."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    meses = np.sort(pd.unique(np.asarray(periodo)))
    idx_mes = [np.where(np.asarray(periodo) == m)[0] for m in meses]
    n, L = len(meses), max(2, int(round(len(meses) ** (1 / 3))))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / L))
    r2s = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n - L + 1, k)
        sel = np.concatenate([np.concatenate(idx_mes[s:s + L]) for s in st])
        r2s[i] = r2_score(y[sel], pred[sel])
    lo, hi = np.percentile(r2s, [2.5, 97.5])
    return float(r2_score(y, pred)), float(lo), float(hi), L, int(n)


def reporte_prueba(periodo_te, y_te, pred_final, nombre_final, refs: dict):
    """Escribe tabla_significancia_vs_baseline.csv y tabla_ic_r2_prueba.csv."""
    t = comparar_pronosticos(periodo_te, y_te, pred_final, refs, nombre_final)
    t.to_csv(os.path.join(OUT_TAB, "tabla_significancia_vs_baseline.csv"), index=False)
    r2, lo, hi, L, n = ic_r2_bloques(periodo_te, y_te, pred_final)
    ic = pd.DataFrame([{
        "Modelo": nombre_final, "R2_prueba": r2, "IC95_inf": lo, "IC95_sup": hi,
        "Umbral": UMBRAL_R2_OFICIAL,
        "Decision_HGa": "Se rechaza (IC superior < umbral)" if hi < UMBRAL_R2_OFICIAL else
                        ("Se acepta (IC inferior ≥ umbral)" if lo >= UMBRAL_R2_OFICIAL else "No concluyente (el IC contiene el umbral)"),
        "n_meses": n, "bloque_meses": L}])
    ic.to_csv(os.path.join(OUT_TAB, "tabla_ic_r2_prueba.csv"), index=False)
    print(f"[Significancia] R² prueba = {r2:.4f}  IC95% [{lo:.4f}, {hi:.4f}] vs umbral {UMBRAL_R2_OFICIAL}")
    for _, r in t.iterrows():
        print(f"[Significancia] vs {r['Referencia']:<28} ΔMAE%={r['Reduccion_MAE_pct']:+.1f} "
              f"| IC95 dif mensual [{r['IC95_inf']:+.4f}, {r['IC95_sup']:+.4f}] | p_Holm={r['p_holm']:.3f}")
    return t, ic
