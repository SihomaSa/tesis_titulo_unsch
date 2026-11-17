"""
reporte_json.py — Construye reporte_contrastacion_hipotesis.json con la lógica de la tesis:
  - HG-a (eficacia): se decide con el IC95 del R² frente al criterio de referencia
  - HG-b (baselines): persistencia (todos / con precio previo real) y media móvil, con Holm
  - H1–H4: estado de 3 niveles (validación temporal, prueba LightGBM directo, modelo final)
"""
import json

import numpy as np

from src.config import ALPHA, MATRIZ_CONSISTENCIA, UMBRAL_R2_OFICIAL

# Dimensiones del contraste sobre el modelo final que corresponden a cada dimensión
_FINAL = {
    "X1_Temporal": ["X1_Temporal"],
    "X2_Productiva": ["X2_Productiva"],
    "X3_Espacial_Producto": ["X3a_Territorio", "X3b_Producto"],
    "X4_Economica_Historica": [],
}
_ETIQ_FINAL = {"X3a_Territorio": "territorio", "X3b_Producto": "producto"}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fila(tab, col, valor):
    if tab is None or col not in tab.columns:
        return None
    r = tab[tab[col] == valor]
    return None if r.empty else r.iloc[0]


def _col(fila, nombre):
    return _f(fila[nombre]) if (fila is not None and nombre in fila.index) else None


# ------------------------------------------------------------------ HG-a
def _hg_a(T, m):
    r2 = _f(m["R2"])
    ic = T.get("ic")
    ic95, estado = None, None
    if ic is not None and len(ic):
        r = ic.iloc[0]
        inf, sup = _f(r["IC95_inf"]), _f(r["IC95_sup"])
        ic95 = [inf, sup]
        if sup is not None and sup < UMBRAL_R2_OFICIAL:
            estado = "NO RESPALDADA"
        elif inf is not None and inf >= UMBRAL_R2_OFICIAL:
            estado = "RESPALDADA"
        else:
            estado = "NO CONCLUYENTE"
    else:  # sin IC (corrida --sin-extras): se usa el valor puntual
        estado = "RESPALDADA" if (r2 is not None and r2 >= UMBRAL_R2_OFICIAL) else "NO RESPALDADA"
    return {"estado": estado, "criterio_referencia": UMBRAL_R2_OFICIAL,
            "regla": "se rechaza si el límite superior del IC95 del R² queda por debajo del criterio de referencia"}, ic95


# ------------------------------------------------------------------ HG-b
def _hg_b(T, m, base, mm3):
    sg, sh = T.get("sig"), T.get("sig_h")
    p = _fila(sg, "Referencia", "Persistencia")
    q = _fila(sg, "Referencia", "Media móvil 3")
    ph = _fila(sh, "Referencia", "Persistencia")

    red_p = _col(p, "Reduccion_MAE_pct")
    red_q = _col(q, "Reduccion_MAE_pct")
    if red_p is None:
        red_p = _f((base["MAE"] - m["MAE"]) / base["MAE"] * 100)
    if red_q is None:
        red_q = _f((mm3["MAE"] - m["MAE"]) / mm3["MAE"] * 100)

    p_p, p_q, p_ph = _col(p, "p_holm"), _col(q, "p_holm"), _col(ph, "p_holm")
    estado = None
    if p_p is not None and p_q is not None:
        ok_p, ok_q = p_p < ALPHA, p_q < ALPHA
        ok_ph = (p_ph is not None) and (p_ph < ALPHA)
        estado = "RESPALDADA" if (ok_p and ok_ph and ok_q) else ("NO RESPALDADA" if not (ok_p or ok_q) else "PARCIALMENTE RESPALDADA")
    return {"estado": estado,
            "reduccion_persistencia_pct": red_p, "p_holm_persistencia": p_p,
            "reduccion_persistencia_con_historial_pct": _col(ph, "Reduccion_MAE_pct"), "p_holm_persistencia_con_historial": p_ph,
            "reduccion_media_movil_pct": red_q, "p_holm_media_movil": p_q}


# ------------------------------------------------------------------ H1-H4
def _niveles(T, k, ab_fila):
    niv = []
    # (i) validación temporal, LightGBM directo
    niv.append({"nivel": "Validación temporal",
                "dMAE": _col(ab_fila, "Diferencia_MAE"),
                "p_holm": _col(ab_fila, "p_mae_holm") if (ab_fila is not None and "p_mae_holm" in ab_fila.index) else _col(ab_fila, "p_holm")})
    # (ii) prueba, LightGBM directo
    c = _fila(T.get("c_lgbm"), "Dimension", k)
    niv.append({"nivel": "Prueba (LightGBM directo)", "dMAE": _col(c, "Dif_MAE_mensual"), "p_holm": _col(c, "p_holm")})
    # (iii) prueba, modelo final
    dims = _FINAL.get(k, [])
    if not dims:
        niv.append({"nivel": "Modelo final", "dMAE": None, "p_holm": None, "nota": "No aplica: X4 es la base del modelo final"})
    else:
        mejor = None
        for d in dims:
            r = _fila(T.get("c_res"), "Dimension", d)
            pv = _col(r, "p_holm")
            if r is not None and pv is not None and (mejor is None or pv < mejor[1]):
                mejor = (d, pv, _col(r, "Dif_MAE_mensual"))
        if mejor:
            etiqueta = "Modelo final" + (f" ({_ETIQ_FINAL[mejor[0]]})" if mejor[0] in _ETIQ_FINAL else "")
            niv.append({"nivel": etiqueta, "dMAE": mejor[2], "p_holm": mejor[1]})
        else:
            niv.append({"nivel": "Modelo final", "dMAE": None, "p_holm": None, "nota": "Sin dato"})
    return niv


def _shap_frac(T, shp):
    if shp:
        return {k: float(v) for k, v in shp.items()}
    s = T.get("shap")
    if s is None:
        return {}
    g = s.groupby("Dimension")["Pct_total"].sum()
    return {k: float(v) / 100 for k, v in g.items()}


# ------------------------------------------------------------------ principal
def construir_reporte(T, nom, m, base, mm3, n_final, n_test, h_act, n_splits, imp_dim, shp, clasif, ab):
    hg_a, ic95 = _hg_a(T, m)
    hg_b = _hg_b(T, m, base, mm3)

    abl_t = T.get("abl_lgbm")
    abl_t = abl_t.set_index("Dimension") if abl_t is not None else ab
    shp_f = _shap_frac(T, shp)

    evid = f"R² en prueba = {m['R2']:.4f}"
    if ic95 and None not in ic95:
        evid += f" (IC95 [{ic95[0]:.4f}; {ic95[1]:.4f}])"
    evid += f"; MAE persistencia = {base['MAE']:.4f} vs modelo = {m['MAE']:.4f}"

    especificas = []
    for d in MATRIZ_CONSISTENCIA["dimensiones"]:
        k = d["clave"]
        fila = abl_t.loc[k] if (abl_t is not None and k in abl_t.index) else None
        niveles = _niveles(T, k, fila)
        especificas.append({
            "codigo": d["codigo"], "clave": k, "texto": d["hipotesis_especifica"],
            "estado": (clasif or {}).get(k, {}).get("estado"),
            "niveles": niveles,
            "diferencia_MAE": niveles[0]["dMAE"], "diferencia_R2": _col(fila, "Diferencia_R2"),
            "p_valor": niveles[0]["p_holm"],
            "importancia_ganancia": _f(imp_dim.get(k)) if imp_dim else None,
            "importancia_shap": _f(shp_f.get(k)),
        })

    return {
        "n_registros_validos": int(n_final), "modelo_seleccionado": nom,
        "horizonte_meses": h_act, "n_splits_ablation": int(n_splits),
        "modelo_final": {
            "r2_final": _f(m["R2"]), "r2_ic95": ic95,
            "mae_final": _f(m["MAE"]), "rmse_final": _f(m["RMSE"]), "mape_final": _f(m["MAPE"]),
            "mae_persistencia": _f(base["MAE"]), "mae_media_movil": _f(mm3["MAE"]), "n_test": int(n_test),
        },
        "hipotesis_general": {
            "texto": MATRIZ_CONSISTENCIA["hipotesis_general"],
            "hg_a": hg_a, "hg_b": hg_b, "evidencia": evid,
        },
        "hipotesis_especificas": especificas,
    }


def guardar_reporte(reporte, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)