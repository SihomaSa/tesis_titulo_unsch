"""
texto_robustez.py — Redacción AUTOMÁTICA (con los números de outputs/tables/) de lo que exigió el jurado:

  Obs. 1   hipótesis separadas: eficacia (HG-a), superación de baselines (HG-b), contribución por dimensión (H1–H4)
  Obs. 2,9 contrastes en validación, en prueba y sobre el modelo final; X3 separada
  Obs. 3   significancia frente a baselines (pérdidas pareadas por mes, HAC, bloques)
  Obs. 4   definición del horizonte y error según la brecha al precio previo
  Obs. 8   variables disponibles al pronosticar (X2 rezagada)
  Obs. 10  sesgo por depuración y sensibilidad
  Obs. 11  particiones
  Obs. 12  interpretación de importancia (predictiva ≠ incremental ≠ causal)
  Obs. 14  alcance de las conclusiones   Obs. 15 umbral   Obs. 16 año del título

Todo lo que dependa de datos que NO están en el pipeline (fechas de publicación, fuente del umbral,
antecedentes de la literatura) queda marcado como [COMPLETAR] / [VERIFICAR]: no se inventa.
Si falta una tabla, el bloque correspondiente se omite sin fallar.
"""
import os

import numpy as np
import pandas as pd

from src.config import OUT_TAB

N = lambda v, d=4: f"{v:.{d}f}"
P = lambda v: ("<0.001" if v < 0.001 else f"{v:.3f}")
PP = lambda v: ("p < 0.001" if v < 0.001 else f"p = {v:.3f}")          # para prosa
PPH = lambda v: ("p Holm < 0.001" if v < 0.001 else f"p Holm = {v:.3f}")
MILES = lambda n: f"{int(n):,}".replace(",", " ")
NOM_DIM = {"X1_Temporal": "temporal (X1)", "X2_Productiva": "productiva (X2)", "X3_Geografica": "geográfica (X3)",
           "X3a_Territorio": "territorio (X3a)", "X3b_Producto": "producto/cultivo (X3b)",
           "X4_Economica_Historica": "económica histórica (X4)"}
MES = lambda p: f"{str(int(p))[:4]}-{str(int(p))[4:]}"

ARCHIVOS = {
    "part": "tabla_particiones.csv", "folds": "tabla_folds.csv", "motivos": "tabla_motivos_exclusion.csv",
    "sesgo": "tabla_sesgo_depuracion.csv", "chi2": "tabla_sesgo_chi2.csv", "chi2_cos": "tabla_sesgo_chi2_regla_cosecha.csv",
    "sens": "tabla_sensibilidad_depuracion.csv", "sig": "tabla_significancia_vs_baseline.csv",
    "sig_h": "tabla_significancia_vs_baseline_con_historial.csv", "sig_rob": "tabla_significancia_robustez.csv",
    "sig_rob_h": "tabla_significancia_robustez_con_historial.csv", "ic": "tabla_ic_r2_prueba.csv",
    "ic_h": "tabla_ic_r2_con_historial.csv", "brecha": "tabla_mae_por_brecha.csv",
    "abl_lgbm": "tabla_ablation_wilcoxon_lgbm.csv", "abl_x3": "tabla_ablation_wilcoxon_lgbm_x3_separada.csv",
    "abl_rf": "tabla_ablation_wilcoxon.csv", "c_lgbm": "tabla_contribucion_dimensiones_prueba_lgbm.csv",
    "c_x3": "tabla_contribucion_dimensiones_prueba_lgbm_x3_separada.csv",
    "c_res": "tabla_contribucion_dimensiones_prueba_lgbm_res.csv", "x2_fin": "tabla_variante_x2_modelo_final.csv",
    "x2_dir": "tabla_variante_x2_rezagada.csv", "fold_res": "tabla_ablacion_por_fold_resumen.csv",
    "imp": "tabla_importancia_comparada.csv", "shap": "tabla_shap_detalle.csv",
}


def cargar(out_tab=None):
    d = out_tab or OUT_TAB
    T = {}
    for k, f in ARCHIVOS.items():
        ruta = os.path.join(d, f)
        T[k] = pd.read_csv(ruta) if os.path.exists(ruta) else None
    return T


def _md(df, fmt=None):
    fmt = fmt or {}
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = []
    for _, r in df.iterrows():
        cel = []
        for c in df.columns:
            v = r[c]
            if c in fmt and pd.notna(v):
                cel.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cel.append("—" if pd.isna(v) else N(v))
            else:
                cel.append("—" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
        filas.append("| " + " | ".join(cel) + " |")
    return "\n".join([cab, sep] + filas)


# ---------------------------------------------------------------------------------------------- 4.1
def bloque_particiones(T):
    t = T["part"]
    if t is None:
        return []
    L = ["**Tabla K.** Particiones cronológicas, tamaño y disponibilidad de rezagos\n"]
    v = pd.DataFrame({
        "Conjunto": t["Conjunto"], "Desde": t["Desde"].map(MES), "Hasta": t["Hasta"].map(MES),
        "n": t["n_registros"].map(MILES), "% total": t["pct_del_total"].map(lambda x: N(x, 1)),
        "Meses": t["meses_distintos"], "Series": t["series_cultivo_distrito"],
        "% con historial previo": t["pct_con_historial_previo"].map(lambda x: N(x, 1)),
        "Brecha mediana (meses)": t["brecha_mediana_meses_al_precio_previo"].map(lambda x: N(x, 0))})
    L.append(_md(v) + "\n")
    L.append("*Nota.* La partición es cronológica y respeta límites de mes. Serie = combinación cultivo–provincia–distrito. "
             "Los rezagos (PRECIO_LAG1, MEDIA_MOVIL_3, VARIACION_PRECIO_PREVIA) se calculan con los precios OBSERVADOS de la serie, "
             "también durante la prueba: el estudio evalúa un pronóstico de un paso adelante en el que el precio previo se conoce al emitir "
             "la predicción; no se encadenan predicciones propias. Elaboración propia.\n")
    pr = t.iloc[-1]
    L.append("**Definición del horizonte de predicción.** El modelo predice el precio en chacra de la *siguiente observación disponible* de cada serie "
             "cultivo–provincia–distrito, dado el último precio observado de esa serie. Como las series son irregulares (solo hay precio cuando hubo "
             f"cosecha), la distancia entre ambas observaciones no es siempre un mes: en el conjunto de prueba la mediana es de {N(pr['brecha_mediana_meses_al_precio_previo'],0)} meses. "
             "Los períodos sin cosecha no generan observación y no se imputan; la brecha se incorpora al modelo mediante MESES_DESDE_PRECIO_PREVIO y TIENE_HISTORIAL, "
             "y su efecto sobre el error se reporta en la Tabla M.\n")
    if T["folds"] is not None:
        L.append("El detalle de los folds de validación temporal (fechas y tamaños de entrenamiento y validación del ajuste de hiperparámetros y de la ablación) "
                 "se presenta en el Anexo [COMPLETAR: Anexo] (`tabla_folds.csv`).\n")
    return L


def bloque_depuracion(T):
    mo = T["motivos"]
    if mo is None:
        return []
    L = ["**Tabla Q.** Motivos de exclusión de registros y su naturaleza\n"]
    m = mo.copy()
    m["n"] = m["n"].map(MILES)
    m["Pct"] = m["Pct"].map(lambda x: N(x, 1))
    m.columns = ["Motivo", "n", "%", "Naturaleza"]
    L.append(_md(m) + "\n")
    txt = ("De los registros iniciales solo se retiene una fracción pequeña. Se distinguen tres naturalezas: (i) *estructural*, cuando no existe precio de chacra "
           "(el objetivo no está observado, no es un dato inválido); (ii) *registro inválido*, cuando faltan variables del registro agrícola; y "
           "(iii) *regla de consistencia*, cuando COSECHA = 0 y PRODUCCIÓN > 0, que puede reflejar cosecha de otra campaña o mes y no necesariamente un error.")
    ch = T["chi2_cos"]
    cu = None if ch is None else ch[ch["Variable"] == "CULTIVO"]
    if cu is not None and len(cu):
        txt += (f" Entre los registros con precio y variables completas, la regla de cosecha depende fuertemente del cultivo "
                f"(V de Cramér = {N(float(cu['V_Cramer'].iloc[0]),2)}), por lo que la muestra de modelado no es representativa de todos los cultivos de Piura.")
    L.append(txt + "\n")
    s = T["sens"]
    if s is not None:
        L.append("**Tabla R.** Sensibilidad a la regla de depuración (mismo modelo, prueba cronológica de cada escenario)\n")
        r = s.copy()
        r = r[["Escenario", "Hiperparametros", "n_registros", "n_prueba", "R2_modelo", "MAE_modelo", "MAE_persistencia", "Reduccion_MAE_pct"]]
        r.columns = ["Escenario", "Hiperparámetros", "n", "n prueba", "R²", "MAE modelo", "MAE persistencia", "Reducción MAE (%)"]
        L.append(_md(r, {"n": MILES, "n prueba": MILES, "Reducción MAE (%)": lambda x: N(x, 1)}) + "\n")
        aj = s[s["Hiperparametros"] == "ajustados"]
        pd_ = s[s["Hiperparametros"] != "ajustados"]
        t2 = (f"*Nota.* En ambos escenarios el modelo con hiperparámetros ajustados supera a la persistencia en MAE "
              f"({', '.join(N(x,1) + ' %' for x in aj['Reduccion_MAE_pct'])}), de modo que la conclusión cualitativa no depende de la regla de cosecha, aunque el margen cambia. "
              f"Con hiperparámetros por defecto el modelo no supera a la persistencia ({', '.join(N(x,1) + ' %' for x in pd_['Reduccion_MAE_pct'])}): la ventaja depende del ajuste. "
              "Los hiperparámetros del escenario relajado se tomaron del ajuste con la muestra estricta. Elaboración propia.\n")
        L.append(t2)
    return L


# ---------------------------------------------------------------------------------------------- 4.2
def bloque_significancia(T, nom, base, m, umbral):
    L = []
    ic, sg, sh = T["ic"], T["sig"], T["sig_h"]
    if ic is not None:
        r = ic.iloc[0]
        L.append(f"**Eficacia frente al umbral.** El R² de prueba ({N(r['R2_prueba'])}) tiene un intervalo de confianza del 95 % de [{N(r['IC95_inf'])}; {N(r['IC95_sup'])}] "
                 f"(bootstrap por bloques de {int(r['bloque_meses'])} meses sobre {int(r['n_meses'])} meses de prueba). Decisión frente al umbral de {umbral}: {r['Decision_HGa'][0].lower() + r['Decision_HGa'][1:]}.\n")
    if sg is not None:
        L.append("**Tabla L.** Reducción del MAE frente a baselines y otros modelos (diferencias de pérdida pareadas por mes)\n")
        x = sg[["Referencia", "MAE_referencia", "MAE_modelo", "Reduccion_MAE_pct", "IC95_inf", "IC95_sup", "p_holm"]].copy()
        x.insert(1, "Muestra", "Todos los registros")
        if sh is not None:
            y = sh[["Referencia", "MAE_referencia", "MAE_modelo", "Reduccion_MAE_pct", "IC95_inf", "IC95_sup", "p_holm"]].copy()
            y.insert(1, "Muestra", "Solo con precio previo real")
            x = pd.concat([x, y], ignore_index=True)
            x["_o1"] = x["Referencia"].map({"Persistencia": 0, "Media móvil 3": 1}).fillna(2)
            x["_o2"] = (x["Muestra"] != "Todos los registros").astype(int)
            x = x.sort_values(["_o1", "Referencia", "_o2"]).drop(columns=["_o1", "_o2"])
        x.columns = ["Referencia", "Muestra", "MAE ref.", "MAE modelo", "Reducción (%)", "IC95 inf.", "IC95 sup.", "p (Holm)"]
        L.append(_md(x, {"Reducción (%)": lambda v: N(v, 1), "IC95 inf.": lambda v: N(v, 4), "IC95 sup.": lambda v: N(v, 4), "p (Holm)": P}) + "\n")
        L.append("*Nota.* IC95 de la diferencia media mensual de pérdida absoluta (referencia − modelo; > 0 favorece al modelo), por bootstrap de bloques móviles. "
                 "Contraste unilateral con error estándar HAC (Newey–West), corrección de Holm por las comparaciones de cada muestra. Con pocos meses de prueba el poder es limitado. Elaboración propia.\n")
        p = sg[sg["Referencia"] == "Persistencia"].iloc[0]
        txt = (f"Frente a la persistencia, el {nom} reduce el MAE en {N(p['Reduccion_MAE_pct'],1)} % (IC95 de la diferencia mensual [{N(p['IC95_inf'],3)}; {N(p['IC95_sup'],3)}]; {PPH(p['p_holm'])}). ")
        if sh is not None:
            ph = sh[sh["Referencia"] == "Persistencia"].iloc[0]
            txt += (f"Sin embargo, al excluir los registros sin precio previo —para los cuales la «persistencia» es solo un valor de respaldo (mediana) y no un pronóstico comparable— "
                    f"la reducción es de {N(ph['Reduccion_MAE_pct'],1)} % y {'sigue siendo significativa' if ph['p_holm'] < 0.05 else 'deja de ser estadísticamente concluyente'} (IC95 [{N(ph['IC95_inf'],3)}; {N(ph['IC95_sup'],3)}]; {PP(ph['p_holm'])}). ")
        rb = T["sig_rob_h"] if T["sig_rob_h"] is not None else T["sig_rob"]
        if rb is not None:
            sub = rb[rb["Referencia"] == "Persistencia"]
            cu = sub[sub["Perdida"] == "cuadrática"]
            if len(cu):
                txt += (f"Con pérdida cuadrática (que penaliza más los errores grandes) la diferencia frente a la persistencia sí es significativa en {int((cu['IC_excluye_0'] == 'Sí').sum())} de {len(cu)} configuraciones de rezagos y bloques, "
                        "por lo que la conclusión depende de la métrica elegida. ")
            ml = [r_ for r_ in rb["Referencia"].unique() if r_ not in ("Persistencia", "Media móvil 3", "Regresión Lineal")]
            sin_sig = []
            for r_ in ml:
                c_ = rb[(rb["Referencia"] == r_) & (rb["Perdida"] == "cuadrática")]
                if len(c_) and (c_["IC_excluye_0"] == "Sí").sum() == 0:
                    sin_sig.append(r_)
            if sin_sig:
                txt += (f"Frente a {', '.join(sin_sig)}, la ventaja se observa con pérdida absoluta pero no con pérdida cuadrática; el modelo se seleccionó por MAE y no es superior en términos de errores grandes. ")
        L.append(txt.strip() + "\n")
    return L


def bloque_brecha(T):
    b = T["brecha"]
    if b is None:
        return []
    L = ["### 4.2.6. Error según la distancia al último precio observado\n",
         "**Tabla M.** Desempeño en prueba según los meses transcurridos desde el último precio de la serie\n"]
    x = b[["Grupo", "N", "Pct", "MAE_modelo", "MAE_persistencia", "Reduccion_MAE_pct", "R2_modelo"]].copy()
    x.columns = ["Grupo", "N", "% de la prueba", "MAE modelo", "MAE persistencia", "Reducción (%)", "R² modelo"]
    L.append(_md(x, {"N": MILES, "% de la prueba": lambda v: N(v, 1), "Reducción (%)": lambda v: N(v, 1)}) + "\n")
    L.append("*Nota.* En «sin historial previo» la persistencia usa la mediana de desarrollo y no es comparable. Elaboración propia.\n")
    c = b.dropna(subset=["Reduccion_MAE_pct"])
    peor = c[c["Reduccion_MAE_pct"] < 0]
    mejor = c[c["Reduccion_MAE_pct"] > 0]
    if len(peor) and len(mejor):
        L.append(f"El modelo no supera a la persistencia cuando el último precio es reciente ({'; '.join(f'{g}: {N(r,1)} %' for g, r in zip(peor['Grupo'], peor['Reduccion_MAE_pct']))}) "
                 f"y su ventaja aparece cuando la brecha es mayor ({'; '.join(f'{g}: +{N(r,1)} %' for g, r in zip(mejor['Grupo'], mejor['Reduccion_MAE_pct']))}). "
                 "La mejora global, por tanto, proviene sobre todo de series con precios previos antiguos, donde la persistencia es menos informativa.\n")
    return L


# ---------------------------------------------------------------------------------------------- 4.3
def clasificar(T, alpha):
    ab, ct, cr = T["abl_lgbm"], T["c_lgbm"], T["c_res"]
    if ab is None or ct is None:
        return None
    ab, ct = ab.set_index("Dimension"), ct.set_index("Dimension")
    cr = None if cr is None else cr.set_index("Dimension")
    mapa = {"X1_Temporal": ["X1_Temporal"], "X2_Productiva": ["X2_Productiva"],
            "X3_Geografica": ["X3a_Territorio", "X3b_Producto"], "X4_Economica_Historica": []}
    out = {}
    for k in ab.index:
        val = bool(ab.loc[k, "p_mae_holm"] < alpha) if "p_mae_holm" in ab.columns else bool(ab.loc[k, "p_holm"] < alpha)
        pru = bool(ct.loc[k, "p_holm"] < alpha)
        fin = None
        if cr is not None and mapa.get(k):
            fin = any(bool(cr.loc[d, "p_holm"] < alpha) for d in mapa[k] if d in cr.index)
        res = [val, pru] + ([] if fin is None else [fin])
        estado = "RESPALDADA" if all(res) else ("NO RESPALDADA" if not any(res) else "PARCIALMENTE RESPALDADA")
        out[k] = {"validacion": val, "prueba": pru, "final": fin, "estado": estado}
    return out


def bloque_hipotesis(T, clasif, alpha, nom, base, m, mm3, umbral, cumple):
    L = []
    L.append("### 4.3.1. Formulación operativa de las hipótesis\n")
    L.append("Las hipótesis se contrastan como *contribución predictiva* y no como efecto causal, porque el estudio es observacional: "
             "que una dimensión mejore el pronóstico fuera de muestra no demuestra que determine el precio.\n")
    L.append(f"- **HG-a (eficacia):** el modelo seleccionado alcanza R² ≥ {umbral} en el conjunto de prueba. Se contrasta con el R² puntual y su intervalo de confianza por bootstrap de bloques de meses.\n"
             "- **HG-b (superación de baselines):** el modelo reduce el error de pronóstico frente a la persistencia y la media móvil. Se contrasta con diferencias de pérdida pareadas por mes (errores HAC, IC por bloques, Holm).\n"
             "- **H1–H4 (contribución por dimensión):** excluir la dimensión empeora el pronóstico fuera de muestra. Se contrasta en tres niveles: (i) validación temporal con LightGBM directo (ΔMAE y ΔR², Wilcoxon unilateral, Holm); "
             "(ii) conjunto de prueba con LightGBM directo (pérdidas pareadas por mes); (iii) conjunto de prueba sobre el modelo final (LightGBM sobre persistencia), donde X4 no puede retirarse porque es su ancla.\n")
    L.append("Criterio de reporte: una hipótesis se declara *respaldada* si resulta significativa (p Holm < " + str(alpha) + ") en todos los contrastes aplicables, "
             "*parcialmente respaldada* si lo es solo en alguno, y *no respaldada* si en ninguno. [COMPLETAR en 3.7.1: este criterio de consistencia debe figurar en la metodología; indica con honestidad que se adoptó al revisar las observaciones.]\n")

    L.append("### 4.3.2. Resultados de los contrastes\n")
    ab = T["abl_lgbm"]
    if ab is not None:
        L.append("**Tabla I.** Ablación por dimensión en validación temporal (LightGBM directo)\n")
        a = ab[["Dimension", "R2_completo", "R2_sin_dim", "Diferencia_R2", "p_holm", "MAE_completo", "MAE_sin_dim", "Diferencia_MAE", "p_mae_holm"]].copy()
        a.columns = ["Dimensión excluida", "R² completo", "R² sin dim.", "ΔR²", "p Holm (R²)", "MAE completo", "MAE sin dim.", "ΔMAE", "p Holm (MAE)"]
        L.append(_md(a, {"p Holm (R²)": P, "p Holm (MAE)": P}) + "\n")
        L.append("*Nota.* Wilcoxon pareado unilateral sobre los folds temporales; ΔMAE = MAE sin la dimensión − MAE completo (> 0 favorece incluir la dimensión). "
                 "Los folds de TimeSeriesSplit se solapan y no son independientes: los p-valores deben leerse con esa cautela. Elaboración propia.\n")
    if T["abl_x3"] is not None:
        L.append("**Tabla O.** Ablación con la dimensión geográfica separada en territorio y producto (Holm sobre cinco pruebas)\n")
        a = T["abl_x3"][["Dimension", "Diferencia_R2", "p_holm", "Diferencia_MAE", "p_mae_holm"]].copy()
        a.columns = ["Dimensión excluida", "ΔR²", "p Holm (R²)", "ΔMAE", "p Holm (MAE)"]
        L.append(_md(a, {"p Holm (R²)": P, "p Holm (MAE)": P}) + "\n")
        L.append("*Nota.* El cultivo identifica el producto, no el territorio; por eso se evalúa por separado de provincia y distrito. Elaboración propia.\n")
    if T["c_x3"] is not None:
        L.append("**Tabla N.** Contribución de cada dimensión en el conjunto de prueba (pérdidas pareadas por mes)\n")
        a = T["c_x3"][["Dimension", "Dif_MAE_mensual", "IC95_inf", "IC95_sup", "p_holm"]].copy()
        a.insert(0, "Modelo", "LightGBM directo")
        if T["c_res"] is not None:
            b = T["c_res"][["Dimension", "Dif_MAE_mensual", "IC95_inf", "IC95_sup", "p_holm"]].copy()
            b.insert(0, "Modelo", "Modelo final (LightGBM sobre persistencia)")
            a = pd.concat([a, b], ignore_index=True)
        a.columns = ["Modelo", "Dimensión excluida", "Dif. MAE mensual", "IC95 inf.", "IC95 sup.", "p (Holm)"]
        L.append(_md(a, {"Dif. MAE mensual": lambda v: N(v, 4), "IC95 inf.": lambda v: N(v, 4), "IC95 sup.": lambda v: N(v, 4), "p (Holm)": P}) + "\n")
        L.append("*Nota.* Diferencia = MAE sin la dimensión − MAE completo, promediada por mes; > 0 indica que la dimensión ayuda. "
                 "En el modelo final X4 no se retira porque es la base de la persistencia. Elaboración propia.\n")
    fr = T["fold_res"]
    if fr is not None and "X3b_Producto" in set(fr["Dimension"]):
        r = fr[fr["Dimension"] == "X3b_Producto"].iloc[0]
        L.append(f"**Inestabilidad de la dimensión geográfica.** En la validación temporal el efecto de X3 proviene del cultivo (producto) y no del territorio, mientras que en la prueba ocurre lo contrario. "
                 f"El efecto del producto es mayor en los folds con menos registros con historial (correlación de Spearman entre ΔMAE y % con historial = {N(r['Spearman_dif_vs_pct_historial'],2)}; {PP(r['p_Spearman'])}), "
                 "lo que sugiere que el cultivo actúa como sustituto del nivel de precio cuando falta el precio previo y se vuelve redundante cuando existe. Es una interpretación compatible con los datos, no una prueba causal.\n")
    cl = T["c_lgbm"]
    cx = T["c_x3"]
    if cl is not None and cx is not None and "X3_Geografica" in set(cl["Dimension"]) and "X3a_Territorio" in set(cx["Dimension"]):
        g = cl[cl["Dimension"] == "X3_Geografica"].iloc[0]
        a_ = cx[cx["Dimension"] == "X3a_Territorio"].iloc[0]
        b_ = cx[cx["Dimension"] == "X3b_Producto"].iloc[0]
        if a_["p_holm"] < alpha and g["p_holm"] >= alpha:
            L.append(f"**Efecto no aditivo.** En prueba, retirar solo el territorio empeora el pronóstico (ΔMAE = {N(a_['Dif_MAE_mensual'],4)}; {PPH(a_['p_holm'])}) y retirar solo el producto no lo hace ({PPH(b_['p_holm'])}), "
                     f"pero retirar ambos a la vez tampoco produce una pérdida significativa (ΔMAE = {N(g['Dif_MAE_mensual'],4)}; {PPH(g['p_holm'])}). "
                     "Territorio y cultivo se sustituyen mutuamente (redundancia: cada uno identifica en parte la misma serie), por lo que sus efectos no se suman y no pueden atribuirse por separado con seguridad.\n")
    x2 = T["x2_fin"]
    if x2 is not None:
        L.append("**Tabla P.** Variante con la dimensión productiva rezagada (modelo final, conjunto de prueba)\n")
        a = x2[["Escenario", "n_variables", "R2", "MAE", "Dif_MAE_vs_original", "IC95_inf", "IC95_sup", "p_HAC_original_mejor"]].copy()
        a.columns = ["Escenario", "Variables", "R²", "MAE", "Dif. MAE vs original", "IC95 inf.", "IC95 sup.", "p (HAC)"]
        L.append(_md(a, {"Variables": lambda v: str(int(v)), "p (HAC)": P}) + "\n")
        L.append("*Nota.* «X2 rezagada» usa siembra, cosecha, producción y rendimiento de la observación previa de la serie, que están disponibles al emitir el pronóstico. "
                 "Si la diferencia no es significativa, el desempeño no depende de información contemporánea al precio. Elaboración propia.\n")

    # decisiones
    L.append("### 4.3.3. Decisión sobre las hipótesis\n")
    ic, sg, sh = T["ic"], T["sig"], T["sig_h"]
    if ic is not None:
        r = ic.iloc[0]
        ha = "NO RESPALDADA" if r["IC95_sup"] < umbral else ("RESPALDADA" if r["IC95_inf"] >= umbral else "NO CONCLUYENTE")
        L.append(f"**HG-a (eficacia): {ha}.** R² = {N(r['R2_prueba'])} (IC95 [{N(r['IC95_inf'])}; {N(r['IC95_sup'])}]) frente al umbral de {umbral}. "
                 "No alcanzar el umbral no invalida el estudio: indica que el modelo no cumple el criterio de eficacia fijado para este conjunto de datos.\n")
    if sg is not None:
        p = sg[sg["Referencia"] == "Persistencia"].iloc[0]
        q = sg[sg["Referencia"] == "Media móvil 3"].iloc[0]
        ok_p = p["p_holm"] < alpha
        ok_ph = (sh is not None) and (sh[sh["Referencia"] == "Persistencia"].iloc[0]["p_holm"] < alpha)
        ok_q = q["p_holm"] < alpha
        est = "RESPALDADA" if (ok_p and ok_ph and ok_q) else ("NO RESPALDADA" if not (ok_p or ok_q) else "PARCIALMENTE RESPALDADA")
        L.append(f"**HG-b (superación de baselines): {est}.** Frente a la media móvil la reducción es de {N(q['Reduccion_MAE_pct'],1)} % ({PPH(q['p_holm'])}). "
                 f"Frente a la persistencia es de {N(p['Reduccion_MAE_pct'],1)} % ({PPH(p['p_holm'])}) con todos los registros"
                 + (f", pero no es concluyente con registros que tienen precio previo real ({PP(sh[sh['Referencia']=='Persistencia'].iloc[0]['p_holm'])})" if (sh is not None and not ok_ph) else "") + ".\n")
    if clasif:
        nombres = {"X1_Temporal": "H1 — Dimensión temporal", "X2_Productiva": "H2 — Dimensión productiva",
                   "X3_Geografica": "H3 — Dimensión geográfica", "X4_Economica_Historica": "H4 — Dimensión económica histórica"}
        filas = []
        for k, c in clasif.items():
            sn = lambda b: "—" if b is None else ("Sí" if b else "No")
            filas.append([nombres[k], sn(c["validacion"]), sn(c["prueba"]), sn(c["final"]), c["estado"]])
        d = pd.DataFrame(filas, columns=["Hipótesis", "Significativa en validación temporal", "Significativa en prueba (LightGBM directo)", "Significativa en prueba (modelo final)", "Estado"])
        L.append("**Tabla S.** Consistencia de los contrastes por dimensión\n")
        L.append(_md(d) + "\n")
        L.append("*Nota.* «—» = no aplicable (X4 es la base del modelo final). Significancia: p Holm < " + str(alpha) + ". La no aceptación de una hipótesis indica ausencia de aporte predictivo incremental detectable en este conjunto de datos, no ausencia de relación económica. Elaboración propia.\n")
        for k, c in clasif.items():
            L.append(f"**{nombres[k]}: {c['estado']}.** " + _frase_dim(k, c, T) + "\n")
    # interpretación
    L.append("### 4.3.4. Cómo interpretar la importancia de las dimensiones\n")
    imp = T["imp"]
    txt = ("La importancia por ganancia, SHAP y permutación mide cuánto usa el modelo cada dimensión (*importancia predictiva*); la ablación mide cuánto empeora el pronóstico al retirarla (*aporte incremental*); "
           "ninguna de las dos estima un *efecto causal*. Pueden diferir porque las variables están correlacionadas. ")
    if imp is not None and T["shap"] is not None:
        s = T["shap"]
        dim = s.groupby("Dimension")["Pct_total"].sum()
        top = dim.idxmax()
        txt += (f"El porcentaje SHAP de cada dimensión se calcula como la suma del valor absoluto medio de SHAP de sus variables dividida por la suma de todas las variables (X4 = {N(dim[top],1)} %), "
                f"sobre una muestra de {MILES(s['n_registros_muestra'].iloc[0])} registros de prueba con historial, y está expresado en la escala log(precio / último precio): es el cambio relativo respecto de la persistencia, no soles. ")
        x2 = imp[imp["Dimension"] == "X2_Productiva"].iloc[0]
        txt += (f"Un ejemplo de la divergencia es X2: {N(x2['Ganancia_pct'],1)} % por ganancia, {N(x2['SHAP_pct'],1)} % por SHAP y un aporte incremental de {N(x2['Aporte_incremental_R2'],4)} de R².")
    L.append(txt + "\n")
    return L


def _frase_dim(k, c, T):
    if k in ("X1_Temporal", "X2_Productiva"):
        s = "No se detecta aporte en validación, en prueba ni en el modelo final."
        if k == "X2_Productiva" and T["x2_fin"] is not None:
            r = T["x2_fin"]
            lag = r[r["Escenario"].str.contains("rezagada")]
            if len(lag):
                s += (f" La variante con X2 rezagada no cambia el desempeño de forma concluyente (ΔMAE = {N(float(lag['Dif_MAE_vs_original'].iloc[0]),4)}; {PP(float(lag['p_HAC_original_mejor'].iloc[0]))}), "
                      "por lo que no hay evidencia de que el resultado dependa de información contemporánea al precio.")
        return s
    if k == "X3_Geografica":
        return ("El aporte aparece en la validación temporal de los modelos directos, pero no se replica sobre el conjunto de prueba con el mismo algoritmo ni sobre el modelo final; "
                "al separar territorio y producto, el origen del efecto cambia entre validación y prueba. La evidencia es inestable y no permite afirmar que la dimensión geográfica contribuya al modelo desplegado.")
    return ("Excluir la historia de precios empeora de forma significativa el pronóstico en validación y en prueba. Es la dimensión que concentra la capacidad predictiva, lo que es coherente con la inercia de los precios; "
            "en el modelo final es además su ancla estructural y no puede retirarse.")


# ---------------------------------------------------------------------------------------------- 4.4
def bloque_discusion(T, clasif, nom, m, base, umbral, cumple, mejor_ml):
    L = ["## 4.4. DISCUSIÓN\n",
         "*Estructura por objetivo (hallazgo → antecedentes → coincidencias y diferencias → explicación técnica → aporte y límites). Los antecedentes deben completarse con fuentes verificadas: "
         "no se incluyen citas que no figuren verificadas en el borrador.*\n"]
    ic = T["ic"]
    r2ic = "" if ic is None else f" (IC95 [{N(ic.iloc[0]['IC95_inf'])}; {N(ic.iloc[0]['IC95_sup'])}])"
    L.append("### 4.4.1. Objetivo general: eficacia predictiva\n")
    L.append(f"**Hallazgo.** El {nom} alcanzó R² = {N(m['R2'])}{r2ic}, MAE = {N(m['MAE'])} S/ por kg y MAPE = {N(m['MAPE'],1)} % en un conjunto de prueba cronológico, y no cumple el umbral de {umbral}. "
             f"En R² es superado por {mejor_ml[0]} ({N(mejor_ml[1])}); su ventaja está en el MAE. "
             "**Antecedentes.** [COMPLETAR: 2–3 estudios nacionales e internacionales sobre precios agrícolas con autor, año, cultivos, algoritmo, horizonte, esquema de validación y métrica reportada.] "
             "**Coincidencias y diferencias.** [COMPLETAR] Los R² de otros estudios solo son comparables si el esquema de validación es cronológico y el objetivo es el mismo: la validación aleatoria, el horizonte, la agregación por cultivo y la presencia de rezagos del propio precio inflan el R². "
             f"En este estudio la persistencia ya alcanza R² = {N(base['R2'])}, de modo que gran parte de la varianza explicada proviene de la inercia del precio. "
             "**Aporte y límite.** El aporte es una evaluación cronológica, con pérdidas pareadas e incertidumbre, sobre series irregulares de Piura; no autoriza a generalizar a otras regiones ni a afirmar beneficios económicos.\n")
    L.append("### 4.4.2. Superación de baselines y horizonte\n")
    sg = T["sig"]
    if sg is not None:
        p = sg[sg["Referencia"] == "Persistencia"].iloc[0]
        L.append(f"**Hallazgo.** La reducción del MAE frente a la persistencia es de {N(p['Reduccion_MAE_pct'],1)} % ({PPH(p['p_holm'])}), pero depende de la muestra, de la métrica y de la distancia al último precio (Tablas L y M). "
                 "**Explicación técnica.** Cuando el último precio es reciente, repetirlo es difícil de superar; el modelo aporta cuando ese precio es antiguo. "
                 "**Antecedentes.** [COMPLETAR: estudios que comparen contra persistencia o modelos ingenuos; si no existen, dilo.] "
                 "**Aporte y límite.** Un modelo de Machine Learning es valioso aquí sobre todo en series con precios previos antiguos; la mejora global es modesta.\n")
    nombres = {"X1_Temporal": "temporal", "X2_Productiva": "productiva", "X3_Geografica": "geográfica", "X4_Economica_Historica": "económica histórica"}
    for i, (k, c) in enumerate((clasif or {}).items(), 3):
        L.append(f"### 4.4.{i}. Dimensión {nombres[k]}\n")
        L.append(f"**Hallazgo.** {c['estado'].capitalize()}: " + _frase_dim(k, c, T) + " "
                 "**Antecedentes.** [COMPLETAR: estudios que incorporen esta clase de variables, con su dataset, algoritmo, horizonte y validación.] "
                 "**Coincidencias y diferencias.** [COMPLETAR] Si un estudio reporta aporte de esta dimensión, revisa si usó validación aleatoria, otro cultivo único o variables que ya contenían el precio. "
                 "**Aporte y límite.** Contribución predictiva en este conjunto de datos; no es evidencia de causalidad, y la ausencia de aporte no implica ausencia de relación económica.\n")
    return L


def limitaciones(T):
    L = ["### Limitaciones (4.4.7)\n",
         "- No se incorporaron variables climáticas (SENAMHI, ONI) ni de mercado externas.\n"
         "- La dimensión geográfica usa codificación ordinal de provincia, distrito y cultivo, sin coordenadas ni embeddings; su efecto es inestable entre validación y prueba.\n"
         "- Los precios solo existen cuando hubo cosecha; las series son irregulares y la distancia al último precio varía (Tabla M).\n"
         "- La muestra de modelado es una fracción pequeña del registro inicial y no es representativa de todos los cultivos (Tablas Q y R); los resultados no se generalizan a toda la región.\n"
         "- Pocos meses de prueba limitan la potencia de los contrastes; los folds de validación temporal se solapan y no son independientes.\n"
         "- El desempeño es menor en cultivos de alto valor y baja frecuencia (Tabla H).\n"
         "- La validación es retrospectiva (pronóstico de un paso con rezagos observados); no se evaluó deriva conceptual ni beneficio económico.\n"]
    return L


# ---------------------------------------------------------------------------------------------- 5
def conclusiones(T, clasif, nom, m, base, ft, umbral, cumple, mejor_ml, shp_txt, anom, n_test):
    sg, ic = T["sig"], T["ic"]
    C = []
    ics = "" if ic is None else f" (IC95 [{N(ic.iloc[0]['IC95_inf'])}; {N(ic.iloc[0]['IC95_sup'])}])"
    C.append(f"1. **Objetivo general.** Se desarrolló y evaluó, con una metodología de nueve fases, un modelo de predicción del precio en chacra para Piura. El modelo seleccionado en validación, {nom}, obtuvo en prueba "
             f"(n = {MILES(n_test)}) R² = {N(m['R2'])}{ics}, MAE = {N(m['MAE'])} S/ por kg y MAPE = {N(m['MAPE'],1)} %; el criterio R² ≥ {umbral} {'se cumple' if cumple else 'no se cumple'}.")
    if sg is not None:
        p = sg[sg["Referencia"] == "Persistencia"].iloc[0]
        sh = T["sig_h"]
        ph = None if sh is None else sh[sh["Referencia"] == "Persistencia"].iloc[0]
        C.append(f"2. **Comparación con referencias.** El modelo reduce el MAE frente a la persistencia en {N(p['Reduccion_MAE_pct'],1)} % ({PPH(p['p_holm'])})"
                 + (f", aunque la reducción es de {N(ph['Reduccion_MAE_pct'],1)} % y no concluyente ({PP(ph['p_holm'])}) al considerar solo registros con precio previo real" if (ph is not None and ph['p_holm'] >= 0.05) else "")
                 + f"; supera a la media móvil y a la regresión lineal en MAE. En R², {mejor_ml[0]} ({N(mejor_ml[1])}) supera al modelo seleccionado."
                 + _clausula_ml(T)
                 + " La mejora técnica es modesta y depende de la métrica y de la distancia al último precio.")
    b = T["brecha"]
    if b is not None:
        c = b.dropna(subset=["Reduccion_MAE_pct"])
        C.append("3. **Horizonte.** El modelo predice la siguiente observación disponible de cada serie; su ventaja sobre la persistencia se concentra en brechas largas "
                 + "(" + "; ".join(f"{g}: {r:+.1f} %" for g, r in zip(c['Grupo'], c['Reduccion_MAE_pct'])) + ").")
    if clasif:
        nm = {"X1_Temporal": "temporal", "X2_Productiva": "productiva", "X3_Geografica": "geográfica", "X4_Economica_Historica": "económica histórica"}
        for i, (k, c) in enumerate(clasif.items(), 4):
            C.append(f"{i}. **Dimensión {nm[k]}.** Hipótesis {c['estado'].lower()}. " + _frase_dim(k, c, T))
        n0 = 4 + len(clasif)
    else:
        n0 = 4
    C.append(f"{n0}. **Interpretabilidad.** {shp_txt} Las medidas de importancia describen el uso predictivo de las variables y no efectos causales.")
    C.append(f"{n0+1}. **Anomalías.** Isolation Forest marcó {N(anom['pct'],2)} % de los registros; los anómalos tienen un precio medio de S/ {N(anom['precio_anomalo'],3)} frente a S/ {N(anom['precio_normal'],3)} de los normales.")
    C.append(f"{n0+2}. **Alcance.** Los resultados demuestran un desempeño técnico en un conjunto de prueba retrospectivo (la reducción del MAE frente a la persistencia equivale a unos S/ {N(base['MAE'] - m['MAE'], 2)} por kg); "
             "no demuestran beneficios económicos ni mejores decisiones agrícolas, que requieren evaluación con usuarios o datos operativos.")
    C.append(f"{n0+3}. **Sistema web.** *(Completar con mediciones propias: tiempo de respuesta, disponibilidad y SUS; no provienen de este pipeline.)*")
    return C


def _clausula_ml(T):
    """Frente a otros modelos de ML la ventaja del MAE no se sostiene con pérdida cuadrática."""
    rb_ = T["sig_rob"]
    if rb_ is None:
        return ""
    sin = []
    for r_ in rb_["Referencia"].unique():
        if r_ in ("Persistencia", "Media móvil 3", "Regresión Lineal"):
            continue
        c_ = rb_[(rb_["Referencia"] == r_) & (rb_["Perdida"] == "cuadrática")]
        if len(c_) and (c_["IC_excluye_0"] == "Sí").sum() == 0:
            sin.append(r_)
    if not sin:
        return ""
    return f" Frente a {', '.join(sin)} la ventaja se observa con pérdida absoluta (MAE) pero no con pérdida cuadrática."


# ---------------------------------------------------------------------------------------------- Cap. III
def bloque_capitulo3(T, umbral, anio_datos="2015–2024"):
    L = ["# TEXTO PARA EL CAPÍTULO III (metodología) — responde Obs. 4, 8, 11, 15 y 16\n",
         "## 3.x Disponibilidad de los predictores al momento del pronóstico (Obs. 8)\n"]
    d = pd.DataFrame([
        ["X1 Temporal", "Calendario (año, mes, estación)", "Conocida de antemano", "Sí"],
        ["X2 Productiva", "Registro agrícola regional (siembra, cosecha, producción, rendimiento)", "[COMPLETAR: periodicidad y fecha de publicación]", "[VERIFICAR]"],
        ["X3 Geográfica", "Catálogo de cultivo, provincia y distrito", "Estática", "Sí"],
        ["X4 Económica histórica", "Precios de chacra de observaciones anteriores de la misma serie", "[COMPLETAR: rezago de publicación del precio previo]", "[VERIFICAR]"]],
        columns=["Dimensión", "Fuente", "Disponibilidad", "¿Disponible antes del pronóstico?"])
    L.append(_md(d) + "\n")
    x2 = T["x2_fin"]
    if x2 is not None:
        lag = x2[x2["Escenario"].str.contains("rezagada")]
        if len(lag):
            L.append(f"Como control, se repitió la evaluación del modelo final con la dimensión productiva rezagada (valores de la observación previa) y sin ella: la diferencia de MAE frente al original es de {N(float(lag['Dif_MAE_vs_original'].iloc[0]),4)} ({PP(float(lag['p_HAC_original_mejor'].iloc[0]))}), "
                     "es decir, el desempeño no depende de información contemporánea al precio (Tabla P).\n")
    L.append("## 3.x Operacionalización de la dimensión geográfica (Obs. 9)\n")
    L.append("La dimensión geográfica se definió con provincia, distrito y cultivo codificados de forma ordinal. El cultivo identifica el *producto* y no el territorio; "
             "se mantuvo dentro de X3 para conservar la matriz de consistencia original [COMPLETAR: justifica por qué el cultivo se clasificó como geográfico o, si prefieres, muévelo a una dimensión de producto]. "
             "Como análisis complementario, X3 se separó en territorio (provincia y distrito) y producto (cultivo) y se repitieron la ablación y el contraste en prueba con corrección de Holm sobre cinco pruebas (Tablas O y N).\n")
    L.append("## 3.x Definición del horizonte y de la validación temporal (Obs. 4 y 11)\n")
    L.append("El modelo pronostica la siguiente observación disponible de cada serie cultivo–provincia–distrito dado su último precio observado (pronóstico de un paso adelante). "
             "Los intervalos sin cosecha no tienen precio y no se imputan; la distancia al último precio se incorpora como variable y se reporta como estratificación del error. "
             "La partición cronológica 70/15/15, las fechas, los tamaños y el uso de rezagos con precios observados se detallan en la Tabla K.\n")
    L.append("## 3.x Criterio de eficacia (Obs. 15)\n")
    L.append(f"El criterio R² ≥ {umbral} se adoptó como umbral de aceptabilidad porque [COMPLETAR con la verdad: antecedente comparable (cita verificada), estándar técnico o decisión metodológica fijada antes de evaluar el modelo en prueba]. "
             "No se modifica después de conocer los resultados. No alcanzarlo no invalida el estudio; se interpreta como que el modelo no cumple el criterio de eficacia fijado y se complementa con criterios relativos (mejora frente a baselines con intervalos de confianza).\n")
    L.append("## 3.x Alcance temporal (Obs. 16)\n")
    L.append(f"El año 2025 del título corresponde a [COMPLETAR: año de ejecución / contexto de aplicación]. Los registros agrícolas analizados abarcan {anio_datos}; no se evaluaron registros de 2025. "
             "Alternativa: titular «Piura, 2015–2024».\n")
    return L


CORRECCIONES_EXTRA = (
    "- **Obs. 13 (referencias):** Tabla A y marco teórico: sustituye Sokolova y Lapalme (2009) —métricas de clasificación— por una fuente de métricas de regresión verificada (p. ej. Hyndman y Koehler, 2006); "
    "corrige o elimina Enrique y Castillo (2020), Mena Salgado et al. (2025) y Yuan (2024) según `docs/verificacion_referencias.md`.\n"
    "- **Obs. 1:** reemplaza «influye significativamente» por «contribuye predictivamente» en la matriz de consistencia y en 1.6.1; separa HG-a y HG-b (ver 4.3.1).\n"
    "- **Obs. 14:** las conclusiones hablan de desempeño técnico; elimina cualquier afirmación de beneficio económico sin evidencia.\n")


# ---------------------------------------------------------------------------------------------- guía de sustentación
def qa_baseline(T, m, base, mm3):
    """Respuesta a «¿es mejor que repetir el último precio?» con los números y la incertidumbre."""
    txt = (f"En prueba el MAE del modelo es {N(m['MAE'])} frente a {N(base['MAE'])} de la persistencia (R² {N(m['R2'])} vs {N(base['R2'])}); "
           f"la media móvil de 3 obtiene un R² casi igual ({N(mm3['R2'])}). ")
    sg, sh = T["sig"], T["sig_h"]
    if sg is not None:
        p = sg[sg["Referencia"] == "Persistencia"].iloc[0]
        txt += f"La reducción del MAE es de {N(p['Reduccion_MAE_pct'],1)} % (IC95 de la diferencia mensual [{N(p['IC95_inf'],3)}; {N(p['IC95_sup'],3)}]; {PPH(p['p_holm'])}). "
        if sh is not None:
            ph = sh[sh["Referencia"] == "Persistencia"].iloc[0]
            txt += (f"Pero con solo los registros que tienen precio previo real la reducción es de {N(ph['Reduccion_MAE_pct'],1)} % "
                    f"({PPH(ph['p_holm'])}; IC95 [{N(ph['IC95_inf'],3)}; {N(ph['IC95_sup'],3)}]), "
                    f"{'todavía significativa' if ph['p_holm'] < 0.05 else 'no concluyente'}. ")
    b = T["brecha"]
    if b is not None:
        c = b.dropna(subset=["Reduccion_MAE_pct"])
        peor = c[c["Reduccion_MAE_pct"] < 0]
        if len(peor):
            txt += ("Además, cuando el último precio es reciente ("
                    + "; ".join(f"{g}: {r:+.1f} %" for g, r in zip(peor["Grupo"], peor["Reduccion_MAE_pct"]))
                    + ") el modelo no supera a la persistencia; su ventaja está en brechas largas (Tabla M). ")
    return txt + "Hay que presentarlo como una mejora modesta y dependiente de la métrica, no como superioridad rotunda."


def qa_hipotesis(clasif, T):
    """Respuesta a «¿qué hipótesis específicas se respaldaron?» según la Tabla S."""
    if not clasif:
        return "Corre el pipeline completo (sin --sin-extras) para obtener la consistencia de los contrastes (Tabla S)."
    nm = {"X1_Temporal": "temporal (H1)", "X2_Productiva": "productiva (H2)", "X3_Geografica": "geográfica (H3)", "X4_Economica_Historica": "económica histórica (H4)"}
    partes = [f"{nm[k]}: {c['estado'].lower()}" for k, c in clasif.items()]
    txt = "Según la consistencia entre validación temporal, prueba y modelo final (Tabla S): " + "; ".join(partes) + ". "
    if clasif.get("X4_Economica_Historica", {}).get("estado") != "NO RESPALDADA":
        txt += ("La información predictiva se concentra en la historia de precios; las demás dimensiones no añaden un aporte detectable y consistente una vez que se conoce el precio previo. ")
    if clasif.get("X3_Geografica", {}).get("estado", "").startswith("PARCIAL"):
        txt += ("La dimensión geográfica es inestable: en validación el efecto es del cultivo y en prueba del territorio, y no se sostiene en el modelo final. ")
    return txt + "No respaldar una hipótesis específica no invalida el trabajo: indica en qué dimensiones hay información predictiva incremental en este conjunto de datos, y no prueba ausencia de relación económica."
