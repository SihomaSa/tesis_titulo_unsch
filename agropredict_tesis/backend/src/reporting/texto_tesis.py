"""
texto_tesis.py — Genera el texto de los capítulos IV y V con los NÚMEROS REALES
de la corrida. El redactado de aceptación/rechazo de hipótesis depende de los
resultados: si cambia el dataset o la semilla, el texto cambia con ellos y
nunca queda desfasado de las tablas.

Salidas (outputs/tesis_texto/):
  - texto_resultados_y_conclusiones.md / .docx : pegar en el borrador
  - guia_sustentacion.md                       : preguntas probables + respuestas con tus números
"""
import os

import pandas as pd

from src.config import ALPHA, MATRIZ_CONSISTENCIA, OUT_TXT, UMBRAL_R2_OFICIAL
from src.models.pipelines import NOMBRES
from src.reporting import texto_robustez as rb

N = lambda v, d=4: f"{v:.{d}f}"
P = lambda v: ("<0.001" if v < 0.001 else f"{v:.3f}")
MILES = lambda n: f"{int(n):,}".replace(",", " ")
BONITO = {"X1_Temporal": "temporal", "X2_Productiva": "productiva", "X3_Geografica": "geográfica", "X4_Economica_Historica": "económica histórica"}
MES = lambda p: f"{str(int(p))[:4]}-{str(int(p))[4:]}"


def _md_tabla(df, formatos=None):
    formatos = formatos or {}
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = []
    for _, r in df.iterrows():
        cel = []
        for c in df.columns:
            v = r[c]
            if c in formatos and pd.notna(v):
                cel.append(formatos[c](v))
            elif isinstance(v, float):
                cel.append("—" if pd.isna(v) else N(v))
            else:
                cel.append(str(v))
        filas.append("| " + " | ".join(cel) + " |")
    return "\n".join([cab, sep] + filas)


def _inline(par, texto, size=None, negrita=False):
    """Convierte **negrita**, *cursiva* y `código` de markdown en formato real de Word."""
    import re
    from docx.shared import Pt
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`)", texto):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size)
        if negrita:
            r.bold = True


def _a_docx(md: str, ruta: str):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(12)
    lineas = md.split("\n")
    i = 0
    while i < len(lineas):
        l = lineas[i]
        if l.startswith("|"):
            bloque = []
            while i < len(lineas) and lineas[i].startswith("|"):
                bloque.append(lineas[i]); i += 1
            filas = [[c.strip() for c in b.strip("|").split("|")] for b in bloque if not set(b) <= set("|- ")]
            t = doc.add_table(rows=len(filas), cols=len(filas[0])); t.style = "Table Grid"
            for r, fila in enumerate(filas):
                for c, val in enumerate(fila):
                    celda = t.cell(r, c)
                    celda.text = ""
                    _inline(celda.paragraphs[0], val, size=9, negrita=(r == 0))
            continue
        if l.startswith("### "): doc.add_heading(l[4:], 3)
        elif l.startswith("## "): doc.add_heading(l[3:], 2)
        elif l.startswith("# "): doc.add_heading(l[2:], 1)
        elif l.startswith("- "): _inline(doc.add_paragraph(style="List Bullet"), l[2:])
        elif l.strip(): _inline(doc.add_paragraph(), l)
        i += 1
    doc.save(ruta)


def generar_texto(df, embudo, info_split, res, abl, interp, t_prov, t_seg, anom, features, dimensiones, rapido):
    T = rb.cargar()
    nom = res["nombre_sel"]
    ft, fv = res["filas_test"], res["filas_val"]
    m = ft[nom]
    base = ft["Persistencia (baseline)"]
    mm3 = ft["Media móvil 3 (baseline)"]
    n_final = int(embudo["Registros"].iloc[-1])
    n_ini = int(embudo["Registros"].iloc[0])
    sp = info_split
    ab = abl["tabla"].set_index("Dimension")
    claves = list(dimensiones)
    cumple = m["R2"] >= UMBRAL_R2_OFICIAL
    mejora_mae = (base["MAE"] - m["MAE"]) / base["MAE"] * 100
    supera = m["MAE"] < base["MAE"]
    imp_dim = interp["imp_dim"]
    shp = interp["shap"]["por_dimension"] if interp["shap"] else None
    ref_dim = shp or imp_dim
    dom = max(ref_dim, key=ref_dim.get)
    sig = [k for k in claves if ab.loc[k, "p_holm"] < ALPHA]
    rech = [k for k in claves if ab.loc[k, "p_holm"] >= ALPHA]
    _ml = {k: v for k, v in ft.items() if "baseline" not in k}
    _mk = max(_ml, key=lambda k: _ml[k]["R2"])
    mejor_r2_test = (_mk, _ml[_mk]["R2"])
    seg_bajo, seg_alto = float(t_seg.iloc[0]["R2"]), float(t_seg.iloc[1]["R2"])
    hp = pd.read_csv(os.path.join(os.path.dirname(OUT_TXT), "tables", "tabla_hiperparametros_finales.csv"))

    aviso = ("> **PRUEBA DE HUMO (--rapido):** estos números provienen de una corrida reducida; "
             "vuelve a ejecutar `python main.py` sin `--rapido` antes de pegar nada en la tesis.\n\n") if rapido else ""

    L = []
    L.append("# TEXTO PARA LA TESIS — generado automáticamente con los resultados de tu corrida\n")
    L.append(aviso)
    L.append("**Cómo usarlo:** pega cada sección en el capítulo indicado. Los números ya coinciden con las tablas y "
             "figuras de `outputs/`. Reemplaza las tablas y figuras antiguas por las nuevas. "
             "Las tablas K–S (particiones, significancia, brecha, contribuciones, sensibilidad, decisión) tienen letras provisionales: renumera todas en orden de aparición en tu documento. "
             "Donde veas [COMPLETAR] o [VERIFICAR] falta información que el pipeline no tiene (antecedentes, fechas de publicación, fuente del umbral): no la inventes.\n")

    # ------------------------------------------------------------------ 4.1
    L.append("# CAPÍTULO IV — RESULTADOS Y DISCUSIÓN\n")
    L.append("## 4.1. METODOLOGÍA DE DESARROLLO DEL MODELO PREDICTIVO\n")
    L.append("El modelo predictivo se desarrolló siguiendo un enfoque basado en CRISP-DM (Chapman et al., 2000), "
             "complementado con el proceso KDD (Fayyad et al., 1996) y con las prácticas de Machine Learning descritas por Géron (2023), "
             "organizado en nueve fases secuenciales con retroalimentación iterativa (Figura A). Esta organización reemplaza el flujo "
             "de elaboración propia presentado en el borrador anterior y garantiza que cada decisión técnica tenga respaldo en la literatura.\n")
    L.append("**Figura A.** Fases de la metodología de desarrollo del modelo predictivo. *Nota.* Elaboración propia con base en Chapman et al. (2000), Fayyad et al. (1996) y Géron (2023). → `figura_flujo_metodologico_9_fases.png`\n")
    cuadro = pd.DataFrame([
        ["1. Comprensión del problema", "Definición del problema, variable objetivo y dimensiones X1–X4", "Matriz de consistencia", "Chapman et al. (2000); Géron (2023)"],
        ["2. Comprensión de los datos", "Inspección, depuración y análisis exploratorio", "pandas; embudo de depuración", "Han et al. (2022)"],
        ["3. Preparación de los datos", "Limpieza, variables derivadas, partición cronológica 70/15/15", "pandas; ingeniería de características", "Géron (2023); Kuhn y Johnson (2013)"],
        ["4. Construcción del pipeline", "Imputación, escalado y codificación encapsuladas con el modelo", "Pipeline de scikit-learn", "Pedregosa et al. (2011)"],
        ["5. Modelado", "Random Forest, XGBoost, LightGBM y Regresión Lineal frente a dos baselines", "scikit-learn, xgboost, lightgbm", "Chen y Guestrin (2016)"],
        ["6. Ajuste de hiperparámetros", "Búsqueda aleatoria con validación cruzada temporal", "RandomizedSearchCV + TimeSeriesSplit", "Bergstra y Bengio (2012)"],
        ["7. Evaluación", "R², MAE, RMSE y MAPE; contraste de hipótesis (Wilcoxon + Holm); pérdidas pareadas con errores HAC", "scikit-learn, SciPy", "Hyndman y Koehler (2006); Holm (1979); Newey y West (1987) [VERIFICAR cada cita]"],
        ["8. Interpretabilidad", "Valores SHAP y permutación agrupada por dimensión", "shap", "Lundberg y Lee (2017)"],
        ["9. Validación y despliegue", "Evaluación final en prueba, API REST y aplicación web", "FastAPI, Firebase, Railway", "Géron (2023)"],
    ], columns=["Fase", "Descripción", "Técnicas / herramientas", "Base científica"])
    L.append("**Tabla A.** Fases de la metodología y su sustento científico\n")
    L.append(_md_tabla(cuadro) + "\n")
    L.append("*Nota.* Elaboración propia.\n")

    # Fase 1
    L.append("### 4.1.1. Fase 1 — Comprensión del problema\n")
    L.append(f"El problema es predecir el precio en chacra (S/ por kg) de los cultivos de la región Piura, un problema de regresión supervisada. "
             f"La variable dependiente es el precio en chacra y las variables independientes se organizan en cuatro dimensiones del entorno "
             f"agroeconómico: temporal (X1), productiva (X2), geográfica (X3) y económica histórica (X4) (Figura B).\n")
    L.append("**Figura B.** Variables predictoras por dimensión del entorno agroeconómico. *Nota.* Elaboración propia. → `figura_variables_por_dimension.png`\n")

    # Fase 2-3
    L.append("### 4.1.2. Fase 2 — Comprensión de los datos\n")
    L.append(f"Los datos provienen del portal de Datos Abiertos del Gobierno Regional de Piura (Campaña Agrícola de los Principales Cultivos, 2015–2024). "
             f"El archivo contiene {MILES(n_ini)} registros. Se aplicó un protocolo de depuración documentado paso a paso (Tabla B); "
             f"la muestra final de modelado quedó en {MILES(n_final)} registros, correspondientes a {df['CULTIVO'].nunique()} cultivos y "
             f"{df['PROVINCIA'].nunique()} provincias. El precio en chacra de la muestra oscila entre S/ {N(df['PRECIO_CHACRA'].min(),2)} y "
             f"S/ {N(df['PRECIO_CHACRA'].max(),2)} por kg, con media de S/ {N(df['PRECIO_CHACRA'].mean(),2)}.\n")
    L.append("**Tabla B.** Embudo de depuración de los datos\n")
    L.append(_md_tabla(embudo, {"Registros": MILES, "Eliminados": MILES}) + "\n")
    L.append("*Nota.* Los registros sin precio no se imputaron: un precio inventado no es una observación y sesgaría la variable que se desea predecir. Elaboración propia.\n")
    L.extend(rb.bloque_depuracion(T))

    L.append("### 4.1.3. Fase 3 — Preparación de los datos\n")
    L.append(f"Se corrigió el formato de la variable MES del dataset oficial (AAAAMM) para extraer el mes calendario, y se construyeron las "
             f"{len(features)} variables de entrada distribuidas en las cuatro dimensiones (Tabla C). Todas las variables de la dimensión económica "
             f"histórica se calculan exclusivamente con precios de períodos anteriores del mismo cultivo, provincia y distrito, de modo que ninguna "
             f"variable contiene información del precio que se desea predecir (control de fuga de datos). "
             f"Los valores faltantes (el primer registro de cada serie carece de historial) se imputan con la mediana calculada únicamente con "
             f"datos de entrenamiento, dentro del pipeline.\n")
    tv = pd.DataFrame([[d["codigo"], d["nombre"], len(dimensiones[d["clave"]]), ", ".join(dimensiones[d["clave"]])]
                       for d in MATRIZ_CONSISTENCIA["dimensiones"]], columns=["Dimensión", "Nombre", "N.º", "Variables"])
    L.append("**Tabla C.** Variables predictoras por dimensión\n")
    L.append(_md_tabla(tv, {"MAPE": lambda v: N(v, 1), "WAPE": lambda v: N(v, 1)}) + "\n")
    L.append(f"Los datos se ordenaron cronológicamente y se dividieron en tres conjuntos sin mezclar períodos: entrenamiento "
             f"({MILES(sp['n_train'])} registros, {MES(sp['train_desde'])} a {MES(sp['train_hasta'])}), validación "
             f"({MILES(sp['n_val'])}, hasta {MES(sp['val_hasta'])}) y prueba ({MILES(sp['n_test'])}, hasta {MES(sp['test_hasta'])}), "
             f"aproximadamente 70 %, 15 % y 15 %. El conjunto de prueba no intervino en el ajuste ni en la selección del modelo.\n")
    L.extend(rb.bloque_particiones(T))

    L.append("### 4.1.4. Fase 4 — Construcción del pipeline\n")
    L.append("El preprocesamiento (imputación por mediana, escalado y codificación one-hot para la regresión lineal) y el estimador se encapsularon en "
             "un único Pipeline de scikit-learn (Pedregosa et al., 2011). Así, en cada partición de la validación cruzada las transformaciones se "
             "ajustan solo con los datos de entrenamiento de esa partición, lo que asegura reproducibilidad y evita fuga de información.\n")

    L.append("### 4.1.5. Fase 5 — Modelado\n")
    L.append("Se entrenaron cuatro algoritmos —Random Forest, XGBoost, LightGBM y Regresión Lineal— y se compararon con dos pronósticos ingenuos: "
             "la **persistencia** (el precio de hoy es el último precio observado) y la **media móvil de tres observaciones**. Los baselines son la vara "
             "mínima: un modelo de Machine Learning solo aporta valor si los supera.\n")

    L.append("### 4.1.6. Fase 6 — Ajuste de hiperparámetros\n")
    L.append("Los hiperparámetros (valores de configuración fijados antes del entrenamiento, a diferencia de los parámetros que el modelo aprende de los datos) "
             f"se optimizaron con búsqueda aleatoria (Bergstra y Bengio, 2012) y validación cruzada temporal (TimeSeriesSplit), usando únicamente el tramo de "
             f"entrenamiento y el MAE como criterio. La Tabla D muestra la configuración elegida.\n")
    L.append("**Tabla D.** Hiperparámetros seleccionados\n")
    def _cfg(r):
        partes = []
        for c in hp.columns:
            if c in ("modelo", "cv_mae") or pd.isna(r[c]):
                continue
            v = r[c]
            partes.append(f"{c} = {int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v}")
        return "; ".join(partes) if partes else "—"
    hp_t = pd.DataFrame({"Modelo": hp["modelo"],
                         "MAE de la validación cruzada temporal": [("—" if pd.isna(v) else N(v)) for v in hp["cv_mae"]],
                         "Hiperparámetros seleccionados": [_cfg(r) for _, r in hp.iterrows()]})
    L.append(_md_tabla(hp_t) + "\n")
    L.append("*Nota.* Elaboración propia. La Regresión Lineal no tiene hiperparámetros relevantes.\n")

    L.append("## 4.2. RESULTADOS\n")
    L.append("### 4.2.1. Fase 7 — Evaluación y selección del modelo\n")
    tv = res["tabla_val"].copy()
    tv = tv[["Modelo", "R2", "MAE", "RMSE", "MAPE", "WAPE", "R2_CV_fold_media"]].rename(columns={"R2_CV_fold_media": "R² CV temporal"})
    L.append(f"La selección del modelo se hizo en el conjunto de validación, con el MAE como criterio. Resultó seleccionado **{nom}** "
             f"(MAE de validación = {N(fv[nom]['MAE'])} S/ por kg; la persistencia obtuvo {N(fv['Persistencia (baseline)']['MAE'])}).\n")
    L.append("**Tabla E.** Comparación de modelos en el conjunto de validación\n")
    L.append(_md_tabla(tv, {"MAPE": lambda v: N(v, 1), "WAPE": lambda v: N(v, 1)}) + "\n")
    L.append("*Nota.* R², MAE, RMSE (S/ por kg) y MAPE (%). Elaboración propia.\n")
    tt = res["tabla_test"][["Modelo", "R2", "MAE", "RMSE", "MAPE", "WAPE", "R2_log"]]
    L.append(f"Con el modelo ya elegido y reentrenado con entrenamiento y validación, se evaluó una única vez el conjunto de prueba "
             f"(n = {MILES(sp['n_test'])}, los meses más recientes, nunca vistos). {nom} alcanzó R² = {N(m['R2'])}, MAE = {N(m['MAE'])} S/ por kg, "
             f"RMSE = {N(m['RMSE'])} S/ por kg y MAPE = {N(m['MAPE'],2)} %. "
             + (f"Este valor {'supera' if cumple else 'no alcanza'} el criterio de aceptabilidad establecido (R² ≥ {UMBRAL_R2_OFICIAL}). " )
             + (f"Frente al pronóstico ingenuo de persistencia (R² = {N(base['R2'])}; MAE = {N(base['MAE'])}), el modelo "
                f"{'reduce el error absoluto medio en ' + N(mejora_mae,1) + ' %' if supera else 'no logra reducir el error absoluto medio'} "
                f"(media móvil: R² = {N(mm3['R2'])}; MAE = {N(mm3['MAE'])}).\n"))
    L.append("**Tabla F.** Resultados en el conjunto de prueba (hold-out cronológico)\n")
    L.append(_md_tabla(tt, {"MAPE": lambda v: N(v, 1), "WAPE": lambda v: N(v, 1)}) + "\n")
    L.append("*Nota.* La selección del modelo se realizó en validación; la prueba se usó una sola vez. WAPE = error absoluto total / precio total (%); R²_log = R² sobre el logaritmo del precio, menos sensible a los pocos cultivos de precio alto. Elaboración propia.\n")
    L.append("**Figura C.** Valores reales frente a predichos en el conjunto de prueba. → `figura_real_vs_predicho.png`\n")
    L.append("**Figura D.** Comparación de modelos y baselines en el conjunto de prueba. → `figura_comparacion_modelos_prueba.png`\n")
    L.append("**Figura E.** Evolución del R² por fold temporal. → `figura_r2_por_fold.png`\n")

    L.append("### 4.2.2. Rendimiento por segmento\n")
    L.append("**Tabla G.** Rendimiento del modelo final por provincia (conjunto de prueba)\n")
    L.append(_md_tabla(t_prov, {"N_test": MILES}) + "\n")
    L.append("**Tabla H.** Rendimiento según el nivel de precio del cultivo\n")
    L.append(_md_tabla(t_seg, {"N_test": MILES, "Pct_test": lambda v: N(v, 1)}) + "\n")
    if len(t_seg) == 2:
        a, b = t_seg.iloc[0], t_seg.iloc[1]
        L.append(f"El desempeño es heterogéneo: el modelo alcanza R² = {N(a['R2'])} en el segmento de precios bajos ({N(a['Pct_test'],1)} % de la prueba) y "
                 f"R² = {N(b['R2'])} en el de precios altos (MAE = {N(b['MAE'])} S/ por kg). Los cultivos de alto valor son menos frecuentes y más volátiles, "
                 f"lo que se reconoce como limitación (sección 4.4.4).\n")

    L.append("### 4.2.3. Fase 8 — Interpretabilidad\n")
    s_txt = (f" El análisis con valores SHAP (Lundberg y Lee, 2017) atribuye " + ", ".join(
        f"{k.split('_')[0]} {N(v*100,1)} %" for k, v in shp.items()) + " de la contribución media absoluta a las predicciones.") if shp else ""
    pm = interp["permutacion"].set_index("Dimension")
    L.append(f"La importancia por ganancia del modelo asigna " + ", ".join(f"{k.split('_')[0]} {N(v*100,1)} %" for k, v in imp_dim.items()) + "." + s_txt +
             f" Como la ganancia puede repartirse arbitrariamente entre variables correlacionadas, se verificó con permutación agrupada: al desordenar "
             f"las variables de {dom.split('_')[0]} el R² cae {N(pm.loc[dom,'caida_R2_permutacion'])}; en las demás dimensiones la caída es " +
             ", ".join(f"{k.split('_')[0]} {N(pm.loc[k,'caida_R2_permutacion'])}" for k in claves if k != dom) + ".\n")
    L.append("**Figura F.** Importancia por dimensión (ganancia, SHAP y permutación). → `figura_importancia_por_dimension.png`; **Figura G.** Resumen SHAP → `figura_shap_resumen.png`\n")

    L.append("### 4.2.4. Detección de anomalías\n")
    L.append(f"Isolation Forest con contaminación de {N(anom['contaminacion']*100,0)} % identificó {MILES(anom['n_anomalias'])} registros anómalos "
             f"({N(anom['pct'],2)} % de {MILES(n_final)}). El precio promedio fue S/ {N(anom['precio_normal'],3)} por kg en los registros normales y "
             f"S/ {N(anom['precio_anomalo'],3)} en los anómalos (diferencia de S/ {N(anom['delta'],3)}). Se exportó una muestra aleatoria de 50 anomalías "
             f"para revisión manual: **la proporción de errores de registro y eventos reales debe volver a calcularla el autor sobre esa muestra** "
             f"(`muestra_50_anomalias_revision_manual.csv`), pues el conjunto de anomalías cambió respecto de la versión anterior.\n")
    L.append("**Figura H.** Distribución de precios normales y anómalos. → `figura_anomalias_isolation_forest.png`\n")
    L.append("### 4.2.5. Significancia de la mejora frente a los baselines\n")
    L.extend(rb.bloque_significancia(T, nom, base, m, UMBRAL_R2_OFICIAL))
    L.extend(rb.bloque_brecha(T))

    # ------------------------------------------------------------------ 4.3 hipótesis
    L.append("## 4.3. CONTRASTACIÓN DE HIPÓTESIS\n")
    _alg = {"lgbm": "LightGBM (misma familia que el modelo final)", "rf": "Random Forest"}.get(abl.get("clave"), "el algoritmo indicado")
    clasif = rb.clasificar(T, ALPHA)
    L.append(f"Para evaluar la contribución predictiva de cada dimensión se entrenaron, con los hiperparámetros ya optimizados, variantes de {_alg} que excluyen una dimensión \n"
             f"a la vez, y se compararon con el modelo completo en {abl['n_splits']} particiones temporales del tramo de desarrollo (entrenamiento y validación; la prueba no se usó en este contraste). \n"
             f"Se aplicó la prueba de rangos con signo de Wilcoxon para muestras pareadas, unilateral, con corrección de Holm (α = {ALPHA}). Con {abl['n_splits']} particiones el p-valor mínimo alcanzable es {2**-abl['n_splits']:.5f}. \n"
             "Los resultados se replicaron en el conjunto de prueba y sobre el modelo final con pérdidas pareadas por mes, y con Random Forest como análisis de robustez (`tabla_ablation_wilcoxon.csv`).\n")
    L.extend(rb.bloque_hipotesis(T, clasif, ALPHA, nom, base, m, mm3, UMBRAL_R2_OFICIAL, cumple))
    if T["abl_lgbm"] is None:   # respaldo si no se corrieron los extras (--sin-extras)
        tabla = ab.reset_index()[["Dimension", "R2_completo", "R2_sin_dim", "Diferencia_R2", "p_valor", "p_holm", "Significativa_estadistica"]]
        tabla.columns = ["Dimensión excluida", "R² completo", "R² sin la dimensión", "ΔR²", "p (Wilcoxon)", "p (Holm)", "¿Significativa?"]
        L.append("**Tabla I.** Ablation por dimensión y significancia estadística\n")
        L.append(_md_tabla(tabla, {"p (Wilcoxon)": P, "p (Holm)": P}) + "\n")
        L.append("*Nota.* Corrida sin análisis extra (`--sin-extras`): faltan los contrastes en prueba, sobre el modelo final y las significancias frente a baselines.\n")
    L.append("**Figura I.** Ablation por dimensión. → `figura_ablation_por_dimension.png`\n")
    nombres_h = {"X1_Temporal": "H1 — Dimensión temporal", "X2_Productiva": "H2 — Dimensión productiva",
                 "X3_Geografica": "H3 — Dimensión geográfica", "X4_Economica_Historica": "H4 — Dimensión económica histórica"}

    # ------------------------------------------------------------------ 4.4 discusión
    L.extend(rb.bloque_discusion(T, clasif, nom, m, base, UMBRAL_R2_OFICIAL, cumple, mejor_r2_test))
    L.extend(rb.limitaciones(T))

    # ------------------------------------------------------------------ 5 conclusiones
    L.append("# CAPÍTULO V — CONCLUSIONES\n")
    _shp = (f"La dimensión {BONITO[dom]} es la de mayor peso ({N(ref_dim[dom]*100,1)} % de la contribución SHAP, en escala log-ratio respecto de la persistencia)." if shp
            else f"La dimensión {BONITO[dom]} es la de mayor peso ({N(ref_dim[dom]*100,1)} % de la importancia por ganancia).")
    for linea in rb.conclusiones(T, clasif, nom, m, base, ft, UMBRAL_R2_OFICIAL, cumple, mejor_r2_test, _shp, anom, sp["n_test"]):
        L.append(linea + "\n")
    L.extend(rb.bloque_capitulo3(T, UMBRAL_R2_OFICIAL))

    # ------------------------------------------------------------------ correcciones
    L.append("# CORRECCIONES QUE DEBES HACER EN OTRAS SECCIONES (no se pegan: se aplican)\n")
    L.append(f"- **Resumen/Abstract y 3.4:** la cifra de registros debe ser coherente: {MILES(n_ini)} iniciales y {MILES(n_final)} de modelado (el borrador dice 116,000 en el resumen).\n"
             f"- **3.7.1 y 3.7.3:** cambia a *{abl['n_splits']} particiones temporales*, prueba *unilateral* y *corrección de Holm*.\n"
             f"- **3.7.2.2:** elimina «estratificando por cultivo» (la partición es cronológica 70/15/15), el «grid manual» (ahora es búsqueda aleatoria con validación temporal) y las métricas R²=0.892/MAE=0.287; los resultados no van en la metodología.\n"
             f"- **3.7.4:** la conclusión sobre dimensiones contradice el capítulo IV; reemplázala por una referencia a la sección 4.3.\n"
             f"- **Tabla 2/variables X1 y X4:** cambia MES por MES_NUM; VARIACION_PRECIO por VARIACION_PRECIO_PREVIA; añade MESES_DESDE_PRECIO_PREVIO y TIENE_HISTORIAL. X3: CULTIVO se codifica desde COD_CULTIVO (no «target encoding»).\n"
             f"- **Figura 3 (variables predictoras):** sustitúyela por la Figura B (original), y quita el título dentro de la imagen; los títulos van solo como leyenda.\n"
             f"- **Figura 12 (flujo):** sustitúyela por la Figura A (nueve fases).\n"
             f"- **LSTM:** el borrador menciona LSTM (R²=0.0812) pero el código no lo contiene; elimina esas frases o añade el experimento.\n"
             f"- **Tablas 3, 4 y 5:** las tres tienen el mismo título; usa las Tablas D, E y F.\n"
             f"- **3.7.2.3:** intervalos de confianza, Redis y Recharts/Leaflet/Angular: la API ahora devuelve un intervalo empírico del 90 %; Redis y Angular no están implementados en el repositorio — retíralos del texto o impleméntalos.\n"
             f"- **Tabla 10 (SUS, uptime, 287 ms):** no provienen del pipeline; ten a mano la evidencia (encuestas, logs).\n"
             + rb.CORRECCIONES_EXTRA)
    md = "\n".join(L)
    ruta_md = os.path.join(OUT_TXT, "texto_resultados_y_conclusiones.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(md)
    ruta_docx = os.path.join(OUT_TXT, "texto_resultados_y_conclusiones.docx")
    _a_docx(md, ruta_docx)

    reporte = {
        "n_registros_validos": n_final, "modelo_seleccionado": nom, "n_splits_ablation": abl["n_splits"],
        "modelo_final": {"r2_final": m["R2"], "mae_final": m["MAE"], "rmse_final": m["RMSE"], "mape_final": m["MAPE"]},
        "hipotesis_general": {
            "texto": MATRIZ_CONSISTENCIA["hipotesis_general"],
            "estado_vs_umbral_OFICIAL": "ACEPTADA" if cumple else "RECHAZADA",
            "evidencia": f"R² en prueba = {m['R2']:.4f} (umbral oficial = {UMBRAL_R2_OFICIAL}); MAE persistencia = {base['MAE']:.4f} vs modelo = {m['MAE']:.4f}",
        },
        "hipotesis_especificas": [{
            "codigo": d["codigo"], "texto": d["hipotesis_especifica"],
            "R2_sin_dimension": float(ab.loc[d["clave"], "R2_sin_dim"]),
            "diferencia_R2": float(ab.loc[d["clave"], "Diferencia_R2"]),
            "p_valor": float(ab.loc[d["clave"], "p_holm"]),
            "significativa_wilcoxon": "Sí" if ab.loc[d["clave"], "p_holm"] < ALPHA else "No",
            "importancia_gini_dimension": float(imp_dim[d["clave"]]),
            "estado_consistencia": (clasif or {}).get(d["clave"], {}).get("estado", "no evaluado (corrida sin extras)"),
        } for d in MATRIZ_CONSISTENCIA["dimensiones"]],
    }
    import json
    with open(os.path.join(os.path.dirname(OUT_TXT), "tables", "reporte_contrastacion_hipotesis.json"), "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ guía de sustentación
    G = ["# GUÍA DE SUSTENTACIÓN — preguntas probables y respuestas con tus números\n"]
    qa = [
        ("¿Qué metodología siguió para construir el modelo?",
         "CRISP-DM complementado con KDD y las prácticas de Géron (2023), en nueve fases: problema, datos, preparación, pipeline, modelado, ajuste de hiperparámetros, evaluación, interpretabilidad y validación/despliegue. Cada fase tiene respaldo bibliográfico (Tabla A)."),
        ("¿Cómo evitó la fuga de datos?",
         "Tres controles: (1) las variables históricas usan solo precios de períodos anteriores; (2) la partición es cronológica 70/15/15 sin mezclar períodos; (3) imputación y escalado viven dentro del Pipeline y se ajustan solo con entrenamiento. La prueba se usó una sola vez. Además, como la dimensión productiva (X2) es contemporánea al precio, se repitió la evaluación con X2 rezagada y sin X2: el desempeño no cambia de forma concluyente (Tabla P)."),
        ("¿Qué son los hiperparámetros y cómo los eligió?",
         "Son valores que se fijan antes de entrenar (profundidad, número de árboles, tasa de aprendizaje); los parámetros, en cambio, los aprende el modelo. Se optimizaron con búsqueda aleatoria y validación cruzada temporal solo en entrenamiento (Tabla D)."),
        ("¿Su modelo es mejor que simplemente repetir el último precio?",
         rb.qa_baseline(T, m, base, mm3)),
        (f"¿Por qué se eligió {nom} si otro modelo tiene mayor R² en prueba?",
         f"La regla de selección (menor MAE en validación) se fijó antes de ver la prueba; elegir en prueba la contaminaría. Mejor R² en prueba entre los modelos de ML: {mejor_r2_test[0]} ({N(mejor_r2_test[1])}); el elegido logra el menor MAE ({N(m['MAE'])}) y el mayor R²_log ({N(m['R2_log'])})." if mejor_r2_test[0] != nom else "El modelo elegido en validación coincide con el de mejor R² en prueba."),
        ("¿Qué hipótesis específicas se respaldaron y cuáles no?",
         rb.qa_hipotesis(clasif, T)),
        ("¿Por qué el R² completo de la ablation no coincide con el de prueba?",
         f"La ablation principal se hace con {'LightGBM (misma familia que el modelo final)' if abl.get('clave') == 'lgbm' else 'Random Forest'} en particiones temporales del tramo de desarrollo (entrenamiento + validación), no en el conjunto de prueba; sirve para comparar dimensiones entre sí, no para reportar el desempeño final."),
        ("¿Por qué LightGBM/XGBoost/RF y no una red LSTM?",
         "Los datos son series cortas, irregulares y de muchos cultivos; los modelos de árboles manejan bien variables mixtas y pocos datos. Una LSTM global mezclaría cultivos heterogéneos; se deja como trabajo futuro por cultivo."),
        ("¿Cómo sabe que su R² no depende de los cultivos baratos?",
         "La Tabla H desagrega el desempeño por nivel de precio; el rendimiento en cultivos de alto valor es menor y se declara como limitación."),
        ("¿Qué aporta SHAP?",
         "Explica cuánto contribuye cada variable a cada predicción. Se agregó por dimensión y se contrastó con la permutación agrupada, que es robusta a la correlación entre variables."),
        ("¿Se imputó la variable objetivo?",
         "No. Los registros sin precio se descartaron y se reportan en el embudo de depuración (Tabla B)."),
        ("¿Qué pasa con el primer registro de cada serie, que no tiene historial?",
         "Se marca con TIENE_HISTORIAL = 0 y el pipeline imputa la mediana de entrenamiento; el modelo aprende que en esos casos no hay información previa."),
    ]
    if not cumple:
        qa.insert(0, (f"¿Por qué el R² ({N(m['R2'])}) no alcanza el criterio de {UMBRAL_R2_OFICIAL}?",
            f"El criterio figura en la sección 3.6.3 y se reporta tal cual: la hipótesis de eficacia no se respalda con ese umbral [VERIFICA y di con honestidad de dónde sale: antecedente, estándar técnico o decisión metodológica previa a ver la prueba]. Contexto verificable: "
            f"(1) repetir el último precio ya da R² = {N(base['R2'])}: la inercia de los precios explica gran parte de la varianza; "
            f"(2) el R² global se apoya en distinguir cultivos baratos de caros: dentro de cada segmento el R² es menor (Tabla H: {N(seg_bajo)} en precios bajos y {N(seg_alto)} en altos); "
            f"(3) el estudio de Fuentes et al. (2025), citado en el marco referencial, reporta R² = 0.788 [VERIFICA la cifra y la cita contra la fuente antes de usarla]; compara con cautela porque cultivos, horizonte y validación difieren. "
            f"No se debe modificar el umbral después de ver los resultados sin acuerdo explícito del asesor y del jurado; lo defendible es discutir la brecha y proponer mejoras (variables climáticas, modelos por cultivo)."))
    qa += [
        ("¿Por qué las hipótesis específicas ya no dicen «influye»?",
         "Porque el estudio es observacional: una dimensión que mejora el pronóstico fuera de muestra tiene contribución predictiva, pero eso no prueba causalidad. Las hipótesis se contrastan en validación temporal, en prueba y sobre el modelo final (Tabla S)."),
        ("¿Cómo sabe que X2 no filtra información del futuro?",
         "X2 es contemporánea al precio, así que se repitió la evaluación con X2 rezagada y sin X2 (Tabla P): no hay diferencia concluyente. El desempeño no depende de información contemporánea y, de hecho, X2 no aporta."),
        ("¿Por qué la dimensión geográfica cambia entre validación y prueba?",
         "En validación el efecto es del cultivo y aparece cuando hay menos precios previos; en prueba es del territorio. El cultivo funciona como sustituto del nivel de precio cuando falta historial. La evidencia es inestable y no se afirma que X3 contribuya al modelo final."),
        ("¿La muestra final no está sesgada si conserva solo el 11.6 %?",
         "Sí, hay selección fuerte por cultivo (V de Cramér alta en la regla de cosecha). Se reportan los motivos de exclusión, la distribución antes/después y una sensibilidad que relaja la regla (Tablas Q y R); la conclusión cualitativa se mantiene con margen menor, pero los resultados no se generalizan a todos los cultivos."),
        ("¿Qué horizonte predice el modelo?",
         "La siguiente observación disponible de cada serie cultivo–provincia–distrito, dado su último precio observado. La distancia puede ser mayor de un mes; por eso se reporta el error por brecha (Tabla M)."),
        ("¿Sus resultados demuestran beneficios económicos para el agricultor?",
         "No. Demuestran desempeño técnico en un conjunto de prueba retrospectivo. El beneficio económico requiere evaluación con usuarios o datos operativos."),
    ]
    for q, a in qa:
        G.append(f"## {q}\n{a}\n")
    ruta_g = os.path.join(OUT_TXT, "guia_sustentacion.md")
    with open(ruta_g, "w", encoding="utf-8") as f:
        f.write("\n".join(G))
    return [ruta_md, ruta_docx, ruta_g]
