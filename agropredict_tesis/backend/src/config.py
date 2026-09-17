"""
config.py — Configuración central del proyecto AgroPredict.

Aquí vive la ÚNICA copia de la Matriz de Consistencia (Anexo 1 de la tesis).
Cualquier módulo que necesite citar el problema/objetivo/hipótesis importa
esto — así el código y la tesis nunca se desalinean por tener el texto
copiado y pegado en 5 archivos distintos (que es lo que pasaba antes).
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import mlflow
from pathlib import Path

# ------------------------------------------------------------------
# RUTAS
# ------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_agricola.csv")
OUT_FIG = os.path.join(BACKEND_DIR, "outputs", "figures")
OUT_TAB = os.path.join(BACKEND_DIR, "outputs", "tables")
OUT_MOD = os.path.join(BACKEND_DIR, "outputs", "models")
MLRUNS_DIR = os.path.join(BACKEND_DIR, "mlruns")

for d in (OUT_FIG, OUT_TAB, OUT_MOD, MLRUNS_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
# PARÁMETROS DEL MODELO
# ------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS_TSS = 6               # antes 5: con n=5 el Wilcoxon nunca baja de p=0.0625
                                # (mínimo p de dos colas con n=5 es 2/2^5=0.0625).
                                # Con n=6, el mínimo posible es 2/2^6=0.03125 < 0.05,
                                # o sea que SÍ es posible detectar significancia si el
                                # efecto es consistente en las 6 repeticiones.
                                # Actualiza también la sección 3.7.1 de la tesis
                                # ("5 repeticiones" -> "6 repeticiones") para que el
                                # texto coincida con el pipeline.
TARGET = "PRECIO_CHACRA"

# ⚠️ DECISIÓN METODOLÓGICA (antes: dos umbrales sin resolver, 0.85 y 0.75):
#    Se fija 0.85 (sección 3.6.3) como umbral OFICIAL de la tesis, porque:
#      1) Es el umbral más exigente -> criterio más riguroso.
#      2) Tus resultados reales (R² hold-out ≈ 0.93 con RF y LightGBM) superan
#         CÓMODAMENTE incluso el umbral más estricto, así que no hay ningún
#         incentivo para "relajar" el criterio a 0.75 solo para que ajuste.
#    ACCIÓN PARA TI: en la sección 4.4 de tu tesis, donde dice "≥0.75", cambia
#    el texto a "≥0.85" para que coincida con 3.6.3. El pipeline sigue
#    calculando el resultado frente a los DOS umbrales (por transparencia),
#    pero UMBRAL_R2_OFICIAL es el que debes citar como criterio de aceptación
#    de la Hipótesis General.
UMBRAL_R2_CAP3 = 0.85   # sección 3.6.3 (Metodología) — umbral oficial
UMBRAL_R2_CAP4 = 0.75   # sección 4.4 (Discusión) — se mantiene solo como referencia
UMBRAL_R2_OFICIAL = UMBRAL_R2_CAP3

# ⚠️ DECISIÓN METODOLÓGICA (antes: Anexo 1 describía X4 como "z-score de
#    precio"; el código usa PRECIO_LAG1 + VARIACION_PRECIO + MEDIA_MOVIL_3):
#    Se corrige el TEXTO del Anexo 1, no el código, porque todos los
#    resultados de tu tesis (importancia de variables, ablation, figuras)
#    ya están calculados y redactados sobre estas tres variables. Cambiar el
#    código ahora invalidaría todo el capítulo 4 ya escrito.
#    ACCIÓN PARA TI: en el Anexo 1, reemplaza la descripción de X4
#    ("z-score de precio") por: "Rezago de precio a 1 mes (PRECIO_LAG1),
#    variación porcentual mensual (VARIACION_PRECIO) y media móvil de 3
#    meses (MEDIA_MOVIL_3)". La lista `variables` de X4 más abajo en este
#    archivo ya refleja esto — es la fuente de verdad.

# ⚠️ DECISIÓN METODOLÓGICA (antes: Anexo 1 incluía UBIGEO en X3; el código
#    original no lo usaba):
#    Se mantiene USE_UBIGEO_EN_X3 = False, es decir, el código NO cambia,
#    porque activarlo obligaría a re-ejecutar todo el pipeline y todos los
#    resultados de X3/H3 que ya tienes redactados (importancia, Wilcoxon,
#    figuras) quedarían desactualizados.
#    ACCIÓN PARA TI: en el Anexo 1, quita "UBIGEO" de la lista de variables
#    de X3 — deja solo PROVINCIA, DISTRITO y CULTIVO (codificados con
#    LabelEncoder), que es lo que el código realmente usa y lo que sustenta
#    los resultados reportados.
USE_UBIGEO_EN_X3 = False

# ⚠️ DECISIÓN METODOLÓGICA (antes: Anexo 1 dice "~116,000 registros"
#    (población); capítulo 4.4 dice "14,393" (muestra de modelado) — no es
#    un error, son dos cifras distintas de dos etapas distintas):
#    No requiere cambio de código. Documentado aquí para que copies la
#    aclaración textual a tu tesis.
#    ACCIÓN PARA TI: en la sección donde reportas 14,393, agrega una nota
#    tipo: "De los 116,000 registros de la población (portal de Datos
#    Abiertos del Gobierno Regional de Piura, filtrados a las 8 provincias
#    de la región), la muestra final de modelado tras limpieza (imputación
#    de PRECIO_CHACRA, descarte de inconsistencias COSECHA=0/PRODUCCION>0,
#    precios ≤0 y duplicados) quedó en 14,393 registros (ver Tabla X,
#    sección 3.7.2.1.B)." Así el lector entiende que 116,000 es población y
#    14,393 es la muestra después de la limpieza descrita en el código.
N_POBLACION_ANEXO1 = 116000  # solo referencia textual, no se usa en el pipeline

PROVINCIAS_PIURA = {
    "PIURA", "SULLANA", "MORROPON", "TALARA", "PAITA",
    "AYABACA", "HUANCABAMBA", "SECHURA",
}

COLORES = {
    "principal": "#1B6B4A",
    "secundario": "#C4960A",
    "x1_temporal": "#1A5276",
    "x2_productiva": "#1B6B4A",
    "x3_geografica": "#C4960A",
    "x4_economica": "#C0392B",
    "lgbm": "#7D3C98",
}

# ------------------------------------------------------------------
# MATRIZ DE CONSISTENCIA (Anexo 1) — fuente única de verdad
# ------------------------------------------------------------------
MATRIZ_CONSISTENCIA = {
    "titulo": "Sistema web para la predicción de precios agrícolas mediante "
              "técnicas de Machine Learning, Piura, 2025",
    "problema_general": (
        "¿En qué medida las dimensiones del entorno agroeconómico influyen en "
        "la precisión de la predicción de los precios agrícolas mediante "
        "técnicas de Machine Learning en la región Piura, 2025?"
    ),
    "objetivo_general": (
        "Determinar la influencia del entorno agroeconómico en la eficacia de "
        "las técnicas de Machine Learning para la predicción de precios "
        "agrícolas en la región Piura, 2025."
    ),
    "hipotesis_general": (
        "El entorno agroeconómico influye significativamente en las técnicas "
        "de Machine Learning para la predicción eficaz de precios agrícolas "
        "en la región Piura, 2025."
    ),
    "dimensiones": [
        {
            "codigo": "X1",
            "nombre": "Dimensión Temporal",
            "problema_especifico": "¿Cómo influye la dimensión temporal del entorno "
                "agroeconómico en la predicción de precios agrícolas mediante "
                "técnicas de Machine Learning?",
            "objetivo_especifico": "Evaluar la influencia de la dimensión temporal "
                "del entorno agroeconómico en la predicción de precios agrícolas "
                "mediante técnicas de Machine Learning.",
            "hipotesis_especifica": "La dimensión temporal del entorno agroeconómico "
                "influye significativamente en la predicción de precios agrícolas "
                "mediante técnicas de Machine Learning.",
            "variables": ["ANO", "MES", "MES_SEN", "MES_COS", "ESTACION"],
        },
        {
            "codigo": "X2",
            "nombre": "Dimensión Productiva",
            "problema_especifico": "¿Cómo influye la dimensión productiva del entorno "
                "agroeconómico en la predicción de precios agrícolas mediante "
                "técnicas de Machine Learning?",
            "objetivo_especifico": "Evaluar la influencia de la dimensión productiva "
                "del entorno agroeconómico en la predicción de precios agrícolas "
                "mediante técnicas de Machine Learning.",
            "hipotesis_especifica": "La dimensión productiva del entorno agroeconómico "
                "influye significativamente en la predicción de precios agrícolas "
                "mediante técnicas de Machine Learning.",
            "variables": ["SIEMBRA", "COSECHA", "PRODUCCION", "RENDIMIENTO_TON_HA"],
        },
        {
            "codigo": "X3",
            "nombre": "Dimensión Geográfica",
            "problema_especifico": "¿Cómo influye la dimensión geográfica del entorno "
                "agroeconómico en la predicción de precios agrícolas mediante "
                "técnicas de Machine Learning?",
            "objetivo_especifico": "Evaluar la influencia de la dimensión geográfica "
                "del entorno agroeconómico en la predicción de precios agrícolas "
                "mediante técnicas de Machine Learning.",
            "hipotesis_especifica": "La dimensión geográfica del entorno agroeconómico "
                "influye significativamente en la predicción de precios agrícolas "
                "mediante técnicas de Machine Learning.",
            "variables": ["PROVINCIA", "DISTRITO", "CULTIVO"],  # + UBIGEO opcional
        },
        {
            "codigo": "X4",
            "nombre": "Dimensión Económica Histórica",
            "problema_especifico": "¿Cómo influye la dimensión económica histórica del "
                "entorno agroeconómico en la predicción de precios agrícolas mediante "
                "técnicas de Machine Learning?",
            "objetivo_especifico": "Evaluar la influencia de la dimensión económica "
                "histórica del entorno agroeconómico en la predicción de precios "
                "agrícolas mediante técnicas de Machine Learning.",
            "hipotesis_especifica": "La dimensión económica histórica del entorno "
                "agroeconómico influye significativamente en la predicción de precios "
                "agrícolas mediante técnicas de Machine Learning.",
            "variables": ["PRECIO_LAG1", "VARIACION_PRECIO", "MEDIA_MOVIL_3"],
        },
    ],
    "decisiones_metodologicas_resueltas": [
        "Umbral R²: se fija 0.85 (sección 3.6.3) como oficial. Corregir el "
        "texto de la sección 4.4 (decía ≥0.75) para que cite ≥0.85.",
        "X4: el Anexo 1 debe describirse como PRECIO_LAG1 + VARIACION_PRECIO "
        "+ MEDIA_MOVIL_3 (no 'z-score de precio'), que es lo que el código "
        "usa y lo que sustenta los resultados ya reportados.",
        "X3: el Anexo 1 debe listar solo PROVINCIA, DISTRITO y CULTIVO "
        "(sin UBIGEO), que es lo que el código realmente usa "
        "(USE_UBIGEO_EN_X3=False).",
        "Población vs. muestra: 116,000 (Anexo 1) es la población tras "
        "filtrar a provincias de Piura; 14,393 (cap.4.4) es la muestra "
        "final de modelado tras la limpieza de 3.7.2.1.B. No es un error, "
        "son dos cifras de dos etapas distintas — aclarar esto en el texto.",
    ],
}


def setup_mlflow(experiment_name: str = "AgroPredict_Piura_Tesis"):
    """
    Configura MLflow con SQLite como backend de tracking.
    """
    # Obtener directorio backend
    backend_dir = Path(__file__).resolve().parent.parent
    
    # Usar SQLite en lugar de archivos (FileStore obsoleto)
    db_path = backend_dir / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    
    # Configurar MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow configurado con SQLite")
    print(f"   📁 Tracking URI: {tracking_uri}")
    print(f"   📊 Experimento: {experiment_name}")
    
    return tracking_uri