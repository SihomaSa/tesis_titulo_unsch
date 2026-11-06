"""
config.py — Configuración central del proyecto AgroPredict.

Contiene rutas, parámetros del experimento y la Matriz de Consistencia
(Anexo 1 de la tesis) como única fuente de verdad: el pipeline, la API y el
texto de resultados generado automáticamente leen de aquí.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

# ------------------------------------------------------------------
# RUTAS
# ------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get("AGRO_DATA_PATH", os.path.join(BACKEND_DIR, "data", "dataset_agricola.csv"))
OUT_FIG = os.path.join(BACKEND_DIR, "outputs", "figures")
OUT_TAB = os.path.join(BACKEND_DIR, "outputs", "tables")
OUT_MOD = os.path.join(BACKEND_DIR, "outputs", "models")
OUT_TXT = os.path.join(BACKEND_DIR, "outputs", "tesis_texto")
for _d in (OUT_FIG, OUT_TAB, OUT_MOD, OUT_TXT):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------
# PARÁMETROS DEL EXPERIMENTO (modo completo; main.py --rapido los reduce)
# ------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "PRECIO_CHACRA"
PROVINCIAS_PIURA = {
    "PIURA", "SULLANA", "MORROPON", "TALARA", "PAITA",
    "AYABACA", "HUANCABAMBA", "SECHURA",
}

# Fase 3 — división cronológica 70/15/15 (metodología del Dr. Janampa)
FRAC_TRAIN, FRAC_VAL, FRAC_TEST = 0.70, 0.15, 0.15

# Fase 6 — búsqueda aleatoria de hiperparámetros con validación temporal
N_SPLITS_TUNING = 5        # TimeSeriesSplit dentro del tramo de entrenamiento
N_ITER_TUNING = 25         # combinaciones evaluadas por modelo

# Contraste de hipótesis (ablation + Wilcoxon). Con n folds, el p mínimo de
# Wilcoxon unilateral es 1/2^n; con n=10 es 0.00098, suficiente incluso
# después de la corrección de Holm para 4 pruebas.
N_SPLITS_ABLATION = 10
ALPHA = 0.05

# Criterio de aceptación de la Hipótesis General (sección 3.6.3 de la tesis)
UMBRAL_R2_OFICIAL = 0.85

# Segmentación por valor del cultivo (limitación documentada en 4.4.4)
UMBRAL_PRECIO_ALTO = 4.5   # S/ por kg

COLORES = {
    "principal": "#1B6B4A", "secundario": "#C4960A",
    "X1": "#1A5276", "X2": "#1B6B4A", "X3": "#C4960A", "X4": "#C0392B",
    "gris": "#7F8C8D", "lgbm": "#7D3C98",
}

# ------------------------------------------------------------------
# MATRIZ DE CONSISTENCIA (Anexo 1)
# ------------------------------------------------------------------
_dim = lambda nombre: {
    "problema_especifico": f"¿Cómo influye la dimensión {nombre} del entorno agroeconómico en la predicción de precios agrícolas mediante técnicas de Machine Learning?",
    "objetivo_especifico": f"Evaluar la influencia de la dimensión {nombre} del entorno agroeconómico en la predicción de precios agrícolas mediante técnicas de Machine Learning.",
    "hipotesis_especifica": f"La dimensión {nombre} del entorno agroeconómico influye significativamente en la predicción de precios agrícolas mediante técnicas de Machine Learning.",
}

MATRIZ_CONSISTENCIA = {
    "titulo": "Sistema web para la predicción de precios agrícolas mediante técnicas de Machine Learning, Piura, 2025",
    "problema_general": "¿En qué medida las dimensiones del entorno agroeconómico influyen en la precisión de la predicción de los precios agrícolas mediante técnicas de Machine Learning en la región Piura, 2025?",
    "objetivo_general": "Determinar la influencia del entorno agroeconómico en la eficacia de las técnicas de Machine Learning para la predicción de precios agrícolas en la región Piura, 2025.",
    "hipotesis_general": "El entorno agroeconómico influye significativamente en las técnicas de Machine Learning para la predicción eficaz de precios agrícolas en la región Piura, 2025.",
    "dimensiones": [
        {"codigo": "X1", "clave": "X1_Temporal", "nombre": "Dimensión Temporal", **_dim("temporal"),
         "variables": ["ANO", "MES_NUM", "MES_SEN", "MES_COS", "ESTACION"]},
        {"codigo": "X2", "clave": "X2_Productiva", "nombre": "Dimensión Productiva", **_dim("productiva"),
         "variables": ["SIEMBRA", "COSECHA", "PRODUCCION", "RENDIMIENTO_TON_HA"]},
        {"codigo": "X3", "clave": "X3_Geografica", "nombre": "Dimensión Geográfica", **_dim("geográfica"),
         "variables": ["PROVINCIA_enc", "DISTRITO_enc", "CULTIVO_enc"]},
        {"codigo": "X4", "clave": "X4_Economica_Historica", "nombre": "Dimensión Económica Histórica", **_dim("económica histórica"),
         "variables": ["PRECIO_LAG1", "MEDIA_MOVIL_3", "VARIACION_PRECIO_PREVIA",
                       "MESES_DESDE_PRECIO_PREVIO", "TIENE_HISTORIAL"]},
    ],
}
