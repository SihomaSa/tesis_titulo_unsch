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
DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_agricola.csv")
_SUB = os.environ.get("AGRO_OUT_SUBDIR", "")   # "" = resultados anteriores; "horizonte_3m" = v2
OUT_FIG = os.path.join(BACKEND_DIR, "outputs", _SUB, "figures")
OUT_TAB = os.path.join(BACKEND_DIR, "outputs", _SUB, "tables")
OUT_MOD = os.path.join(BACKEND_DIR, "outputs", _SUB, "models")
OUT_TXT = os.path.join(BACKEND_DIR, "outputs", _SUB, "tesis_texto")
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
    "problema_especifico": f"¿En qué medida contribuye predictivamente la dimensión {nombre} del entorno agroeconómico en la eficacia de la predicción de precios agrícolas mediante técnicas de Machine Learning?",
    "objetivo_especifico": f"Determinar la contribución predictiva de la dimensión {nombre} del entorno agroeconómico en la eficacia de la predicción de precios agrícolas mediante técnicas de Machine Learning.",
    # Obs. 1 del jurado: «influye significativamente» sugería causalidad; se reformula como contribución predictiva.
    "hipotesis_especifica": f"La dimensión {nombre} del entorno agroeconómico contribuye predictivamente en la eficacia de la predicción de precios agrícolas mediante técnicas de Machine Learning.",
    "hipotesis_especifica_original": f"La dimensión {nombre} del entorno agroeconómico influye significativamente en la predicción de precios agrícolas mediante técnicas de Machine Learning.",
}

MATRIZ_CONSISTENCIA = {
    "titulo": "Sistema web para la predicción de precios agrícolas mediante técnicas de Machine Learning, Piura, 2025",
    "problema_general": "¿En qué medida las dimensiones del entorno agroeconómico contribuyen predictivamente en la eficacia de la predicción de los precios agrícolas mediante técnicas de Machine Learning en la región Piura, 2025?",
    "objetivo_general": "Determinar la contribución predictiva del entorno agroeconómico en la eficacia de las técnicas de Machine Learning para la predicción de precios agrícolas en la región Piura, 2025.",
    # Obs. 1: la hipótesis general se separa en eficacia (HG-a) y superación de baselines (HG-b); las dimensiones se contrastan en H1–H4.
    "hipotesis_general": "HG-a: el modelo de Machine Learning seleccionado alcanza un R² ≥ 0.85 en datos no vistos de la región Piura. HG-b: el modelo reduce el error de pronóstico frente a los baselines de persistencia y media móvil.",
    "hipotesis_general_a": "El modelo de Machine Learning seleccionado alcanza un R² ≥ 0.85 en el conjunto de prueba cronológico (región Piura).",
    "hipotesis_general_b": "El modelo seleccionado reduce el error de pronóstico (MAE) frente a los baselines de persistencia y de media móvil, con diferencias de pérdida pareadas por mes.",
    "hipotesis_general_original": "El entorno agroeconómico influye significativamente en las técnicas de Machine Learning para la predicción eficaz de precios agrícolas en la región Piura, 2025.",
    "dimensiones": [
        {"codigo": "X1", "clave": "X1_Temporal", "nombre": "Dimensión Temporal", **_dim("temporal"),
         "variables": ["ANO", "MES_NUM", "MES_SEN", "MES_COS", "ESTACION"]},
        {"codigo": "X2", "clave": "X2_Productiva", "nombre": "Dimensión Productiva", **_dim("productiva"),
         "variables": ["SIEMBRA", "COSECHA", "PRODUCCION", "RENDIMIENTO_TON_HA"]},
        {"codigo": "X3", "clave": "X3_Espacial_Producto", "nombre": "Dimensión Espacial y de producto", **_dim("espacial y de producto"),
         "variables": ["PROVINCIA_enc", "DISTRITO_enc", "CULTIVO_enc"]},
        {"codigo": "X4", "clave": "X4_Economica_Historica", "nombre": "Dimensión Económica Histórica", **_dim("económica histórica"),
         "variables": ["PRECIO_LAG1", "MEDIA_MOVIL_3", "VARIACION_PRECIO_PREVIA",
                       "MESES_DESDE_PRECIO_PREVIO", "TIENE_HISTORIAL"]},
    ],
}

# --- v2: horizonte fijo (solo se usa con --horizonte) ---
HORIZONTES = [1, 3, 6]          # meses calendario entre el origen del pronóstico y el mes objetivo
X2_MODO = "inmediata"           # "rezagada" (principal) | "contemporanea" (cota superior, no disponible ex ante)
X2_REZAGO_EXTRA_MESES = 0       # súbelo a 12 si SIEMBRA/COSECHA/PRODUCCION se publican anualmente
USAR_ANO_V2 = True              # ANO no se puede extrapolar con árboles (obs. 16)