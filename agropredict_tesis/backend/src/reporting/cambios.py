"""cambios.py — Registro de cambios v2, base del cuadro 'corrección hecha / páginas' que pidió el jurado."""
import os

import pandas as pd

from src.config import OUT_TAB

EFECTO = "Ninguno sobre lo entregado: salida separada; el análisis anterior se conserva como complementario"
CAMBIOS = [
    {"Obs_IA": "10", "Archivo": "data/feature_engineering.py", "Cambio": "Horizonte fijo h (1, 3, 6 meses calendario): X2 y X4 se calculan con información de meses <= T-h; MESES_DESDE_PRECIO_PREVIO se mide desde el origen; prueba anti-fuga automática"},
    {"Obs_IA": "9", "Archivo": "data/feature_engineering.py + config.py", "Cambio": "Modelo principal con X2 rezagada (disponible en el origen); X2 contemporánea (*_ACT) queda solo como cota superior; ANO excluido en v2"},
    {"Obs_IA": "11", "Archivo": "data/splits.py + analysis/ablation_study.py", "Cambio": "Brecha de h-1 meses entre entrenamiento y validación/prueba; folds por meses completos con brecha (MesesExpansivo)"},
    {"Obs_IA": "21-24", "Archivo": "analysis/interpretabilidad.py", "Cambio": "SHAP separado por modelo y escala, con IC bootstrap, SHAP agrupado, procedimiento documentado; permutación con 30 repeticiones, IC y ancla fija"},
]


def escribir(h):
    t = pd.DataFrame(CAMBIOS)
    t["Horizonte_meses"] = h
    t["Efecto_sobre_lo_entregado"] = EFECTO
    t.to_csv(os.path.join(OUT_TAB, "registro_cambios_v2.csv"), index=False, encoding="utf-8-sig")