"""
pipelines.py — FASE 4 (construcción del pipeline) y espacios de búsqueda de la
FASE 6. Todo el preprocesamiento (imputación, escalado, codificación) vive
DENTRO del Pipeline de scikit-learn, así que se ajusta solo con datos de
entrenamiento en cada fold (reproducibilidad y cero fuga; Pedregosa et al., 2011).
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import RANDOM_STATE
from src.models.residual import PersistenciaResidual

CAT_COLS = ["PROVINCIA_enc", "DISTRITO_enc", "CULTIVO_enc"]
NOMBRES = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "lr": "Regresión Lineal",
           "lgbm_res": "LightGBM sobre persistencia"}


def build_pipeline(clave: str, features: list, **params) -> Pipeline:
    if clave == "rf":
        est = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    elif clave == "xgb":
        est = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, tree_method="hist")
    elif clave == "lgbm":
        est = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    elif clave == "lr":
        cats = [c for c in CAT_COLS if c in features]
        nums = [c for c in features if c not in cats]
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ])
        pipe = Pipeline([("prep", prep), ("model", LinearRegression())])
        return pipe
    elif clave == "lgbm_res":
        mk = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("model", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                                                       n_estimators=300, learning_rate=0.05, num_leaves=31))])
        est = PersistenciaResidual(base=mk(), fallback=mk())
        if params:
            est.set_params(**params)
        return est
    else:
        raise ValueError(clave)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("model", est)])
    if params:
        pipe.set_params(**params)
    return pipe


def espacio_busqueda(clave: str) -> dict:
    if clave == "rf":
        return {"model__n_estimators": [100, 200, 300], "model__max_depth": [6, 10, 14, None],
                "model__min_samples_leaf": [1, 2, 5], "model__max_features": [0.5, 0.8, 1.0]}
    if clave == "xgb":
        return {"model__n_estimators": [200, 400, 600], "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.02, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0], "model__min_child_weight": [1, 5, 10],
                "model__reg_lambda": [1, 5, 10]}
    if clave == "lgbm":
        return {"model__n_estimators": [200, 400, 600], "model__max_depth": [-1, 4, 6, 8],
                "model__num_leaves": [15, 31, 63], "model__learning_rate": [0.02, 0.05, 0.1],
                "model__min_child_samples": [10, 30, 60], "model__subsample": [0.7, 0.85, 1.0],
                "model__subsample_freq": [1], "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_alpha": [0.0, 0.5, 2.0], "model__reg_lambda": [0.0, 0.5, 2.0]}
    if clave == "lgbm_res":
        return {"base__model__n_estimators": [200, 400, 600], "base__model__max_depth": [-1, 4, 6],
                "base__model__num_leaves": [15, 31, 63], "base__model__learning_rate": [0.02, 0.05, 0.1],
                "base__model__min_child_samples": [10, 30, 60], "base__model__subsample": [0.7, 1.0],
                "base__model__subsample_freq": [1], "base__model__colsample_bytree": [0.7, 1.0],
                "base__model__reg_lambda": [0.0, 2.0],
                "base__model__objective": ["regression", "regression_l1", "huber"]}
    return {}


def pipeline_arbol(modelo):
    """Devuelve el Pipeline (imputador + árbol) que explica al modelo, sea directo o residual."""
    return modelo.base_ if hasattr(modelo, "base_") else modelo


def importancias_modelo_dummy():  # pragma: no cover
    return None


def importancias_modelo(pipe, features):
    """Importancia normalizada (ganancia) del estimador final de un Pipeline."""
    est = pipeline_arbol(pipe).named_steps["model"]
    if hasattr(est, "booster_"):
        raw = est.booster_.feature_importance(importance_type="gain")
    else:
        raw = est.feature_importances_
    raw = np.asarray(raw, dtype=float)
    return raw / raw.sum() if raw.sum() > 0 else raw
