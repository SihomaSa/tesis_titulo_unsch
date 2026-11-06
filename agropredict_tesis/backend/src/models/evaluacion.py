"""evaluacion.py — FASE 7: métricas (R², MAE, RMSE, MAPE) y baselines ingenuos."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metricas(y, pred) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = y != 0
    return {
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE": float(np.mean(np.abs((y[m] - pred[m]) / y[m])) * 100),
        "WAPE": float(np.sum(np.abs(y - pred)) / np.sum(np.abs(y)) * 100),
        "R2_log": float(r2_score(np.log(np.clip(y, 1e-6, None)), np.log(np.clip(pred, 1e-6, None)))),
    }


def baseline_persistencia(X, y_train_mediana: float):
    """Pronóstico ingenuo: el precio de hoy = último precio observado."""
    return X["PRECIO_LAG1"].fillna(y_train_mediana).to_numpy()


def baseline_media_movil(X, y_train_mediana: float):
    return X["MEDIA_MOVIL_3"].fillna(y_train_mediana).to_numpy()
