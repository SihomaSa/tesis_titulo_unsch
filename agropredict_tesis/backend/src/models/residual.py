"""
residual.py — Modelo "sobre persistencia": en lugar de predecir el precio, predice
el CAMBIO relativo respecto del último precio observado, log(precio / PRECIO_LAG1),
y luego reconstruye el precio = PRECIO_LAG1 · exp(cambio). Así el modelo parte del
pronóstico ingenuo y solo aprende lo que lo corrige. Para registros sin historial
usa un modelo directo de respaldo.

Es un candidato más de la Fase 5: la selección en validación decide si gana.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class PersistenciaResidual(BaseEstimator, RegressorMixin):
    def __init__(self, base=None, fallback=None):
        self.base = base
        self.fallback = fallback

    @staticmethod
    def _mascara(X):
        return ((X["TIENE_HISTORIAL"].to_numpy() == 1) & (X["PRECIO_LAG1"].to_numpy() > 0)
                & np.isfinite(X["PRECIO_LAG1"].to_numpy()))

    def fit(self, X, y):
        y = np.asarray(y, float)
        m = self._mascara(X)
        lag = X["PRECIO_LAG1"].to_numpy()
        self.base_ = clone(self.base).fit(X[m], np.log(y[m] / lag[m]))
        self.fallback_ = clone(self.fallback).fit(X, y)
        return self

    def predict(self, X):
        out = np.asarray(self.fallback_.predict(X), float)
        m = self._mascara(X)
        if m.any():
            lag = X["PRECIO_LAG1"].to_numpy()
            out[m] = lag[m] * np.exp(self.base_.predict(X[m]))
        return out
