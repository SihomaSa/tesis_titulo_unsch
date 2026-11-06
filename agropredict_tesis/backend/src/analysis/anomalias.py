"""anomalias.py — Isolation Forest (módulo de control de calidad del sistema)."""
import os

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import OUT_TAB, RANDOM_STATE, TARGET


def detectar(df, X, contaminacion=0.05):
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    iso = IsolationForest(contamination=contaminacion, n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    pred = iso.fit_predict(Xs)
    n_an = int((pred == -1).sum())
    muestra = df.loc[pred == -1].sample(min(50, max(n_an, 1)), random_state=RANDOM_STATE)
    muestra.to_csv(os.path.join(OUT_TAB, "muestra_50_anomalias_revision_manual.csv"), index=False)
    pn, pa = float(df.loc[pred == 1, TARGET].mean()), float(df.loc[pred == -1, TARGET].mean())
    fila = {"contaminacion": contaminacion, "n_anomalias": n_an, "pct": n_an / len(X) * 100,
            "precio_normal": pn, "precio_anomalo": pa, "delta": pa - pn}
    pd.DataFrame([fila]).to_csv(os.path.join(OUT_TAB, "tabla_anomalias.csv"), index=False)
    print(f"[Anomalías] {n_an:,} ({fila['pct']:.2f}%) | precio normal S/{pn:.3f} vs anómalo S/{pa:.3f}")
    return {**fila, "pred": pred}
