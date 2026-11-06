"""
entrenamiento.py — FASES 5, 6 y 7.

  Fase 5  Modelado: RF, XGBoost, LightGBM y Regresión Lineal + 2 baselines
          ingenuos (persistencia y media móvil) que el jurado debe poder usar
          como vara de comparación.
  Fase 6  Tuning: RandomizedSearchCV con TimeSeriesSplit SOLO sobre el tramo
          de entrenamiento (Bergstra & Bengio, 2012).
  Fase 7  Evaluación: selección del modelo en VALIDACIÓN; el conjunto de
          PRUEBA se usa una sola vez, con el modelo ya elegido y reentrenado
          con entrenamiento+validación.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src import tracking
from src.config import (N_ITER_TUNING, N_SPLITS_TUNING, OUT_MOD, OUT_TAB, RANDOM_STATE)
from src.models.evaluacion import baseline_media_movil, baseline_persistencia, metricas
from src.models.pipelines import NOMBRES, build_pipeline, espacio_busqueda


def _limpiar(params):
    return {k.split("__")[-1]: v for k, v in params.items()}


def entrenar_y_comparar(X, y, idx_tr, idx_va, idx_te, features, catalogo, info_split, rapido=False):
    n_iter = 4 if rapido else N_ITER_TUNING
    n_spl = 3 if rapido else N_SPLITS_TUNING
    Xtr, ytr = X.iloc[idx_tr], y.iloc[idx_tr]
    Xva, yva = X.iloc[idx_va], y.iloc[idx_va]
    Xte, yte = X.iloc[idx_te], y.iloc[idx_te]
    mediana_tr = float(ytr.median())
    tss = TimeSeriesSplit(n_splits=n_spl)

    modelos, mejores_params, cv_mae, r2_folds = {}, {}, {}, {}
    with tracking.run("02_tuning_y_comparacion"):
        for clave in ["lr", "rf", "xgb", "lgbm", "lgbm_res"]:
            pipe = build_pipeline(clave, features)
            if clave == "lr":
                best = pipe.fit(Xtr, ytr)
                mejores_params[clave] = {}
                cv_mae[clave] = np.nan
            else:
                print(f"[Fase 6] Tuning {NOMBRES[clave]} ({n_iter} combinaciones x {n_spl} folds temporales)...")
                bs = RandomizedSearchCV(pipe, espacio_busqueda(clave), n_iter=n_iter, cv=tss,
                                        scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
                                        n_jobs=1, refit=True)
                bs.fit(Xtr, ytr)
                best = bs.best_estimator_
                mejores_params[clave] = _limpiar(bs.best_params_)
                cv_mae[clave] = float(-bs.best_score_)
                print(f"         mejor MAE CV={cv_mae[clave]:.4f} | {mejores_params[clave]}")
            modelos[clave] = best
            # R² por fold temporal (insumo de la Figura "Evolución del R² por fold")
            fr = []
            for a, b in tss.split(Xtr):
                p = clone(best).fit(Xtr.iloc[a], ytr.iloc[a])
                fr.append(float(r2_score(ytr.iloc[b], p.predict(Xtr.iloc[b]))))
            r2_folds[clave] = fr

        # ---------- Evaluación en VALIDACIÓN ----------
        filas_val = {}
        filas_val["Persistencia (baseline)"] = metricas(yva, baseline_persistencia(Xva, mediana_tr))
        filas_val["Media móvil 3 (baseline)"] = metricas(yva, baseline_media_movil(Xva, mediana_tr))
        for clave, m in modelos.items():
            filas_val[NOMBRES[clave]] = metricas(yva, m.predict(Xva))
        tabla_val = pd.DataFrame(filas_val).T.reset_index().rename(columns={"index": "Modelo"})
        tabla_val["R2_CV_fold_media"] = [np.mean(r2_folds[k]) if k in r2_folds and NOMBRES.get(k) == n else np.nan
                                         for n, k in zip(tabla_val["Modelo"], [None] * len(tabla_val))]
        inv = {v: k for k, v in NOMBRES.items()}
        tabla_val["R2_CV_fold_media"] = [np.mean(r2_folds[inv[n]]) if n in inv else np.nan for n in tabla_val["Modelo"]]
        tabla_val["R2_CV_fold_std"] = [np.std(r2_folds[inv[n]]) if n in inv else np.nan for n in tabla_val["Modelo"]]
        tabla_val.to_csv(os.path.join(OUT_TAB, "tabla_comparacion_validacion.csv"), index=False)

        # ---------- Selección en validación (criterio: menor MAE) ----------
        candidatos = {k: filas_val[NOMBRES[k]]["MAE"] for k in modelos}
        clave_sel = min(candidatos, key=candidatos.get)
        nombre_sel = NOMBRES[clave_sel]
        print(f"[Fase 7] Modelo seleccionado en VALIDACIÓN (menor MAE): {nombre_sel}")

        # Intervalo empírico 90% con residuos de validación (modelo entrenado solo con train)
        res_abs = np.abs(yva.to_numpy() - modelos[clave_sel].predict(Xva))
        q90 = float(np.quantile(res_abs, 0.90))

        # ---------- Reentrenamiento train+val y evaluación ÚNICA en PRUEBA ----------
        Xdev = X.iloc[np.concatenate([idx_tr, idx_va])]
        ydev = y.iloc[np.concatenate([idx_tr, idx_va])]
        finales = {k: clone(m).fit(Xdev, ydev) for k, m in modelos.items()}
        filas_test = {}
        med_dev = float(ydev.median())
        filas_test["Persistencia (baseline)"] = metricas(yte, baseline_persistencia(Xte, med_dev))
        filas_test["Media móvil 3 (baseline)"] = metricas(yte, baseline_media_movil(Xte, med_dev))
        for k, m in finales.items():
            filas_test[NOMBRES[k]] = metricas(yte, m.predict(Xte))
        tabla_test = pd.DataFrame(filas_test).T.reset_index().rename(columns={"index": "Modelo"})
        tabla_test.to_csv(os.path.join(OUT_TAB, "tabla_resultados_prueba.csv"), index=False)

        modelo_final = finales[clave_sel]
        pred_te = modelo_final.predict(Xte)
        met_final = filas_test[nombre_sel]
        print(f"[Fase 7] PRUEBA (n={len(yte):,}) {nombre_sel}: R²={met_final['R2']:.4f} | MAE={met_final['MAE']:.4f} "
              f"| RMSE={met_final['RMSE']:.4f} | MAPE={met_final['MAPE']:.1f}%")
        print(f"[Fase 7] Baseline persistencia en PRUEBA: R²={filas_test['Persistencia (baseline)']['R2']:.4f}")

        pd.DataFrame([{"modelo": NOMBRES[k], **mejores_params[k], "cv_mae": cv_mae[k]} for k in modelos]
                     ).to_csv(os.path.join(OUT_TAB, "tabla_hiperparametros_finales.csv"), index=False)

        bundle = {"model": modelo_final, "features": features, "nombre": nombre_sel,
                  "catalogo": catalogo, "q90_abs": q90, "metricas_prueba": met_final,
                  "mediana_precio": med_dev}
        ruta = os.path.join(OUT_MOD, "modelo_final.pkl")
        joblib.dump(bundle, ruta)
        tracking.params({"modelo_seleccionado": nombre_sel, **{f"hp_{k}": v for k, v in mejores_params.items()}})
        tracking.metrics({f"test_{k}": v for k, v in met_final.items()})
        tracking.artifact(ruta)

    return {
        "modelos_train": modelos, "modelos_finales": finales, "clave_sel": clave_sel,
        "nombre_sel": nombre_sel, "modelo_final": modelo_final, "pred_test": pred_te,
        "tabla_val": tabla_val, "tabla_test": tabla_test, "filas_val": filas_val, "filas_test": filas_test,
        "mejores_params": mejores_params, "cv_mae": cv_mae, "r2_folds": r2_folds,
        "q90_abs": q90, "info_split": info_split,
    }
