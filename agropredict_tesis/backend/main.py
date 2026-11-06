"""
main.py — Punto de entrada ÚNICO. Ejecuta las 9 fases de la metodología
(CRISP-DM + KDD adaptada a modelos predictivos; Chapman et al., 2000; Fayyad
et al., 1996; Géron, 2023) y genera tablas, figuras, el modelo y el TEXTO de
resultados con los números reales.

    cd backend
    pip install -r requirements.txt
    # coloca tu CSV en data/dataset_agricola.csv
    python main.py

    python main.py --rapido      # prueba de humo (pocos minutos, menos búsqueda)
    python main.py --datos RUTA  # otro CSV
"""
import argparse
import os
import sys


# ---------------------------------------------------------------------------
# FIX: forzar UTF-8 en stdout/stderr ANTES de cualquier print.
# Evita UnicodeEncodeError (cp1252) con caracteres como Δ, ², tildes, etc.
# Debe ejecutarse antes de importar módulos que imprimen al cargarse.
# ---------------------------------------------------------------------------
def _forzar_utf8():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_forzar_utf8()

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import tracking
from src.analysis import anomalias, segmentos
from src.analysis.ablation_study import run_ablation
from src.analysis.interpretabilidad import interpretar
from src.analysis import depuracion_sesgo, documentacion, robustez, significancia
from src.models.evaluacion import baseline_media_movil, baseline_persistencia
from src.config import DATA_PATH, MATRIZ_CONSISTENCIA, TARGET
from src.data.data_loader import load_and_clean
from src.data.feature_engineering import build_features
from src.data.splits import split_cronologico
from src.models.entrenamiento import entrenar_y_comparar
from src.models.pipelines import NOMBRES
from src.reporting.texto_tesis import generar_texto
from src.visualization import figuras as fg
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba de humo con menos iteraciones")
    ap.add_argument("--datos", default=DATA_PATH)
    ap.add_argument("--sin-extras", action="store_true", help="omite los análisis de robustez/significancia (Obs. del jurado)")
    a = ap.parse_args()

    print("=" * 78)
    print("AGROPREDICT — Metodología predictiva en 9 fases (CRISP-DM + KDD)")
    print("=" * 78)
    if not os.path.exists(a.datos):
        sys.exit(f"No se encontró el dataset en {a.datos}. Copia tu CSV a backend/data/dataset_agricola.csv")
    tracking.setup()

    print(f"\n[Fase 1] Problema: {MATRIZ_CONSISTENCIA['problema_general']}")
    print(f"[Fase 1] Variable objetivo: {TARGET} (regresión supervisada)")

    df, embudo = load_and_clean(a.datos)                                   # Fases 2-3
    df, features, dimensiones, catalogo = build_features(df)               # Fase 3
    X = df[features].astype(float)
    y = df[TARGET].astype(float)
    idx_tr, idx_va, idx_te, info_split = split_cronologico(df)             # Fase 3

    res = entrenar_y_comparar(X, y, idx_tr, idx_va, idx_te, features, catalogo, info_split, a.rapido)  # Fases 4-7
    rf_params = res["mejores_params"]["rf"]
    idx_dev = np.concatenate([idx_tr, idx_va])

    # Obs. 2: la ablación PRINCIPAL se hace con LightGBM (misma familia que el modelo final, hiperparámetros ya
    # ajustados). Random Forest se conserva solo como análisis de robustez (tabla_ablation_wilcoxon.csv).
    print("\n[Fase 7] Contraste de hipótesis: ablation + Wilcoxon + Holm con LightGBM (principal)...")
    abl = run_ablation(X, y, idx_dev, features, dimensiones, res["mejores_params"]["lgbm"], a.rapido,
                       clave="lgbm", sufijo="_lgbm")
    print("\n[Fase 7] Robustez: la misma ablación con Random Forest...")
    abl_rf = run_ablation(X, y, idx_dev, features, dimensiones, rf_params, a.rapido, clave="rf", sufijo="")

    print("\n[Fase 8] Interpretabilidad (SHAP + permutación agrupada)...")
    interp = interpretar(res, X, y, idx_va, idx_te, features, dimensiones)

    print("\n[Fase 9] Validación final y análisis complementarios...")
    df_te, y_te = df.iloc[idx_te], y.iloc[idx_te]
    t_prov = segmentos.por_provincia(df_te, y_te, res["pred_test"])
    t_seg = segmentos.por_segmento_precio(y_te, res["pred_test"])
    anom = anomalias.detectar(df, X)

    if not a.sin_extras:
        print("\n[Fase 9b] Observaciones del jurado: significancia, robustez, documentación y sesgo de depuración...")
        df_te, y_te_, X_te = df.iloc[idx_te], y.iloc[idx_te], X.iloc[idx_te]
        med_dev = float(y.iloc[idx_dev].median())
        per_te = df_te["PERIODO"].to_numpy()
        refs = {"Persistencia": baseline_persistencia(X_te, med_dev), "Media móvil 3": baseline_media_movil(X_te, med_dev)}
        for k, m_ in res["modelos_finales"].items():
            if k != res["clave_sel"]:
                refs[NOMBRES[k]] = m_.predict(X_te)
        print("\n[Obs.1,3] Significancia frente a baselines y otros modelos (pérdidas pareadas por mes, HAC, bloques)")
        significancia.reporte_prueba(per_te, y_te_.to_numpy(), res["pred_test"], res["nombre_sel"], refs)
        print("\n[Obs.3] Robustez de la significancia: pérdida absoluta y cuadrática, L-HAC, bloque, prueba de signos")
        significancia.robustez_significancia(per_te, y_te_.to_numpy(), res["pred_test"], refs, res["nombre_sel"])

        p_lgbm = res["mejores_params"]["lgbm"]
        p_res = res["mejores_params"]["lgbm_res"]
        print("\n[Obs.9] Ablación con X3 separada en territorio y producto (Holm con 5 pruebas)")
        run_ablation(X, y, idx_dev, features, robustez.dimensiones_x3_separada(dimensiones), p_lgbm, a.rapido,
                     clave="lgbm", sufijo="_lgbm_x3_separada")
        print("\n[Obs.1] Contribución de cada dimensión EN PRUEBA (LightGBM directo, pareado por mes)")
        robustez.contribucion_en_prueba(df, y, idx_dev, idx_te, features, dimensiones, p_lgbm, a.rapido)
        print("\n[Obs.9] Contribución EN PRUEBA con X3 separada en territorio y producto")
        robustez.contribucion_en_prueba(df, y, idx_dev, idx_te, features, robustez.dimensiones_x3_separada(dimensiones),
                                        p_lgbm, a.rapido, etiqueta="lgbm_x3_separada")
        print("\n[Obs.2] Contribución X1–X3 EN PRUEBA sobre el MODELO FINAL (LightGBM sobre persistencia)")
        robustez.contribucion_en_prueba_residual(df, y, idx_dev, idx_te, features,
                                                 robustez.dimensiones_x3_separada(dimensiones), p_res, a.rapido)
        print("\n[Obs.8] Variante con X2 rezagada")
        robustez.variante_x2_rezagada(df, y, idx_dev, idx_te, features, dimensiones, p_lgbm, a.rapido)

        print("\n[Obs.11] Particiones y folds")
        documentacion.tabla_particiones(df, idx_tr, idx_va, idx_te)
        documentacion.tabla_folds(df, idx_tr, idx_dev, a.rapido)
        print("\n[Obs.4] MAE por brecha al precio previo")
        documentacion.mae_por_brecha(df_te, y_te_, res["pred_test"], refs["Persistencia"])
        print("\n[Obs.12] SHAP e importancia comparada")
        documentacion.shap_detalle(interp, features, dimensiones)
        documentacion.importancia_comparada(interp, abl, dimensiones)

        print("\n[Obs.10] Sesgo por depuración")
        depuracion_sesgo.diagnostico(a.datos)
        depuracion_sesgo.sensibilidad(a.datos, params_res=p_res, rapido=a.rapido)

    # Figuras
    fg.fig_flujo_metodologico()
    fg.fig_dimensiones(dimensiones)
    fg.fig_comparacion(res["tabla_val"], res["tabla_test"])
    fg.fig_r2_folds(res["r2_folds"], NOMBRES)
    fg.fig_real_vs_predicho(y_te.to_numpy(), res["pred_test"], res["nombre_sel"], res["filas_test"][res["nombre_sel"]]["R2"])
    fg.fig_importancia(interp["imp_dim"], interp["shap"]["por_dimension"] if interp["shap"] else None, interp["permutacion"])
    fg.fig_ablation(abl["tabla"])
    fg.fig_provincias(t_prov)
    fg.fig_anomalias(df, anom["pred"], TARGET)
    if interp["shap"]:
        fg.fig_shap_resumen(interp["shap"])

    rutas = generar_texto(df=df, embudo=embudo, info_split=info_split, res=res, abl=abl, interp=interp,
                          t_prov=t_prov, t_seg=t_seg, anom=anom, features=features, dimensiones=dimensiones,
                          rapido=a.rapido)
    # copia las figuras que usa el dashboard al frontend (si existe la carpeta)
    import shutil
    destino = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "assets", "figures")
    if os.path.isdir(os.path.dirname(destino)):
        os.makedirs(destino, exist_ok=True)
        for n in ("figura_comparacion_modelos_prueba.png", "figura_ablation_por_dimension.png",
                  "figura_importancia_por_dimension.png", "figura_real_vs_predicho.png"):
            origen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures", n)
            if os.path.exists(origen):
                shutil.copy(origen, destino)

    print("\n" + "=" * 78)
    print("LISTO. Revisa:")
    print("  backend/outputs/tesis_texto/   -> texto de resultados/conclusiones listo para pegar en Word")
    print("  backend/outputs/tables/        -> todas las tablas (CSV)")
    print("  backend/outputs/figures/       -> figuras PNG 300 dpi")
    print("  backend/outputs/models/        -> modelo_final.pkl (lo usa la API)")
    for r in rutas:
        print("   ·", r)
    print("=" * 78)


if __name__ == "__main__":
    main()