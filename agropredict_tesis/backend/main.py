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

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import tracking
from src.analysis import anomalias, segmentos
from src.analysis.ablation_study import run_ablation
from src.analysis.interpretabilidad import interpretar
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

    print("\n[Fase 7] Contraste de hipótesis (ablation + Wilcoxon + Holm)...")
    abl = run_ablation(X, y, idx_dev, features, dimensiones, rf_params, a.rapido)

    print("\n[Fase 8] Interpretabilidad (SHAP + permutación agrupada)...")
    interp = interpretar(res, X, y, idx_va, idx_te, features, dimensiones)

    print("\n[Fase 9] Validación final y análisis complementarios...")
    df_te, y_te = df.iloc[idx_te], y.iloc[idx_te]
    t_prov = segmentos.por_provincia(df_te, y_te, res["pred_test"])
    t_seg = segmentos.por_segmento_precio(y_te, res["pred_test"])
    anom = anomalias.detectar(df, X)

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
