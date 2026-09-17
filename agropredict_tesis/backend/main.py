"""
main.py — Punto de entrada ÚNICO del pipeline de tesis AgroPredict.

Ejecuta, en orden, todo lo que necesitas para el capítulo 4:
  1. Carga y limpieza de datos            -> src/data/data_loader.py
  2. Ingeniería de características X1-X4  -> src/data/feature_engineering.py
  3. Comparación de 4 modelos (OG)        -> src/models/compare_models.py
  4. Modelo final hold-out (PG/OG/HG)     -> src/models/final_model.py
  5. Ablation study por dimensión (H1-H4) -> src/analysis/ablation_study.py
  6. Isolation Forest (anomalías)         -> src/analysis/anomaly_detection.py
  7. Reporte de contrastación de hipótesis-> src/analysis/hypothesis_report.py

Cada corrida queda registrada en MLflow (backend/mlruns/). Para ver y
exportar las figuras a tu tesis:

    mlflow ui --backend-store-uri backend/mlruns

y abre http://localhost:5000 -> entra a cada run -> pestaña "Artifacts".

Uso:
    cd backend
    pip install -r requirements.txt
    python main.py
"""

# 🔧 CAMBIO 1: Forzar backend Agg ANTES de cualquier import de matplotlib
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # <- esto evita el crash de Tkinter

import matplotlib
matplotlib.use("Agg")  # doble seguridad

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import setup_mlflow, MATRIZ_CONSISTENCIA
from src.data.data_loader import load_and_clean
from src.data.feature_engineering import build_features
from src.models.compare_models import compare_models
from src.models.final_model import train_final_model
from src.analysis.ablation_study import run_ablation
from src.analysis.anomaly_detection import detect_anomalies
from src.analysis.rendimiento_provincial import rendimiento_por_provincia
from src.analysis.rendimiento_por_frecuencia_cultivo import rendimiento_por_frecuencia
from src.analysis.hypothesis_report import build_report


def main():
    print("=" * 80)
    print("🚀 PIPELINE DE TESIS — AGROPREDICT (Piura, 2025)")
    print("=" * 80)
    print(f"\nPROBLEMA GENERAL:\n{MATRIZ_CONSISTENCIA['problema_general']}\n")

    setup_mlflow()

    # 1-2. Datos y features
    df = load_and_clean()
    df, features, dimensiones = build_features(df)
    X = df[features].fillna(0)
    y = df["PRECIO_CHACRA"]

    # 3. Comparación de modelos (Objetivo General)
    print("\n" + "-" * 80)
    print("[OBJETIVO GENERAL] Comparación de técnicas de Machine Learning")
    print("-" * 80)
    resultados_modelos, tss = compare_models(X, y)

    # 4. Modelo final hold-out (Problema/Objetivo/Hipótesis General)
    print("\n" + "-" * 80)
    print("[HIPÓTESIS GENERAL] Modelo final hold-out 80/20")
    print("-" * 80)
    final = train_final_model(X, y, features, dimensiones)

    # 5. Ablation study (Hipótesis específicas H1-H4)
    print("\n" + "-" * 80)
    print("[HIPÓTESIS ESPECÍFICAS H1-H4] Ablation study por dimensión")
    print("-" * 80)
    ablation_resultados = run_ablation(X, y, features, dimensiones, tss)

    # 6. Anomalías
    print("\n" + "-" * 80)
    print("[INSTRUMENTO] Isolation Forest — detección de anomalías")
    print("-" * 80)
    anomaly_info = detect_anomalies(df, X)

    # 6b. Rendimiento por provincia (objetivo específico c / dimensión geográfica)
    print("\n" + "-" * 80)
    print("[OBJETIVO ESPECÍFICO c] Rendimiento del modelo final por provincia")
    print("-" * 80)
    tabla_provincial = rendimiento_por_provincia(
        df, X, y, final["model"], final["split_idx"]
    )

    # 6c. Rendimiento por frecuencia de cultivo (comunes vs. alto valor)
    print("\n" + "-" * 80)
    print("[LIMITACIÓN DETECTADA] Rendimiento por frecuencia de cultivo")
    print("-" * 80)
    resumen_frecuencia = rendimiento_por_frecuencia(
        df, X, y, final["model"], final["split_idx"]
    )

    # 7. Reporte final de contrastación de hipótesis
    print("\n" + "-" * 80)
    print("[REPORTE] Contrastación de hipótesis (capítulo 4.4)")
    print("-" * 80)
    reporte = build_report(
        n_valido=len(df),
        final_metrics=final["metrics"],
        ablation_resultados=ablation_resultados,
        importancia_por_dimension=final["importancia_por_dimension"],
        anomaly_info=anomaly_info,
        modelo_seleccionado=final["modelo_seleccionado"],
        metrics_por_modelo=final["metrics_por_modelo"],
        resumen_frecuencia_cultivo=resumen_frecuencia,
    )

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETADO")
    print(f"   HG (umbral oficial 0.85): {reporte['hipotesis_general']['estado_vs_umbral_OFICIAL_0.85']}")
    for h in reporte["hipotesis_especificas"]:
        print(f"   {h['codigo']}: Δ={h['diferencia_R2']:.4f} p={h['p_valor']:.4f} "
              f"({h['significativa_wilcoxon']})")
    print("\n📁 Tablas: backend/outputs/tables/")
    print("📁 Figuras: backend/outputs/figures/")
    print("📁 MLflow: mlflow ui --backend-store-uri backend/mlruns")
    print("=" * 80)


if __name__ == "__main__":
    main()