"""
hypothesis_report.py — Genera el reporte de "Contrastación de hipótesis"
(equivalente a tu sección 4.4.1) a partir de resultados REALES de la
corrida, no de números fijos en el código. Así el texto de tu tesis y el
resultado del pipeline nunca quedan desincronizados.
"""
import json
import os
from datetime import datetime

from src.config import MATRIZ_CONSISTENCIA, UMBRAL_R2_CAP3, UMBRAL_R2_CAP4, \
    UMBRAL_R2_OFICIAL, OUT_TAB


def build_report(n_valido: int, final_metrics: dict, ablation_resultados: list,
                  importancia_por_dimension: dict, anomaly_info: dict,
                  modelo_seleccionado: str = None, metrics_por_modelo: dict = None,
                  resumen_frecuencia_cultivo: dict = None) -> dict:
    r2_final = final_metrics["r2_final"]
    hg_estado_cap4 = "ACEPTADA" if r2_final >= UMBRAL_R2_CAP4 else "RECHAZADA"
    hg_estado_cap3 = "ACEPTADA" if r2_final >= UMBRAL_R2_CAP3 else "RECHAZADA"
    hg_estado_oficial = "ACEPTADA" if r2_final >= UMBRAL_R2_OFICIAL else "RECHAZADA"

    hipotesis_especificas = []
    for dim_cfg, res in zip(MATRIZ_CONSISTENCIA["dimensiones"], ablation_resultados):
        hipotesis_especificas.append({
            "codigo": dim_cfg["codigo"],
            "texto": dim_cfg["hipotesis_especifica"],
            "R2_sin_dimension": res["R2_sin_dim"],
            "diferencia_R2": res["Diferencia_R2"],
            "p_valor": res["p_valor"],
            "significativa_wilcoxon": res["Significativa_estadistica"],
            "importancia_gini_dimension": importancia_por_dimension.get(res["Dimension"]),
        })

    reporte = {
        "fecha_generacion": datetime.now().isoformat(),
        "n_registros_validos": n_valido,
        "modelo_final": final_metrics,
        "modelo_seleccionado": modelo_seleccionado,
        "metrics_por_modelo_MISMO_holdout": metrics_por_modelo,
        "hipotesis_general": {
            "texto": MATRIZ_CONSISTENCIA["hipotesis_general"],
            "estado_vs_umbral_OFICIAL_0.85": hg_estado_oficial,
            "estado_vs_umbral_0.75_cap4_referencia": hg_estado_cap4,
            "estado_vs_umbral_0.85_cap3_referencia": hg_estado_cap3,
            "evidencia": f"R² final = {r2_final:.4f} (umbral oficial = {UMBRAL_R2_OFICIAL})",
        },
        "hipotesis_especificas": hipotesis_especificas,
        "anomalias": anomaly_info,
        "rendimiento_por_frecuencia_cultivo": resumen_frecuencia_cultivo,
        "decisiones_metodologicas_resueltas": MATRIZ_CONSISTENCIA["decisiones_metodologicas_resueltas"],
    }

    path = os.path.join(OUT_TAB, "reporte_contrastacion_hipotesis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
    print(f"[hypothesis_report] ✅ {path}")

    return reporte
