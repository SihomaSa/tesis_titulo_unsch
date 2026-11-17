"""
Regenera outputs/<subdir>/tables/reporte_contrastacion_hipotesis.json SIN reentrenar,
leyendo los CSV ya generados por main.py.

    cd backend
    python regenerar_reporte.py                 # usa outputs/horizonte_3m
    python regenerar_reporte.py --subdir ""     # usa outputs/ (análisis anterior)
"""
import argparse
import json
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--subdir", default="horizonte_3m")
_a = _ap.parse_args()
os.environ["AGRO_OUT_SUBDIR"] = _a.subdir          # ANTES de importar src.config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from src.config import ALPHA, OUT_TAB  # noqa: E402
from src.reporting import texto_robustez as rb  # noqa: E402
from src.reporting.reporte_json import construir_reporte, guardar_reporte  # noqa: E402

ruta_json = os.path.join(OUT_TAB, "reporte_contrastacion_hipotesis.json")
if not os.path.exists(ruta_json):
    sys.exit(f"No existe {ruta_json}. Corre antes: python main.py --horizonte 3")
with open(ruta_json, encoding="utf-8") as f:
    viejo = json.load(f)

T = rb.cargar()
tt = pd.read_csv(os.path.join(OUT_TAB, "tabla_resultados_prueba.csv")).set_index("Modelo")
nom = viejo["modelo_seleccionado"]
m = tt.loc[nom]
base = tt.loc["Persistencia (baseline)"]
mm3 = tt.loc["Media móvil 3 (baseline)"]

n_test = int(T["part"]["n_registros"].iloc[-1]) if T["part"] is not None else 0
imp_dim = ({r["Dimension"]: float(r["Ganancia_pct"]) / 100 for _, r in T["imp"].iterrows()}
           if T["imp"] is not None else {})
clasif = rb.clasificar(T, ALPHA)
ab = T["abl_lgbm"].set_index("Dimension") if T["abl_lgbm"] is not None else None

rep = construir_reporte(T=T, nom=nom, m=m, base=base, mm3=mm3, n_final=viejo["n_registros_validos"],
                        n_test=n_test, h_act=rb._horizonte(T), n_splits=viejo.get("n_splits_ablation", 10),
                        imp_dim=imp_dim, shp=None, clasif=clasif, ab=ab)
guardar_reporte(rep, ruta_json)
print("OK ->", ruta_json)
print("HG-a:", rep["hipotesis_general"]["hg_a"]["estado"], "| HG-b:", rep["hipotesis_general"]["hg_b"]["estado"])
for h in rep["hipotesis_especificas"]:
    print(h["codigo"], h["estado"])