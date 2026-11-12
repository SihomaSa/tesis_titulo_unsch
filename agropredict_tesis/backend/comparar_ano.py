import pandas as pd
A = "outputs/RESPALDO_horizonte_3m/tables/"  # sin ANO (principal)
B = "outputs/horizonte_3m/tables/"           # con ANO (sensibilidad)
M = "LightGBM sobre persistencia"
rd = lambda d, f, i: pd.read_csv(d + f).set_index(i)

filas = []
for nom, d in [("Principal (sin ANO, 16 var.)", A), ("Sensibilidad (con ANO, 17 var.)", B)]:
    r = rd(d, "tabla_resultados_prueba.csv", "Modelo").loc[M]
    s = rd(d, "tabla_significancia_vs_baseline.csv", "Referencia").loc["Persistencia"]
    sh = rd(d, "tabla_significancia_vs_baseline_con_historial.csv", "Referencia").loc["Persistencia"]
    c = rd(d, "tabla_contribucion_dimensiones_prueba_lgbm_res.csv", "Dimension")
    filas.append([nom, r.R2, r.MAE, r.RMSE, s.Reduccion_MAE_pct, s.p_holm, sh.p_holm,
                  c.loc["X1_Temporal", "p_holm"]])
t = pd.DataFrame(filas, columns=["Especificación", "R²", "MAE", "RMSE", "Reducción MAE vs pers. (%)",
                                 "p Holm (todos)", "p Holm (con historial)", "p Holm X1 en modelo final"])
txt = "**Tabla T.** Sensibilidad a la inclusión de ANO (conjunto de prueba)\n\n" + t.round(4).to_string(index=False)

aa, ab = rd(A, "tabla_ablation_wilcoxon_lgbm.csv", "Dimension"), rd(B, "tabla_ablation_wilcoxon_lgbm.csv", "Dimension")
u = pd.DataFrame({"ΔMAE sin ANO": aa.Diferencia_MAE, "p Holm sin ANO": aa.p_mae_holm,
                  "ΔMAE con ANO": ab.Diferencia_MAE, "p Holm con ANO": ab.p_mae_holm}).round(4)
txt += "\n\n**Tabla U.** Ablación en validación temporal con y sin ANO\n\n" + u.to_string()
open("outputs/comparacion_ano.md", "w", encoding="utf-8").write(txt)
print(txt)