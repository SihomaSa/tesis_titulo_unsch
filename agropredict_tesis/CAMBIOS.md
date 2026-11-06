# Qué se corrigió y por qué

## Errores que invalidaban resultados
1. **Fuga de datos:** `VARIACION_PRECIO` se calculaba con el precio actual (el objetivo). Con ella, el modelo reconstruye la respuesta; explicaba el R² de 0.93 y el 82–96 % de importancia de X4. Ahora X4 usa solo períodos anteriores (`VARIACION_PRECIO_PREVIA`).
2. **MES venía como AAAAMM** (p. ej. 202002): `ESTACION` valía 3 en todas las filas y `MES_SEN/MES_COS` eran ruido. La dimensión temporal no tenía información, y de ahí salió el rechazo de H1. Ahora se usa `MES_NUM` (1–12).
3. **Rezago mal definido:** el rezago se calculaba por cultivo-provincia sobre filas de varios distritos, así que `PRECIO_LAG1` podía ser el precio de otro distrito del mismo mes. Ahora la serie es cultivo-provincia-distrito y el rezago viene de un mes estrictamente anterior.
4. **Selección con el conjunto de prueba:** el modelo final se elegía por R² en el hold-out. Ahora: entrenamiento 70 % → validación 15 % (selección) → prueba 15 % (una sola vez).
5. **Imputación de la variable objetivo** con la mediana: eliminada (los registros sin precio se descartan y se reportan).
6. **Frontend:** enviaba `COD_CULTIVO` como si fuera el código del LabelEncoder, y mantenía a mano mapas de codificación. La API ahora recibe nombres y código de cultivo, y el catálogo viaja con el modelo.
7. **API:** leía `random_forest_final.pkl` mientras el pipeline guardaba `modelo_final.pkl`.

## Lo que pedía el jurado
- Metodología de 9 fases (CRISP-DM + KDD) aplicada al pipeline y al texto.
- Hiperparámetros: búsqueda aleatoria con validación temporal (antes eran valores fijos).
- Baselines ingenuos, SHAP y permutación agrupada por dimensión.
- Hipótesis: Wilcoxon unilateral + Holm, poder predictivo propio y aporte incremental, para poder explicar el rechazo en las conclusiones.
- Figuras originales (flujo de 9 fases y variables por dimensión) sin título incrustado.
