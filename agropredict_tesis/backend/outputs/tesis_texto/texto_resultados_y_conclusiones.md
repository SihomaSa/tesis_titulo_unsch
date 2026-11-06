# TEXTO PARA LA TESIS — generado automáticamente con los resultados de tu corrida


**Cómo usarlo:** pega cada sección en el capítulo indicado. Los números ya coinciden con las tablas y figuras de `outputs/`. Reemplaza las tablas y figuras antiguas por las nuevas.

# CAPÍTULO IV — RESULTADOS Y DISCUSIÓN

## 4.1. METODOLOGÍA DE DESARROLLO DEL MODELO PREDICTIVO

El modelo predictivo se desarrolló siguiendo un enfoque basado en CRISP-DM (Chapman et al., 2000), complementado con el proceso KDD (Fayyad et al., 1996) y con las prácticas de Machine Learning descritas por Géron (2023), organizado en nueve fases secuenciales con retroalimentación iterativa (Figura A). Esta organización reemplaza el flujo de elaboración propia presentado en el borrador anterior y garantiza que cada decisión técnica tenga respaldo en la literatura.

**Figura A.** Fases de la metodología de desarrollo del modelo predictivo. *Nota.* Elaboración propia con base en Chapman et al. (2000), Fayyad et al. (1996) y Géron (2023). → `figura_flujo_metodologico_9_fases.png`

**Tabla A.** Fases de la metodología y su sustento científico

| Fase | Descripción | Técnicas / herramientas | Base científica |
|---|---|---|---|
| 1. Comprensión del problema | Definición del problema, variable objetivo y dimensiones X1–X4 | Matriz de consistencia | Chapman et al. (2000); Géron (2023) |
| 2. Comprensión de los datos | Inspección, depuración y análisis exploratorio | pandas; embudo de depuración | Han et al. (2022) |
| 3. Preparación de los datos | Limpieza, variables derivadas, partición cronológica 70/15/15 | pandas; ingeniería de características | Géron (2023); Kuhn y Johnson (2013) |
| 4. Construcción del pipeline | Imputación, escalado y codificación encapsuladas con el modelo | Pipeline de scikit-learn | Pedregosa et al. (2011) |
| 5. Modelado | Random Forest, XGBoost, LightGBM y Regresión Lineal frente a dos baselines | scikit-learn, xgboost, lightgbm | Chen y Guestrin (2016) |
| 6. Ajuste de hiperparámetros | Búsqueda aleatoria con validación cruzada temporal | RandomizedSearchCV + TimeSeriesSplit | Bergstra y Bengio (2012) |
| 7. Evaluación | R², MAE, RMSE y MAPE; contraste de hipótesis (Wilcoxon + Holm) | scikit-learn, SciPy | Sokolova y Lapalme (2009) |
| 8. Interpretabilidad | Valores SHAP y permutación agrupada por dimensión | shap | Lundberg y Lee (2017) |
| 9. Validación y despliegue | Evaluación final en prueba, API REST y aplicación web | FastAPI, Firebase, Railway | Géron (2023) |

*Nota.* Elaboración propia.

### 4.1.1. Fase 1 — Comprensión del problema

El problema es predecir el precio en chacra (S/ por kg) de los cultivos de la región Piura, un problema de regresión supervisada. La variable dependiente es el precio en chacra y las variables independientes se organizan en cuatro dimensiones del entorno agroeconómico: temporal (X1), productiva (X2), geográfica (X3) y económica histórica (X4) (Figura B).

**Figura B.** Variables predictoras por dimensión del entorno agroeconómico. *Nota.* Elaboración propia. → `figura_variables_por_dimension.png`

### 4.1.2. Fase 2 — Comprensión de los datos

Los datos provienen del portal de Datos Abiertos del Gobierno Regional de Piura (Campaña Agrícola de los Principales Cultivos, 2015–2024). El archivo contiene 124 514 registros. Se aplicó un protocolo de depuración documentado paso a paso (Tabla B); la muestra final de modelado quedó en 14 393 registros, correspondientes a 80 cultivos y 7 provincias. El precio en chacra de la muestra oscila entre S/ 0.10 y S/ 18.00 por kg, con media de S/ 1.81.

**Tabla B.** Embudo de depuración de los datos

| Etapa | Registros | Eliminados |
|---|---|---|
| Registros iniciales del dataset | 124 514 | 0 |
| En provincias de la región Piura | 124 514 | 0 |
| Con PRECIO_CHACRA informado (sin imputar el objetivo) | 124 514 | 0 |
| Con PRECIO_CHACRA > 0 | 38 644 | 85 870 |
| Con SIEMBRA, COSECHA y PRODUCCION informadas | 35 293 | 3 351 |
| Sin inconsistencias (COSECHA=0 y PRODUCCION>0) | 14 393 | 20 900 |
| Sin duplicados exactos (muestra final de modelado) | 14 393 | 0 |

*Nota.* Los registros sin precio no se imputaron: un precio inventado no es una observación y sesgaría la variable que se desea predecir. Elaboración propia.

### 4.1.3. Fase 3 — Preparación de los datos

Se corrigió el formato de la variable MES del dataset oficial (AAAAMM) para extraer el mes calendario, y se construyeron las 17 variables de entrada distribuidas en las cuatro dimensiones (Tabla C). Todas las variables de la dimensión económica histórica se calculan exclusivamente con precios de períodos anteriores del mismo cultivo, provincia y distrito, de modo que ninguna variable contiene información del precio que se desea predecir (control de fuga de datos). Los valores faltantes (el primer registro de cada serie carece de historial) se imputan con la mediana calculada únicamente con datos de entrenamiento, dentro del pipeline.

**Tabla C.** Variables predictoras por dimensión

| Dimensión | Nombre | N.º | Variables |
|---|---|---|---|
| X1 | Dimensión Temporal | 5 | ANO, MES_NUM, MES_SEN, MES_COS, ESTACION |
| X2 | Dimensión Productiva | 4 | SIEMBRA, COSECHA, PRODUCCION, RENDIMIENTO_TON_HA |
| X3 | Dimensión Geográfica | 3 | PROVINCIA_enc, DISTRITO_enc, CULTIVO_enc |
| X4 | Dimensión Económica Histórica | 5 | PRECIO_LAG1, MEDIA_MOVIL_3, VARIACION_PRECIO_PREVIA, MESES_DESDE_PRECIO_PREVIO, TIENE_HISTORIAL |

Los datos se ordenaron cronológicamente y se dividieron en tres conjuntos sin mezclar períodos: entrenamiento (10 143 registros, 2015-08 a 2022-05), validación (2 102, hasta 2023-08) y prueba (2 148, hasta 2024-12), aproximadamente 70 %, 15 % y 15 %. El conjunto de prueba no intervino en el ajuste ni en la selección del modelo.

### 4.1.4. Fase 4 — Construcción del pipeline

El preprocesamiento (imputación por mediana, escalado y codificación one-hot para la regresión lineal) y el estimador se encapsularon en un único Pipeline de scikit-learn (Pedregosa et al., 2011). Así, en cada partición de la validación cruzada las transformaciones se ajustan solo con los datos de entrenamiento de esa partición, lo que asegura reproducibilidad y evita fuga de información.

### 4.1.5. Fase 5 — Modelado

Se entrenaron cuatro algoritmos —Random Forest, XGBoost, LightGBM y Regresión Lineal— y se compararon con dos pronósticos ingenuos: la **persistencia** (el precio de hoy es el último precio observado) y la **media móvil de tres observaciones**. Los baselines son la vara mínima: un modelo de Machine Learning solo aporta valor si los supera.

### 4.1.6. Fase 6 — Ajuste de hiperparámetros

Los hiperparámetros (valores de configuración fijados antes del entrenamiento, a diferencia de los parámetros que el modelo aprende de los datos) se optimizaron con búsqueda aleatoria (Bergstra y Bengio, 2012) y validación cruzada temporal (TimeSeriesSplit), usando únicamente el tramo de entrenamiento y el MAE como criterio. La Tabla D muestra la configuración elegida.

**Tabla D.** Hiperparámetros seleccionados

| Modelo | MAE de la validación cruzada temporal | Hiperparámetros seleccionados |
|---|---|---|
| Regresión Lineal | — | — |
| Random Forest | 0.3658 | n_estimators = 100; min_samples_leaf = 2; max_features = 0.5; max_depth = 10 |
| XGBoost | 0.3561 | n_estimators = 600; max_depth = 4; subsample = 0.85; reg_lambda = 5; min_child_weight = 5; learning_rate = 0.02; colsample_bytree = 0.7 |
| LightGBM | 0.3588 | n_estimators = 200; max_depth = 6; subsample = 0.85; reg_lambda = 2; learning_rate = 0.05; colsample_bytree = 0.7; subsample_freq = 1; reg_alpha = 2; num_leaves = 31; min_child_samples = 10 |
| LightGBM sobre persistencia | 0.3272 | n_estimators = 600; max_depth = 6; subsample = 0.7; reg_lambda = 0; learning_rate = 0.02; colsample_bytree = 1; subsample_freq = 1; num_leaves = 63; min_child_samples = 30; objective = regression_l1 |

*Nota.* Elaboración propia. La Regresión Lineal no tiene hiperparámetros relevantes.

## 4.2. RESULTADOS

### 4.2.1. Fase 7 — Evaluación y selección del modelo

La selección del modelo se hizo en el conjunto de validación, con el MAE como criterio. Resultó seleccionado **LightGBM sobre persistencia** (MAE de validación = 0.4190 S/ por kg; la persistencia obtuvo 0.4315).

**Tabla E.** Comparación de modelos en el conjunto de validación

| Modelo | R2 | MAE | RMSE | MAPE | WAPE | R² CV temporal |
|---|---|---|---|---|---|---|
| Persistencia (baseline) | 0.6574 | 0.4315 | 0.9615 | 21.5 | 20.7 | — |
| Media móvil 3 (baseline) | 0.6878 | 0.4736 | 0.9178 | 22.6 | 22.7 | — |
| Regresión Lineal | 0.6896 | 0.5175 | 0.9153 | 27.7 | 24.8 | 0.8199 |
| Random Forest | 0.6822 | 0.4585 | 0.9260 | 23.0 | 22.0 | 0.7710 |
| XGBoost | 0.6900 | 0.4561 | 0.9147 | 23.9 | 21.8 | 0.7974 |
| LightGBM | 0.7022 | 0.4514 | 0.8965 | 23.8 | 21.6 | 0.7976 |
| LightGBM sobre persistencia | 0.7085 | 0.4190 | 0.8869 | 21.5 | 20.1 | 0.7897 |

*Nota.* R², MAE, RMSE (S/ por kg) y MAPE (%). Elaboración propia.

Con el modelo ya elegido y reentrenado con entrenamiento y validación, se evaluó una única vez el conjunto de prueba (n = 2 148, los meses más recientes, nunca vistos). LightGBM sobre persistencia alcanzó R² = 0.7775, MAE = 0.4189 S/ por kg, RMSE = 0.9366 S/ por kg y MAPE = 21.66 %. Este valor no alcanza el criterio de aceptabilidad establecido (R² ≥ 0.85). Frente al pronóstico ingenuo de persistencia (R² = 0.7319; MAE = 0.4392), el modelo reduce el error absoluto medio en 4.6 % (media móvil: R² = 0.7762; MAE = 0.4565).

**Tabla F.** Resultados en el conjunto de prueba (hold-out cronológico)

| Modelo | R2 | MAE | RMSE | MAPE | WAPE | R2_log |
|---|---|---|---|---|---|---|
| Persistencia (baseline) | 0.7319 | 0.4392 | 1.0279 | 22.3 | 19.5 | 0.7291 |
| Media móvil 3 (baseline) | 0.7762 | 0.4565 | 0.9392 | 21.7 | 20.2 | 0.7781 |
| Regresión Lineal | 0.7514 | 0.5308 | 0.9899 | 29.5 | 23.5 | 0.7321 |
| Random Forest | 0.7893 | 0.4583 | 0.9113 | 24.7 | 20.3 | 0.7870 |
| XGBoost | 0.7992 | 0.4679 | 0.8897 | 28.9 | 20.8 | 0.7449 |
| LightGBM | 0.7845 | 0.4844 | 0.9217 | 29.2 | 21.5 | 0.7454 |
| LightGBM sobre persistencia | 0.7775 | 0.4189 | 0.9366 | 21.7 | 18.6 | 0.7929 |

*Nota.* La selección del modelo se realizó en validación; la prueba se usó una sola vez. WAPE = error absoluto total / precio total (%); R²_log = R² sobre el logaritmo del precio, menos sensible a los pocos cultivos de precio alto. Elaboración propia.

**Figura C.** Valores reales frente a predichos en el conjunto de prueba. → `figura_real_vs_predicho.png`

**Figura D.** Comparación de modelos y baselines en el conjunto de prueba. → `figura_comparacion_modelos_prueba.png`

**Figura E.** Evolución del R² por fold temporal. → `figura_r2_por_fold.png`

### 4.2.2. Rendimiento por segmento

**Tabla G.** Rendimiento del modelo final por provincia (conjunto de prueba)

| Provincia | N_test | R2 | MAE |
|---|---|---|---|
| AYABACA | 443 | 0.8525 | 0.4385 |
| SECHURA | 242 | 0.8349 | 0.3952 |
| PIURA | 388 | 0.7652 | 0.4715 |
| HUANCABAMBA | 472 | 0.7636 | 0.3381 |
| MORROPON | 241 | 0.5904 | 0.6022 |
| PAITA | 169 | 0.3859 | 0.3382 |
| SULLANA | 193 | 0.2698 | 0.3368 |

**Tabla H.** Rendimiento según el nivel de precio del cultivo

| Segmento | N_test | Pct_test | R2 | MAE |
|---|---|---|---|---|
| Precio ≤ S/ 4.5/kg | 2 001 | 93.2 | 0.5070 | 0.3213 |
| Precio > S/ 4.5/kg | 147 | 6.8 | 0.1507 | 1.7471 |

El desempeño es heterogéneo: el modelo alcanza R² = 0.5070 en el segmento de precios bajos (93.2 % de la prueba) y R² = 0.1507 en el de precios altos (MAE = 1.7471 S/ por kg). Los cultivos de alto valor son menos frecuentes y más volátiles, lo que se reconoce como limitación (sección 4.4.4).

### 4.2.3. Fase 8 — Interpretabilidad

La importancia por ganancia del modelo asigna X1 18.4 %, X2 23.2 %, X3 18.7 %, X4 39.8 %. El análisis con valores SHAP (Lundberg y Lee, 2017) atribuye X1 22.6 %, X2 6.3 %, X3 11.6 %, X4 59.6 % de la contribución media absoluta a las predicciones. Como la ganancia puede repartirse arbitrariamente entre variables correlacionadas, se verificó con permutación agrupada: al desordenar las variables de X4 el R² cae 1.3210; en las demás dimensiones la caída es X1 0.0000, X2 0.0043, X3 0.0250.

**Figura F.** Importancia por dimensión (ganancia, SHAP y permutación). → `figura_importancia_por_dimension.png`; **Figura G.** Resumen SHAP → `figura_shap_resumen.png`

### 4.2.4. Detección de anomalías

Isolation Forest con contaminación de 5 % identificó 720 registros anómalos (5.00 % de 14 393). El precio promedio fue S/ 1.684 por kg en los registros normales y S/ 4.190 en los anómalos (diferencia de S/ 2.506). Se exportó una muestra aleatoria de 50 anomalías para revisión manual: **la proporción de errores de registro y eventos reales debe volver a calcularla el autor sobre esa muestra** (`muestra_50_anomalias_revision_manual.csv`), pues el conjunto de anomalías cambió respecto de la versión anterior.

**Figura H.** Distribución de precios normales y anómalos. → `figura_anomalias_isolation_forest.png`

## 4.3. CONTRASTACIÓN DE HIPÓTESIS

Para evaluar la influencia de cada dimensión se entrenaron, con los hiperparámetros ya optimizados, variantes del Random Forest que excluyen una dimensión a la vez, y se compararon con el modelo completo en 10 particiones temporales del tramo de desarrollo (entrenamiento y validación; la prueba no se usó). Se aplicó la prueba de rangos con signo de Wilcoxon para muestras pareadas, unilateral (la hipótesis plantea que excluir la dimensión empeora el R²), con corrección de Holm por las cuatro comparaciones (α = 0.05). Con 10 particiones el p-valor mínimo alcanzable es 0.00098, de modo que la prueba tiene poder para detectar efectos consistentes, incluso después de la corrección de Holm.

**Tabla I.** Ablation por dimensión y significancia estadística

| Dimensión excluida | R² completo | R² sin la dimensión | ΔR² | p (Wilcoxon) | p (Holm) | ¿Significativa? |
|---|---|---|---|---|---|---|
| X1_Temporal | 0.7333 | 0.7330 | 0.0003 | 0.461 | 0.922 | No |
| X2_Productiva | 0.7333 | 0.7457 | -0.0124 | 0.903 | 0.922 | No |
| X3_Geografica | 0.7333 | 0.7027 | 0.0307 | <0.001 | 0.004 | Sí |
| X4_Economica_Historica | 0.7333 | 0.6319 | 0.1015 | <0.001 | 0.004 | Sí |

*Nota.* Prueba unilateral; Holm sobre 4 comparaciones. Elaboración propia.

Para interpretar los resultados se añadieron dos análisis complementarios: el poder predictivo de cada dimensión por sí sola y su aporte incremental cuando ya se conoce la historia de precios (X4).

**Tabla J.** Poder predictivo propio y aporte incremental de cada dimensión

| Dimensión | R² solo con la dimensión | R² de X4 + dimensión | Aporte sobre X4 | p (Holm) |
|---|---|---|---|---|
| X1_Temporal | -0.0167 | 0.7239 | -0.0014 | 1.000 |
| X2_Productiva | 0.1318 | 0.7058 | -0.0195 | 1.000 |
| X3_Geografica | 0.5816 | 0.7314 | 0.0061 | 0.196 |
| X4_Economica_Historica | 0.7253 | 0.7253 | — | — |

**Figura I.** Ablation por dimensión. → `figura_ablation_por_dimension.png`

### 4.3.1. Hipótesis general

**NO ACEPTADA.** El modelo que integra las cuatro dimensiones del entorno agroeconómico alcanzó R² = 0.7775 en datos no vistos, inferior al criterio de R² ≥ 0.85, y reduce el error frente al pronóstico ingenuo de persistencia en 4.6 %.

### 4.3.2. Hipótesis específicas

**H1 — Dimensión temporal: NO ACEPTADA** (ΔR² = 0.0003; p = 0.461; p Holm = 0.922). No hay evidencia de que excluir la dimensión empeore de manera significativa el modelo. Por sí sola su capacidad predictiva es limitada (R² = -0.0167) y, con X4 presente, aporta -0.0014 de R² adicional.

**H2 — Dimensión productiva: NO ACEPTADA** (ΔR² = -0.0124; p = 0.903; p Holm = 0.922). No hay evidencia de que excluir la dimensión empeore de manera significativa el modelo. Por sí sola su capacidad predictiva es limitada (R² = 0.1318) y, con X4 presente, aporta -0.0195 de R² adicional.

**H3 — Dimensión geográfica: ACEPTADA** (ΔR² = 0.0307; p = <0.001; p Holm = 0.004). Excluir la dimensión reduce el R² de 0.7333 a 0.7027, de forma consistente en las particiones temporales. Por sí sola alcanza R² = 0.5816.

**H4 — Dimensión económica histórica: ACEPTADA** (ΔR² = 0.1015; p = <0.001; p Holm = 0.004). Excluir la dimensión reduce el R² de 0.7333 a 0.6319, de forma consistente en las particiones temporales. Por sí sola alcanza R² = 0.7253.

*Nota.* La no aceptación de una hipótesis específica indica ausencia de aporte predictivo incremental detectable en este conjunto de datos, no ausencia de relación económica.

## 4.4. DISCUSIÓN (párrafos de apoyo)

Los resultados muestran que la económica histórica concentra la mayor parte de la capacidad predictiva del modelo, lo que concuerda con la fuerte inercia de los precios agrícolas reportada en la literatura. El modelo supera al pronóstico ingenuo de persistencia en MAE (mejora modesta, de 4.6 %, sin prueba de significancia), por lo que el valor agregado de las técnicas de Machine Learning en este contexto es acotado: la mayor parte de la capacidad predictiva proviene de la propia historia de precios. Las comparaciones con estudios previos deben hacerse con cautela, porque las métricas dependen del número de cultivos, del horizonte y del esquema de validación.

### Limitaciones (4.4.4)

- No se incorporaron variables climáticas (SENAMHI, ONI) ni de mercado externas.
- La dimensión geográfica usa codificación ordinal de provincia, distrito y cultivo, sin coordenadas ni embeddings.
- Los precios solo existen cuando hubo cosecha efectiva; las series por cultivo-distrito son irregulares (variable MESES_DESDE_PRECIO_PREVIO).
- El desempeño es menor en cultivos de alto valor y baja frecuencia (Tabla H).
- La validación es retrospectiva; no se evaluó deriva conceptual en producción.

# CAPÍTULO V — CONCLUSIONES

1. **Objetivo general.** Se desarrolló y validó, con una metodología de nueve fases (CRISP-DM + KDD), un modelo de predicción del precio en chacra para la región Piura. El modelo seleccionado en validación fue LightGBM sobre persistencia, que en el conjunto de prueba (n = 2 148) obtuvo R² = 0.7775, MAE = 0.4189 S/ por kg, RMSE = 0.9366 S/ por kg y MAPE = 21.66 %; el criterio R² ≥ 0.85 no se cumple.

2. **Comparación con referencias.** Frente al pronóstico ingenuo de persistencia, LightGBM sobre persistencia reduce el MAE en 4.6 % y mejora el R² (0.7775 frente a 0.7319); frente a la regresión lineal, el R² es 0.7775 contra 0.7514.

3. **Dimensión temporal.** No se encontró aporte incremental significativo (ΔR² = 0.0003; p Holm = 0.922). Por sí sola su capacidad predictiva es nula o muy baja (R² = -0.0167) y, con la historia de precios presente, no mejora el modelo (aporte = -0.0014); en este conjunto de datos la dimensión no aporta información que el precio pasado no contenga, sin que ello implique ausencia de relación económica. 

4. **Dimensión productiva.** No se encontró aporte incremental significativo (ΔR² = -0.0124; p Holm = 0.922). Por sí sola su capacidad predictiva es nula o muy baja (R² = 0.1318) y, con la historia de precios presente, no mejora el modelo (aporte = -0.0195); en este conjunto de datos la dimensión no aporta información que el precio pasado no contenga, sin que ello implique ausencia de relación económica. 

5. **Dimensión geográfica.** Influye de forma estadísticamente significativa: excluirla reduce el R² en 0.0307 (p Holm = 0.004). 

6. **Dimensión económica histórica.** Influye de forma estadísticamente significativa: excluirla reduce el R² en 0.1015 (p Holm = 0.004). 

7. **Interpretabilidad.** La dimensión económica histórica es la de mayor peso (59.6 % de la contribución SHAP).

8. **Anomalías.** Isolation Forest marcó 5.00 % de los registros; los anómalos tienen un precio medio de S/ 4.190 frente a S/ 1.684 de los normales.

9. **Sistema web.** *(Completar con las mediciones propias: tiempo de respuesta, disponibilidad y SUS; no provienen de este pipeline y deben respaldarse con evidencia.)*

# CORRECCIONES QUE DEBES HACER EN OTRAS SECCIONES (no se pegan: se aplican)

- **Resumen/Abstract y 3.4:** la cifra de registros debe ser coherente: 124 514 iniciales y 14 393 de modelado (el borrador dice 116,000 en el resumen).
- **3.7.1 y 3.7.3:** cambia a *10 particiones temporales*, prueba *unilateral* y *corrección de Holm*.
- **3.7.2.2:** elimina «estratificando por cultivo» (la partición es cronológica 70/15/15), el «grid manual» (ahora es búsqueda aleatoria con validación temporal) y las métricas R²=0.892/MAE=0.287; los resultados no van en la metodología.
- **3.7.4:** la conclusión sobre dimensiones contradice el capítulo IV; reemplázala por una referencia a la sección 4.3.
- **Tabla 2/variables X1 y X4:** cambia MES por MES_NUM; VARIACION_PRECIO por VARIACION_PRECIO_PREVIA; añade MESES_DESDE_PRECIO_PREVIO y TIENE_HISTORIAL. X3: CULTIVO se codifica desde COD_CULTIVO (no «target encoding»).
- **Figura 3 (variables predictoras):** sustitúyela por la Figura B (original), y quita el título dentro de la imagen; los títulos van solo como leyenda.
- **Figura 12 (flujo):** sustitúyela por la Figura A (nueve fases).
- **LSTM:** el borrador menciona LSTM (R²=0.0812) pero el código no lo contiene; elimina esas frases o añade el experimento.
- **Tablas 3, 4 y 5:** las tres tienen el mismo título; usa las Tablas D, E y F.
- **3.7.2.3:** intervalos de confianza, Redis y Recharts/Leaflet/Angular: la API ahora devuelve un intervalo empírico del 90 %; Redis y Angular no están implementados en el repositorio — retíralos del texto o impleméntalos.
- **Tabla 10 (SUS, uptime, 287 ms):** no provienen del pipeline; ten a mano la evidencia (encuestas, logs).
