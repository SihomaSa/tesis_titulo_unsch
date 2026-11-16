# TEXTO PARA LA TESIS — generado automáticamente con los resultados de tu corrida


**Cómo usarlo:** pega cada sección en el capítulo indicado. Los números ya coinciden con las tablas y figuras de `outputs/`. Reemplaza las tablas y figuras antiguas por las nuevas. Las tablas K–T (particiones, significancia, brecha, contribuciones, sensibilidad, decisión, efecto de ANO) tienen letras provisionales: renumera todas en orden de aparición en tu documento. Donde veas [COMPLETAR] o [VERIFICAR] falta información que el pipeline no tiene (antecedentes, fechas de publicación, fuente del umbral): no la inventes.

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
| 6. Ajuste de hiperparámetros | Búsqueda aleatoria con validación cruzada temporal | RandomizedSearchCV + validación temporal por meses | Bergstra y Bengio (2012) |
| 7. Evaluación | R², MAE, RMSE y MAPE; contraste de hipótesis (Wilcoxon + Holm); pérdidas pareadas con errores HAC | scikit-learn, SciPy | Hyndman y Koehler (2006); Holm (1979); Newey y West (1987) [VERIFICAR cada cita] |
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

**Tabla Q.** Motivos de exclusión de registros y su naturaleza

| Motivo | n | % | Naturaleza |
|---|---|---|---|
| Precio = 0 (sin precio registrado) | 85 870 | 69.0 | Estructural: no hubo precio de chacra registrado (el objetivo no existe) |
| COSECHA=0 y PRODUCCION>0 | 20 900 | 16.8 | Regla de consistencia: posible cosecha de otra campaña/mes (discutible, ver sensibilidad) |
| Retenido | 14 393 | 11.6 | Retenido (muestra de modelado) |
| Faltan SIEMBRA/COSECHA/PRODUCCION | 3 351 | 2.7 | Registro inválido: variables del registro agrícola incompletas |

De los registros iniciales solo se retiene una fracción pequeña. Se distinguen tres naturalezas: (i) *estructural*, cuando no existe precio de chacra (el objetivo no está observado, no es un dato inválido); (ii) *registro inválido*, cuando faltan variables del registro agrícola; y (iii) *regla de consistencia*, cuando COSECHA = 0 y PRODUCCIÓN > 0, que puede reflejar cosecha de otra campaña o mes y no necesariamente un error. Entre los registros con precio y variables completas, la regla de cosecha depende fuertemente del cultivo (V de Cramér = 0.79), por lo que la muestra de modelado no es representativa de todos los cultivos de Piura.

**Tabla R.** Sensibilidad a la regla de depuración (mismo modelo, prueba cronológica de cada escenario)

| Escenario | Hiperparámetros | n | n prueba | R² | MAE modelo | MAE persistencia | Reducción MAE (%) |
|---|---|---|---|---|---|---|---|
| Regla estricta (la del estudio) | ajustados | 14 393 | 2 148 | 0.7820 | 0.4205 | 0.4392 | 4.3 |
| Regla estricta (la del estudio) | por defecto | 14 393 | 2 148 | 0.7479 | 0.4708 | 0.4392 | -7.2 |
| Se conserva COSECHA=0 y PRODUCCION>0 | ajustados | 35 293 | 5 090 | 0.7752 | 0.4032 | 0.4109 | 1.9 |
| Se conserva COSECHA=0 y PRODUCCION>0 | por defecto | 35 293 | 5 090 | 0.7520 | 0.4444 | 0.4109 | -8.2 |

*Nota.* En ambos escenarios el modelo con hiperparámetros ajustados supera a la persistencia en MAE (4.3 %, 1.9 %), de modo que la conclusión cualitativa no depende de la regla de cosecha, aunque el margen cambia. Con hiperparámetros por defecto el modelo no supera a la persistencia (-7.2 %, -8.2 %): la ventaja depende del ajuste. Los hiperparámetros del escenario relajado se tomaron del ajuste con la muestra estricta. Esta sensibilidad se calcula con horizonte de un mes (h = 1) y sin brecha entre conjuntos, por lo que sus valores absolutos no son comparables con los del modelo con horizonte de la Tabla F. Elaboración propia.

**Tabla 6a.** Distribución por provincia antes y después de la depuración

| Categoría | % inicial | % retenido | Dif. (pp) | Tasa de retención (%) |
|---|---|---|---|---|
| SECHURA | 6.0 | 8.9 | 2.8 | 17.0 |
| HUANCABAMBA | 20.8 | 22.8 | 1.9 | 12.6 |
| PAITA | 6.0 | 7.2 | 1.2 | 13.9 |
| AYABACA | 19.9 | 20.4 | 0.5 | 11.9 |
| SULLANA | 9.7 | 8.6 | -1.1 | 10.3 |
| PIURA | 18.7 | 17.4 | -1.3 | 10.7 |
| MORROPON | 18.8 | 14.7 | -4.1 | 9.0 |

**Tabla 6b.** Distribución por año antes y después de la depuración

| Categoría | % inicial | % retenido | Dif. (pp) | Tasa de retención (%) |
|---|---|---|---|---|
| 2015 | 4.2 | 6.6 | 2.3 | 17.9 |
| 2016 | 10.2 | 11.5 | 1.3 | 13.0 |
| 2017 | 10.1 | 10.4 | 0.3 | 11.9 |
| 2018 | 10.2 | 10.1 | -0.1 | 11.5 |
| 2019 | 10.6 | 9.6 | -1.0 | 10.5 |
| 2020 | 10.7 | 10.0 | -0.7 | 10.8 |
| 2021 | 11.0 | 10.3 | -0.7 | 10.8 |
| 2022 | 11.2 | 10.7 | -0.5 | 11.0 |
| 2023 | 10.9 | 11.6 | 0.6 | 12.2 |
| 2024 | 10.8 | 9.3 | -1.5 | 9.9 |

**Tabla 6c.** Cultivos que más pierden y que más ganan peso tras la depuración (10 y 10)

| Grupo | Categoría | % inicial | % retenido | Dif. (pp) | Tasa de retención (%) |
|---|---|---|---|---|---|
| Pierden peso | PACAE O GUABO | 2.5 | 0.6 | -1.9 | 2.8 |
| Pierden peso | PALTO | 2.9 | 1.2 | -1.7 | 4.8 |
| Pierden peso | NARANJO | 2.9 | 1.3 | -1.7 | 5.0 |
| Pierden peso | PLATANOS Y BANANAS PARA COCCIÓN (M X paradisiaca) | 4.5 | 3.0 | -1.5 | 7.7 |
| Pierden peso | LIMON SUTIL | 3.8 | 2.4 | -1.5 | 7.2 |
| Pierden peso | CIRUELO (CIRUELA SPONDIAS) | 2.4 | 1.0 | -1.4 | 4.7 |
| Pierden peso | MANGO | 3.3 | 2.0 | -1.4 | 6.8 |
| Pierden peso | PASTO ELEFANTE | 3.2 | 1.8 | -1.4 | 6.6 |
| Pierden peso | TUNA | 1.2 | 0.0 | -1.2 | 0.0 |
| Pierden peso | OTROS PASTOS | 2.0 | 1.0 | -1.1 | 5.5 |
| Ganan peso | MAIZ AMARILLO DURO | 4.0 | 11.9 | 7.9 | 34.5 |
| Ganan peso | ARROZ | 3.2 | 8.0 | 4.8 | 28.6 |
| Ganan peso | PAPA (agrupa mejoradas y nativas) | 1.4 | 4.7 | 3.3 | 38.5 |
| Ganan peso | YUCA | 3.2 | 6.4 | 3.2 | 23.1 |
| Ganan peso | FRIJOL CAUPI CHICLAYO CASTILLA CHILENO GRANO SECO | 1.6 | 3.8 | 2.1 | 26.8 |
| Ganan peso | SANDIA | 1.4 | 3.1 | 1.7 | 26.0 |
| Ganan peso | ARVEJA GRANO SECO | 1.2 | 2.6 | 1.4 | 25.5 |
| Ganan peso | MAIZ AMILACEO | 1.6 | 3.0 | 1.4 | 21.9 |
| Ganan peso | TRIGO BLANDO | 1.1 | 2.3 | 1.2 | 24.0 |
| Ganan peso | CEBOLLA | 1.0 | 2.1 | 1.1 | 24.3 |

*Nota.* % inicial = participación entre todos los registros de Piura; % retenido = participación en la muestra de modelado; Dif. (pp) = retenido − inicial; tasa de retención = registros retenidos / registros iniciales de la categoría. Elaboración propia.

Para examinar cómo la depuración modificó la composición de los datos, se compararon las distribuciones de provincias, años y cultivos antes y después de la exclusión (Tablas 6a a 6c). La participación de las provincias cambió entre -4.1 y 2.8 puntos porcentuales, y la de los años entre -1.5 y 2.3. En cultivos, la depuración sobrerrepresentó a maiz amarillo duro, arroz y papa (agrupa mejoradas y nativas) y subrepresentó a pacae o guabo, palto y naranjo; la asociación entre la composición y la condición de retenido resultó significativa en cultivo (χ² = 11924.7; p < 0.001; V de Cramér = 0.31), provincia (χ² = 469.4; p < 0.001; V = 0.06) y año (χ² = 308.9; p < 0.001; V = 0.05). Al aislar solo la regla de cosecha, la dependencia es fuerte con el cultivo (V = 0.79), moderada con el año (V = 0.25) y débil con la provincia (V = 0.14). En consecuencia, la muestra de modelado no reproduce la composición del registro inicial: los resultados son válidos para los registros con cosecha efectiva y precio informado, y no deben generalizarse a todos los cultivos ni a todos los períodos de la región.

[Sección 5 — frase para la respuesta a la Obs. 10] Además, se agregaron las Tablas 6a–6c, que comparan la distribución de provincias, años y cultivos antes y después de la depuración (provincias: -4.1 a 2.8 pp; años: -1.5 a 2.3 pp).

### 4.1.3. Fase 3 — Preparación de los datos

Se corrigió el formato de la variable MES del dataset oficial (AAAAMM) para extraer el mes calendario, y se construyeron las 17 variables de entrada distribuidas en las cuatro dimensiones (Tabla C). Todas las variables de la dimensión económica histórica se calculan exclusivamente con precios de períodos anteriores del mismo cultivo, provincia y distrito, de modo que ninguna variable contiene información del precio que se desea predecir (control de fuga de datos). Los valores faltantes (el primer registro de cada serie carece de historial) se imputan con la mediana calculada únicamente con datos de entrenamiento, dentro del pipeline.

**Tabla C.** Variables predictoras por dimensión

| Dimensión | Nombre | N.º | Variables |
|---|---|---|---|
| X1 | Dimensión Temporal | 5 | ANO, MES_NUM, MES_SEN, MES_COS, ESTACION |
| X2 | Dimensión Productiva | 4 | SIEMBRA, COSECHA, PRODUCCION, RENDIMIENTO_TON_HA |
| X3 | Dimensión Espacial y de producto | 3 | PROVINCIA_enc, DISTRITO_enc, CULTIVO_enc |
| X3a | ↳ Territorio (subconjunto de X3) | 2 | PROVINCIA_enc, DISTRITO_enc |
| X3b | ↳ Producto (subconjunto de X3) | 1 | CULTIVO_enc |
| X4 | Dimensión Económica Histórica | 5 | PRECIO_LAG1, MEDIA_MOVIL_3, VARIACION_PRECIO_PREVIA, MESES_DESDE_PRECIO_PREVIO, TIENE_HISTORIAL |

*Nota.* Total: 17 variables (X3a y X3b son subconjuntos de X3 y no se suman aparte). X3a: provincia y distrito, codificados de forma ordinal; X3b: cultivo, codificado desde COD_CULTIVO, nivel nominal. UBIGEO solo se usó para verificar la pertenencia a Piura (criterio de inclusión), no como predictor. Con horizonte h = 3, las variables de X2 corresponden a la observación previa disponible en el origen del pronóstico. ANO se conserva conforme a la matriz de consistencia; su efecto se evalúa en la sección 4.3.5 (Tabla T). Elaboración propia.

Los datos se ordenaron cronológicamente y se dividieron en tres conjuntos sin mezclar períodos: entrenamiento (10 002 registros, 2015-08 a 2022-05), validación (2 102, hasta 2023-08) y prueba (2 148, hasta 2024-12), aproximadamente 70 %, 15 % y 15 %. El conjunto de prueba no intervino en el ajuste ni en la selección del modelo.

**Tabla K.** Particiones cronológicas, tamaño y disponibilidad de rezagos

| Conjunto | Desde | Hasta | n | % total | Meses | Series | % con historial previo | Brecha mediana (meses) |
|---|---|---|---|---|---|---|---|---|
| Entrenamiento | 2015-08 | 2022-03 | 10 002 | 69.5 | 80 | 1169 | 85.5 | 10 |
| Validación | 2022-06 | 2023-08 | 2 102 | 14.6 | 15 | 889 | 97.2 | 10 |
| Prueba | 2023-09 | 2024-12 | 2 148 | 14.9 | 16 | 830 | 97.7 | 7 |

*Nota.* La partición es cronológica y respeta límites de mes. Serie = combinación cultivo–provincia–distrito. Sí: pronóstico a h=3 mes(es). Último precio, media móvil, variación y variables productivas usan solo observaciones con mes <= (mes objetivo − 3), también durante la prueba; no se usan predicciones previas. Elaboración propia.

**Definición del horizonte de predicción.** El modelo pronostica el precio en chacra del mes T con un horizonte fijo de h = 3 mes(es) calendario. Todas las variables que dependen de la serie (último precio, media móvil, variación y variables productivas) usan solo observaciones con mes ≤ T − 3. Solo existe objetivo cuando hubo cosecha con precio informado en el mes T; el pronóstico es, por tanto, condicional a esa ocurrencia, y los meses sin cosecha no se imputan. En prueba, la distancia mediana entre el último precio y el mes objetivo es de 7 meses (≥ 3 por construcción). Entre entrenamiento y validación/prueba se dejó una brecha de 2 mes(es), de modo que ninguna etiqueta de entrenamiento es posterior al origen del pronóstico.

El detalle de los folds de validación temporal (fechas y tamaños de entrenamiento y validación del ajuste de hiperparámetros y de la ablación) se presenta en el Anexo [COMPLETAR: Anexo] (`tabla_folds.csv`).

### 4.1.4. Fase 4 — Construcción del pipeline

El preprocesamiento (imputación por mediana, escalado y codificación one-hot para la regresión lineal) y el estimador se encapsularon en un único Pipeline de scikit-learn (Pedregosa et al., 2011). Así, en cada partición de la validación cruzada las transformaciones se ajustan solo con los datos de entrenamiento de esa partición, lo que asegura reproducibilidad y evita fuga de información.

### 4.1.5. Fase 5 — Modelado

Se entrenaron cuatro algoritmos —Random Forest, XGBoost, LightGBM y Regresión Lineal— y se compararon con dos pronósticos ingenuos: la **persistencia** (el precio de hoy es el último precio observado) y la **media móvil de tres observaciones**. Los baselines son la vara mínima: un modelo de Machine Learning solo aporta valor si los supera.

### 4.1.6. Fase 6 — Ajuste de hiperparámetros

Los hiperparámetros (valores de configuración fijados antes del entrenamiento, a diferencia de los parámetros que el modelo aprende de los datos) se optimizaron con búsqueda aleatoria (Bergstra y Bengio, 2012) y validación cruzada temporal con folds expansivos por meses completos y brecha entre entrenamiento y validación, usando únicamente el tramo de entrenamiento y el MAE como criterio. La Tabla D muestra la configuración elegida.

**Tabla D.** Hiperparámetros seleccionados

| Modelo | MAE de la validación cruzada temporal | Hiperparámetros seleccionados |
|---|---|---|
| Regresión Lineal | — | — |
| Random Forest | 0.4044 | n_estimators = 200; min_samples_leaf = 2; max_features = 1; max_depth = 14 |
| XGBoost | 0.3916 | n_estimators = 200; max_depth = 6; subsample = 0.7; reg_lambda = 5; min_child_weight = 10; learning_rate = 0.05; colsample_bytree = 0.85 |
| LightGBM | 0.3919 | n_estimators = 200; max_depth = 6; subsample = 0.85; reg_lambda = 2; learning_rate = 0.05; colsample_bytree = 0.7; subsample_freq = 1; reg_alpha = 2; num_leaves = 31; min_child_samples = 10 |
| LightGBM sobre persistencia | 0.3749 | n_estimators = 200; max_depth = 4; subsample = 1; reg_lambda = 2; learning_rate = 0.1; colsample_bytree = 1; subsample_freq = 1; num_leaves = 31; min_child_samples = 30; objective = regression_l1 |

*Nota.* Elaboración propia. La Regresión Lineal no tiene hiperparámetros relevantes.

## 4.2. RESULTADOS

### 4.2.1. Fase 7 — Evaluación y selección del modelo

La selección del modelo se hizo en el conjunto de validación, con el MAE como criterio. Resultó seleccionado **LightGBM sobre persistencia** (MAE de validación = 0.4680 S/ por kg; la persistencia obtuvo 0.4923).

**Tabla E.** Comparación de modelos en el conjunto de validación

| Modelo | R2 | MAE | RMSE | MAPE | WAPE | R² CV temporal |
|---|---|---|---|---|---|---|
| Persistencia (baseline) | 0.6374 | 0.4923 | 0.9892 | 24.8 | 23.6 | — |
| Media móvil 3 (baseline) | 0.6626 | 0.5117 | 0.9542 | 24.4 | 24.5 | — |
| Regresión Lineal | 0.6734 | 0.5416 | 0.9388 | 29.1 | 25.9 | 0.7775 |
| Random Forest | 0.6603 | 0.5052 | 0.9575 | 25.5 | 24.2 | 0.7464 |
| XGBoost | 0.6677 | 0.4932 | 0.9469 | 26.0 | 23.6 | 0.7571 |
| LightGBM | 0.6802 | 0.4913 | 0.9289 | 25.5 | 23.5 | 0.7556 |
| LightGBM sobre persistencia | 0.6949 | 0.4680 | 0.9073 | 23.9 | 22.4 | 0.7671 |

*Nota.* R², MAE, RMSE (S/ por kg) y MAPE (%). Elaboración propia.

Con el modelo ya elegido y reentrenado con entrenamiento y validación, se evaluó una única vez el conjunto de prueba (n = 2 148, los meses más recientes, nunca vistos). LightGBM sobre persistencia alcanzó R² = 0.7360, MAE = 0.4936 S/ por kg, RMSE = 1.0200 S/ por kg y MAPE = 25.09 %. Este valor no alcanza el criterio de aceptabilidad establecido (R² ≥ 0.85). Frente al pronóstico ingenuo de persistencia (R² = 0.6862; MAE = 0.5276), el modelo reduce el error absoluto medio en 6.5 % (media móvil: R² = 0.7397; MAE = 0.5071).

**Tabla F.** Resultados en el conjunto de prueba (hold-out cronológico)

| Modelo | R2 | MAE | RMSE | MAPE | WAPE | R2_log |
|---|---|---|---|---|---|---|
| Persistencia (baseline) | 0.6862 | 0.5276 | 1.1121 | 26.4 | 23.4 | 0.6838 |
| Media móvil 3 (baseline) | 0.7397 | 0.5071 | 1.0129 | 24.0 | 22.5 | 0.7472 |
| Regresión Lineal | 0.7162 | 0.5639 | 1.0577 | 30.3 | 25.0 | 0.7152 |
| Random Forest | 0.6370 | 0.6027 | 1.1961 | 30.5 | 26.7 | 0.7141 |
| XGBoost | 0.6845 | 0.5596 | 1.1151 | 28.5 | 24.8 | 0.7433 |
| LightGBM | 0.6652 | 0.5681 | 1.1487 | 28.5 | 25.2 | 0.7414 |
| LightGBM sobre persistencia | 0.7360 | 0.4936 | 1.0200 | 25.1 | 21.9 | 0.7590 |

*Nota.* La selección del modelo se realizó en validación; la prueba se usó una sola vez. WAPE = error absoluto total / precio total (%); R²_log = R² sobre el logaritmo del precio, menos sensible a los pocos cultivos de precio alto. Elaboración propia.

**Figura C.** Valores reales frente a predichos en el conjunto de prueba. → `figura_real_vs_predicho.png`

**Figura D.** Comparación de modelos y baselines en el conjunto de prueba. → `figura_comparacion_modelos_prueba.png`

**Figura E.** Evolución del R² por fold temporal. → `figura_r2_por_fold.png`

### 4.2.2. Rendimiento por segmento

**Tabla G.** Rendimiento del modelo final por provincia (conjunto de prueba)

| Provincia | N_test | R2 | MAE |
|---|---|---|---|
| AYABACA | 443 | 0.8297 | 0.4928 |
| SECHURA | 242 | 0.7965 | 0.4989 |
| PIURA | 388 | 0.7677 | 0.5389 |
| HUANCABAMBA | 472 | 0.6195 | 0.4427 |
| MORROPON | 241 | 0.5355 | 0.6754 |
| PAITA | 169 | 0.2348 | 0.3880 |
| SULLANA | 193 | 0.1942 | 0.3873 |

**Tabla H.** Rendimiento según el nivel de precio del cultivo

| Segmento | N_test | Pct_test | R2 | MAE |
|---|---|---|---|---|
| Precio ≤ S/ 4.5/kg | 2 001 | 93.2 | 0.5354 | 0.3747 |
| Precio > S/ 4.5/kg | 147 | 6.8 | -0.1646 | 2.1116 |

El desempeño es heterogéneo: el modelo alcanza R² = 0.5354 en el segmento de precios bajos (93.2 % de la prueba) y R² = -0.1646 en el de precios altos (MAE = 2.1116 S/ por kg). Los cultivos de alto valor son menos frecuentes y más volátiles, lo que se reconoce como limitación (sección 4.4.4).

### 4.2.3. Fase 8 — Interpretabilidad

La importancia por ganancia del modelo asigna X1 19.5 %, X2 16.5 %, X3 18.4 %, X4 45.6 %. El análisis con valores SHAP (Lundberg y Lee, 2017) atribuye X1 22.3 %, X2 8.6 %, X3 10.5 %, X4 58.6 % de la contribución media absoluta a las predicciones. Como la ganancia puede repartirse arbitrariamente entre variables correlacionadas, se verificó con permutación agrupada: al desordenar las variables de X4 el R² cae 0.1536; en las demás dimensiones la caída es X1 0.0012, X2 0.0071, X3 0.0353.

**Figura F.** Importancia por dimensión (ganancia, SHAP y permutación). → `figura_importancia_por_dimension.png`; **Figura G.** Resumen SHAP → `figura_shap_resumen.png`

### 4.2.4. Detección de anomalías

Isolation Forest con contaminación de 5 % identificó 720 registros anómalos (5.00 % de 14 393). El precio promedio fue S/ 1.708 por kg en los registros normales y S/ 3.736 en los anómalos (diferencia de S/ 2.028). Se exportó una muestra aleatoria de 50 anomalías para revisión manual: **la proporción de errores de registro y eventos reales debe volver a calcularla el autor sobre esa muestra** (`muestra_50_anomalias_revision_manual.csv`), pues el conjunto de anomalías cambió respecto de la versión anterior.

**Figura H.** Distribución de precios normales y anómalos. → `figura_anomalias_isolation_forest.png`

### 4.2.5. Significancia de la mejora frente a los baselines

**Eficacia frente al umbral.** El R² de prueba (0.7360) tiene un intervalo de confianza del 95 % de [0.6470; 0.7739] (bootstrap por bloques de 3 meses sobre 16 meses de prueba). Decisión frente al umbral de 0.85: se rechaza (IC superior < umbral).

**Tabla L.** Reducción del MAE frente a baselines y otros modelos (diferencias de pérdida pareadas por mes)

| Referencia | Muestra | MAE ref. | MAE modelo | Reducción (%) | IC95 inf. | IC95 sup. | p (Holm) |
|---|---|---|---|---|---|---|---|
| Persistencia | Todos los registros | 0.5276 | 0.4936 | 6.5 | 0.0170 | 0.0568 | 0.013 |
| Persistencia | Solo con precio previo real | 0.5107 | 0.4888 | 4.3 | 0.0121 | 0.0372 | 0.012 |
| Media móvil 3 | Todos los registros | 0.5071 | 0.4936 | 2.7 | -0.0031 | 0.0287 | 0.071 |
| Media móvil 3 | Solo con precio previo real | 0.4897 | 0.4888 | 0.2 | -0.0121 | 0.0111 | 0.316 |
| LightGBM directo | Todos los registros | 0.5681 | 0.4936 | 13.1 | 0.0143 | 0.1295 | 0.071 |
| LightGBM directo | Solo con precio previo real | 0.5643 | 0.4888 | 13.4 | 0.0122 | 0.1297 | 0.075 |
| Random Forest | Todos los registros | 0.6027 | 0.4936 | 18.1 | 0.0473 | 0.1584 | 0.018 |
| Random Forest | Solo con precio previo real | 0.5948 | 0.4888 | 17.8 | 0.0460 | 0.1581 | 0.023 |
| Regresión Lineal | Todos los registros | 0.5639 | 0.4936 | 12.5 | 0.0273 | 0.1160 | 0.032 |
| Regresión Lineal | Solo con precio previo real | 0.5560 | 0.4888 | 12.1 | 0.0240 | 0.1151 | 0.045 |
| XGBoost | Todos los registros | 0.5596 | 0.4936 | 11.8 | 0.0216 | 0.1107 | 0.056 |
| XGBoost | Solo con precio previo real | 0.5543 | 0.4888 | 11.8 | 0.0203 | 0.1110 | 0.062 |

*Nota.* IC95 de la diferencia media mensual de pérdida absoluta (referencia − modelo; > 0 favorece al modelo), por bootstrap de bloques móviles. Contraste unilateral con error estándar HAC (Newey–West), corrección de Holm por las comparaciones de cada muestra. Con pocos meses de prueba el poder es limitado. «LightGBM directo» es el LightGBM sin ancla de persistencia; el modelo evaluado es LightGBM sobre persistencia. Elaboración propia.

Frente a la persistencia, el LightGBM sobre persistencia reduce el MAE en 6.5 % (IC95 de la diferencia mensual [0.017; 0.057]; p Holm = 0.013). Al excluir los registros sin precio previo —para los cuales la «persistencia» es solo un valor de respaldo (mediana) y no un pronóstico comparable— la reducción es de 4.3 % y sigue siendo significativa (IC95 [0.012; 0.037]; p Holm = 0.012). Con pérdida cuadrática (que penaliza más los errores grandes) el IC95 de la diferencia frente a la persistencia excluye 0 en 9 de 9 configuraciones de rezagos y bloques (con pérdida absoluta, en 9 de 9). Frente a Regresión Lineal, la ventaja se observa con pérdida absoluta pero no con pérdida cuadrática; el modelo se seleccionó por MAE y no es superior en términos de errores grandes.

### 4.2.6. Error según la distancia al último precio observado

**Tabla M.** Desempeño en prueba según los meses transcurridos desde el último precio de la serie

| Grupo | N | % de la prueba | MAE modelo | MAE persistencia | Reducción (%) | R² modelo |
|---|---|---|---|---|---|---|
| Sin historial previo | 49 | 2.3 | 0.6992 | — | — | 0.4269 |
| Último precio en el mes T-h (distancia mínima) (distancia 3 m) | 492 | 22.9 | 0.2530 | 0.2365 | -7.0 | 0.8854 |
| Último precio con 1–3 m de antigüedad (distancia 4–6 m) | 513 | 23.9 | 0.3121 | 0.3352 | 6.9 | 0.8899 |
| Con 4–12 m de antigüedad (distancia 7–15 m) | 943 | 43.9 | 0.6544 | 0.6952 | 5.9 | 0.7135 |
| Con más de 12 m de antigüedad (distancia > 15 m) | 151 | 7.0 | 0.8231 | 0.8489 | 3.0 | 0.5332 |

*Nota.* En «sin historial previo» la persistencia usa la mediana de desarrollo y no es comparable. Elaboración propia.

El modelo no supera a la persistencia cuando el último precio es reciente (Último precio en el origen (distancia 3 m): -7.0 %) y su ventaja aparece cuando la brecha es mayor (Último precio con 1–3 m de antigüedad (distancia 4–6 m): +6.9 %; Con 4–12 m de antigüedad (distancia 7–15 m): +5.9 %; Con más de 12 m de antigüedad (distancia > 15 m): +3.0 %). La mejora global, por tanto, proviene sobre todo de series con precios previos antiguos, donde la persistencia es menos informativa.

## 4.3. CONTRASTACIÓN DE HIPÓTESIS

Para evaluar la contribución predictiva de cada dimensión se entrenaron, con los hiperparámetros ya optimizados, variantes de LightGBM (misma familia que el modelo final) que excluyen una dimensión 
a la vez, y se compararon con el modelo completo en 10 particiones temporales del tramo de desarrollo (entrenamiento y validación; la prueba no se usó en este contraste). 
Se aplicó la prueba de rangos con signo de Wilcoxon para muestras pareadas, unilateral, con corrección de Holm (α = 0.05). Con 10 particiones el p-valor mínimo alcanzable es 0.00098. 
Los resultados se replicaron en el conjunto de prueba y sobre el modelo final con pérdidas pareadas por mes, y con Random Forest como análisis de robustez (`tabla_ablation_wilcoxon.csv`).

### 4.3.1. Formulación operativa de las hipótesis

Las hipótesis se contrastan como *contribución predictiva* y no como efecto causal, porque el estudio es observacional: que una dimensión mejore el pronóstico fuera de muestra no demuestra que determine el precio.

- **HG-a (eficacia):** el modelo seleccionado alcanza R² ≥ 0.85 en el conjunto de prueba. Se contrasta con el R² puntual y su intervalo de confianza por bootstrap de bloques de meses.
- **HG-b (superación de baselines):** el modelo reduce el error de pronóstico frente a la persistencia y la media móvil. Se contrasta con diferencias de pérdida pareadas por mes (errores HAC, IC por bloques, Holm).
- **H1–H4 (contribución por dimensión):** excluir la dimensión empeora el pronóstico fuera de muestra. Se contrasta en tres niveles: (i) validación temporal con LightGBM directo (ΔMAE y ΔR², Wilcoxon unilateral, Holm); (ii) conjunto de prueba con LightGBM directo (pérdidas pareadas por mes); (iii) conjunto de prueba sobre el modelo final (LightGBM sobre persistencia), donde X4 no puede retirarse porque es su ancla.

Criterio de reporte: una hipótesis se declara *respaldada* si resulta significativa (p Holm < 0.05) en todos los contrastes aplicables, *parcialmente respaldada* si lo es solo en alguno, y *no respaldada* si en ninguno. [COMPLETAR en 3.7.1: este criterio de consistencia debe figurar en la metodología; indica con honestidad que se adoptó al revisar las observaciones.]

### 4.3.2. Resultados de los contrastes

**Tabla I.** Ablación por dimensión en validación temporal (LightGBM directo)

| Dimensión excluida | R² completo | R² sin dim. | ΔR² | p Holm (R²) | MAE completo | MAE sin dim. | ΔMAE | p Holm (MAE) |
|---|---|---|---|---|---|---|---|---|
| X1_Temporal | 0.7385 | 0.7347 | 0.0038 | 0.322 | 0.4020 | 0.4024 | 0.0005 | 0.922 |
| X2_Productiva | 0.7385 | 0.7414 | -0.0029 | 0.920 | 0.4020 | 0.4031 | 0.0011 | 0.922 |
| X3_Espacial_Producto | 0.7385 | 0.6592 | 0.0793 | 0.004 | 0.4020 | 0.4650 | 0.0630 | 0.004 |
| X4_Economica_Historica | 0.7385 | 0.7206 | 0.0179 | 0.196 | 0.4020 | 0.4425 | 0.0405 | 0.004 |

*Nota.* Wilcoxon pareado unilateral sobre los folds temporales; ΔMAE = MAE sin la dimensión − MAE completo (> 0 favorece incluir la dimensión). Los folds son expansivos por meses completos con brecha de 2 mes(es); comparten datos de entrenamiento y son pocos, por lo que el Wilcoxon tiene poder limitado. Es un contraste exploratorio, no confirmatorio. Elaboración propia.

**Tabla O.** Ablación con la dimensión geográfica separada en territorio y producto (Holm sobre cinco pruebas)

| Dimensión excluida | ΔR² | p Holm (R²) | ΔMAE | p Holm (MAE) |
|---|---|---|---|---|
| X1_Temporal | 0.0038 | 0.322 | 0.0005 | 0.922 |
| X2_Productiva | -0.0029 | 0.920 | 0.0011 | 0.922 |
| X3a_Territorio | 0.0063 | 0.027 | 0.0060 | 0.005 |
| X3b_Producto | 0.0808 | 0.005 | 0.0650 | 0.005 |
| X4_Economica_Historica | 0.0179 | 0.196 | 0.0405 | 0.005 |

*Nota.* El cultivo identifica el producto, no el territorio; por eso se evalúa por separado de provincia y distrito. Elaboración propia.

**Tabla N.** Contribución de cada dimensión en el conjunto de prueba (pérdidas pareadas por mes)

| Modelo | Dimensión excluida | Dif. MAE mensual | IC95 inf. | IC95 sup. | p (Holm) |
|---|---|---|---|---|---|
| LightGBM directo | X1_Temporal | -0.0057 | -0.0160 | 0.0060 | 1.000 |
| LightGBM directo | X2_Productiva | -0.0073 | -0.0249 | 0.0148 | 1.000 |
| LightGBM directo | X3a_Territorio | 0.0099 | 0.0003 | 0.0219 | 0.182 |
| LightGBM directo | X3b_Producto | 0.0111 | -0.0039 | 0.0302 | 0.335 |
| LightGBM directo | X4_Economica_Historica | 0.0371 | 0.0187 | 0.0528 | <0.001 |
| Modelo final (LightGBM sobre persistencia) | X1_Temporal | 0.0186 | 0.0061 | 0.0306 | 0.016 |
| Modelo final (LightGBM sobre persistencia) | X2_Productiva | -0.0010 | -0.0080 | 0.0059 | 0.904 |
| Modelo final (LightGBM sobre persistencia) | X3a_Territorio | 0.0003 | -0.0037 | 0.0047 | 0.904 |
| Modelo final (LightGBM sobre persistencia) | X3b_Producto | 0.0078 | 0.0005 | 0.0196 | 0.192 |

*Nota.* Diferencia = MAE sin la dimensión − MAE completo, promediada por mes; > 0 indica que la dimensión ayuda. En el modelo final X4 no se retira porque es la base de la persistencia. Elaboración propia.

**Inestabilidad de la dimensión geográfica.** Las contribuciones de territorio y producto no coinciden entre la validación temporal y la prueba (Tablas N y O). El efecto del producto es mayor en los folds con menos registros con historial (correlación de Spearman entre ΔMAE y % con historial = -0.63; p = 0.050), lo que sugiere que el cultivo actúa como sustituto del nivel de precio cuando falta el precio previo y se vuelve redundante cuando existe. Es una interpretación compatible con los datos, no una prueba causal.

**Tabla P.** Variante con la dimensión productiva rezagada (modelo final, conjunto de prueba)

| Escenario | Variables | R² | MAE | Dif. MAE vs original | IC95 inf. | IC95 sup. | p (HAC) |
|---|---|---|---|---|---|---|---|
| Original (X2 según horizonte activo) | 17 | 0.7360 | 0.4936 | — | — | — | — |
| X2 rezagada (observación previa) | 17 | 0.7347 | 0.4931 | -0.0005 | -0.0020 | 0.0010 | 0.791 |
| Sin X2 | 13 | 0.7331 | 0.4952 | 0.0016 | -0.0080 | 0.0059 | 0.617 |

*Nota.* La variante X2 rezagada utiliza las variables de siembra, cosecha, producción y rendimiento correspondientes a la observación previa de la misma serie, por lo que representa un escenario con información productiva histórica disponible al momento de emitir la predicción. La variante sin X2 permite evaluar el desempeño sin incorporar dichas variables. Las diferencias observadas frente al escenario original X2 rezagada: ΔMAE = -0.0005, p = 0.791; sin X2: ΔMAE = 0.0016, p = 0.617; sin embargo, estos resultados no demuestran equivalencia entre escenarios ni garantizan la disponibilidad operativa de las variables contemporáneas. Elaboración propia.

### 4.3.3. Decisión sobre las hipótesis

**HG-a (eficacia): NO RESPALDADA.** R² = 0.7360 (IC95 [0.6470; 0.7739]) frente al umbral de 0.85. No alcanzar el umbral no invalida el estudio: indica que el modelo no cumple el criterio de eficacia fijado para este conjunto de datos.

**HG-b (superación de baselines): PARCIALMENTE RESPALDADA.** Frente a la media móvil la reducción es de 2.7 % (p Holm = 0.071). Frente a la persistencia es de 6.5 % (p Holm = 0.013) con todos los registros.

**Tabla S.** Consistencia de los contrastes por dimensión

| Hipótesis | Significativa en validación temporal | Significativa en prueba (LightGBM directo) | Significativa en prueba (modelo final) | Estado |
|---|---|---|---|---|
| H1 — Dimensión temporal | No | No | Sí | PARCIALMENTE RESPALDADA |
| H2 — Dimensión productiva | No | No | No | NO RESPALDADA |
| H3 — Dimensión espacial y de producto | Sí | Sí | No | PARCIALMENTE RESPALDADA |
| H4 — Dimensión económica histórica | Sí | Sí | — | RESPALDADA |

*Nota.* «—» = no aplicable (X4 es la base del modelo final). Significancia: p Holm < 0.05. La no aceptación de una hipótesis indica ausencia de aporte predictivo incremental detectable en este conjunto de datos, no ausencia de relación económica. Elaboración propia.

**H1 — Dimensión temporal: PARCIALMENTE RESPALDADA.** Aporte no significativo en validación temporal, no significativo en prueba (LightGBM directo) y significativo sobre el modelo final. Incluye el año calendario (ANO), cuyo efecto con hiperparámetros fijos se detalla en la sección 4.3.5 (Tabla T); su contribución es inestable entre contrastes y no es evidencia de causalidad.

**H2 — Dimensión productiva: NO RESPALDADA.** Aporte no significativo en validación temporal, no significativo en prueba (LightGBM directo) y no significativo sobre el modelo final. La X2 del modelo principal está rezagada (disponible en el origen); la versión contemporánea se reporta como cota superior en la Tabla P.

**H3 — Dimensión espacial y de producto: PARCIALMENTE RESPALDADA.** Aporte significativo en validación temporal, significativo en prueba (LightGBM directo) y no significativo sobre el modelo final. Territorio y cultivo se sustituyen parcialmente (identifican en parte la misma serie), por lo que sus efectos no se suman ni se atribuyen por separado con seguridad. La evidencia debe leerse junto con la separación en X3a y X3b (Tablas N y O).

**H4 — Dimensión económica histórica: RESPALDADA.** Aporte significativo en validación temporal, significativo en prueba (LightGBM directo). Concentra la capacidad predictiva, coherente con la inercia de los precios; en el modelo final es además su ancla estructural y no puede retirarse.

### 4.3.4. Cómo interpretar la importancia de las dimensiones

La importancia por ganancia, SHAP y permutación mide cuánto usa el modelo cada dimensión (*importancia predictiva*); la ablación mide cuánto empeora el pronóstico al retirarla (*aporte incremental*); ninguna de las dos estima un *efecto causal*. Pueden diferir porque las variables están correlacionadas. El porcentaje SHAP de cada dimensión se calcula como la suma del valor absoluto medio de SHAP de sus variables dividida por la suma de todas las variables (X4 = 58.6 %), sobre una muestra de 1 500 registros de prueba con historial, y está expresado en la escala log(precio / último precio): es el cambio relativo respecto de la persistencia, no soles. Un ejemplo de la divergencia es X2: 16.5 % por ganancia, 8.6 % por SHAP y un aporte incremental de -0.0029 de R². Otro ejemplo es X1: 22.3 % por SHAP frente a una caída de R² por permutación de 0.0012. No es una contradicción: SHAP se expresa en escala log(precio / último precio) y la permutación en R² del precio, de modo que miden cosas distintas.

### 4.3.5. Sensibilidad a la inclusión del año calendario (ANO)

La matriz de consistencia incluye el año calendario (ANO) en la dimensión temporal (X1). Como los modelos de árboles no extrapolan fuera del rango de valores vistos en entrenamiento y la prueba cronológica contiene años posteriores a los del entrenamiento, ANO podría actuar como una simple etiqueta de período. Para aislar su efecto se reentrenó el modelo final con los mismos hiperparámetros, con y sin ANO, en 5 semillas, de modo que la única diferencia entre especificaciones es la lista de variables (Tabla T).

**Tabla T.** Efecto de ANO con hiperparámetros fijos (conjunto de prueba, modelo final)

| Especificación | Variables | R² (media ± DE) | MAE (media ± DE) | Ventaja mensual de MAE vs persistencia | p (HAC) vs persistencia |
|---|---|---|---|---|---|
| Con ANO | 17 | 0.7360 ± 0.0000 | 0.4936 ± 0.0000 | 0.0349 | 0.002 |
| Sin ANO | 16 | 0.7297 ± 0.0000 | 0.5030 ± 0.0000 | 0.0210 | 0.034 |

*Nota.* DE = desviación estándar entre semillas. El contraste con ANO frente a sin ANO usa la predicción promedio de las semillas y pérdidas pareadas por mes (HAC, IC95 por bloques). Los p frente a la persistencia de esta tabla no llevan corrección de Holm: la inferencia principal frente a los baselines es la Tabla L. Elaboración propia.

Con ANO el R² en prueba fue 0.7360 y el MAE 0.4936; sin ANO, 0.7297 y 0.5030. La diferencia mensual de MAE a favor de incluir ANO es 0.0140 (IC95 [0.0071; 0.0186]; p < 0.001), y la diferencia de MAE medio (0.0094 S/ por kg) supera la variabilidad entre semillas. El efecto de ANO es, por tanto, estadísticamente detectable pero de magnitud pequeña. Una posible explicación, no contrastada directamente, es que ANO capta el nivel de precios de períodos ya vistos en entrenamiento; para años posteriores el árbol lo trata como el último año conocido. Por ello el modelo debe reentrenarse periódicamente para pronosticar períodos futuros. Los hiperparámetros se fijaron intencionalmente: en corridas completas con y sin ANO la búsqueda aleatoria elige configuraciones distintas, por lo que comparar esas corridas mezclaría el efecto de ANO con el de los hiperparámetros.

**Figura I.** Ablation por dimensión. → `figura_ablation_por_dimension.png`

## 4.4. DISCUSIÓN

*Estructura por objetivo (hallazgo → antecedentes → coincidencias y diferencias → explicación técnica → aporte y límites). Los antecedentes deben completarse con fuentes verificadas: no se incluyen citas que no figuren verificadas en el borrador.*

### 4.4.1. Objetivo general: eficacia predictiva

**Hallazgo.** El LightGBM sobre persistencia alcanzó R² = 0.7360 (IC95 [0.6470; 0.7739]), MAE = 0.4936 S/ por kg y MAPE = 25.1 % en un conjunto de prueba cronológico, y no cumple el umbral de 0.85. En R² es superado por Regresión Lineal (0.7162); su ventaja está en el MAE. **Antecedentes.** [COMPLETAR: 2–3 estudios nacionales e internacionales sobre precios agrícolas con autor, año, cultivos, algoritmo, horizonte, esquema de validación y métrica reportada.] **Coincidencias y diferencias.** [COMPLETAR] Los R² de otros estudios solo son comparables si el esquema de validación es cronológico y el objetivo es el mismo: la validación aleatoria, el horizonte, la agregación por cultivo y la presencia de rezagos del propio precio inflan el R². En este estudio la persistencia ya alcanza R² = 0.6862, de modo que gran parte de la varianza explicada proviene de la inercia del precio. **Aporte y límite.** El aporte es una evaluación cronológica, con pérdidas pareadas e incertidumbre, sobre series irregulares de Piura; no autoriza a generalizar a otras regiones ni a afirmar beneficios económicos.

### 4.4.2. Superación de baselines y horizonte

**Hallazgo.** La reducción del MAE frente a la persistencia es de 6.5 % (p Holm = 0.013), pero depende de la muestra, de la métrica y de la distancia al último precio (Tablas L y M). **Explicación técnica.** Cuando el último precio es reciente, repetirlo es difícil de superar; el modelo aporta cuando ese precio es antiguo. **Antecedentes.** [COMPLETAR: estudios que comparen contra persistencia o modelos ingenuos; si no existen, dilo.] **Aporte y límite.** Un modelo de Machine Learning es valioso aquí sobre todo en series con precios previos antiguos; la mejora global es modesta.

### 4.4.3. Dimensión temporal

**Hallazgo.** Parcialmente respaldada: Aporte no significativo en validación temporal, no significativo en prueba (LightGBM directo) y significativo sobre el modelo final. Incluye el año calendario (ANO), cuyo efecto con hiperparámetros fijos se detalla en la sección 4.3.5 (Tabla T); su contribución es inestable entre contrastes y no es evidencia de causalidad. **Antecedentes.** [COMPLETAR: estudios que incorporen esta clase de variables, con su dataset, algoritmo, horizonte y validación.] **Coincidencias y diferencias.** [COMPLETAR] Si un estudio reporta aporte de esta dimensión, revisa si usó validación aleatoria, otro cultivo único o variables que ya contenían el precio. **Aporte y límite.** Contribución predictiva en este conjunto de datos; no es evidencia de causalidad, y la ausencia de aporte no implica ausencia de relación económica.

### 4.4.4. Dimensión productiva

**Hallazgo.** No respaldada: Aporte no significativo en validación temporal, no significativo en prueba (LightGBM directo) y no significativo sobre el modelo final. La X2 del modelo principal está rezagada (disponible en el origen); la versión contemporánea se reporta como cota superior en la Tabla P. **Antecedentes.** [COMPLETAR: estudios que incorporen esta clase de variables, con su dataset, algoritmo, horizonte y validación.] **Coincidencias y diferencias.** [COMPLETAR] Si un estudio reporta aporte de esta dimensión, revisa si usó validación aleatoria, otro cultivo único o variables que ya contenían el precio. **Aporte y límite.** Contribución predictiva en este conjunto de datos; no es evidencia de causalidad, y la ausencia de aporte no implica ausencia de relación económica.

### 4.4.5. Dimensión espacial y de producto

**Hallazgo.** Parcialmente respaldada: Aporte significativo en validación temporal, significativo en prueba (LightGBM directo) y no significativo sobre el modelo final. Territorio y cultivo se sustituyen parcialmente (identifican en parte la misma serie), por lo que sus efectos no se suman ni se atribuyen por separado con seguridad. La evidencia debe leerse junto con la separación en X3a y X3b (Tablas N y O). **Antecedentes.** [COMPLETAR: estudios que incorporen esta clase de variables, con su dataset, algoritmo, horizonte y validación.] **Coincidencias y diferencias.** [COMPLETAR] Si un estudio reporta aporte de esta dimensión, revisa si usó validación aleatoria, otro cultivo único o variables que ya contenían el precio. **Aporte y límite.** Contribución predictiva en este conjunto de datos; no es evidencia de causalidad, y la ausencia de aporte no implica ausencia de relación económica.

### 4.4.6. Dimensión económica histórica

**Hallazgo.** Respaldada: Aporte significativo en validación temporal, significativo en prueba (LightGBM directo). Concentra la capacidad predictiva, coherente con la inercia de los precios; en el modelo final es además su ancla estructural y no puede retirarse. **Antecedentes.** [COMPLETAR: estudios que incorporen esta clase de variables, con su dataset, algoritmo, horizonte y validación.] **Coincidencias y diferencias.** [COMPLETAR] Si un estudio reporta aporte de esta dimensión, revisa si usó validación aleatoria, otro cultivo único o variables que ya contenían el precio. **Aporte y límite.** Contribución predictiva en este conjunto de datos; no es evidencia de causalidad, y la ausencia de aporte no implica ausencia de relación económica.

### Limitaciones (4.4.7)

- No se incorporaron variables climáticas (SENAMHI, ONI) ni de mercado externas.
- La dimensión geográfica usa codificación ordinal de provincia, distrito y cultivo, sin coordenadas ni embeddings; su efecto es inestable entre validación y prueba.
- Los precios solo existen cuando hubo cosecha; las series son irregulares y la distancia al último precio varía (Tabla M).
- La muestra de modelado es una fracción pequeña del registro inicial y no es representativa de todos los cultivos (Tablas Q, R y 6a–6c); los resultados no se generalizan a toda la región.
- Los folds son expansivos por meses completos con brecha de 2 mes(es); comparten datos de entrenamiento y son pocos, por lo que el Wilcoxon tiene poder limitado. Es un contraste exploratorio, no confirmatorio.
- El desempeño es menor en cultivos de alto valor y baja frecuencia (Tabla H).
- La validación es retrospectiva: pronóstico a h = 3 mes(es) con información disponible en el origen. No se evaluó deriva conceptual ni beneficio económico.

- ANO es una variable de tendencia que los árboles no extrapolan: aporta una mejora pequeña dentro del período observado (Tabla T), pero no es proyectable a años posteriores al entrenamiento, por lo que el modelo requiere reentrenamiento periódico para uso operativo.

# CAPÍTULO V — CONCLUSIONES

1. **Objetivo general.** Se desarrolló y evaluó, con una metodología de nueve fases, un modelo de predicción del precio en chacra para Piura. El modelo seleccionado en validación, LightGBM sobre persistencia, obtuvo en prueba (n = 2 148) R² = 0.7360 (IC95 [0.6470; 0.7739]), MAE = 0.4936 S/ por kg y MAPE = 25.1 %; el criterio R² ≥ 0.85 no se cumple.

2. **Comparación con referencias.** Frente a la persistencia el MAE se reduce 6.5 % (p Holm = 0.013), diferencia significativa; frente a la media móvil, 2.7 % (p Holm = 0.071), no concluyente; frente a la regresión lineal, 12.5 % (p Holm = 0.032). En R², Regresión Lineal (0.7162) supera al modelo seleccionado.  Frente a Regresión Lineal la ventaja se observa con pérdida absoluta (MAE) pero no con pérdida cuadrática. La mejora técnica es modesta y depende de la métrica y de la distancia al último precio.

3. **Horizonte.** El modelo pronostica el precio a h = 3 mes(es), condicionado a que exista cosecha con precio en el mes objetivo; su desempeño relativo a la persistencia según la antigüedad del último precio es: T-h (distancia 3 m): -7.0 %; Último precio con 1–3 m de antigüedad (distancia 4–6 m): +6.9 %; Con 4–12 m de antigüedad (distancia 7–15 m): +5.9 %; Con más de 12 m de antigüedad (distancia > 15 m): +3.0 %.

4. **Dimensión temporal.** Hipótesis parcialmente respaldada. Aporte no significativo en validación temporal, no significativo en prueba (LightGBM directo) y significativo sobre el modelo final. Incluye el año calendario (ANO), cuyo efecto con hiperparámetros fijos se detalla en la sección 4.3.5 (Tabla T); su contribución es inestable entre contrastes y no es evidencia de causalidad.

5. **Dimensión productiva.** Hipótesis no respaldada. Aporte no significativo en validación temporal, no significativo en prueba (LightGBM directo) y no significativo sobre el modelo final. La X2 del modelo principal está rezagada (disponible en el origen); la versión contemporánea se reporta como cota superior en la Tabla P.

6. **Dimensión geográfica.** Hipótesis parcialmente respaldada. Aporte significativo en validación temporal, significativo en prueba (LightGBM directo) y no significativo sobre el modelo final. Territorio y cultivo se sustituyen parcialmente (identifican en parte la misma serie), por lo que sus efectos no se suman ni se atribuyen por separado con seguridad. La evidencia debe leerse junto con la separación en X3a y X3b (Tablas N y O).

7. **Dimensión económica histórica.** Hipótesis respaldada. Aporte significativo en validación temporal, significativo en prueba (LightGBM directo). Concentra la capacidad predictiva, coherente con la inercia de los precios; en el modelo final es además su ancla estructural y no puede retirarse.

8. **Año calendario.** Con hiperparámetros fijos, incluir ANO cambia el MAE de 0.5030 a 0.4936 (diferencia mensual 0.0140; p < 0.001); el efecto es pequeño y no es proyectable a años posteriores al entrenamiento.

9. **Interpretabilidad.** La dimensión económica histórica es la de mayor peso (58.6 % de la contribución SHAP, en escala log-ratio respecto de la persistencia). Las medidas de importancia describen el uso predictivo de las variables y no efectos causales.

10. **Anomalías.** Isolation Forest marcó 5.00 % de los registros; los anómalos tienen un precio medio de S/ 3.736 frente a S/ 1.708 de los normales.

11. **Alcance.** Los resultados demuestran un desempeño técnico en un conjunto de prueba retrospectivo (la reducción del MAE frente a la persistencia equivale a unos S/ 0.03 por kg); no demuestran beneficios económicos ni mejores decisiones agrícolas, que requieren evaluación con usuarios o datos operativos.

12. **Sistema web.** *(Completar con mediciones propias: tiempo de respuesta, disponibilidad y SUS; no provienen de este pipeline.)*

# TEXTO PARA EL CAPÍTULO III (metodología) — responde Obs. 4, 8, 11, 15 y 16

## 3.x Disponibilidad de los predictores al momento del pronóstico (Obs. 8)

| Dimensión | Fuente | Disponibilidad | ¿Disponible antes del pronóstico? |
|---|---|---|---|
| X1 Temporal | Calendario (año, mes, estación) | Conocida de antemano | Sí |
| X2 Productiva | Registro agrícola regional, rezagada h = 3 mes(es) [+ X2_REZAGO_EXTRA_MESES] | [COMPLETAR: periodicidad y fecha de publicación de la fuente] | [VERIFICAR] |
| X3 Geográfica | Catálogo de cultivo, provincia y distrito | Estática | Sí |
| X4 Económica histórica | Último precio observado con mes ≤ T − 3 | [COMPLETAR: rezago de publicación del precio previo] | [VERIFICAR] |

Como análisis de sensibilidad, se repitió la evaluación del modelo final con la dimensión productiva rezagada (valores de la observación previa) y sin ella: la diferencia de MAE frente al original es de -0.0005 (p = 0.791). Esta evidencia debe interpretarse como un análisis de sensibilidad retrospectivo y no como una demostración de disponibilidad anticipada de las variables productivas; para una aplicación operativa antes de la publicación del registro agrícola, sería necesario utilizar exclusivamente los predictores históricos disponibles hasta la fecha de emisión del pronóstico (Tabla P).

## 3.x Variable ANO en la dimensión temporal

La dimensión temporal (X1) conserva el año calendario (ANO) junto con MES_NUM, MES_SEN, MES_COS y ESTACION, conforme a la matriz de consistencia. Dado que los modelos de árboles no extrapolan fuera del rango de entrenamiento, se evaluó su efecto mediante un experimento controlado con hiperparámetros fijos y varias semillas (sección 4.3.5, Tabla T).

## 3.x Operacionalización de la dimensión geográfica (Obs. 9)

La dimensión geográfica se definió con provincia, distrito y cultivo codificados de forma ordinal. El cultivo identifica el *producto* y no el territorio; se mantuvo dentro de X3 para conservar la matriz de consistencia original [COMPLETAR: justifica por qué el cultivo se clasificó como geográfico o, si prefieres, muévelo a una dimensión de producto]. Como análisis complementario, X3 se separó en territorio (provincia y distrito) y producto (cultivo) y se repitieron la ablación y el contraste en prueba con corrección de Holm sobre cinco pruebas (Tablas O y N).

## 3.x Definición del horizonte y de la validación temporal (Obs. 4 y 11)

El modelo pronostica el precio en chacra del mes T con un horizonte fijo de h = 3 mes(es). Todas las variables que dependen de la serie usan solo observaciones con mes ≤ T − 3. Solo existe objetivo cuando hubo cosecha con precio informado en el mes T; el pronóstico es condicional a esa ocurrencia. Se dejó una brecha de 2 mes(es) entre entrenamiento y validación/prueba. La partición cronológica 70/15/15, las fechas, los tamaños y el uso de rezagos con precios observados se detallan en la Tabla K.

## 3.x Criterio de eficacia (Obs. 15)

El criterio R² ≥ 0.85 se adoptó como umbral de aceptabilidad porque [COMPLETAR con la verdad: antecedente comparable (cita verificada), estándar técnico o decisión metodológica fijada antes de evaluar el modelo en prueba]. No se modifica después de conocer los resultados. No alcanzarlo no invalida el estudio; se interpreta como que el modelo no cumple el criterio de eficacia fijado y se complementa con criterios relativos (mejora frente a baselines con intervalos de confianza).

## 3.x Alcance temporal (Obs. 16)

El año 2025 del título corresponde a [COMPLETAR: año de ejecución / contexto de aplicación]. Los registros agrícolas analizados abarcan 2015–2024; no se evaluaron registros de 2025. Alternativa: titular «Piura, 2015–2024».

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
- **Obs. 13 (referencias):** Tabla A y marco teórico: sustituye Sokolova y Lapalme (2009) —métricas de clasificación— por una fuente de métricas de regresión verificada (p. ej. Hyndman y Koehler, 2006); corrige o elimina Enrique y Castillo (2020), Mena Salgado et al. (2025) y Yuan (2024) según `docs/verificacion_referencias.md`.
- **Obs. 1:** reemplaza «influye significativamente» por «contribuye predictivamente» en la matriz de consistencia y en 1.6.1; separa HG-a y HG-b (ver 4.3.1).
- **Obs. 14:** las conclusiones hablan de desempeño técnico; elimina cualquier afirmación de beneficio económico sin evidencia.
- **Variables (Tabla 2 / Cap. III):** el modelo usa 17 variables e incluye ANO en X1 (ANO, MES_NUM, MES_SEN, MES_COS, ESTACION). Mantén 17 en todo el documento.
- **ANO:** si el jurado pregunta, remite a la sección 4.3.5 (Tabla T). No uses la comparación de corridas completas con y sin ANO: difieren en hiperparámetros.
