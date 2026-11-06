# GUÍA DE SUSTENTACIÓN — preguntas probables y respuestas con tus números

## ¿Por qué el R² (0.7775) no alcanza el criterio de 0.85?
El criterio se fijó a priori en la sección 3.6.3 y se reporta tal cual: la hipótesis general no se acepta con ese umbral. Contexto verificable: (1) repetir el último precio ya da R² = 0.7319: la inercia de los precios explica gran parte de la varianza; (2) el R² global se apoya en distinguir cultivos baratos de caros: dentro de cada segmento el R² es menor (Tabla H: 0.5070 en precios bajos y 0.1507 en altos); (3) el estudio de Fuentes et al. (2025), citado en el marco referencial, reporta R² = 0.788; compara con cautela porque cultivos, horizonte y validación difieren. No se debe modificar el umbral después de ver los resultados sin acuerdo explícito del asesor y del jurado; lo defendible es discutir la brecha y proponer mejoras (variables climáticas, modelos por cultivo).

## ¿Qué metodología siguió para construir el modelo?
CRISP-DM complementado con KDD y las prácticas de Géron (2023), en nueve fases: problema, datos, preparación, pipeline, modelado, ajuste de hiperparámetros, evaluación, interpretabilidad y validación/despliegue. Cada fase tiene respaldo bibliográfico (Tabla A).

## ¿Cómo evitó la fuga de datos?
Tres controles: (1) las variables históricas usan solo precios de períodos anteriores; (2) la partición es cronológica 70/15/15 sin mezclar períodos; (3) imputación y escalado viven dentro del Pipeline y se ajustan solo con entrenamiento. La prueba se usó una sola vez.

## ¿Qué son los hiperparámetros y cómo los eligió?
Son valores que se fijan antes de entrenar (profundidad, número de árboles, tasa de aprendizaje); los parámetros, en cambio, los aprende el modelo. Se optimizaron con búsqueda aleatoria y validación cruzada temporal solo en entrenamiento (Tabla D).

## ¿Su modelo es mejor que simplemente repetir el último precio?
Ligeramente: en prueba el MAE del modelo es 0.4189 frente a 0.4392 de la persistencia (R² 0.7775 vs 0.7319). La reducción del MAE es de 4.6 %, una mejora modesta cuya significancia (pérdidas pareadas por mes, errores HAC, IC por bootstrap de bloques) está en tabla_significancia_vs_baseline.csv; además la media móvil de 3 obtiene un R² casi igual (0.7762). Hay que presentarlo como mejora acotada, no como superioridad rotunda.

## ¿Por qué se eligió LightGBM sobre persistencia si otro modelo tiene mayor R² en prueba?
La regla de selección (menor MAE en validación) se fijó antes de ver la prueba; elegir en prueba la contaminaría. Mejor R² en prueba entre los modelos de ML: XGBoost (0.7992); el elegido logra el menor MAE (0.4189) y el mayor R²_log (0.7929).

## ¿Por qué no se aceptaron 2 de las hipótesis específicas?
Las dimensiones temporal, productiva no alcanzaron significancia tras la corrección de Holm. Temporal: sola alcanza R² = -0.0116 y con X4 aporta 0.0026, es decir, no añade información que el precio pasado no tenga. Productiva: sola alcanza R² = 0.1323 y con X4 aporta -0.0370, es decir, no añade información que el precio pasado no tenga. No aceptar una hipótesis específica no invalida el trabajo: indica en qué dimensiones está la información predictiva (geográfica, económica histórica).

## ¿Por qué el R² completo de la ablation no coincide con el de prueba?
La ablation principal se hace con LightGBM (misma familia que el modelo final) en particiones temporales del tramo de desarrollo (entrenamiento + validación), no en el conjunto de prueba; sirve para comparar dimensiones entre sí, no para reportar el desempeño final.

## ¿Por qué LightGBM/XGBoost/RF y no una red LSTM?
Los datos son series cortas, irregulares y de muchos cultivos; los modelos de árboles manejan bien variables mixtas y pocos datos. Una LSTM global mezclaría cultivos heterogéneos; se deja como trabajo futuro por cultivo.

## ¿Cómo sabe que su R² no depende de los cultivos baratos?
La Tabla H desagrega el desempeño por nivel de precio; el rendimiento en cultivos de alto valor es menor y se declara como limitación.

## ¿Qué aporta SHAP?
Explica cuánto contribuye cada variable a cada predicción. Se agregó por dimensión y se contrastó con la permutación agrupada, que es robusta a la correlación entre variables.

## ¿Se imputó la variable objetivo?
No. Los registros sin precio se descartaron y se reportan en el embudo de depuración (Tabla B).

## ¿Qué pasa con el primer registro de cada serie, que no tiene historial?
Se marca con TIENE_HISTORIAL = 0 y el pipeline imputa la mediana de entrenamiento; el modelo aprende que en esos casos no hay información previa.
