# GUÍA DE SUSTENTACIÓN — preguntas probables y respuestas con tus números

## ¿Por qué el R² (0.7775) no alcanza el criterio de 0.85?
El criterio figura en la sección 3.6.3 y se reporta tal cual: la hipótesis de eficacia no se respalda con ese umbral [VERIFICA y di con honestidad de dónde sale: antecedente, estándar técnico o decisión metodológica previa a ver la prueba]. Contexto verificable: (1) repetir el último precio ya da R² = 0.7319: la inercia de los precios explica gran parte de la varianza; (2) el R² global se apoya en distinguir cultivos baratos de caros: dentro de cada segmento el R² es menor (Tabla H: 0.5070 en precios bajos y 0.1507 en altos); (3) el estudio de Fuentes et al. (2025), citado en el marco referencial, reporta R² = 0.788 [VERIFICA la cifra y la cita contra la fuente antes de usarla]; compara con cautela porque cultivos, horizonte y validación difieren. No se debe modificar el umbral después de ver los resultados sin acuerdo explícito del asesor y del jurado; lo defendible es discutir la brecha y proponer mejoras (variables climáticas, modelos por cultivo).

## ¿Qué metodología siguió para construir el modelo?
CRISP-DM complementado con KDD y las prácticas de Géron (2023), en nueve fases: problema, datos, preparación, pipeline, modelado, ajuste de hiperparámetros, evaluación, interpretabilidad y validación/despliegue. Cada fase tiene respaldo bibliográfico (Tabla A).

## ¿Cómo evitó la fuga de datos?
Tres controles: (1) las variables históricas usan solo precios de períodos anteriores; (2) la partición es cronológica 70/15/15 sin mezclar períodos; (3) imputación y escalado viven dentro del Pipeline y se ajustan solo con entrenamiento. La prueba se usó una sola vez. Además, como la dimensión productiva (X2) es contemporánea al precio, se repitió la evaluación con X2 rezagada y sin X2: el desempeño no cambia de forma concluyente (Tabla P).

## ¿Qué son los hiperparámetros y cómo los eligió?
Son valores que se fijan antes de entrenar (profundidad, número de árboles, tasa de aprendizaje); los parámetros, en cambio, los aprende el modelo. Se optimizaron con búsqueda aleatoria y validación cruzada temporal solo en entrenamiento (Tabla D).

## ¿Su modelo es mejor que simplemente repetir el último precio?
En prueba el MAE del modelo es 0.4189 frente a 0.4392 de la persistencia (R² 0.7775 vs 0.7319); la media móvil de 3 obtiene un R² casi igual (0.7762). La reducción del MAE es de 4.6 % (IC95 de la diferencia mensual [0.006; 0.041]; p Holm = 0.033). Pero con solo los registros que tienen precio previo real la reducción es de 2.2 % (p Holm = 0.097; IC95 [-0.000; 0.024]), no concluyente. Además, cuando el último precio es reciente (Precio previo hace 1 mes: -8.6 %; Hace 2–3 meses: -3.8 %) el modelo no supera a la persistencia; su ventaja está en brechas largas (Tabla M). Hay que presentarlo como una mejora modesta y dependiente de la métrica, no como superioridad rotunda.

## ¿Por qué se eligió LightGBM sobre persistencia si otro modelo tiene mayor R² en prueba?
La regla de selección (menor MAE en validación) se fijó antes de ver la prueba; elegir en prueba la contaminaría. Mejor R² en prueba entre los modelos de ML: XGBoost (0.7992); el elegido logra el menor MAE (0.4189) y el mayor R²_log (0.7929).

## ¿Qué hipótesis específicas se respaldaron y cuáles no?
Según la consistencia entre validación temporal, prueba y modelo final (Tabla S): temporal (H1): no respaldada; productiva (H2): no respaldada; geográfica (H3): parcialmente respaldada; económica histórica (H4): respaldada. La información predictiva se concentra en la historia de precios; las demás dimensiones no añaden un aporte detectable y consistente una vez que se conoce el precio previo. La dimensión geográfica es inestable: en validación el efecto es del cultivo y en prueba del territorio, y no se sostiene en el modelo final. No respaldar una hipótesis específica no invalida el trabajo: indica en qué dimensiones hay información predictiva incremental en este conjunto de datos, y no prueba ausencia de relación económica.

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

## ¿Por qué las hipótesis específicas ya no dicen «influye»?
Porque el estudio es observacional: una dimensión que mejora el pronóstico fuera de muestra tiene contribución predictiva, pero eso no prueba causalidad. Las hipótesis se contrastan en validación temporal, en prueba y sobre el modelo final (Tabla S).

## ¿Cómo sabe que X2 no filtra información del futuro?
X2 es contemporánea al precio, así que se repitió la evaluación con X2 rezagada y sin X2 (Tabla P): no hay diferencia concluyente. El desempeño no depende de información contemporánea y, de hecho, X2 no aporta.

## ¿Por qué la dimensión geográfica cambia entre validación y prueba?
En validación el efecto es del cultivo y aparece cuando hay menos precios previos; en prueba es del territorio. El cultivo funciona como sustituto del nivel de precio cuando falta historial. La evidencia es inestable y no se afirma que X3 contribuya al modelo final.

## ¿La muestra final no está sesgada si conserva solo el 11.6 %?
Sí, hay selección fuerte por cultivo (V de Cramér alta en la regla de cosecha). Se reportan los motivos de exclusión, la distribución antes/después y una sensibilidad que relaja la regla (Tablas Q y R); la conclusión cualitativa se mantiene con margen menor, pero los resultados no se generalizan a todos los cultivos.

## ¿Qué horizonte predice el modelo?
La siguiente observación disponible de cada serie cultivo–provincia–distrito, dado su último precio observado. La distancia puede ser mayor de un mes; por eso se reporta el error por brecha (Tabla M).

## ¿Sus resultados demuestran beneficios económicos para el agricultor?
No. Demuestran desempeño técnico en un conjunto de prueba retrospectivo. El beneficio económico requiere evaluación con usuarios o datos operativos.
