# GUÍA DE SUSTENTACIÓN — preguntas probables y respuestas con tus números

## ¿Por qué el R² (0.7384) no alcanza el criterio de 0.85?
El criterio figura en la sección 3.6.3 y se reporta tal cual: la hipótesis de eficacia no se respalda con ese umbral [VERIFICA y di con honestidad de dónde sale: antecedente, estándar técnico o decisión metodológica previa a ver la prueba]. Contexto verificable: (1) repetir el último precio ya da R² = 0.6862: la inercia de los precios explica gran parte de la varianza; (2) el R² global se apoya en distinguir cultivos baratos de caros: dentro de cada segmento el R² es menor (Tabla H: 0.5242 en precios bajos y -0.1343 en altos); (3) el estudio de Fuentes et al. (2025), citado en el marco referencial, reporta R² = 0.788 [VERIFICA la cifra y la cita contra la fuente antes de usarla]; compara con cautela porque cultivos, horizonte y validación difieren. No se debe modificar el umbral después de ver los resultados sin acuerdo explícito del asesor y del jurado; lo defendible es discutir la brecha y proponer mejoras (variables climáticas, modelos por cultivo).

## ¿Qué metodología siguió para construir el modelo?
CRISP-DM complementado con KDD y las prácticas de Géron (2023), en nueve fases: problema, datos, preparación, pipeline, modelado, ajuste de hiperparámetros, evaluación, interpretabilidad y validación/despliegue. Cada fase tiene respaldo bibliográfico (Tabla A).

## ¿Cómo evitó la fuga de datos?
Tres controles: (1) las variables históricas y las de X2 usan solo información con mes ≤ T − h (h = 3); (2) la partición es cronológica 70/15/15 con brecha entre conjuntos; (3) imputación y escalado viven dentro del Pipeline y se ajustan solo con entrenamiento. La prueba se usó una sola vez; además hay una prueba anti-fuga automática.

## ¿Qué son los hiperparámetros y cómo los eligió?
Son valores que se fijan antes de entrenar (profundidad, número de árboles, tasa de aprendizaje); los parámetros, en cambio, los aprende el modelo. Se optimizaron con búsqueda aleatoria y validación cruzada temporal solo en entrenamiento (Tabla D).

## ¿Su modelo es mejor que simplemente repetir el último precio?
En prueba el MAE del modelo es 0.4966 frente a 0.5276 de la persistencia (R² 0.7384 vs 0.6862); la media móvil de 3 obtiene un R² casi igual (0.7397). La reducción del MAE es de 5.9 % (IC95 de la diferencia mensual [0.016; 0.051]; p Holm = 0.010). Con solo los registros que tienen precio previo real la reducción es de 3.8 % (p Holm = 0.016; IC95 [0.010; 0.032]), todavía significativa. Además, cuando el último precio es reciente (Último precio en el origen (distancia 3 m): -9.6 %) el modelo no supera a la persistencia; su ventaja está en brechas largas (Tabla M). Hay que presentarlo como una mejora modesta y dependiente de la métrica, no como superioridad rotunda.

## ¿Por qué se eligió LightGBM sobre persistencia si otro modelo tiene mayor R² en prueba?
La regla de selección (menor MAE en validación) se fijó antes de ver la prueba; elegir en prueba la contaminaría. Mejor R² en prueba entre los modelos de ML: Regresión Lineal (0.7160); el elegido logra el menor MAE (0.4966) y el mayor R²_log (0.7562).

## ¿Qué hipótesis específicas se respaldaron y cuáles no?
Según la consistencia entre validación temporal, prueba y modelo final (Tabla S): temporal (H1): parcialmente respaldada; productiva (H2): no respaldada; geográfica (H3): parcialmente respaldada; económica histórica (H4): respaldada. La información predictiva se concentra en la historia de precios; las demás dimensiones no añaden un aporte detectable y consistente una vez que se conoce el precio previo. La evidencia de la dimensión geográfica no es consistente entre validación, prueba y modelo final. La de la dimensión temporal tampoco es consistente y depende en parte de ANO (sección 4.3.5). No respaldar una hipótesis específica no invalida el trabajo: indica en qué dimensiones hay información predictiva incremental en este conjunto de datos, y no prueba ausencia de relación económica.

## ¿Por qué incluyó ANO si los árboles no extrapolan?
Se conservó porque forma parte de la matriz de consistencia, y verifiqué su efecto con un experimento controlado (mismos hiperparámetros, varias semillas): MAE 0.4979 con ANO frente a 0.5045 sin ANO; diferencia mensual 0.0133 (IC95 [0.0032; 0.0215]; p = 0.005). El efecto es pequeño. Reconozco que los árboles no extrapolan, así que para pronosticar años futuros el modelo debe reentrenarse.

## ¿Por qué el R² completo de la ablation no coincide con el de prueba?
La ablation principal se hace con LightGBM (misma familia que el modelo final) en particiones temporales del tramo de desarrollo (entrenamiento + validación), no en el conjunto de prueba; sirve para comparar dimensiones entre sí, no para reportar el desempeño final.

## ¿Por qué LightGBM/XGBoost/RF y no una red LSTM?
Los datos son series cortas, irregulares y de muchos cultivos; los modelos de árboles manejan bien variables mixtas y pocos datos. Una LSTM global mezclaría cultivos heterogéneos; se deja como trabajo futuro por cultivo.

## ¿Cómo sabe que su R² no depende de los cultivos baratos?
La Tabla H desagrega el desempeño por nivel de precio; el rendimiento en cultivos de alto valor es menor y se declara como limitación.

## ¿Qué aporta SHAP y por qué no coincide con la permutación?
SHAP explica cuánto contribuye cada variable a cada predicción; se agregó por dimensión. La permutación agrupada mide la caída de R² del precio, mientras que SHAP del modelo final está en escala log(precio / último precio); miden cosas distintas y por eso pueden diferir (sección 4.3.4).

## ¿Se imputó la variable objetivo?
No. Los registros sin precio se descartaron y se reportan en el embudo de depuración (Tabla B).

## ¿Qué pasa con el primer registro de cada serie, que no tiene historial?
Se marca con TIENE_HISTORIAL = 0 y el pipeline imputa la mediana de entrenamiento; el modelo aprende que en esos casos no hay información previa.

## ¿Por qué las hipótesis específicas ya no dicen «influye»?
Porque el estudio es observacional: una dimensión que mejora el pronóstico fuera de muestra tiene contribución predictiva, pero eso no prueba causalidad. Las hipótesis se contrastan en validación temporal, en prueba y sobre el modelo final (Tabla S).

## ¿Cómo sabe que X2 no filtra información del futuro?
En el modelo con horizonte, X2 está rezagada: solo usa registros disponibles en el origen del pronóstico. Además se comparó con la variante sin X2 (Tabla P) y la diferencia no es concluyente; X2 no aporta de forma detectable.

## ¿Por qué la dimensión geográfica cambia entre validación y prueba?
Territorio y cultivo se sustituyen parcialmente porque cada uno identifica en parte la misma serie; sus efectos no son aditivos y cambian según el contraste (Tablas N y O). La evidencia es inestable y no se afirma que X3 contribuya al modelo final.

## ¿La muestra final no está sesgada si conserva solo el 11.6 %?
Sí, hay selección fuerte por cultivo (V de Cramér alta en la regla de cosecha). Se reportan los motivos de exclusión, la distribución antes/después y una sensibilidad que relaja la regla (Tablas Q y R); la conclusión cualitativa se mantiene con margen menor, pero los resultados no se generalizan a todos los cultivos.

## ¿Qué horizonte predice el modelo?
El precio del mes T con un horizonte fijo de h = 3 meses: todas las variables de la serie y de X2 usan solo información con mes ≤ T − 3. El pronóstico es condicional a que exista cosecha con precio en el mes objetivo, y se reporta el error según la antigüedad del último precio (Tabla M).

## ¿Sus resultados demuestran beneficios económicos para el agricultor?
No. Demuestran desempeño técnico en un conjunto de prueba retrospectivo. El beneficio económico requiere evaluación con usuarios o datos operativos.
