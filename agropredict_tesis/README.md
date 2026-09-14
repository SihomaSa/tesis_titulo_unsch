# AgroPredict — Proyecto reorganizado para la tesis

Sistema web para la predicción de precios agrícolas mediante técnicas de
Machine Learning — Región Piura, 2025 (UNSCH, Sihomara Sandy Ochoa Cisneros).

Este proyecto reemplaza la carpeta `backend/` desordenada (con más de 15
variantes de `pipeline_*.py` sueltos en la raíz) por una estructura única,
modular, y con un frontend mínimo listo para conectarse a la API.

---

## 1. Análisis desde cero: ¿cuál archivo era el que realmente usabas?

Con solo los nombres de archivo de tus capturas no puedo abrir el contenido
real de tu código anterior — pero cruzando esos nombres con el texto que ya
escribiste en tu tesis (103 páginas, capítulos 3 y 4) encontré evidencia
fuerte de cuál pipeline generó tus resultados oficiales:

**Evidencia clave: la numeración de tablas coincide.**
En tu carpeta `backend/data/` existen estos archivos:
```
tabla_4.1_rendimiento_modelos.csv
tabla_4.1_resumen_modelos.csv
tabla_4.2_r2_por_fold.csv
tabla_4.3_feature_importance.csv
tabla_4.4_wilcoxon_hipotesis.csv
reporte_final_tesis.json
reporte_tabla3_completo.json
resumen_tss.json
```
Estos nombres (`tabla_4.1`, `4.2`, `4.3`, `4.4`) **coinciden exactamente**
con la numeración de tu capítulo 4 (Resultados) y con la subsección 4.4.1
"Contrastación de hipótesis" que ya redactaste, con los mismos valores
(R²=0.7940, p=0.0625 para H1/H3, p=0.1875 para H2, X4=82.86% de importancia).
Es decir: **el script que produjo esas tablas es el que realmente sustenta
tu tesis**, no necesariamente el más reciente ni el de nombre más "definitivo".

Los candidatos más probables, por ubicación y nombre, son los de
`backend/data/mlflow_pipeline/`:
- `6_full_model_comparison.py`
- `pipeline_full_comparison_tss.py`
- o el combo `5_model_comparison.py` + `generate_final_tables.py` /
  `exportar_tablas_tesis.py` (en la raíz de `backend/`)

En cambio, estos archivos de la raíz de `backend/` muestran señales de ser
**iteraciones abandonadas** (mismo propósito, nombres repetidos con sufijos
"mejorado"/"definitivo", sin numeración de tabla asociada):
```
pipeline_AgroPredict_mejorado.py
pipeline_AgroPredict1_mejorado.py
pipeline_definitivo.py
pipeline_mejorado_agropredict.py
pipeline_precio_chacra_mejorado.py
```
El texto que me compartiste como "el código" (con `R² final = 0.7940`) es
casi con certeza uno de estos — probablemente `pipeline_AgroPredict_mejorado.py`,
que además fue el que un Claude anterior identificó como "el archivo
principal que estás usando". Los números coinciden con tu capítulo 4, así
que **no está mal**, pero es una rama distinta a la que generó las
`tabla_4.x` numeradas. Ambas ramas convergen aproximadamente al mismo
resultado, lo cual es tranquilizador, pero confirma que hay código
duplicado que hacía lo mismo dos veces.

**Cómo confirmarlo tú mismo, en 30 segundos, cuando tengas los archivos
reales delante:**
```bash
# El que tenga la fecha de modificación más reciente Y coincida con
# los valores ya redactados en tu tesis es el "real".
ls -lt backend/*.py backend/data/mlflow_pipeline/*.py | head -10
grep -l "0.7940\|0.7855\|14393\|14,393" backend/*.py backend/data/mlflow_pipeline/*.py
```

**Otros tres candidatos que no pude descartar** (aparecen en tus capturas
como carpetas de resultados, `backend/reports/`):
```
resultados_matriz_consistencia/
resultados_pipeline_definitivo/
resultados_tesis_piura/
```
Si me dices qué CSV tiene adentro cada una (o subes 2-3 archivos reales),
puedo confirmarte con certeza cuál es la fuente única de verdad y cuáles
puedes borrar sin miedo.

**Recomendación práctica:** usa el proyecto de esta carpeta (`agropredict_tesis/`)
como el reemplazo definitivo de todo `backend/`. Corre `backend/main.py` una
vez con tu CSV real, compara el `R²` y el `n` de registros con los que ya
tienes en tu tesis, y si coinciden (o están cerca), borra los `pipeline_*.py`
sueltos y los notebooks duplicados — ya no los necesitas.

---

## 2. Inconsistencias detectadas entre la tesis y el código

Ver también `backend/src/config.py` → `MATRIZ_CONSISTENCIA["inconsistencias_detectadas"]`
(se sirve también por la API en `/api/consistency-matrix` y se pinta en el
frontend, pestaña "Matriz de Consistencia").

| # | Inconsistencia | Dónde aparece | Qué hacer |
|---|---|---|---|
| 1 | Umbral de aceptación R² | Cap. 3.6.3 dice **≥0.85**; Cap. 4.4 y el código usan **≥0.75** | Elegir uno como oficial y corregir el otro capítulo. El pipeline reporta el resultado frente a ambos. |
| 2 | Composición de X4 | Anexo 1 dice "z-score de precio"; el código usa `MEDIA_MOVIL_3` | Corregir el **texto del Anexo** (no el código: ya tienes X4=82.86% de importancia redactado con `MEDIA_MOVIL_3`). |
| 3 | Composición de X3 | Anexo 1 incluye `UBIGEO`; el código no lo usaba | Disponible como flag `USE_UBIGEO_EN_X3` en `config.py`. Actívalo solo si vas a repetir el ablation de H3 (cambiará el R² ya reportado). |
| 4 | Tamaño de muestra | Anexo 1: "116,000 registros válidos" (población); Cap. 4.4: "14,393 registros válidos" (muestra final de modelado) | No es necesariamente un error — aclara en el texto que son dos cifras distintas (población tras limpieza básica vs. muestra final tras filtrar a Piura + dropna). Verifica cuál produce tu `main.py`. |
| 5 | LightGBM | Aparece en resultados (Cap. 4.4) pero no en la metodología (3.7.2.2, que solo menciona RF/XGBoost/RL) | Agregado al pipeline (`compare_models.py`). Agrega una frase en 3.7.2.2 mencionando LightGBM. |

---

## 3. Estructura del proyecto

```
agropredict_tesis/
├── backend/
│   ├── main.py                      ← ÚNICO script a correr para reproducir el capítulo 4
│   ├── requirements.txt
│   ├── data/                        ← coloca aquí tu dataset_agricola.csv
│   ├── outputs/
│   │   ├── figures/                 ← PNG a 300dpi, listos para pegar en Word
│   │   ├── tables/                  ← tabla_4.1 ... tabla_4.4, reporte_contrastacion_hipotesis.json
│   │   └── models/                  ← random_forest_final.pkl
│   ├── mlruns/                      ← historial de MLflow (abrir con `mlflow ui`)
│   └── src/
│       ├── config.py                ← Matriz de Consistencia (fuente única de verdad)
│       ├── data/
│       │   ├── data_loader.py       ← Carga y limpieza (3.7.2.1.B)
│       │   └── feature_engineering.py  ← X1-X4 (3.7.2.1.C)
│       ├── models/
│       │   ├── compare_models.py    ← RF/XGBoost/LightGBM/RL (Objetivo General)
│       │   └── final_model.py       ← Modelo final hold-out (Hipótesis General)
│       ├── analysis/
│       │   ├── ablation_study.py    ← H1-H4 + Wilcoxon (Hipótesis específicas)
│       │   ├── anomaly_detection.py ← Isolation Forest
│       │   └── hypothesis_report.py ← Genera el JSON de "Contrastación de hipótesis"
│       └── api/
│           └── main.py              ← FastAPI: sirve todo al frontend
└── frontend/
    ├── index.html                   ← Dashboard (Matriz + Resultados + Predicción)
    ├── css/styles.css
    └── js/{config.js, app.js}
```

---

## 4. Cómo correrlo

### Backend — generar los resultados de la tesis
```bash
cd backend
python -m venv venv && source venv/bin/activate   # o venv\Scripts\activate en Windows
pip install -r requirements.txt

# coloca tu CSV real en data/dataset_agricola.csv
python main.py
```
Esto genera todas las tablas (`outputs/tables/`), figuras (`outputs/figures/`)
y el modelo (`outputs/models/random_forest_final.pkl`), y registra todo en
MLflow.

### Ver y exportar los gráficos de MLflow para pegarlos en la tesis
```bash
mlflow ui --backend-store-uri backend/mlruns
```
Abre `http://localhost:5000` → elige el experimento `AgroPredict_Piura_Tesis`
→ entra a cada run (`01_comparacion_modelos`, `02_modelo_final_holdout`,
`03_ablation_study_hipotesis`, `04_isolation_forest`) → pestaña **Artifacts**
→ ahí están las figuras en PNG a 300dpi, listas para insertar en Word.

### API — para que el frontend consuma resultados y predicciones
```bash
cd backend
uvicorn src.api.main:app --reload --port 8000
```

### Frontend — dashboard
Solo abre `frontend/index.html` en el navegador (o sírvelo con cualquier
servidor estático, por ejemplo `python -m http.server` dentro de `frontend/`).
Si despliegas la API en otro dominio (Railway, Render, Cloud Run), cambia
`API_BASE_URL` en `frontend/js/config.js`.

Este `index.html` es un dashboard mínimo en HTML/JS puro (sin build step)
para que puedas verlo funcionando de inmediato. Si tu tesis exige
específicamente Angular (como dice 3.7.2.3.A), usa este mismo consumo de
API (`/api/consistency-matrix`, `/api/results/*`, `/api/predict`) como
referencia para portar la lógica a tus componentes Angular — el contrato
de la API no cambia.

---

## 5. Qué mostrar en tu sustentación

- Pestaña **Matriz de Consistencia** del frontend: PG/OG/HG arriba, las 4
  dimensiones (con su PE/OE/HE) abajo, coloreadas igual que tus figuras de
  ablation (azul=temporal, verde=productiva, dorado=geográfica, rojo=económica).
- Pestaña **Resultados**: tabla y gráfico de comparación de 4 modelos +
  ablation study, generados en vivo desde tus CSV reales.
- Pestaña **Contrastación de Hipótesis**: HG y H1-H4 con su ΔR², p-valor y
  estado, calculados por el pipeline (no hardcodeados) — así nunca hay
  desfase entre lo que dice la pantalla y lo que dice tu tesis.
