# AgroPredict — Predicción de precios agrícolas (Piura, 2025)

Tesis de título, Escuela Profesional de Ingeniería de Sistemas, UNSCH.
Autora: Sihomara Sandy Ochoa Cisneros · Asesora: Mg. Celia Edith Martínez Cordova.

El modelo se desarrolla con una metodología de **nueve fases** basada en CRISP-DM
(Chapman et al., 2000) y KDD (Fayyad et al., 1996), con las prácticas de Géron (2023):
problema → datos → preparación → pipeline → modelado → ajuste de hiperparámetros →
evaluación → interpretabilidad (SHAP) → validación y despliegue.

## Ejecutar

```bash
cd backend
python -m venv venv
source venv/Scripts/activate      # Git Bash en Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
# copia tu CSV oficial a backend/data/dataset_agricola.csv
python main.py              # corrida completa
python main.py --rapido     # prueba de humo: SOBRESCRIBE outputs/; vuelve a correr main.py completo antes de usar nada
```

Resultados en `backend/outputs/`:

| Carpeta | Contenido |
|---|---|
| `tesis_texto/` | texto de resultados y conclusiones con tus números (`.md` y `.docx`) y guía de sustentación |
| `tables/` | todas las tablas en CSV |
| `figures/` | figuras PNG a 300 dpi (sin título incrustado) |
| `models/` | `modelo_final.pkl` (lo carga la API) |

## API y dashboard

```bash
cd backend && uvicorn src.api.main:app --reload --port 8000
# en otra terminal
cd frontend && python -m http.server 5500
```

`POST /api/predict` recibe provincia y distrito por nombre y el código de cultivo; devuelve el
precio y un intervalo empírico del 90 %. Despliegue: backend en Railway (Dockerfile), frontend en Firebase Hosting.

## Decisiones metodológicas clave

- **Sin fuga de datos:** las variables históricas usan solo períodos anteriores; la variable objetivo no se imputa.
- **Partición cronológica 70/15/15**; el conjunto de prueba se usa una sola vez.
- **Pipeline de scikit-learn:** imputación, escalado y codificación se ajustan solo con entrenamiento.
- **Baselines ingenuos** (persistencia y media móvil) como vara de comparación.
- **Hipótesis específicas:** ablation por dimensión, Wilcoxon unilateral con corrección de Holm, más poder predictivo propio y aporte incremental de cada dimensión.
