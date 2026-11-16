import os
os.environ["AGRO_OUT_SUBDIR"] = "horizonte_12m"
import numpy as np
from src.config import DATA_PATH, TARGET
from src.data.data_loader import load_and_clean
from src.data.feature_engineering import build_features
from src.data.splits import split_cronologico
from src.models.pipelines import build_pipeline
from src.models.evaluacion import metricas

# 1. Cargar datos con horizonte 12
df, _ = load_and_clean(DATA_PATH)
df, feats, _, _ = build_features(df, horizonte=12)
X, y = df[feats].astype(float), df[TARGET].astype(float)

# 2. Dividir
idx_tr, idx_va, idx_te, _ = split_cronologico(df, horizonte=12)
idx_dev = np.concatenate([idx_tr, idx_va])

# 3. Entrenar un modelo directo simple para ver el desempeño
print("Entrenando LightGBM directo con h=12...")
modelo = build_pipeline("lgbm", feats).fit(X.iloc[idx_dev], y.iloc[idx_dev])
pred = modelo.predict(X.iloc[idx_te])

# 4. Calcular métricas y comparar con la persistencia
res = metricas(y.iloc[idx_te], pred)
persistencia = X["PRECIO_LAG1"].iloc[idx_te].fillna(y.iloc[idx_dev].median())
res_p = metricas(y.iloc[idx_te], persistencia)

print(f"R² modelo (h=12): {res['R2']:.4f} | MAE: {res['MAE']:.4f}")
print(f"R² persistencia (h=12): {res_p['R2']:.4f} | MAE: {res_p['MAE']:.4f}")