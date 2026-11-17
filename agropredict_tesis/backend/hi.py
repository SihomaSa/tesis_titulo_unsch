import numpy as np
from src.config import DATA_PATH
from src.data.data_loader import load_and_clean
from src.data.splits import split_cronologico
from src.data.feature_engineering import build_features

# 1. Cargar datos (igual que en main.py)
df, _ = load_and_clean(DATA_PATH)
df, features, dimensiones, catalogo = build_features(df, horizonte=3)

# 2. Obtener los índices del split
idx_tr, idx_va, idx_te, info_split = split_cronologico(df, horizonte=3)

# 3. Calcular los registros que NO se usaron
usados = np.concatenate([idx_tr, idx_va, idx_te])
excluidos = df.loc[~df.index.isin(usados)]

print(f"Total de registros válidos: {len(df)}")
print(f"Total de registros usados en train+val+test: {len(usados)}")
print(f"Diferencia (excluidos): {len(df) - len(usados)}")
print("\nRegistros excluidos por PERIODO (AAAAMM):")
print(excluidos["PERIODO"].value_counts().sort_index())