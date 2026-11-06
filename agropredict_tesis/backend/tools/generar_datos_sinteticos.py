"""
SOLO PARA PRUEBAS DE HUMO. Genera un CSV sintético con el MISMO esquema que
'dataset_agricola.csv' (incluida la columna MES en formato AAAAMM) para
verificar que el pipeline corre de punta a punta. NO usar sus resultados en
la tesis: los números reales salen de tu dataset oficial.

    python tools/generar_datos_sinteticos.py data/dataset_sintetico.csv
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
PROV = {"PIURA": ["PIURA", "CASTILLA", "CATACAOS"], "SULLANA": ["SULLANA", "MARCAVELICA"],
        "AYABACA": ["AYABACA", "FRIAS"], "PAITA": ["PAITA", "COLAN"], "SECHURA": ["SECHURA", "VICE"],
        "MORROPON": ["CHULUCANAS"], "HUANCABAMBA": ["HUANCABAMBA"], "TALARA": ["PARIÑAS"]}
CULT = {"ARROZ": (14010020000, 1.1), "MAIZ AMARILLO DURO": (14010070000, 0.9), "MANGO": (13010210000, 1.6),
        "LIMON": (13010170102, 2.0), "CEBOLLA": (14030100000, 1.2), "CAFETO": (13020030000, 8.5),
        "CACAO": (13020020000, 7.5), "PLATANO": (15010040000, 0.8), "YUCA": (14040140000, 0.9),
        "AJO": (14030030000, 6.0)}
filas = []
for prov, dists in PROV.items():
    for d in dists:
        for cn, (cod, base) in CULT.items():
            if rng.rand() < 0.35:
                continue
            p = base * rng.uniform(0.8, 1.3)
            for ano in range(2015, 2025):
                for mes in range(1, 13):
                    p = max(0.1, 0.85 * p + 0.15 * base * (1 + 0.15 * np.sin(2 * np.pi * mes / 12 + hash(cn) % 6))
                            + rng.normal(0, 0.05 * base))
                    siembra = rng.gamma(2, 50) if rng.rand() < .4 else 0.0
                    cosecha = rng.gamma(2, 40)
                    prod = cosecha * rng.uniform(1, 25)
                    precio = p if rng.rand() < 0.35 else (0.0 if rng.rand() < 0.9 else np.nan)
                    filas.append(["20250527", f"{ano}1231", "PIURA", prov, d, 200000, ano, ano * 100 + mes,
                                  cod, cn, siembra, cosecha, prod, cosecha, precio])
df = pd.DataFrame(filas, columns=["FECHA_CORTE", "FECHA_MUESTRA", "DEPARTAMENTO", "PROVINCIA", "DISTRITO", "UBIGEO",
                                  "ANO", "MES", "COD_CULTIVO", "CULTIVO", "SIEMBRA", "COSECHA", "PRODUCCION",
                                  "VERDE_ACTUAL", "PRECIO_CHACRA"])
df.loc[rng.rand(len(df)) < 0.03, "COSECHA"] = 0
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "data/dataset_sintetico.csv", index=False)
print(f"{len(df):,} filas sintéticas")
