"""
data_loader.py — Carga y limpieza (sección 3.6.2 y 3.7.2.1.B de la tesis).

Instrumento: "Formato_dataset_productos_dra.csv" — 124,514 registros / 15
columnas, portal Datos Abiertos del Gobierno Regional de Piura.
"""
import pandas as pd

from src.config import DATA_PATH, TARGET, PROVINCIAS_PIURA


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga el CSV oficial y aplica la limpieza descrita en 3.7.2.1.B:
    - Filtra a provincias de la región Piura.
    - Imputa nulos de PRECIO_CHACRA con la mediana global.
    - Descarta registros inconsistentes (COSECHA=0 y PRODUCCION>0).
    - Descarta precios <= 0 y duplicados.

    Devuelve el DataFrame limpio y agrega logging de auditoría en consola
    para poder comparar contra los números que ya reportaste en el
    capítulo 4 (Anexo 1: ~116,000 registros población; cap.4: 14,393
    registros válidos en la muestra final de modelado).
    """
    df = pd.read_csv(path, low_memory=False)
    n_inicial = len(df)
    print(f"[data_loader] Registros iniciales: {n_inicial:,}")

    df = df[df["PROVINCIA"].isin(PROVINCIAS_PIURA)]
    print(f"[data_loader] Registros en provincias de Piura: {len(df):,}")

    df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    df = df.dropna(subset=["COSECHA", "PRODUCCION"])
    df = df[~((df["COSECHA"] == 0) & (df["PRODUCCION"] > 0))]
    df = df[df[TARGET] > 0]
    df = df.drop_duplicates()

    n_valido = len(df)
    print(f"[data_loader] Registros válidos finales: {n_valido:,}")
    print(f"[data_loader] NOTA: compara este número con el 14,393 del capítulo 4.4 "
          f"y con el 116,000 del Anexo 1 (población vs. muestra de modelado).")

    # ⚠️ CORRECCIÓN (antes: sort_values(["CULTIVO","PROVINCIA","ANO","MES"])):
    # ordenar por CULTIVO primero rompe la validación hold-out "cronológica"
    # de final_model.py: el split 80/20 por posición terminaba dejando en el
    # 20% de prueba a cultivos (Q-Z alfabéticamente: QUINUA, SANDIA, SOYA,
    # TOMATE, TRIGO, YUCA, ZAPALLO...) que el modelo NUNCA vio en
    # entrenamiento. Eso no es "no generaliza en el tiempo", es que nunca
    # aprendió esos cultivos. Ordenar por ANO, MES primero hace que el split
    # 80/20 sea realmente temporal (el groupby de feature_engineering para
    # PRECIO_LAG1/MEDIA_MOVIL_3 sigue siendo correcto: no depende del orden
    # global, solo del orden relativo dentro de cada grupo CULTIVO+PROVINCIA,
    # que se preserva porque ANO,MES son las claves de orden principales).
    return df.sort_values(["ANO", "MES", "CULTIVO", "PROVINCIA"]).reset_index(drop=True)
