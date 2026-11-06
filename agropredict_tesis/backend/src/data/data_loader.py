"""
data_loader.py — FASE 2 (comprensión de datos) y primera parte de la FASE 3
(limpieza). Devuelve el DataFrame limpio y la tabla del embudo de depuración
(124 514 → n final), que sustenta el protocolo descrito en 4.1.1.

Correcciones respecto a la versión anterior:
  * La columna MES del dataset oficial viene como AAAAMM (p. ej. 202002).
    Antes se usaba tal cual en sin/cos y ESTACION, lo que dejaba la
    dimensión temporal sin información (ESTACION=3 en todas las filas).
    Ahora se extrae el mes real (MES_NUM) y se construye PERIODO.
  * Ya NO se imputa la variable objetivo con la mediana: un precio inventado
    no es una observación. Los registros sin precio se descartan y se
    reportan en el embudo.
"""
import numpy as np
import pandas as pd

from src.config import DATA_PATH, TARGET, PROVINCIAS_PIURA, OUT_TAB
import os


def _mes_real(df: pd.DataFrame) -> pd.Series:
    """MES puede venir como 1-12 o como AAAAMM; devuelve siempre 1-12."""
    mes = pd.to_numeric(df["MES"], errors="coerce")
    return np.where(mes > 12, mes % 100, mes)


def load_and_clean(path: str = DATA_PATH):
    df = pd.read_csv(path, low_memory=False)
    embudo = [("Registros iniciales del dataset", len(df))]

    df = df[df["PROVINCIA"].astype(str).str.strip().str.upper().isin(PROVINCIAS_PIURA)].copy()
    df["PROVINCIA"] = df["PROVINCIA"].astype(str).str.strip().str.upper()
    embudo.append(("En provincias de la región Piura", len(df)))

    df = df[df[TARGET].notna()]
    embudo.append((f"Con {TARGET} informado (sin imputar el objetivo)", len(df)))

    df = df[df[TARGET] > 0]
    embudo.append((f"Con {TARGET} > 0", len(df)))

    df = df.dropna(subset=["COSECHA", "PRODUCCION", "SIEMBRA"])
    embudo.append(("Con SIEMBRA, COSECHA y PRODUCCION informadas", len(df)))

    df = df[~((df["COSECHA"] == 0) & (df["PRODUCCION"] > 0))]
    embudo.append(("Sin inconsistencias (COSECHA=0 y PRODUCCION>0)", len(df)))

    df = df.drop_duplicates()
    embudo.append(("Sin duplicados exactos (muestra final de modelado)", len(df)))

    df["MES_NUM"] = _mes_real(df).astype(int)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype(int)
    df["PERIODO"] = df["ANO"] * 100 + df["MES_NUM"]
    df["MES_IDX"] = df["ANO"] * 12 + df["MES_NUM"] - 1   # índice mensual continuo
    for c in ("DISTRITO", "CULTIVO"):
        df[c] = df[c].astype(str).str.strip()

    df = df.sort_values(["PERIODO", "CULTIVO", "PROVINCIA", "DISTRITO"]).reset_index(drop=True)

    tabla = pd.DataFrame(embudo, columns=["Etapa", "Registros"])
    tabla["Eliminados"] = (-tabla["Registros"].diff()).fillna(0).astype(int)
    tabla.to_csv(os.path.join(OUT_TAB, "tabla_embudo_depuracion.csv"), index=False)

    print("[Fase 2-3] Embudo de depuración:")
    for _, r in tabla.iterrows():
        print(f"   {r['Etapa']:<62} {int(r['Registros']):>9,}  (-{int(r['Eliminados']):,})")
    return df, tabla
