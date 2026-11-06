"""splits.py — FASE 3: partición cronológica 70/15/15 en límites de período
(un mismo mes nunca queda repartido entre dos conjuntos)."""
import numpy as np
import pandas as pd

from src.config import FRAC_TRAIN, FRAC_VAL


def split_cronologico(df: pd.DataFrame):
    n = len(df)
    conteo = df.groupby("PERIODO").size().sort_index().cumsum()
    corte_tr = conteo[conteo >= FRAC_TRAIN * n].index[0]
    corte_va = conteo[conteo >= (FRAC_TRAIN + FRAC_VAL) * n].index[0]
    idx_tr = np.where(df["PERIODO"] <= corte_tr)[0]
    idx_va = np.where((df["PERIODO"] > corte_tr) & (df["PERIODO"] <= corte_va))[0]
    idx_te = np.where(df["PERIODO"] > corte_va)[0]
    info = {
        "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
        "train_hasta": int(corte_tr), "val_hasta": int(corte_va),
        "train_desde": int(df["PERIODO"].min()), "test_hasta": int(df["PERIODO"].max()),
    }
    print(f"[Fase 3] Split cronológico: train={len(idx_tr):,} (hasta {corte_tr}) | "
          f"val={len(idx_va):,} (hasta {corte_va}) | test={len(idx_te):,}")
    return idx_tr, idx_va, idx_te, info
