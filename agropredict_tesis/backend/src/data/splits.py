"""splits.py — FASE 3: partición cronológica 70/15/15 en límites de período, con brecha para h>1,
y validación cruzada expansiva por MESES completos."""
import numpy as np
import pandas as pd

from src.config import FRAC_TRAIN, FRAC_VAL


def split_cronologico(df: pd.DataFrame, horizonte: int = 1):
    n = len(df)
    conteo = df.groupby("PERIODO").size().sort_index().cumsum()
    corte_tr = conteo[conteo >= FRAC_TRAIN * n].index[0]
    corte_va = conteo[conteo >= (FRAC_TRAIN + FRAC_VAL) * n].index[0]
    gap = max(int(horizonte) - 1, 0)
    mes, per = df["MES_IDX"].to_numpy(), df["PERIODO"].to_numpy()
    m_tr = int(df.loc[df["PERIODO"] == corte_tr, "MES_IDX"].iloc[0])
    m_va = int(df.loc[df["PERIODO"] == corte_va, "MES_IDX"].iloc[0])

    idx_tr_completo = np.where(per <= corte_tr)[0]
    idx_tr = np.where((per <= corte_tr) & (mes <= m_tr - gap))[0]      # entrena SOLO con etiquetas conocidas
    idx_va = np.where((per > corte_tr) & (per <= corte_va))[0]         # (al origen más temprano de validación)
    idx_te = np.where(per > corte_va)[0]
    idx_dev = np.where((per <= corte_va) & (mes <= m_va - gap))[0]     # train+val para el ajuste final / ablación
    info = {
        "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
        "train_hasta": int(corte_tr), "val_hasta": int(corte_va),
        "train_desde": int(df["PERIODO"].min()), "test_hasta": int(df["PERIODO"].max()),
        "horizonte": int(horizonte), "gap_meses": gap,
        "n_descartados_train_por_gap": int(len(idx_tr_completo) - len(idx_tr)),
        "idx_dev": idx_dev,
    }
    print(f"[Fase 3] Split cronológico (h={horizonte}, brecha={gap} m): train={len(idx_tr):,} (hasta {corte_tr}) | "
          f"val={len(idx_va):,} (hasta {corte_va}) | test={len(idx_te):,} | dev={len(idx_dev):,}")
    return idx_tr, idx_va, idx_te, info


class MesesExpansivo:
    """CV expansiva por MESES completos con brecha (reemplaza TimeSeriesSplit).
    Validación del fold k = meses en (e_k, e_{k+1}]; entrenamiento = meses <= e_k - gap.
    `mes_idx` va alineado con las filas que se pasarán a split()."""

    def __init__(self, mes_idx, n_splits=5, gap=0):
        self.mes = np.asarray(mes_idx)
        self.n_splits, self.gap = int(n_splits), int(gap)

    def get_n_splits(self, X=None, y=None, groups=None):
        return sum(1 for _ in self.split(X, y, groups))

    def split(self, X=None, y=None, groups=None):
        meses = np.unique(self.mes)
        conteo = np.array([(self.mes == m).sum() for m in meses]).cumsum()
        tot = conteo[-1]
        edges = [meses[min(int(np.searchsorted(conteo, tot * k / (self.n_splits + 1))), len(meses) - 1)]
                 for k in range(1, self.n_splits + 2)]
        for k in range(self.n_splits):
            lo, hi = edges[k], edges[k + 1]
            tr = np.where(self.mes <= lo - self.gap)[0]
            va = np.where((self.mes > lo) & (self.mes <= hi))[0]
            if len(tr) and len(va):
                yield tr, va