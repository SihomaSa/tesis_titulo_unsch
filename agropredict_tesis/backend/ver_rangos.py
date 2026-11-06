from src.config import N_ITER_TUNING, N_SPLITS_TUNING
from src.models.pipelines import NOMBRES, espacio_busqueda

def rango(d):
    if hasattr(d, "dist"):  # distribución de scipy.stats
        n, a = d.dist.name, d.args
        if n == "randint":
            return f"entero {a[0]}–{a[1]-1}"
        if n == "uniform":
            return f"continuo {a[0]}–{round(a[0]+a[1], 4)}"
        if n == "loguniform":
            return f"log-uniforme {a[0]}–{a[1]}"
        return f"{n}{a}"
    return "{" + ", ".join(str(x) for x in d) + "}"  # lista de valores

print(f"n_iter = {N_ITER_TUNING} | particiones = {N_SPLITS_TUNING}")
for clave in ["rf", "xgb", "lgbm", "lgbm_res"]:
    print(f"\n{NOMBRES[clave]}")
    for p, d in espacio_busqueda(clave).items():
        print(f"  {p.split('__')[-1]}: {rango(d)}")