"""tracking.py — Registro opcional en MLflow (no rompe el pipeline si falta)."""
import contextlib
import os

try:
    import mlflow
    _OK = True
except Exception:  # pragma: no cover
    mlflow = None
    _OK = False


def setup(experiment="AgroPredict_Piura_Tesis"):
    if not _OK:
        return
    from src.config import BACKEND_DIR
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(BACKEND_DIR, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment(experiment)


@contextlib.contextmanager
def run(name):
    if not _OK:
        yield
        return
    with mlflow.start_run(run_name=name):
        yield


def params(d):
    if _OK:
        mlflow.log_params({k: str(v)[:250] for k, v in d.items()})


def metrics(d):
    if _OK:
        mlflow.log_metrics({k: float(v) for k, v in d.items() if v == v})


def artifact(path):
    if _OK and os.path.exists(path):
        mlflow.log_artifact(path)
