"""Shared helpers for notebooks 04-07 and the Streamlit app.

Put this file in  <project>/src/pda_utils.py
Notebooks add <project>/src to sys.path and `import pda_utils as U`.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# MLflow imports GitPython; without a git executable it prints a long warning for every run.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

SEED = 42


def find_root(start=None):
    """Project root. Prefers the location of this file (<project>/src/pda_utils.py), so the result
    does not depend on the working directory (notebooks/, app/, or anywhere else).
    Falls back to the working directory if the file is not inside a `src` folder."""
    if start is None:
        here = Path(__file__).resolve().parent
        if here.name == "src":
            return here.parent
        start = Path.cwd()
    p = Path(start).resolve()
    return p.parent if p.name == "notebooks" else p


ROOT = find_root()
PROC = ROOT / "data" / "processed"
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
for _d in (PROC, RES, FIG, MODELS):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:          # read-only filesystem (e.g. a hosted app): the app never writes here
        pass


def load_manifest():
    """Paths, transforms and feature sets written by notebook 03."""
    path = RES / "03_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebook 03 first.")
    return json.loads(path.read_text())


def read_processed(key, columns=None):
    """key: 'case1' | 'case2' | 'case3'."""
    name = load_manifest()["files"][key]
    path = PROC / name
    if name.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, compression="gzip", usecols=columns)


# Stateless transforms. Must stay identical to notebook 03.
STATELESS = {
    "none": lambda x: x,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "log10": lambda x: np.log10(x + 1e-6),
}


def transform_features(raw: pd.DataFrame, transforms=None) -> pd.DataFrame:
    """Raw orbital/physical columns (a, e, i, q, ad, H, albedo, om, w, ma ...) -> model-ready *_tr / sin / cos columns.
    Used by notebooks and by the Streamlit app so training and inference can never drift apart.
    Values outside a transform's domain (e.g. sqrt of a negative H) become NaN, never an error."""
    transforms = transforms or load_manifest()["transforms"]
    out = pd.DataFrame(index=raw.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        for c, t in transforms.items():
            if c in raw.columns:
                out[f"{c}_tr"] = STATELESS[t](raw[c].astype(float))
        for c in ["om", "w", "ma"]:
            if c in raw.columns:
                out[f"{c}_sin"] = np.sin(np.deg2rad(raw[c].astype(float)))
                out[f"{c}_cos"] = np.cos(np.deg2rad(raw[c].astype(float)))
    return out


def validate_inputs(inp: dict):
    """Physical range checks for one object's raw inputs. Returns a list of problems (empty = fine).
    Tree models accept NaN silently, so an out-of-range value would otherwise give a confident wrong answer.
    The limits match what notebook 03 kept (H in [0, 35], bound orbits only)."""
    errs = []

    def has(k):
        return k in inp and inp[k] is not None

    if has("a") and not inp["a"] > 0:
        errs.append("a (semi-major axis) must be > 0 AU")
    if has("e") and not 0 <= inp["e"] < 1:
        errs.append("e (eccentricity) must be in [0, 1): only bound orbits are supported")
    if has("i") and not 0 <= inp["i"] <= 180:
        errs.append("i (inclination) must be between 0 and 180 degrees")
    if has("H") and not 0 <= inp["H"] <= 35:
        errs.append("H (absolute magnitude) must be between 0 and 35")
    if has("albedo") and not 0 < inp["albedo"] <= 1:
        errs.append("albedo must be in (0, 1]")
    for k in ("om", "w", "ma"):
        if has(k) and not 0 <= inp[k] <= 360:
            errs.append(f"{k} must be between 0 and 360 degrees")
    return errs


_MLFLOW_OK = None


def mlflow_log(run_name, params, metrics, model=None, experiment="planetary-defense", log_model=True):
    """Log one run to a local MLflow store (<project>/mlflow.db, artifacts in <project>/mlruns).
    Silently skipped if mlflow is not installed. Tracking never breaks training."""
    global _MLFLOW_OK
    if _MLFLOW_OK is False:
        return
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("mlflow not installed -> experiment tracking skipped (pip install mlflow)")
        _MLFLOW_OK = False
        return
    _MLFLOW_OK = True

    try:
        db_path = ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        # Pin the artifact folder to the project root; otherwise it is created relative to the
        # current directory (notebooks/mlruns when run from a notebook).
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns").as_uri())
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and v == v})
            if model is not None and log_model:
                # MLflow's default format (skops) rejects Random Forest, XGBoost, SMOTE and numpy dtypes
                # as "untrusted types". cloudpickle stores them; only load models you trained yourself.
                try:
                    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
                except TypeError:      # older MLflow: the argument is called artifact_path
                    mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
    except Exception as e:             # tracking must never break training
        print("mlflow logging failed:", repr(e)[:400])