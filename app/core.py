"""Inference + explanation helpers for the Streamlit app (no Streamlit imports here, so it is unit-testable)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE.parent / "src"))
from pda_utils import transform_features          # same transforms as training: no drift possible

_DERIVED = {"q", "ad", "n", "per_y"}              # computed from a and e inside the app


def load_config():
    return json.loads((ASSETS / "config.json").read_text())


def raw_inputs_needed(features):
    """Which user-facing raw inputs a feature list requires (derived orbit quantities are computed, never asked)."""
    need = set()
    for f in features:
        if f.endswith("_tr"):
            need.add(f[:-3])
        elif f.endswith(("_sin", "_cos")):
            need.add(f.rsplit("_", 1)[0])
    if need & _DERIVED:
        need |= {"a", "e"}
    return need - _DERIVED


def derive_raw(inp: dict) -> pd.DataFrame:
    """User inputs -> the raw columns the training pipeline used. q, ad, per_y, n follow from a and e (Kepler)."""
    a, e = float(inp["a"]), float(inp["e"])
    per_y = a ** 1.5
    row = {"a": a, "e": e, "i": float(inp["i"]), "q": a * (1 - e), "ad": a * (1 + e),
           "per_y": per_y, "n": 360.0 / (per_y * 365.25)}
    for k in ("H", "albedo", "om", "w", "ma"):
        if k in inp and inp[k] is not None:
            row[k] = float(inp[k])
    return pd.DataFrame([row])


def make_X(raw, transforms, features):
    X = transform_features(raw, transforms)
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise KeyError(f"missing inputs for features: {missing}")
    return X[features]


# ---------------- SHAP ----------------
def unwrap(pipe, X):
    """Apply every pre-processing step (SMOTE only acts at fit time), return (final estimator, DataFrame)."""
    Xt = X
    for name, step in pipe.steps[:-1]:
        if name == "smote":
            continue
        Xt = step.transform(Xt)
    return pipe.steps[-1][1], pd.DataFrame(np.asarray(Xt), columns=X.columns, index=X.index)


def make_explainer(est, background):
    n = type(est).__name__
    if n in ("RandomForestClassifier", "RandomForestRegressor", "XGBClassifier", "XGBRegressor"):
        return shap.TreeExplainer(est)
    if n in ("LogisticRegression", "LinearRegression"):
        return shap.LinearExplainer(est, background)
    raise ValueError(f"No SHAP explainer wired for {n}")


def _pos(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def explain_row(explainer, est, Xt_row, classification, labels):
    """One-row shap.Explanation. For classifiers returns the positive class."""
    sv = _pos(explainer.shap_values(Xt_row)) if classification else np.asarray(explainer.shap_values(Xt_row))
    ev = np.asarray(explainer.expected_value).ravel()
    base = float(ev[-1] if (classification and ev.size > 1) else ev[0])
    return shap.Explanation(values=sv[0], base_values=base, data=Xt_row.iloc[0].values,
                            feature_names=[labels.get(c, c) for c in Xt_row.columns])


def feature_labels(transforms):
    pretty = {"log10": "log10({})", "sqrt": "sqrt({})", "cbrt": "cbrt({})", "none": "{}"}
    names = {"a": "a (AU)", "e": "e", "i": "i (deg)", "q": "q (AU)", "ad": "Q (AU)", "H": "H", "albedo": "albedo",
             "n": "n", "per_y": "period (y)"}
    out = {}
    for c, t in transforms.items():
        out[f"{c}_tr"] = pretty[t].format(names.get(c, c))
    for c in ("om", "w", "ma"):
        out[f"{c}_sin"] = f"sin({c})"
        out[f"{c}_cos"] = f"cos({c})"
    return out


# ---------------- Case 3 ----------------
class Clusterer:
    def __init__(self, km_bundle, db_bundle, transforms):
        self.km, self.db, self.transforms = km_bundle, db_bundle, transforms
        self.nn = NearestNeighbors(n_neighbors=1).fit(db_bundle["core_X"])

    def _scaled(self, raw, bundle):
        return bundle["scaler"].transform(make_X(raw, self.transforms, bundle["features"])).astype("float32")

    def assign(self, raw):
        km_id = int(self.km["kmeans"].predict(self._scaled(raw, self.km))[0])
        d, j = self.nn.kneighbors(self._scaled(raw, self.db))
        db_id = int(self.db["core_labels"][j[0, 0]]) if d[0, 0] <= self.db["eps"] else -1
        return km_id, db_id, float(d[0, 0])
