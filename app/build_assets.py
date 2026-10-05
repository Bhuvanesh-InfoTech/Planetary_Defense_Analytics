"""Build the small, self-contained asset folder the Streamlit app needs.

Run ONCE from anywhere, after notebooks 04, 05 and 06 have produced the models:
    python app/build_assets.py

Output: app/assets/  (compressed models, config.json, a 25k-point cluster background, a SHAP background).
Commit app/assets/ to GitHub so Streamlit Cloud can load it. The raw data and notebooks' big files are NOT needed at runtime.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                   # pda_utils locates the project from the cwd
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import shap
from sklearn.metrics import precision_recall_curve

import pda_utils as U

ASSETS = ROOT / "app" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(U.SEED)

M = U.load_manifest()
b1 = joblib.load(U.MODELS / "case1_pha_best.joblib")
b2 = joblib.load(U.MODELS / "case2_diameter_best.joblib")
km = joblib.load(U.MODELS / "case3_kmeans.joblib")
db = joblib.load(U.MODELS / "case3_dbscan.joblib")

# ---------------- compressed model copies ----------------
for name, obj in [("case1_pha_best", b1), ("case2_diameter_best", b2), ("case3_kmeans", km), ("case3_dbscan", db)]:
    path = ASSETS / f"{name}.joblib"
    joblib.dump(obj, path, compress=3)
    mb = path.stat().st_size / 1e6
    flag = "   <-- TOO BIG for a normal GitHub push (limit 100 MB): reduce n_estimators/max_depth in the notebook, or use Git LFS" if mb > 90 else ""
    print(f"{path.name:28s} {mb:7.1f} MB{flag}")

# ---------------- Case 1: operating points on the TEST split ----------------
c1 = U.read_processed("case1", columns=["y_pha", "split"] + b1["features"])
t1 = c1[c1["split"] == "test"]
p1 = b1["pipeline"].predict_proba(t1[b1["features"]])[:, 1]
y1 = t1["y_pha"].values
op_points = {}
for key, thr in b1["thresholds"].items():
    pred = p1 >= thr
    tp = int((pred & (y1 == 1)).sum()); fp = int((pred & (y1 == 0)).sum()); fn = int((~pred & (y1 == 1)).sum())
    op_points[key] = dict(threshold=float(thr), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                          false_alarms_per_hit=fp / max(tp, 1), flagged_per_100k=float(pred.mean() * 1e5))
res1 = pd.read_csv(U.RES / "04_case1_results.csv")
row1 = res1[(res1.scenario == b1["scenario"]) & (res1.model == b1["model"]) & (res1.imbalance == b1["imbalance"])].iloc[0]

# ---------------- Case 2: global error band on TEST ----------------
c2 = U.read_processed("case2", columns=["log_diameter", "split", "albedo"] + b2["features"])
t2 = c2[(c2["split"] == "test") & c2["albedo"].notna()]
t2 = t2.sample(min(20_000, len(t2)), random_state=U.SEED)
err2 = np.abs(b2["pipeline"].predict(t2[b2["features"]]) - t2["log_diameter"].values)
res2 = pd.read_csv(U.RES / "05_case2_results.csv")
row2 = res2[(res2.scenario == b2["scenario"]) & (res2.model == b2["model"])].iloc[0]

# ---------------- 'seen in training' ranges (0.01 / 99.99 percentiles; rare NEOs/TNOs must NOT be flagged as out-of-range) ----------------
raw_cols = ["a", "e", "i", "H"]
r1 = U.read_processed("case1", columns=raw_cols)
ranges = {c: [float(r1[c].quantile(0.0001)), float(r1[c].quantile(0.9999))] for c in raw_cols}
alb = c2["albedo"].dropna()
ranges["albedo"] = [float(alb.quantile(0.0001)), float(alb.quantile(0.9999))]

# ---------------- Case 3 background + cluster descriptors ----------------
c3 = U.read_processed("case3", columns=["a", "e", "i"] + km["features"]).sample(25_000, random_state=U.SEED)
c3["km"] = km["kmeans"].predict(km["scaler"].transform(c3[km["features"]]).astype("float32"))
c3[["a", "e", "i", "km"]].round(4).to_csv(ASSETS / "cluster_background.csv.gz", index=False, compression="gzip")
prof = km["profile"].reset_index().rename(columns={"km": "cluster"})
profile_records = prof.to_dict(orient="records")

# ---------------- SHAP backgrounds (only used by linear models; harmless otherwise) ----------------
c1_bg = U.read_processed("case1", columns=b1["features"] + ["split"])
c1_bg[c1_bg["split"] == "train"].sample(300, random_state=U.SEED)[b1["features"]].to_csv(ASSETS / "shap_bg_case1.csv.gz", index=False, compression="gzip")
c2_bg = U.read_processed("case2", columns=b2["features"] + ["split"])
c2_bg[c2_bg["split"] == "train"].dropna(subset=b2["features"]).sample(300, random_state=U.SEED)[b2["features"]].to_csv(ASSETS / "shap_bg_case2.csv.gz", index=False, compression="gzip")

config = {
    "transforms": M["transforms"],
    "ranges": ranges,
    "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__},
    "case1": {"features": b1["features"], "scenario": b1["scenario"], "model": b1["model"], "imbalance": b1["imbalance"],
              "thresholds": b1["thresholds"], "op_points": op_points, "test_prevalence": b1["test_prevalence"],
              "PR_AUC_test": float(row1["PR_AUC_test"]), "ROC_AUC_test": float(row1["ROC_AUC_test"])},
    "case2": {"features": b2["features"], "scenario": b2["scenario"], "model": b2["model"],
              "R2_log_test": float(row2["R2_log_test"]), "RMSE_log_test": float(row2["RMSE_log_test"]),
              "median_rel_err_pct_test": float(row2["median_rel_err_pct_test"]), "within_20pct_test": float(row2["within_20pct_test"]),
              "abs_log_err_q50": float(np.quantile(err2, 0.50)), "abs_log_err_q90": float(np.quantile(err2, 0.90))},
    "case3": {"k": int(km["k"]), "features": km["features"], "eps": db["eps"], "min_samples": db["min_samples"],
              "profile": profile_records},
}
(ASSETS / "config.json").write_text(json.dumps(config, indent=2, default=float))
print("\nversions to pin in app/requirements.txt:", config["versions"])
print("assets written to", ASSETS)
