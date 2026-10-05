"""Planetary Defense Analytics: interactive asteroid classifier, diameter estimator and orbit-cluster finder.

Run locally:   streamlit run app/app.py        (from the project root, after `python app/build_assets.py`)
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

st.set_page_config(page_title="Planetary Defense Analytics", page_icon="☄️", layout="wide")

ASSETS = core.ASSETS
if not (ASSETS / "config.json").exists():
    st.error("`app/assets/` not found. Run `python app/build_assets.py` once (after notebooks 04-06), then reload.")
    st.stop()


# ---------------- cached resources ----------------
@st.cache_resource
def load_all():
    cfg = core.load_config()
    b1 = joblib.load(ASSETS / "case1_pha_best.joblib")
    b2 = joblib.load(ASSETS / "case2_diameter_best.joblib")
    km, db = joblib.load(ASSETS / "case3_kmeans.joblib"), joblib.load(ASSETS / "case3_dbscan.joblib")
    bg1 = pd.read_csv(ASSETS / "shap_bg_case1.csv.gz")
    bg2 = pd.read_csv(ASSETS / "shap_bg_case2.csv.gz")
    est1, Xb1 = core.unwrap(b1["pipeline"], bg1[b1["features"]])
    est2, Xb2 = core.unwrap(b2["pipeline"], bg2[b2["features"]])
    return dict(cfg=cfg, b1=b1, b2=b2, est1=est1, est2=est2,
                ex1=core.make_explainer(est1, Xb1), ex2=core.make_explainer(est2, Xb2),
                clusterer=core.Clusterer(km, db, cfg["transforms"]),
                bg=pd.read_csv(ASSETS / "cluster_background.csv.gz"),
                labels=core.feature_labels(cfg["transforms"]))


S = load_all()
cfg, b1, b2 = S["cfg"], S["b1"], S["b2"]
c1, c2, c3 = cfg["case1"], cfg["case2"], cfg["case3"]
rng_ = cfg["ranges"]

# ---------------- inputs ----------------
PRESETS = {  # approximate textbook values, for orientation only
    "Custom": None,
    "Typical main-belt asteroid": dict(a=2.55, e=0.15, i=5.0, H=16.5, albedo=0.15),
    "Apophis (approx.)": dict(a=0.922, e=0.191, i=3.34, H=19.7, albedo=0.23),
    "Bennu (approx.)": dict(a=1.126, e=0.204, i=6.03, H=20.2, albedo=0.044),
    "Eros (approx.)": dict(a=1.458, e=0.223, i=10.83, H=10.4, albedo=0.25),
    "Ceres (approx.)": dict(a=2.77, e=0.076, i=10.6, H=3.4, albedo=0.09),
    "Jupiter-Trojan-like": dict(a=5.2, e=0.07, i=10.0, H=11.0, albedo=0.07),
    "Kuiper-belt-like": dict(a=43.0, e=0.05, i=5.0, H=5.0, albedo=0.10),
}
DEFAULTS = dict(a=2.55, e=0.15, i=5.0, H=16.5, albedo=0.15, om=180.0, w=180.0, ma=180.0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    p = PRESETS[st.session_state["preset"]]
    if p:
        for k, v in p.items():
            st.session_state[k] = v


need = core.raw_inputs_needed(c1["features"]) | core.raw_inputs_needed(c2["features"]) | {"a", "e", "i"}

st.sidebar.title("☄️ Asteroid parameters")
st.sidebar.selectbox("Example object", list(PRESETS), key="preset", on_change=apply_preset)
st.sidebar.number_input("Semi-major axis a (AU)", min_value=0.3, max_value=1000.0, step=0.05, format="%.3f", key="a",
                        help="Average distance from the Sun. Earth = 1 AU.")
st.sidebar.number_input("Eccentricity e", min_value=0.0, max_value=0.99, step=0.01, format="%.3f", key="e",
                        help="0 = circle. Only bound orbits (e < 1) are supported.")
st.sidebar.number_input("Inclination i (deg)", min_value=0.0, max_value=180.0, step=0.5, format="%.2f", key="i")
if "H" in need:
    st.sidebar.number_input("Absolute magnitude H", min_value=-2.0, max_value=33.0, step=0.1, format="%.2f", key="H",
                            help="Brightness proxy for size: smaller H = brighter = larger.")
if "albedo" in need:
    st.sidebar.number_input("Geometric albedo", min_value=0.005, max_value=1.0, step=0.01, format="%.3f", key="albedo",
                            help="Surface reflectivity, 0-1.")
angles = [x for x in ("om", "w", "ma") if x in need]
if angles:
    with st.sidebar.expander("Orientation angles (minor model inputs)"):
        for x in angles:
            st.number_input(f"{x} (deg)", min_value=0.0, max_value=360.0, step=1.0, key=x)

inp = {k: st.session_state[k] for k in DEFAULTS if k in need or k in ("a", "e", "i")}
raw = core.derive_raw(inp)
q, ad = float(raw["q"].iloc[0]), float(raw["ad"].iloc[0])

warnings_ = []
for k in ("a", "i", "H", "albedo"):
    if k in inp and k in rng_ and not (rng_[k][0] <= inp[k] <= rng_[k][1]):
        warnings_.append(f"{k} = {inp[k]:g} is outside the range seen in training "
                         f"({rng_[k][0]:.3g} to {rng_[k][1]:.3g}, 0.01-99.99 percentile): predictions are extrapolations.")

st.title("Planetary Defense Analytics")
st.caption(f"Derived from your inputs: perihelion q = **{q:.3f} AU**, aphelion Q = **{ad:.3f} AU**, "
           f"period = **{raw['per_y'].iloc[0]:.2f} years**")
for w in warnings_:
    st.warning(w)

tab1, tab2, tab3, tab4 = st.tabs(["🛰️ Hazard (Case 1)", "📏 Diameter (Case 2)", "🧭 Orbit cluster (Case 3)", "ℹ️ About"])


def waterfall_fig(expl, title, max_display=10):
    plt.figure()
    shap.plots.waterfall(expl, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10, y=1.02)
    return fig


# ---------------- Case 1 ----------------
with tab1:
    X1 = core.make_X(raw, cfg["transforms"], c1["features"])
    score = float(b1["pipeline"].predict_proba(X1)[0, 1])

    choice = st.radio("Decision rule", ["Balanced (best F1)", "High recall (catch ≥ 95% of PHAs)", "Default (0.5)"],
                      horizontal=True)
    key = {"Balanced (best F1)": "f1_optimal", "High recall (catch ≥ 95% of PHAs)": "recall_95", "Default (0.5)": "default"}[choice]
    op = c1["op_points"][key]
    is_pha = score >= op["threshold"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Classification", "⚠️ Potentially hazardous" if is_pha else "✅ Not flagged")
    m2.metric("Model score", f"{score * 100:.1f}%", help="Output of a model trained with class balancing. It ranks risk; it is NOT a calibrated probability.")
    m3.metric("Decision threshold", f"{op['threshold']:.3f}")
    st.info(f"**What this rule means (held-out test set):** it finds **{op['recall'] * 100:.0f}%** of true PHAs, with a precision of "
            f"**{op['precision'] * 100:.0f}%**, i.e. about **{op['false_alarms_per_hit']:.1f} false alarms per real PHA**. "
            f"Only {c1['test_prevalence'] * 100:.2f}% of catalogued asteroids are PHAs, so a flag means *schedule a follow-up*, not *this is dangerous*.")
    st.caption("Official definition: MOID ≤ 0.05 AU **and** H ≤ 22. The model never sees MOID (it would leak the label); "
               "it estimates hazard from the orbit's shape and the object's brightness.")

    st.subheader("Why did the model say that? (SHAP)")
    Xt1 = core.unwrap(b1["pipeline"], X1)[1]
    e1 = core.explain_row(S["ex1"], S["est1"], Xt1, True, S["labels"])
    unit = "log-odds" if type(S["est1"]).__name__.startswith("XGB") else "probability"
    st.pyplot(waterfall_fig(e1, f"Contribution of each feature to the PHA score ({unit})"), clear_figure=True)
    st.caption("Red bars push toward *hazardous*, blue away from it. Values on the left are the features' transformed inputs.")

# ---------------- Case 2 ----------------
with tab2:
    X2 = core.make_X(raw, cfg["transforms"], c2["features"])
    pred_log = float(b2["pipeline"].predict(X2)[0])
    d_km = 10 ** pred_log
    q90 = c2["abs_log_err_q90"]
    lo, hi = d_km / 10 ** q90, d_km * 10 ** q90

    def fmt(x):
        return f"{x * 1000:.0f} m" if x < 1 else f"{x:.2f} km"

    m1, m2 = st.columns(2)
    m1.metric("Predicted diameter", fmt(d_km))
    m2.metric("Typical 90% range", f"{fmt(lo)} – {fmt(hi)}",
              help="Global error band measured on the test set (90% of test objects were predicted within this factor). It is not specific to this object.")
    if "albedo" in inp and "H" in inp:
        phys = 1329 / np.sqrt(inp["albedo"]) * 10 ** (-inp["H"] / 5)
        st.caption(f"Physics cross-check, D = 1329/√albedo · 10^(−H/5): **{fmt(phys)}**")

    st.subheader("Why this size? (SHAP)")
    Xt2 = core.unwrap(b2["pipeline"], X2)[1]
    e2 = core.explain_row(S["ex2"], S["est2"], Xt2, False, S["labels"])
    st.pyplot(waterfall_fig(e2, "Contribution of each feature to log10(diameter in km)"), clear_figure=True)
    st.caption(f"Baseline = typical asteroid in the data (≈ {10 ** e2.base_values:.1f} km). Red bars make the object larger, blue smaller.")

# ---------------- Case 3 ----------------
with tab3:
    km_id, db_id, dist = S["clusterer"].assign(raw)
    prof = {r["cluster"]: r for r in c3["profile"]}
    p = prof[km_id]
    m1, m2 = st.columns(2)
    m1.metric(f"K-Means cluster (k = {c3['k']})", f"Cluster {km_id}",
              help=f"Median a ≈ {p['a_AU_median']:.2f} AU, e ≈ {p['e_median']:.2f}, i ≈ {p['i_deg_median']:.1f}°")
    m2.metric("DBSCAN", f"Cluster {db_id}" if db_id >= 0 else "Noise / outlier",
              help="Nearest DBSCAN core sample, accepted only within eps. 'Noise' = lies in no dense region.")
    st.write(f"**Cluster {km_id}** holds {p['share_%']:.1f}% of all asteroids. Median orbit: a ≈ {p['a_AU_median']:.2f} AU, "
             f"e ≈ {p['e_median']:.2f}, i ≈ {p['i_deg_median']:.1f}°. Most common catalogue class: **{p['top_class']}** ({p['top_class_%']:.0f}%).")

    yaxis = st.radio("Plot", ["e", "i"], format_func=lambda x: {"e": "a vs eccentricity", "i": "a vs inclination"}[x], horizontal=True)
    bg = S["bg"]
    fig = go.Figure()
    for cid, g in bg.groupby("km"):
        fig.add_trace(go.Scattergl(x=g["a"], y=g[yaxis], mode="markers", name=f"cluster {cid}",
                                   marker=dict(size=3, opacity=0.45), legendgroup=str(cid)))
    fig.add_trace(go.Scatter(x=[inp["a"]], y=[inp[yaxis]], mode="markers", name="your asteroid",
                             marker=dict(symbol="star", size=20, color="red", line=dict(width=1.5, color="black"))))
    fig.update_xaxes(type="log", title="semi-major axis a (AU, log scale)")
    fig.update_yaxes(title={"e": "eccentricity", "i": "inclination (deg)"}[yaxis])
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    with st.expander("All K-Means clusters"):
        tbl = pd.DataFrame(c3["profile"]).set_index("cluster")[["n", "share_%", "a_AU_median", "e_median", "i_deg_median",
                                                                "top_class", "top_class_%", "PHA_rate_%", "NEO_rate_%"]]
        st.dataframe(tbl.style.apply(lambda r: ["background-color: #ffe9a8" if r.name == km_id else "" for _ in r], axis=1))

# ---------------- About ----------------
with tab4:
    st.markdown(f"""
**What this is.** Three models trained on the NASA/JPL Small-Body Database export (~958k asteroids).

| Task | Model | Held-out test performance |
|---|---|---|
| Hazard classification | {c1['model']} · scenario `{c1['scenario']}` · imbalance: `{c1['imbalance']}` | PR-AUC **{c1['PR_AUC_test']:.3f}** (random baseline = {c1['test_prevalence']:.4f}), ROC-AUC {c1['ROC_AUC_test']:.3f} |
| Diameter | {c2['model']} · scenario `{c2['scenario']}` | R² (log10) **{c2['R2_log_test']:.3f}**, median error **{c2['median_rel_err_pct_test']:.1f}%**, {c2['within_20pct_test']:.0f}% within ±20% |
| Orbit clusters | K-Means (k = {c3['k']}) + DBSCAN | unsupervised, interpreted against catalogue classes |

**Limits worth knowing**
* The PHA model deliberately excludes `MOID` (it defines the label). Scores are rankings, not probabilities.
* Diameter bands are global, not per-object. Objects far from the training distribution (very large, very fast, very high-albedo) are extrapolations.
* Cluster ids come from unsupervised fits; they are statistical groupings, not official orbital families.
* Example objects use rounded values for illustration only.
""")
    st.caption(f"Built with scikit-learn {cfg['versions']['scikit-learn']}, XGBoost {cfg['versions']['xgboost']}, SHAP {cfg['versions']['shap']}.")
