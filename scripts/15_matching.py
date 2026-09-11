"""v4 step 15: does the inverse association between disengagement and acute care use survive
comparison of similar-risk patients?

Exposure: disengagement in days 1 to 90 after enrolment. Outcomes: days 91 to 270, so an acute event
cannot trigger the contact that defines the exposure. Specifications: (1) stratified within acute care
risk decile; (2) exact match on risk decile plus propensity nearest neighbour within caliper;
(3) coarsened exact matching; (4) overlap weights. Negative controls: pre-enrolment cost and
utilization, and dialysis spending."""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED = 20260911
rng = np.random.default_rng(SEED)
out = {}

A = pd.read_parquet(D/"v4_outcomes_A.parquet")
A = A[A.eligible_E2 & A.covered_210].copy()
A["risk_pct"] = pd.to_numeric(A.risk_percentile, errors="coerce")
A = A[A.risk_pct.notna()].copy()
A["util_ok"] = (pd.to_datetime(A.enroll_date) + pd.Timedelta(days=270)) <= pd.Timestamp("2026-06-07")
A["cost_ok"] = A.cost_window_complete_91_270
num = lambda s: pd.to_numeric(s, errors="coerce")
COV = ["age", "risk_pct", "any_bh", "sud", "diabetes", "htn", "chf", "copd", "high_ed_ip", "polypharmacy", "psychosis", "prenatal"]
for c in COV: A[c] = num(A[c])
A["log_pre_cost"] = np.log1p(num(A.total_paid_pre365).clip(lower=0))
A["pre_ed"] = num(A.ed_pre365); A["pre_ip"] = num(A.ip_pre365)
A["risk_decile"] = pd.qcut(A.risk_pct.rank(method="first"), 10, labels=False)
FEAT = COV + ["log_pre_cost", "pre_ed", "pre_ip"]
X = A[FEAT].fillna(A[FEAT].median())
X = pd.concat([X, pd.get_dummies(A.state, prefix="st", drop_first=True).astype(float),
               pd.get_dummies(A.gender.fillna("U"), prefix="sx", drop_first=True).astype(float)], axis=1).astype(float)
T = A.E2_disengaged.to_numpy()
ps = LogisticRegression(max_iter=4000, C=1.0).fit(StandardScaler().fit_transform(X), T).predict_proba(StandardScaler().fit_transform(X))[:, 1]
A["ps"] = ps; A["lps"] = np.log(np.clip(ps, 1e-6, 1-1e-6)/(1-np.clip(ps, 1e-6, 1-1e-6)))
out["n"] = int(len(A)); out["exposed"] = int(T.sum()); out["ps_range"] = [round(float(ps.min()), 3), round(float(ps.max()), 3)]

def smd(x, t, w=None):
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    m1, m0 = np.average(x[t == 1], weights=w[t == 1]), np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1]-m1)**2, weights=w[t == 1]); v0 = np.average((x[t == 0]-m0)**2, weights=w[t == 0])
    s = np.sqrt((v1+v0)/2)
    return 0.0 if s == 0 else float((m1-m0)/s)
def balance(df, w=None):
    t = df.E2_disengaged.to_numpy()
    cols = ["risk_pct", "age", "log_pre_cost", "pre_ed", "pre_ip", "any_bh", "sud", "high_ed_ip"]
    d = {c: round(abs(smd(df[c].to_numpy(float), t, w)), 3) for c in cols}
    return {"max_abs_smd": max(d.values()), "by_covariate": d}

OUTCOMES = [("ed_91_270", "count", "util_ok"), ("ip_91_270", "count", "util_ok"), ("total_paid_91_270", "cost", "cost_ok"),
            ("ed_pre365", "count_negative_control", "util_ok"), ("total_paid_pre180", "cost_negative_control", "cost_ok"),
            ("negctrl_paid_1_180", "cost_negative_control_dialysis", "cost_ok")]

def estimate(df, w=None, label=""):
    res = {}
    for yname, kind, okcol in OUTCOMES:
        d = df[df[okcol]].copy()
        ww = np.ones(int(df[okcol].sum())) if w is None else np.asarray(w, float)[df[okcol].to_numpy()]
        if len(d) < 100 or d.E2_disengaged.nunique() < 2: continue
        Z = sm.add_constant(d[["E2_disengaged"]].astype(float).to_numpy(), has_constant="add")
        y = num(d[yname]).fillna(0).to_numpy(float)
        try:
            if kind.startswith("count"):
                m = sm.GLM(y, Z, family=sm.families.Poisson(), freq_weights=ww).fit(cov_type="HC0")
            else:
                m = sm.WLS(np.log1p(np.maximum(y, 0)), Z, weights=ww).fit(cov_type="HC1")
            b, se = float(m.params[1]), float(np.sqrt(np.diag(m.cov_params())[1]))
            res[yname] = {"kind": kind, "ratio": round(float(np.exp(b)), 3),
                          "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)],
                          "n": int(len(d)), "mean_exposed": round(float(y[d.E2_disengaged == 1].mean()), 1),
                          "mean_unexposed": round(float(y[d.E2_disengaged == 0].mean()), 1)}
        except Exception as e:
            res[yname] = {"error": str(e)[:80]}
    return res

# ---- 1. unadjusted and covariate-adjusted (reference) -------------------------------------------
out["unadjusted"] = {"balance": balance(A), "estimates": estimate(A)}

# ---- 2. stratified within acute care risk decile ---------------------------------------------------
strat = {}
for dec, g in A.groupby("risk_decile"):
    if g.E2_disengaged.nunique() < 2 or len(g) < 150: continue
    e = estimate(g)
    strat[f"decile_{int(dec)+1}"] = {"n": int(len(g)), "exposed": int(g.E2_disengaged.sum()),
                                     "mean_risk_pct": round(float(g.risk_pct.mean()), 1),
                                     "ed_ratio": e.get("ed_91_270", {}).get("ratio"),
                                     "cost_ratio": e.get("total_paid_91_270", {}).get("ratio"),
                                     "mean_cost_exposed": e.get("total_paid_91_270", {}).get("mean_exposed"),
                                     "mean_cost_unexposed": e.get("total_paid_91_270", {}).get("mean_unexposed")}
out["within_risk_decile"] = strat
# pooled within-decile estimate (decile fixed effects)
pooled = {}
for yname, kind, okcol in OUTCOMES:
    d = A[A[okcol]].copy()
    Z = pd.concat([d[["E2_disengaged"]].astype(float).reset_index(drop=True),
                   pd.get_dummies(d.risk_decile, prefix="dec", drop_first=True).astype(float).reset_index(drop=True)], axis=1)
    Z = sm.add_constant(Z, has_constant="add")
    y = num(d[yname]).fillna(0).to_numpy(float)
    try:
        m = (sm.GLM(y, Z.to_numpy(), family=sm.families.Poisson()).fit(cov_type="HC0") if kind.startswith("count")
             else sm.WLS(np.log1p(np.maximum(y, 0)), Z.to_numpy()).fit(cov_type="HC1"))
        b, se = float(m.params[1]), float(np.sqrt(np.diag(m.cov_params())[1]))
        pooled[yname] = {"ratio": round(float(np.exp(b)), 3), "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)], "n": int(len(d))}
    except Exception as e:
        pooled[yname] = {"error": str(e)[:60]}
out["pooled_with_risk_decile_fixed_effects"] = pooled

# ---- 3. exact match on risk decile + propensity nearest neighbour within caliper --------------------
cal = 0.2*np.std(A.lps)
pairs = []
for dec, g in A.groupby("risk_decile"):
    tr = g[g.E2_disengaged == 1].sample(frac=1, random_state=SEED)
    ct = g[g.E2_disengaged == 0].copy()
    used = set()
    ctl = ct.lps.to_numpy(); ctidx = ct.index.to_numpy(); order = np.argsort(ctl)
    ctl_s, ctidx_s = ctl[order], ctidx[order]
    avail = np.ones(len(ctl_s), bool)
    for i, row in tr.iterrows():
        j = np.searchsorted(ctl_s, row.lps)
        best, bestd = -1, np.inf
        for k in range(max(j-40, 0), min(j+40, len(ctl_s))):
            if not avail[k]: continue
            dd = abs(ctl_s[k]-row.lps)
            if dd < bestd: best, bestd = k, dd
        if best >= 0 and bestd <= cal:
            avail[best] = False; pairs.append((i, ctidx_s[best]))
M = A.loc[[i for p in pairs for i in p]].copy()
M["pair"] = [k for k in range(len(pairs)) for _ in (0, 1)]
out["propensity_matched"] = {"pairs": len(pairs), "matched_n": int(len(M)), "caliper_logit_ps": round(float(cal), 3),
                             "balance": balance(M), "estimates": estimate(M)}

# ---- 4. coarsened exact matching ----------------------------------------------------------------------
A["age_band"] = pd.cut(A.age, [0, 34, 49, 64, 200], labels=["18-34", "35-49", "50-64", "65+"])
A["cost_q"] = pd.qcut(A.log_pre_cost.rank(method="first"), 4, labels=False)
A["cem_cell"] = (A.risk_decile.astype(str) + "|" + A.age_band.astype(str) + "|" + A.cost_q.astype(str) + "|" +
                 A.state.astype(str) + "|" + A.any_bh.fillna(0).astype(int).astype(str))
cells = A.groupby("cem_cell").E2_disengaged.agg(["size", "sum"])
good = cells[(cells["sum"] > 0) & (cells["size"] - cells["sum"] > 0)].index
CEM = A[A.cem_cell.isin(good)].copy()
w = np.ones(len(CEM))
for cell, g in CEM.groupby("cem_cell"):                    # CEM weights: reweight controls to treated
    n1, n0 = int(g.E2_disengaged.sum()), int((1-g.E2_disengaged).sum())
    w[(CEM.cem_cell == cell).to_numpy() & (CEM.E2_disengaged == 0).to_numpy()] = n1/max(n0, 1)
out["coarsened_exact_matching"] = {"cells_retained": int(len(good)), "n": int(len(CEM)),
                                   "share_of_cohort": round(float(len(CEM)/len(A)), 3),
                                   "balance": balance(CEM, w), "estimates": estimate(CEM, w)}

# ---- 5. overlap weights --------------------------------------------------------------------------------
ow = np.where(T == 1, 1-A.ps, A.ps)
out["overlap_weights"] = {"effective_sample_size": round(float(ow.sum()**2/np.sum(ow**2)), 0),
                          "balance": balance(A, ow), "estimates": estimate(A, ow)}

json.dump(out, open(R/"matching_v4.json", "w"), indent=1, default=str)
print(json.dumps({k: (v if k in ("n", "exposed", "ps_range") else
                      {kk: vv for kk, vv in v.items() if kk != "by_covariate"}) for k, v in out.items()}, indent=1, default=str)[:4000])
