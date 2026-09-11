"""v4 step 8: Aim 2 - concordance of disengagement risk with acute care risk, and the association of
predicted and observed disengagement with utilization and cost (pre-registration Section 9.2)."""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
out = {}

A = pd.read_parquet(D/"v4_outcomes_A.parquet")
S = pd.read_parquet(D/"v4_score_all_A_full.parquet")[["person_id", "p_ensemble", "dev"]]
A = A.merge(S, on="person_id", how="inner")
A["risk_pct"] = pd.to_numeric(A.risk_percentile, errors="coerce")
A = A[A.risk_pct.notna()].copy()
out["n_with_score_and_risk"] = int(len(A))

# ---------------- 1. rank agreement ---------------------------------------------------------------
rho, p = spearmanr(A.p_ensemble, A.risk_pct)
def topfrac(x, f): 
    k = max(int(round(f*len(x))), 1); return set(np.argsort(-np.asarray(x))[:k])
agree = {}
for f, lbl in [(0.10, "top_decile"), (0.20, "top_quintile")]:
    a, b = topfrac(A.p_ensemble.to_numpy(), f), topfrac(A.risk_pct.to_numpy(), f)
    agree[lbl] = {"jaccard": round(len(a & b)/len(a | b), 3), "overlap_share": round(len(a & b)/len(a), 3)}
A["q_dis"] = pd.qcut(A.p_ensemble, 5, labels=False, duplicates="drop")
A["q_risk"] = pd.qcut(A.risk_pct.rank(method="first"), 5, labels=False, duplicates="drop")
out["rank_agreement"] = {"spearman_rho": round(float(rho), 3), "p": float(p), **agree,
                         "quintile_crosstab": pd.crosstab(A.q_dis, A.q_risk).to_dict()}

# each score's discrimination for the other's outcome
U = A[A.eligible_util].copy()
U["any_acute_180"] = ((U.ed_1_180 + U.ip_1_180) > 0).astype(int)
out["cross_discrimination"] = {
    "disengagement_score_for_acute_care": round(float(roc_auc_score(U.any_acute_180, U.p_ensemble)), 3),
    "risk_score_for_acute_care": round(float(roc_auc_score(U.any_acute_180, U.risk_pct)), 3),
    "disengagement_score_for_disengagement": round(float(roc_auc_score(A.E2_disengaged, A.p_ensemble)), 3),
    "risk_score_for_disengagement": round(float(roc_auc_score(A.E2_disengaged, A.risk_pct)), 3)}

# ---------------- regression helpers ---------------------------------------------------------------
COV = ["age", "risk_pct", "any_bh", "sud", "diabetes", "htn", "chf", "copd", "high_ed_ip", "polypharmacy"]
def design(df, extra):
    X = df[COV].apply(pd.to_numeric, errors="coerce").fillna(df[COV].apply(pd.to_numeric, errors="coerce").median())
    X = pd.concat([X, pd.get_dummies(df.state, prefix="st", drop_first=True).astype(float),
                   pd.get_dummies(df.gender.fillna("U"), prefix="sx", drop_first=True).astype(float),
                   np.log1p(df[["ed_pre365", "ip_pre365", "total_paid_pre365"]].astype(float))], axis=1)
    X = pd.concat([extra.reset_index(drop=True), X.reset_index(drop=True)], axis=1).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan); X = X.fillna(X.median()).fillna(0.0)
    X = X.loc[:, X.std() > 0]                      # drop constants
    keep, seen = [], None
    for c in X.columns:                            # drop columns that add no rank
        trial = X[keep + [c]].to_numpy()
        if np.linalg.matrix_rank(trial) == len(keep) + 1: keep.append(c)
    return sm.add_constant(X[keep], has_constant="add")

def nb_fit(df, yname, exposure, label):
    X = design(df, exposure)
    m = sm.NegativeBinomial(df[yname].astype(float).to_numpy(), X.to_numpy()).fit(disp=0, maxiter=200)
    k = list(X.columns).index(exposure.columns[0])
    se = float(np.sqrt(np.diag(m.cov_params())[k])); b = float(m.params[k])
    return {"outcome": yname, "exposure": label, "irr": round(float(np.exp(b)), 3),
            "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)], "p": round(float(2*(1-__import__("scipy.stats", fromlist=["norm"]).norm.cdf(abs(b/se)))), 4)}

def twopart(df, yname, exposure, label):
    X = design(df, exposure); yv = df[yname].astype(float).to_numpy()
    any_y = (yv > 0).astype(float)
    try:
        sm.Logit(any_y, X.to_numpy()).fit(disp=0, maxiter=200)
        degenerate = False
    except Exception:
        degenerate = True
    if degenerate or any_y.mean() > 0.95 or any_y.mean() < 0.05:
        ols = sm.OLS(np.log1p(np.maximum(yv, 0)), X.to_numpy()).fit(cov_type="HC1")
        k = list(X.columns).index(exposure.columns[0]); b = float(ols.params[k]); se = float(ols.bse[k])
        return {"outcome": yname, "exposure": label, "model": "log-linear (nearly all patients have spending)",
                "cost_ratio": round(float(np.exp(b)), 3), "cost_ratio_ci": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)]}
    lg = sm.Logit(any_y, X.to_numpy()).fit(disp=0, maxiter=200)
    k = list(X.columns).index(exposure.columns[0])
    pos = yv > 0
    gm = sm.GLM(yv[pos], X.to_numpy()[pos], family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    seg = float(np.sqrt(np.diag(gm.cov_params())[k])); bg = float(gm.params[k])
    sel = float(np.sqrt(np.diag(lg.cov_params())[k])); bl = float(lg.params[k])
    return {"outcome": yname, "exposure": label,
            "any_spend_or": round(float(np.exp(bl)), 3), "any_spend_ci": [round(float(np.exp(bl-1.96*sel)), 3), round(float(np.exp(bl+1.96*sel)), 3)],
            "cost_ratio_given_positive": round(float(np.exp(bg)), 3),
            "cost_ratio_ci": [round(float(np.exp(bg-1.96*seg)), 3), round(float(np.exp(bg+1.96*seg)), 3)]}

# ---------------- 2. predicted disengagement risk -> utilization and cost ---------------------------
U = A[A.eligible_util].copy(); C = A[A.eligible_cost].copy()
for df in (U, C):
    df["z_dis"] = (df.p_ensemble - df.p_ensemble.mean())/df.p_ensemble.std()
res2 = []
def safe(fn, *a):
    try: return fn(*a)
    except Exception as e: return {"outcome": a[1], "error": str(e)[:120]}
res2.append(safe(nb_fit, U, "ed_1_180", U[["z_dis"]], "predicted disengagement risk (per SD)"))
res2.append(safe(nb_fit, U, "ip_1_180", U[["z_dis"]], "predicted disengagement risk (per SD)"))
res2.append(safe(twopart, C, "total_paid_1_180", C[["z_dis"]], "predicted disengagement risk (per SD)"))
q = pd.get_dummies(pd.qcut(C.p_ensemble, 5, labels=[f"q{i+1}" for i in range(5)]), drop_first=True).astype(float)
Xq = design(C, q); yv = C.total_paid_1_180.astype(float).to_numpy()
ols = sm.OLS(np.log1p(np.maximum(yv, 0)), Xq.to_numpy()).fit(cov_type="HC1")
out["cost_by_disengagement_quintile"] = {c: {"log_cost_diff_vs_q1": round(float(ols.params[list(Xq.columns).index(c)]), 3),
                                             "ratio_vs_q1": round(float(np.exp(ols.params[list(Xq.columns).index(c)])), 3),
                                             "p": round(float(ols.pvalues[list(Xq.columns).index(c)]), 4)} for c in q.columns}
out["mean_cost_by_quintile"] = {f"q{i+1}": round(float(v), 0) for i, v in enumerate(C.groupby(pd.qcut(C.p_ensemble, 5, labels=False)).total_paid_1_180.mean())}
out["predicted_risk_associations"] = res2

# ---------------- 3. observed disengagement -> utilization and cost (IPTW, days 31-210) -------------
C2 = A[A.eligible_E2 & A.covered_210 & A.cost_window_complete_91_270].copy()
U2 = A[A.eligible_E2 & A.covered_210 & (pd.to_datetime(A.enroll_date) + pd.Timedelta(days=270) <= pd.Timestamp("2026-06-07"))].copy()
def iptw(df):
    X = design(df, pd.DataFrame({"_": np.arange(len(df)) % 2 + 0.0})).drop(columns=["_"], errors="ignore")
    ps = LogisticRegression(max_iter=2000, C=1.0).fit(X, df.E2_disengaged).predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.05, 0.95); t = df.E2_disengaged.to_numpy()
    w = t/ps + (1-t)/(1-ps)
    return w/np.mean(w)
res3 = []
for df, yname, kind in [(U2, "ed_91_270", "count"), (U2, "ip_91_270", "count"), (C2, "total_paid_91_270", "cost"),
                        (U2, "ed_31_210", "count_secondary_overlapping_window"), (C2, "total_paid_31_210", "cost_secondary_overlapping_window"),
                        (C2, "total_paid_pre180", "negative_control_pre_period"), (C2, "negctrl_paid_1_180", "negative_control_dialysis")]:
    d = df.copy(); d["w"] = iptw(d); X = design(d, d[["E2_disengaged"]])
    yv = d[yname].astype(float).to_numpy()
    if kind.startswith("count"):
        m = sm.GLM(yv, X.to_numpy(), family=sm.families.Poisson(), freq_weights=d.w.to_numpy()).fit(cov_type="HC0")
    else:
        # weighted log-linear model: the weighted gamma GLM did not converge on these data
        m = sm.WLS(np.log1p(np.maximum(yv, 0.0)), X.to_numpy(), weights=d.w.to_numpy()).fit(cov_type="HC1")
    k = list(X.columns).index("E2_disengaged"); b = float(m.params[k]); se = float(np.sqrt(np.diag(m.cov_params())[k]))
    res3.append({"outcome": yname, "kind": kind, "scale": "rate ratio" if kind == "count" else "ratio of (1 + cost)", "ratio": round(float(np.exp(b)), 3),
                 "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)], "n": int(len(d)),
                 "mean_disengaged": round(float(d.loc[d.E2_disengaged == 1, yname].mean()), 1),
                 "mean_engaged": round(float(d.loc[d.E2_disengaged == 0, yname].mean()), 1)})
out["observed_disengagement_associations"] = res3
json.dump(out, open(R/"concordance_v4.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ["rank_agreement", "cross_discrimination", "predicted_risk_associations",
                                      "mean_cost_by_quintile", "observed_disengagement_associations"] if k in out}, indent=1, default=str)[:3000])
