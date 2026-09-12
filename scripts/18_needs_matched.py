"""v4 step 18: open medical and social needs at the last contact, among comparable patients.

Everything used for matching is measured at or before the index contact, and disengagement is defined
only afterwards, so these are pre-exposure covariates rather than colliders. Three questions are
separated: (1) are the two groups different kinds of patients clinically, (2) did they have different
amounts of contact and therefore different opportunity for anything to be written down, and (3) among
patients with a care plan of comparable size, what share of it was still open."""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED = 20260912
out = {}

N = pd.read_parquet(D/"v4_needs.parquet")
B = pd.read_parquet(D/"v4_outcomes_B.parquet")
F = pd.read_parquet(D/"v4_featB.parquet")
keep = ["person_id", "encounter_id", "age", "risk_percentile", "any_bh", "sud", "diabetes", "htn", "chf", "copd",
        "high_ed_ip", "polypharmacy", "state", "gender", "days_since_enroll", "n_contact_90d", "n_contact_prior_all",
        "n_attempt_prior_all", "days_since_contact", "n_adt_365d", "paid_prior_365", "contact_index", "n_inperson_90d"]
M = N.merge(F[[c for c in keep if c in F.columns]], on=["person_id", "encounter_id"], how="left")
M = M.merge(B[["person_id", "encounter_id", "era"]], on=["person_id", "encounter_id"], how="left")
M["enc_date"] = pd.to_datetime(M.enc_date)
last = M.sort_values("enc_date").groupby("person_id").tail(1).copy()
num = lambda s: pd.to_numeric(s, errors="coerce")
for c in ["age", "risk_percentile", "any_bh", "sud", "diabetes", "htn", "chf", "copd", "high_ed_ip", "polypharmacy",
          "days_since_enroll", "n_contact_90d", "n_contact_prior_all", "n_attempt_prior_all", "days_since_contact",
          "n_adt_365d", "paid_prior_365", "contact_index", "n_inperson_90d", "n_goals_before"]:
    if c in last.columns: last[c] = num(last[c])
last["log_paid_prior"] = np.log1p(last.paid_prior_365.fillna(0).clip(lower=0))
last["risk_decile"] = pd.qcut(last.risk_percentile.rank(method="first"), 10, labels=False)
last["open_share_of_plan"] = np.where(last.n_goals_before > 0,
                                      (last.open_goal_medical + last.open_goal_social)/last.n_goals_before.replace(0, np.nan), np.nan)
T = last.E1.to_numpy()
out["n"] = int(len(last)); out["disengaged"] = int(T.sum())

CLIN = ["age", "risk_percentile", "any_bh", "sud", "diabetes", "htn", "chf", "copd", "high_ed_ip", "polypharmacy",
        "n_adt_365d", "log_paid_prior"]
DEPTH = ["days_since_enroll", "n_contact_90d", "n_contact_prior_all", "n_attempt_prior_all", "contact_index",
         "n_inperson_90d", "n_goals_before"]
def design(cols):
    X = last[cols].copy()
    X = X.fillna(X.median())
    X = pd.concat([X, pd.get_dummies(last.state, prefix="st", drop_first=True).astype(float).reset_index(drop=True).set_index(X.index),
                   pd.get_dummies(last.gender.fillna("U"), prefix="sx", drop_first=True).astype(float).reset_index(drop=True).set_index(X.index)], axis=1)
    return X.astype(float)

def smd(x, t, w=None):
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    m1, m0 = np.average(x[t == 1], weights=w[t == 1]), np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1]-m1)**2, weights=w[t == 1]); v0 = np.average((x[t == 0]-m0)**2, weights=w[t == 0])
    s = np.sqrt((v1+v0)/2)
    return 0.0 if s == 0 else float((m1-m0)/s)
def balance(X, t, w=None, show=("risk_percentile", "age", "log_paid_prior", "n_contact_prior_all", "n_goals_before", "days_since_enroll")):
    d = {c: round(abs(smd(X[c].to_numpy(float), t, w)), 3) for c in X.columns}
    return {"max_abs_smd": max(d.values()), "key_covariates": {c: d[c] for c in show if c in d}}

OUTCOMES = [("any_open_medical", "any open medical need"), ("any_open_social", "any open social need"),
            ("any_open_need", "any open need"), ("open_share_of_plan", "share of the care plan still open")]
def estimate(df, w=None, subset=None):
    res = []
    for col, lab in OUTCOMES:
        d = df if subset is None else df[subset]
        ww = None if w is None else np.asarray(w, float)[(subset.to_numpy() if subset is not None else slice(None))]
        y = num(d[col]).to_numpy(float); ok = ~np.isnan(y)
        t = d.E1.to_numpy()[ok]; y = y[ok]; ww2 = None if ww is None else np.asarray(ww)[ok]
        if len(y) < 100 or t.sum() < 25 or (1-t).sum() < 25: continue
        w1 = np.ones(len(y)) if ww2 is None else ww2
        m1 = float(np.average(y[t == 1], weights=w1[t == 1])); m0 = float(np.average(y[t == 0], weights=w1[t == 0]))
        Z = sm.add_constant(t.astype(float))
        if col == "open_share_of_plan":
            mod = sm.WLS(y, Z, weights=w1).fit(cov_type="HC1")
            b, se = float(mod.params[1]), float(mod.bse[1])
            eff = {"difference": round(b, 4), "ci_95": [round(b-1.96*se, 4), round(b+1.96*se, 4)]}
        else:
            mod = sm.GLM(y, Z, family=sm.families.Binomial(), freq_weights=w1).fit(cov_type="HC0")
            b, se = float(mod.params[1]), float(mod.bse[1])
            eff = {"odds_ratio": round(float(np.exp(b)), 3), "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)]}
        res.append({"measure": lab, "disengaged": round(100*m1, 1) if col != "open_share_of_plan" else round(m1, 3),
                    "sustained": round(100*m0, 1) if col != "open_share_of_plan" else round(m0, 3),
                    "n": int(len(y)), **eff})
    return res

# ---- 1. unadjusted -----------------------------------------------------------------------------
Xc, Xf = design(CLIN), design(CLIN + DEPTH)
out["unadjusted"] = {"balance_clinical": balance(Xc, T), "balance_full": balance(Xf, T), "estimates": estimate(last)}

# ---- 2. matched on clinical risk only -------------------------------------------------------------
def match(X, label):
    Z = StandardScaler().fit_transform(X)
    ps = LogisticRegression(max_iter=4000, C=1.0).fit(Z, T).predict_proba(Z)[:, 1]
    lps = np.log(np.clip(ps, 1e-6, 1-1e-6)/(1-np.clip(ps, 1e-6, 1-1e-6)))
    tmp = last.copy(); tmp["lps"] = lps
    cal = 0.2*np.std(lps)
    pairs = []
    for dec, g in tmp.groupby("risk_decile"):
        tr = g[g.E1 == 1].sample(frac=1, random_state=SEED); ct = g[g.E1 == 0]
        if not len(ct) or not len(tr): continue
        order = np.argsort(ct.lps.to_numpy()); cl, ci = ct.lps.to_numpy()[order], ct.index.to_numpy()[order]
        avail = np.ones(len(cl), bool)
        for i, row in tr.iterrows():
            j = np.searchsorted(cl, row.lps); best, bd = -1, np.inf
            for k in range(max(j-60, 0), min(j+60, len(cl))):
                if not avail[k]: continue
                dd = abs(cl[k]-row.lps)
                if dd < bd: best, bd = k, dd
            if best >= 0 and bd <= cal:
                avail[best] = False; pairs.append((i, ci[best]))
    idx = [i for p in pairs for i in p]
    Mt = last.loc[idx]
    Xm = X.loc[idx]
    return {"label": label, "pairs": len(pairs), "n": len(Mt), "balance": balance(Xm, Mt.E1.to_numpy()),
            "estimates": estimate(Mt)}, Mt
out["matched_clinical_risk_only"], _ = match(Xc, "acute care risk decile plus clinical propensity match")
out["matched_clinical_and_contact_depth"], Mdepth = match(Xf, "clinical match plus contact history and care-plan size")

# ---- 3. overlap weights on the full covariate set ---------------------------------------------------
Zf = StandardScaler().fit_transform(Xf)
e = np.clip(LogisticRegression(max_iter=4000, C=1.0).fit(Zf, T).predict_proba(Zf)[:, 1], 1e-4, 1-1e-4)
ow = np.where(T == 1, 1-e, e)
out["overlap_weights_full"] = {"effective_sample_size": round(float(ow.sum()**2/np.sum(ow**2)), 0),
                               "balance": balance(Xf, T, ow), "estimates": estimate(last, ow)}

# ---- 4. exact strata: patients with the same number of prior contacts -------------------------------
last["contact_bin"] = pd.cut(last.n_contact_prior_all, [-1, 0, 1, 2, 4, 8, 1e9], labels=["0", "1", "2", "3-4", "5-8", "9+"])
strat = {}
for b, g in last.groupby("contact_bin", observed=True):
    if len(g) < 150 or g.E1.nunique() < 2: continue
    est = estimate(g)
    row = {"n": int(len(g)), "disengaged": int(g.E1.sum())}
    for r in est:
        if r["measure"] == "any open need":
            row |= {"open_need_disengaged_pct": r["disengaged"], "open_need_sustained_pct": r["sustained"],
                    "odds_ratio": r.get("odds_ratio"), "ci_95": r.get("ci_95")}
        if r["measure"] == "share of the care plan still open":
            row |= {"open_share_disengaged": r["disengaged"], "open_share_sustained": r["sustained"],
                    "share_difference": r.get("difference")}
    strat[str(b)] = row
out["by_number_of_prior_contacts"] = strat

json.dump(out, open(R/"needs_matched_v4.json", "w"), indent=1, default=str)
for k in ["unadjusted", "matched_clinical_risk_only", "matched_clinical_and_contact_depth", "overlap_weights_full"]:
    v = out[k]
    print(f"\n=== {k}  n={v.get('n', len(last))} maxSMD={v['balance']['max_abs_smd'] if 'balance' in v else v['balance_full']['max_abs_smd']}")
    for r in v["estimates"]:
        eff = (f"OR {r['odds_ratio']:.3f} ({r['ci_95'][0]:.3f}-{r['ci_95'][1]:.3f})" if "odds_ratio" in r
               else f"diff {r['difference']:+.3f} ({r['ci_95'][0]:+.3f} to {r['ci_95'][1]:+.3f})")
        print(f"   {r['measure']:36s} dis {r['disengaged']} vs sus {r['sustained']}  {eff}")
print("\nby prior contacts:", json.dumps(out["by_number_of_prior_contacts"], indent=1, default=str)[:1200])
