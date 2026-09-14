"""Unfinished clinical work measured from claims rather than from our own care plan.

Every indicator here exists for a patient whether or not the care team wrote anything down, so it is
not a function of how much of the program the patient received. Measured at each patient's last
contact, and compared between patients who then disengaged and patients who stayed, unadjusted and
with overlap weights on baseline covariates."""
import json, pathlib, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
num = lambda s: pd.to_numeric(s, errors="coerce")
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
out = {}

B = pd.read_parquet(D/"v4_outcomes_B.parquet"); B["enc_date"] = pd.to_datetime(B.enc_date)
B = B[B.analysis].copy()
last = B.sort_values("enc_date").groupby("person_id").tail(1)[["person_id", "encounter_id", "enc_date", "E1", "state"]]
F = pd.read_parquet(D/"v4_featB.parquet")[["person_id", "encounter_id", "age", "risk_percentile", "any_bh", "sud",
                                           "diabetes", "htn", "copd", "chf", "n_contact_prior_all", "days_since_enroll",
                                           "paid_prior_365", "n_adt_365d"]]
L = last.merge(F, on=["person_id", "encounter_id"], how="left")
med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["claim_start_date"] = pd.to_datetime(med.claim_start_date)
ph = pd.read_parquet(D/"v4_pharmacy_claim_lines.parquet"); ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date)
et = med.encounter_type.astype(str).str.lower()
S = {"office": med[et.str.contains("office visit")], "ed": med[et.str.contains("emergency")],
     "inpatient": med[et.str.contains("acute inpatient")], "rx": ph}
idx = {k: {p: np.sort(day(g.claim_start_date)) for p, g in v.groupby("person_id")} for k, v in S.items()}
def cnt(k, pid, d0, lo, hi):
    a = idx[k].get(pid)
    return 0 if a is None else int(np.searchsorted(a, d0+hi, "right") - np.searchsorted(a, d0+lo, "left"))

d0 = day(L.enc_date)
L["no_pcp_12mo"] = [int(cnt("office", p, d, -365, 0) == 0) for p, d in zip(L.person_id, d0)]
L["rx_lapse"] = [int(cnt("rx", p, d, -365, -91) >= 2 and cnt("rx", p, d, -90, 0) == 0) for p, d in zip(L.person_id, d0)]
L["ed_without_followup"] = [int(cnt("ed", p, d, -90, 0) > 0 and cnt("office", p, d, -90, 0) == 0) for p, d in zip(L.person_id, d0)]
L["recent_admission"] = [int(cnt("inpatient", p, d, -90, 0) > 0) for p, d in zip(L.person_id, d0)]
L["any_claims_need"] = ((L.no_pcp_12mo + L.rx_lapse + L.ed_without_followup + L.recent_admission) > 0).astype(int)

COV = ["age", "risk_percentile", "any_bh", "sud", "diabetes", "htn", "copd", "chf",
       "n_contact_prior_all", "days_since_enroll", "paid_prior_365", "n_adt_365d"]
X = L[COV].apply(num); X = X.fillna(X.median())
X["log_paid"] = np.log1p(X.paid_prior_365.clip(lower=0)); X = X.drop(columns=["paid_prior_365"])
X = pd.concat([X, pd.get_dummies(L.state, drop_first=True).astype(float)], axis=1).astype(float)
T = L.E1.to_numpy()
Z = StandardScaler().fit_transform(X)
e = np.clip(LogisticRegression(max_iter=4000).fit(Z, T).predict_proba(Z)[:, 1], 1e-4, 1-1e-4)
w = np.where(T == 1, 1-e, e)
def smd(x, t, ww):
    m1, m0 = np.average(x[t == 1], weights=ww[t == 1]), np.average(x[t == 0], weights=ww[t == 0])
    v1 = np.average((x[t == 1]-m1)**2, weights=ww[t == 1]); v0 = np.average((x[t == 0]-m0)**2, weights=ww[t == 0])
    s = np.sqrt((v1+v0)/2)
    return 0.0 if s == 0 else float((m1-m0)/s)
out["overlap_balance_max_abs_smd"] = round(max(abs(smd(X[c].to_numpy(float), T, w)) for c in X.columns), 4)
out["n"] = int(len(L)); out["disengaged"] = int(T.sum())

MEAS = [("no_pcp_12mo", "no office visit in the prior 12 months"),
        ("rx_lapse", "stopped filling a medication they had been filling"),
        ("ed_without_followup", "emergency visit in the prior 90 days with no office visit"),
        ("recent_admission", "inpatient admission in the prior 90 days"),
        ("any_claims_need", "any of these")]
rows = []
for col, lbl in MEAS:
    y = L[col].to_numpy(float)
    m1, m0 = float(np.average(y[T == 1], weights=w[T == 1])), float(np.average(y[T == 0], weights=w[T == 0]))
    mod = sm.GLM(y, sm.add_constant(T.astype(float)), family=sm.families.Binomial(), freq_weights=w).fit(cov_type="HC0")
    b, se = float(mod.params[1]), float(mod.bse[1])
    rows.append({"measure": lbl, "disengaged_pct": round(100*float(y[T == 1].mean()), 1),
                 "sustained_pct": round(100*float(y[T == 0].mean()), 1),
                 "weighted_disengaged_pct": round(100*m1, 1), "weighted_sustained_pct": round(100*m0, 1),
                 "odds_ratio": round(float(np.exp(b)), 3),
                 "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)]})
out["measures"] = rows
# contrast with the care-plan measure, which does depend on documentation
N = pd.read_parquet(D/"v4_needs.parquet")[["person_id", "encounter_id", "any_open_need", "has_care_plan"]]
L2 = L.merge(N, on=["person_id", "encounter_id"], how="left")
out["care_plan_comparison"] = {
 "care_plan_open_need_disengaged_pct": round(100*float(L2.loc[L2.E1 == 1, "any_open_need"].mean()), 1),
 "care_plan_open_need_sustained_pct": round(100*float(L2.loc[L2.E1 == 0, "any_open_need"].mean()), 1),
 "claims_need_disengaged_pct": rows[-1]["disengaged_pct"], "claims_need_sustained_pct": rows[-1]["sustained_pct"]}
json.dump(out, open(R/"claims_based_need_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
