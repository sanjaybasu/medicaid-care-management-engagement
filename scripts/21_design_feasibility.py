"""Feasibility of designs that could answer whether disengaged patients leave with unfinished work.

Checks four things: (1) how much variation assigned staff create in engagement, the first stage for an
examiner design; (2) whether caseload varies enough to serve as a capacity instrument; (3) whether
staff departures are frequent enough for a difference-in-differences; (4) whether claims-based need
measures exist for every patient regardless of what the care team documented."""
import json, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
out = {}
num = lambda s: pd.to_numeric(s, errors="coerce")

B = pd.read_parquet(D/"v4_outcomes_B.parquet"); B["enc_date"] = pd.to_datetime(B.enc_date)
B = B[B.analysis].copy()
enc = pd.read_parquet(D/"v4_encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date)
comp = enc[enc.occurred == "YES"].copy()

# ---- 1. examiner design: does the assigned staff member move engagement? -------------------------
first = comp.sort_values("enc_date").groupby("person_id").first().reset_index()[["person_id", "created_by_id", "roles", "enc_date"]]
first.columns = ["person_id", "staff", "roles", "first_contact"]
A = pd.read_parquet(D/"v4_outcomes_A.parquet"); A = A[A.eligible_E2].copy()
P = A[["person_id", "E2_disengaged", "state", "age", "risk_percentile", "any_bh", "enroll_date"]].merge(first, on="person_id", how="left")
P = P[P.staff.notna()].copy()
cnt = P.groupby("staff").size()
keep = cnt[cnt >= 20].index
Q = P[P.staff.isin(keep)].copy()
g = Q.groupby("staff").E2_disengaged.agg(["mean", "size"])
overall = float(Q.E2_disengaged.mean())
# leave-one-out staff rate, the standard examiner instrument
Q["loo"] = [(g.loc[s, "mean"]*g.loc[s, "size"] - y)/(g.loc[s, "size"]-1) for s, y in zip(Q.staff, Q.E2_disengaged)]
import statsmodels.api as sm
X = sm.add_constant(pd.concat([Q[["loo"]].reset_index(drop=True),
                               pd.get_dummies(Q.state, drop_first=True).astype(float).reset_index(drop=True),
                               num(Q.age).fillna(num(Q.age).median()).reset_index(drop=True),
                               num(Q.risk_percentile).fillna(0).reset_index(drop=True)], axis=1).astype(float))
m = sm.OLS(Q.E2_disengaged.to_numpy(float), X.to_numpy()).fit(cov_type="HC1")
F = float(m.tvalues[1]**2)
out["examiner_first_stage"] = {"staff_with_20plus_patients": int(len(keep)), "patients": int(len(Q)),
                               "disengagement_rate": round(overall, 3),
                               "staff_rate_p10_p90": [round(float(g["mean"].quantile(.1)), 3), round(float(g["mean"].quantile(.9)), 3)],
                               "sd_across_staff": round(float(g["mean"].std()), 3),
                               "leave_one_out_coefficient": round(float(m.params[1]), 3),
                               "first_stage_F": round(F, 1),
                               "usable": bool(F > 10)}
# balance: does the instrument predict baseline risk?
bal = {}
for c in ["age", "risk_percentile", "any_bh"]:
    y = num(Q[c]).fillna(num(Q[c]).median()).to_numpy(float)
    mb = sm.OLS(y, sm.add_constant(Q[["loo"]].to_numpy(float))).fit(cov_type="HC1")
    bal[c] = {"coef": round(float(mb.params[1]), 3), "p": round(float(mb.pvalues[1]), 3)}
out["examiner_balance_on_baseline"] = bal

# ---- 2. caseload as a capacity instrument -------------------------------------------------------
comp["month"] = comp.enc_date.dt.to_period("M")
load = comp.groupby(["created_by_id", "month"]).person_id.nunique().rename("panel").reset_index()
out["caseload_variation"] = {"staff_months": int(len(load)), "median_panel": float(load.panel.median()),
                             "p10_p90": [float(load.panel.quantile(.1)), float(load.panel.quantile(.9))],
                             "within_staff_sd": round(float(load.groupby("created_by_id").panel.std().median()), 2)}

# ---- 3. staff departures for a difference-in-differences ----------------------------------------
span = comp.groupby("created_by_id").enc_date.agg(["min", "max", "size"])
end = comp.enc_date.max()
dep = span[(span["max"] < end - pd.Timedelta(days=90)) & (span["size"] >= 100)]
out["staff_departures"] = {"staff_with_100plus_contacts": int((span["size"] >= 100).sum()),
                           "departed_at_least_90d_before_data_end": int(len(dep)),
                           "median_contacts_before_departure": float(dep["size"].median()) if len(dep) else None,
                           "patients_exposed": int(comp[comp.created_by_id.isin(dep.index)].person_id.nunique())}

# ---- 4. need measured from claims rather than from our own notes ---------------------------------
med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["claim_start_date"] = pd.to_datetime(med.claim_start_date)
ph = pd.read_parquet(D/"v4_pharmacy_claim_lines.parquet"); ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date)
et = med.encounter_type.astype(str).str.lower()
office = med[et.str.contains("office visit")]
last = B.sort_values("enc_date").groupby("person_id").tail(1)[["person_id", "enc_date", "E1"]]
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
def has_in(frame, col, lo, hi):
    by = {p: np.sort(day(g[col])) for p, g in frame.groupby("person_id")}
    def f(pid, d0):
        a = by.get(pid)
        return 0 if a is None else int(np.searchsorted(a, d0+hi, "right") - np.searchsorted(a, d0+lo, "left"))
    return f
fo, fp = has_in(office, "claim_start_date", 0, 0), has_in(ph, "claim_start_date", 0, 0)
d0s = day(last.enc_date)
last = last.assign(pcp_prior_365=[fo(p, d) for p, d in zip(last.person_id, d0s)])
out["claims_based_need_coverage"] = {
 "patients_at_last_contact": int(len(last)),
 "with_any_office_visit_claim_ever": int((last.pcp_prior_365 >= 0).sum()),
 "office_visit_claim_lines": int(len(office)), "pharmacy_claim_lines": int(len(ph)),
 "note": "claims exist for engaged and disengaged patients alike, so a need measure built from them is not a function of our documentation"}
json.dump(out, open(R/"design_feasibility_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
