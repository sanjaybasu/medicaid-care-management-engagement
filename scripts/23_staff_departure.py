"""Staff departure as a natural experiment for engagement.

A care team member leaving is plausibly unrelated to any individual patient's clinical trajectory,
while it disrupts that patient's continuity. Patients whose assigned staff member left during their
first year are compared with patients whose staff member stayed, matched on market, quarter, and
baseline acuity; outcomes come from claims, which do not depend on our documentation."""
import json, pathlib, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
num = lambda s: pd.to_numeric(s, errors="coerce")
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
out = {}

enc = pd.read_parquet(D/"v4_encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date)
comp = enc[enc.occurred == "YES"].copy()
END = comp.enc_date.max()
span = comp.groupby("created_by_id").enc_date.agg(last="max", first="min", n="size")
departed = span[(span["last"] < END - pd.Timedelta(days=90)) & (span["n"] >= 100)]
out["departures"] = {"staff_departed": int(len(departed)), "data_end": str(END.date()),
                     "median_tenure_days": float((departed["last"] - departed["first"]).dt.days.median())}

A = pd.read_parquet(D/"v4_outcomes_A.parquet"); A = A[A.eligible_E2].copy()
A["enroll_date"] = pd.to_datetime(A.enroll_date)
early = comp.merge(A[["person_id", "enroll_date"]], on="person_id")
early = early[(early.enc_date >= early.enroll_date) & (early.enc_date <= early.enroll_date + pd.Timedelta(days=90))]
primary = (early.groupby(["person_id", "created_by_id"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).groupby("person_id").first().reset_index())
primary.columns = ["person_id", "staff", "n_contacts_with_staff"]
P = A.merge(primary, on="person_id", how="inner")
P["staff_last"] = P.staff.map(span["last"])
P["days_to_departure"] = (P.staff_last - P.enroll_date).dt.days
# exposed: the staff member left between day 30 and day 365 of the patient's episode
P["exposed"] = P.staff.isin(departed.index) & P.days_to_departure.between(91, 365)   # every patient gets a full first 90 days
P["control"] = ~P.staff.isin(departed.index)
S = P[P.exposed | P.control].copy()
S["Z"] = S.exposed.astype(int)
S["quarter"] = S.enroll_date.dt.to_period("Q").astype(str)
S["cell"] = S.state + "|" + S.quarter
ok = S.groupby("cell").Z.agg(["size", "sum"])
cells = ok[(ok["size"] >= 30) & (ok["sum"] >= 5) & (ok["size"] - ok["sum"] >= 5)].index
S = S[S.cell.isin(cells)].copy()
out["sample"] = {"patients": int(len(S)), "exposed": int(S.Z.sum()), "cells": int(S.cell.nunique())}

CELL = pd.get_dummies(S.cell, drop_first=True).astype(float).to_numpy()
def demean(v):
    X = sm.add_constant(CELL); v = np.asarray(v, float)
    return v - sm.OLS(v, X).fit().predict(X)
# condition on the first-90-day dose so exposure is not confounded with relationship length
S["dose"] = num(S.n_contacts_with_staff).fillna(0)
DOSE = np.column_stack([S.dose.to_numpy(float), (S.dose**2).to_numpy(float)])
CELL = np.column_stack([CELL, DOSE])
Zr = demean(S.Z.to_numpy(float))
bal = {}
for c, lbl in [("age", "age"), ("risk_percentile", "acute care risk percentile"), ("any_bh", "behavioural health"),
               ("total_paid_pre365", "prior-year spending"), ("ed_pre365", "prior-year emergency visits"),
               ("n_contacts_with_staff", "contacts before exposure")]:
    y = num(S[c]); y = y.fillna(y.median()).to_numpy(float)
    mb = sm.OLS(demean(y), sm.add_constant(Zr)).fit(cov_type="HC1")
    sd = y.std() or 1.0
    bal[lbl] = {"standardized": round(float(mb.params[1]/sd), 3), "p": round(float(mb.pvalues[1]), 3)}
out["balance"] = bal
out["balance_passes"] = all(abs(v["standardized"]) < 0.10 for v in bal.values())

fs = sm.OLS(demean(S.E2_disengaged.to_numpy(float)), sm.add_constant(Zr)).fit(cov_type="HC1")
out["first_stage_effect_on_disengagement"] = {"coefficient": round(float(fs.params[1]), 4),
                                              "ci_95": [round(float(fs.params[1]-1.96*fs.bse[1]), 4), round(float(fs.params[1]+1.96*fs.bse[1]), 4)],
                                              "p": round(float(fs.pvalues[1]), 4), "F": round(float(fs.tvalues[1]**2), 1)}
med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["claim_start_date"] = pd.to_datetime(med.claim_start_date)
ph = pd.read_parquet(D/"v4_pharmacy_claim_lines.parquet"); ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date)
et = med.encounter_type.astype(str).str.lower()
streams = {"office_visits": med[et.str.contains("office visit")], "ed_visits": med[et.str.contains("emergency")],
           "pharmacy_fills": ph}
idx = {k: {p: np.sort(day(g.claim_start_date)) for p, g in f.groupby("person_id")} for k, f in streams.items()}
d0 = day(S.enroll_date)
res = {}
for k in streams:
    S[k] = [int(np.searchsorted(idx[k].get(p, np.array([])), d + 365, "right") -
                np.searchsorted(idx[k].get(p, np.array([])), d + 91, "left")) for p, d in zip(S.person_id, d0)]
    rf = sm.OLS(demean(S[k].to_numpy(float)), sm.add_constant(Zr)).fit(cov_type="HC1")
    res[k] = {"mean_per_patient_days_91_365": round(float(S[k].mean()), 2),
              "reduced_form": round(float(rf.params[1]), 3), "p": round(float(rf.pvalues[1]), 3),
              "ci_95": [round(float(rf.params[1]-1.96*rf.bse[1]), 3), round(float(rf.params[1]+1.96*rf.bse[1]), 3)]}
out["reduced_form_on_claims"] = res
json.dump(out, open(R/"staff_departure_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
