"""Examiner design: is the assigned staff member a usable instrument for sustained engagement,
once we condition on the cell within which assignment happens (market, discipline, quarter)?

Outcomes are taken from claims, which exist whether or not the care team wrote anything down."""
import json, pathlib, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
num = lambda s: pd.to_numeric(s, errors="coerce")
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
out = {}

A = pd.read_parquet(D/"v4_outcomes_A.parquet"); A = A[A.eligible_E2].copy()
A["enroll_date"] = pd.to_datetime(A.enroll_date)
enc = pd.read_parquet(D/"v4_encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date)
comp = enc[enc.occurred == "YES"]
first = comp.sort_values("enc_date").groupby("person_id").first().reset_index()[["person_id", "created_by_id", "roles"]]
first.columns = ["person_id", "staff", "roles"]
P = A.merge(first, on="person_id", how="inner")
P["role"] = np.where(P.roles.astype(str).str.contains("CHW", case=False, na=False), "CHW",
             np.where(P.roles.astype(str).str.contains("PHARM", case=False, na=False), "Pharmacy",
             np.where(P.roles.astype(str).str.contains("THERAP|LCSW|BHS", case=False, na=False), "Therapy", "CC")))
P["quarter"] = P.enroll_date.dt.to_period("Q").astype(str)
P["cell"] = P.state + "|" + P.role + "|" + P.quarter
# keep cells with at least two staff and enough patients, which is where assignment could have gone either way
ok = P.groupby("cell").agg(n=("person_id", "size"), k=("staff", "nunique"))
cells = ok[(ok.n >= 30) & (ok.k >= 3)].index
Q = P[P.cell.isin(cells)].copy()
sc = Q.groupby("staff").size()
Q = Q[Q.staff.isin(sc[sc >= 15].index)].copy()
gg = Q.groupby("staff").E2_disengaged.agg(["mean", "size"])
Q["Z"] = [(gg.loc[s, "mean"]*gg.loc[s, "size"] - y)/(gg.loc[s, "size"]-1) for s, y in zip(Q.staff, Q.E2_disengaged)]
out["sample"] = {"patients": int(len(Q)), "staff": int(Q.staff.nunique()), "cells": int(Q.cell.nunique()),
                 "disengagement_rate": round(float(Q.E2_disengaged.mean()), 3)}

CELL = pd.get_dummies(Q.cell, drop_first=True).astype(float).to_numpy()
def demean(v):
    """residualize on cell fixed effects so the instrument is only within-cell variation"""
    X = sm.add_constant(CELL)
    return np.asarray(v, float) - sm.OLS(np.asarray(v, float), X).fit().predict(X)
Zr = demean(Q.Z.to_numpy(float))
Tr = demean(Q.E2_disengaged.to_numpy(float))

# balance within cells
bal = {}
for c, lbl in [("age", "age"), ("risk_percentile", "acute care risk percentile"), ("any_bh", "behavioural health"),
               ("total_paid_pre365", "prior-year spending"), ("ed_pre365", "prior-year emergency visits")]:
    y = num(Q[c]); y = y.fillna(y.median()).to_numpy(float)
    mb = sm.OLS(demean(y), sm.add_constant(Zr)).fit(cov_type="HC1")
    sd = y.std() or 1.0
    bal[lbl] = {"coef_per_unit_instrument": round(float(mb.params[1]), 3),
                "standardized": round(float(mb.params[1]/sd), 3), "p": round(float(mb.pvalues[1]), 3)}
out["within_cell_balance"] = bal
out["balance_passes"] = all(abs(v["standardized"]) < 0.10 and v["p"] > 0.05 for v in bal.values())

fs = sm.OLS(Tr, sm.add_constant(Zr)).fit(cov_type="HC1")
out["first_stage"] = {"coefficient": round(float(fs.params[1]), 3), "F": round(float(fs.tvalues[1]**2), 1)}

# claims-based outcomes, measured over days 1 to 180 after enrolment
med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["claim_start_date"] = pd.to_datetime(med.claim_start_date)
ph = pd.read_parquet(D/"v4_pharmacy_claim_lines.parquet"); ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date)
et = med.encounter_type.astype(str).str.lower()
streams = {"office_visits": med[et.str.contains("office visit")], "ed_visits": med[et.str.contains("emergency")],
           "pharmacy_fills": ph}
idx = {}
for k, f in streams.items():
    idx[k] = {p: np.sort(day(g.claim_start_date)) for p, g in f.groupby("person_id")}
d0 = day(Q.enroll_date)
for k in streams:
    Q[k] = [int(np.searchsorted(idx[k].get(p, np.array([])), d + 180, "right") -
                np.searchsorted(idx[k].get(p, np.array([])), d + 1, "left")) for p, d in zip(Q.person_id, d0)]
res = {}
for k in streams:
    y = demean(Q[k].to_numpy(float))
    rf = sm.OLS(y, sm.add_constant(Zr)).fit(cov_type="HC1")
    iv = float(rf.params[1]/fs.params[1]) if fs.params[1] else np.nan
    se = float(rf.bse[1]/abs(fs.params[1])) if fs.params[1] else np.nan
    res[k] = {"mean_per_patient": round(float(Q[k].mean()), 2),
              "reduced_form_coef": round(float(rf.params[1]), 3), "reduced_form_p": round(float(rf.pvalues[1]), 3),
              "iv_effect_of_disengagement": round(iv, 3), "iv_ci_95": [round(iv-1.96*se, 3), round(iv+1.96*se, 3)]}
out["iv_estimates"] = res
json.dump(out, open(R/"examiner_design_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
