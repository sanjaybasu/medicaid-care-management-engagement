"""v4 step 19: quantities a reviewer will ask for.

(1) how the excluded patients differ from the analysed cohort; (2) the range of the acute care risk
percentile in a rising-risk cohort, which bounds what that score can discriminate; (3) discrimination
by state, since the validation era is also a market-expansion era; (4) the minimum detectable
difference of the active-comparator contrasts."""
import json, pathlib, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
out = {}
num = lambda s: pd.to_numeric(s, errors="coerce")

enr = pd.read_parquet(D/"v4_enrollment.parquet"); enr["enroll_date"] = pd.to_datetime(enr.enroll_date)
enr = enr[(enr.enroll_date >= "2023-05-23") & (enr.enroll_date <= "2026-06-07")]
A = pd.read_parquet(D/"v4_outcomes_A.parquet")
panel = pd.read_parquet(D/"v4_panel.parquet")
inmart = set(panel.person_id)
enr["in_analysis"] = enr.person_id.isin(set(A[A.eligible_E2].person_id))
enr["has_mart_record"] = enr.person_id.isin(inmart)
enc = pd.read_parquet(D/"v4_encounters.parquet", columns=["person_id", "occurred"])
cc = enc[enc.occurred == "YES"].groupby("person_id").size()
enr["n_contacts"] = enr.person_id.map(cc).fillna(0)
enr["half"] = enr.enroll_date.dt.to_period("2Q").astype(str)
out["excluded_vs_included"] = {
 "enrolled_in_period": int(len(enr)), "in_analysis": int(enr.in_analysis.sum()),
 "no_mart_record": int((~enr.has_mart_record).sum()),
 "mean_completed_contacts_analysed": round(float(enr[enr.in_analysis].n_contacts.mean()), 2),
 "mean_completed_contacts_excluded": round(float(enr[~enr.in_analysis].n_contacts.mean()), 2),
 "share_with_no_contact_analysed": round(float((enr[enr.in_analysis].n_contacts == 0).mean()), 3),
 "share_with_no_contact_excluded": round(float((enr[~enr.in_analysis].n_contacts == 0).mean()), 3),
 "in_analysis_share_by_half_year": {k: round(float(v), 3) for k, v in enr.groupby("half").in_analysis.mean().items()}}

V = pd.read_parquet(D/"v4_valscores_B_full.parquet")
F = pd.read_parquet(D/"v4_featB.parquet")
V = V.merge(F[["person_id", "encounter_id", "risk_percentile", "state"]], on=["person_id", "encounter_id"], how="left")
rp = num(V.risk_percentile)
out["risk_score_range"] = {
 "median": float(rp.median()), "iqr": [float(rp.quantile(.25)), float(rp.quantile(.75))],
 "p05_p95": [float(rp.quantile(.05)), float(rp.quantile(.95))],
 "share_above_80th_percentile": round(float((rp >= 80).mean()), 3),
 "note": "the cohort is restricted to the rising-risk tier, so the acute care risk score varies over a truncated range"}
sc = np.load(R/"val_scores_B_full.npy"); V["p_ens"], V["risk01"], V["y"] = sc[:, 0], sc[:, 1], sc[:, 2]
out["discrimination_by_state"] = {}
for st, g in V.groupby("state"):
    if len(g) < 200 or g.y.nunique() < 2: continue
    out["discrimination_by_state"][st] = {"n": int(len(g)), "event_rate": round(float(g.y.mean()), 3),
                                          "auroc_ensemble": round(float(roc_auc_score(g.y, g.p_ens)), 3),
                                          "auroc_risk_score": round(float(roc_auc_score(g.y, g.risk01)), 3)}
ADV = json.load(open(R/"actions_advanced_v4.json"))
mde = {}
for k, v in ADV.items():
    if not isinstance(v, dict) or "overlap_weighted" not in v or not k.startswith("AC"): continue
    n1 = v["treated"]; n0 = v["n"] - v["treated"]; p = v["outcome_rate"]
    mde[k] = {"n": v["n"], "exposed": n1, "outcome_rate": p,
              "mde_risk_difference_80pct_power": round(float(2.8*np.sqrt(p*(1-p)*(1/max(n1, 1)+1/max(n0, 1)))), 4),
              "observed_rd": v["overlap_weighted"]["rd"], "ci_half_width": round((v["overlap_weighted"]["ci_95"][1]-v["overlap_weighted"]["ci_95"][0])/2, 4)}
out["active_comparator_power"] = mde
json.dump(out, open(R/"reviewer_checks_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
