"""v4 step 2: outcomes E1-E4, program completion, and utilization/cost (pre-registration Section 5).

Writes data_cache/v4_outcomes_A.parquet, v4_outcomes_B.parquet and results/outcomes_v4.json."""
import json, pathlib
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
STUDY_END = pd.Timestamp("2026-06-07")
CLAIMS_COMPLETE = pd.Timestamp("2026-04-30")     # Section 6.3
out = {}

A = pd.read_parquet(D/"v4_taskA.parquet"); B = pd.read_parquet(D/"v4_taskB.parquet")
enc = pd.read_parquet(D/"v4_encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date)
comp = enc[(enc.occurred == "YES") & enc.person_id.isin(set(A.person_id))][["person_id", "enc_date"]].drop_duplicates()
by_pt = {p: np.sort(g.values.astype("datetime64[D]").astype(int)) for p, g in comp.groupby("person_id").enc_date}

sh = pd.read_parquet(D/"v4_status_history.parquet"); sh["updated_at"] = pd.to_datetime(sh.updated_at)
grad = sh[sh.new_status == "GRADUATED"].groupby("person_id").updated_at.min().dt.normalize()
out["patients_with_graduated_status"] = int(grad.index.isin(A.person_id).sum())

g = pd.read_parquet(D/"v4_goals.parquet"); g["updated_at"] = pd.to_datetime(g.updated_at)
careplan = g[(g.goal_type != "HEDIS") & g.status.isin(["COMPLETED", "PROVISIONALLY_COMPLETED"])].copy()
careplan = careplan.sort_values("updated_at")
second_goal = (careplan.groupby("person_id").updated_at
               .apply(lambda x: x.sort_values().iloc[1] if len(x) > 1 else pd.NaT).dropna().dt.normalize())  # date the 2nd care-plan goal completed
out["patients_with_2plus_careplan_goals"] = int(second_goal.index.isin(A.person_id).sum())
gs = set(grad.index); g2 = set(second_goal.index)
out["completion_definition_concordance"] = {"graduated_with_2_goals": round(len(gs & g2)/max(len(gs), 1), 3),
                                            "nongraduated_with_2_goals": round(len(g2 - gs)/max(A.person_id.nunique()-len(gs), 1), 3)}

def next_contact_gap(pid, d0):
    """days from d0 to the next completed contact strictly after d0 (np.inf when none)."""
    arr = by_pt.get(pid);  d = np.datetime64(pd.Timestamp(d0).date()).astype("datetime64[D]").astype(int)
    if arr is None: return np.inf
    nxt = arr[arr > d]
    return float(nxt[0] - d) if len(nxt) else np.inf

def done_by(pid, d0, days, series):
    v = series.get(pid)
    return bool(v is not None and pd.notna(v) and d0 < v <= d0 + pd.Timedelta(days=days))

# ---------------- Task B: E1 (primary) and E4 -------------------------------------------------
B["gap_days"] = [next_contact_gap(p, d) for p, d in zip(B.person_id, B.enc_date)]
B["grad_90"] = [done_by(p, d, 90, grad) for p, d in zip(B.person_id, B.enc_date)]
B["grad_goals_90"] = [done_by(p, d, 90, second_goal) for p, d in zip(B.person_id, B.enc_date)]
B["E1"] = ((B.gap_days > 90) & ~B.grad_90).astype(int)
B["E1_goalsdef"] = ((B.gap_days > 90) & ~B.grad_goals_90).astype(int)
B["E4_time"] = np.minimum(B.gap_days, 90.0)                      # time to next completed contact
B["E4_event"] = (B.gap_days <= 90).astype(int)                   # 1 = returned, 0 = censored at 90 (disengaged)
out["taskB"] = {"analysis_rows": int(B.analysis.sum()), "patients": int(B[B.analysis].person_id.nunique()),
                "E1_rate": round(float(B[B.analysis].E1.mean()), 4),
                "E1_rate_goalsdef": round(float(B[B.analysis].E1_goalsdef.mean()), 4),
                "E1_by_era": {k: round(v, 4) for k, v in B[B.analysis].groupby("era").E1.mean().items()},
                "E1_by_state": {k: round(v, 4) for k, v in B[B.analysis].groupby("state").E1.mean().items()},
                "contacts_per_patient": round(float(B[B.analysis].groupby("person_id").size().mean()), 2)}

# ---------------- Task A: E2, E3 --------------------------------------------------------------
A["gap_days"] = [next_contact_gap(p, d) for p, d in zip(A.person_id, A.enroll_date)]
A["grad_90"] = [done_by(p, d, 90, grad) for p, d in zip(A.person_id, A.enroll_date)]
A["E2_disengaged"] = ((A.gap_days > 90) & ~A.grad_90).astype(int)
A["E3_beyond_enrollment"] = (A.gap_days <= 30).astype(int)
A["graduated_ever"] = A.person_id.isin(gs).astype(int)
A["completed_goals_ever"] = A.person_id.isin(g2).astype(int)
el = A[A.eligible_E2]
out["taskA"] = {"eligible": int(len(el)), "E2_disengaged_rate": round(float(el.E2_disengaged.mean()), 4),
                "E3_beyond_enrollment_rate": round(float(el.E3_beyond_enrollment.mean()), 4),
                "graduated_ever": round(float(el.graduated_ever.mean()), 4),
                "E2_by_era": {k: round(v, 4) for k, v in el.groupby("era").E2_disengaged.mean().items()}}

# ---------------- utilization and cost (Section 5.3) ------------------------------------------
adt = pd.read_parquet(D/"v4_adt.parquet"); adt["admit_date"] = pd.to_datetime(adt.admit_date)
adt = adt.dropna(subset=["admit_date"]).sort_values(["person_id", "admit_date"])
adt["is_ed"] = adt.class_code.astype(str).str.upper().str.startswith("E")
adt["is_ip"] = adt.class_code.astype(str).str.upper().str.startswith("I")
adt["prev"] = adt.groupby(["person_id", "class_code"]).admit_date.shift()
adt["new_stay"] = adt.prev.isna() | ((adt.admit_date - adt.prev).dt.total_seconds() > 24*3600)
stays = adt[adt.new_stay & (adt.is_ed | adt.is_ip)][["person_id", "admit_date", "is_ed", "is_ip"]]

med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["claim_start_date"] = pd.to_datetime(med.claim_start_date)
ph = pd.read_parquet(D/"v4_pharmacy_claim_lines.parquet"); ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date)
et = med.encounter_type.astype(str).str.lower()
med["is_ed_claim"] = et.str.contains("emergency"); med["is_ip_claim"] = et.str.contains("acute inpatient")
med["is_negctrl"] = et.str.contains("dialysis")   # negative control outcome: dialysis spending

def window_sum(frame, valcols, start, end, keycol="claim_start_date"):
    f = frame[(frame[keycol] >= start) & (frame[keycol] <= end)]
    return {c: float(f[c].sum()) for c in valcols}

def per_patient_windows(idx, lo_hi):
    rows = []
    medg = {p: g for p, g in med.groupby("person_id")}; phg = {p: g for p, g in ph.groupby("person_id")}
    stg = {p: g for p, g in stays.groupby("person_id")}
    for pid, d0 in zip(idx.person_id, idx.index_date):
        rec = {"person_id": pid}
        for lbl, (lo, hi) in lo_hi.items():
            s, e = d0 + pd.Timedelta(days=lo), d0 + pd.Timedelta(days=hi)
            st = stg.get(pid); m = medg.get(pid); p = phg.get(pid)
            rec[f"ed_{lbl}"] = int(st[(st.admit_date >= s) & (st.admit_date <= e)].is_ed.sum()) if st is not None else 0
            rec[f"ip_{lbl}"] = int(st[(st.admit_date >= s) & (st.admit_date <= e)].is_ip.sum()) if st is not None else 0
            if m is not None:
                w = m[(m.claim_start_date >= s) & (m.claim_start_date <= e)]
                rec[f"med_paid_{lbl}"] = float(w.paid.sum()); rec[f"ed_claim_{lbl}"] = int(w.is_ed_claim.sum())
                rec[f"ip_claim_{lbl}"] = int(w.is_ip_claim.sum()); rec[f"negctrl_paid_{lbl}"] = float(w.loc[w.is_negctrl, "paid"].sum())
            else:
                rec[f"med_paid_{lbl}"] = 0.0; rec[f"ed_claim_{lbl}"] = 0; rec[f"ip_claim_{lbl}"] = 0; rec[f"negctrl_paid_{lbl}"] = 0.0
            rec[f"rx_paid_{lbl}"] = float(p[(p.claim_start_date >= s) & (p.claim_start_date <= e)].paid.sum()) if p is not None else 0.0
            rec[f"total_paid_{lbl}"] = rec[f"med_paid_{lbl}"] + rec[f"rx_paid_{lbl}"]
        rows.append(rec)
    return pd.DataFrame(rows)

idx = A[["person_id", "enroll_date"]].rename(columns={"enroll_date": "index_date"})
U = per_patient_windows(idx, {"1_180": (1, 180), "31_210": (31, 210), "91_270": (91, 270), "pre180": (-180, -1), "pre365": (-365, -1)})
A = A.merge(U, on="person_id", how="left")
A["cost_window_complete"] = (A.enroll_date + pd.Timedelta(days=180)) <= CLAIMS_COMPLETE
A["cost_window_complete_31_210"] = (A.enroll_date + pd.Timedelta(days=210)) <= CLAIMS_COMPLETE
A["cost_window_complete_91_270"] = (A.enroll_date + pd.Timedelta(days=270)) <= CLAIMS_COMPLETE
A["covered_270"] = [covered(p, d, 270) for p, d in zip(A.person_id, A.enroll_date)]
A["util_window_complete"] = (A.enroll_date + pd.Timedelta(days=180)) <= STUDY_END
A["eligible_cost"] = A.eligible_E2 & A.covered_180 & A.cost_window_complete
A["eligible_util"] = A.eligible_E2 & A.covered_180 & A.util_window_complete
out["aim2"] = {"eligible_cost": int(A.eligible_cost.sum()), "eligible_util": int(A.eligible_util.sum()),
               "mean_total_paid_1_180": round(float(A.loc[A.eligible_cost, "total_paid_1_180"].mean()), 0),
               "mean_ed_1_180": round(float(A.loc[A.eligible_util, "ed_1_180"].mean()), 3),
               "mean_ip_1_180": round(float(A.loc[A.eligible_util, "ip_1_180"].mean()), 3),
               "negctrl_dialysis_share_nonzero": round(float((A.loc[A.eligible_cost, "negctrl_paid_1_180"] > 0).mean()), 3),
               "mean_total_paid_pre180": round(float(A.loc[A.eligible_cost, "total_paid_pre180"].mean()), 0)}

A.to_parquet(D/"v4_outcomes_A.parquet"); B.to_parquet(D/"v4_outcomes_B.parquet")
json.dump(out, open(R/"outcomes_v4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
