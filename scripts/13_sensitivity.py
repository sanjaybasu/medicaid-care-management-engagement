"""v4 step 13: pre-specified sensitivity analyses (pre-registration Section 18)."""
import json, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED, CUT = 20260911, pd.Timestamp("2025-07-01")
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
out = {}

F = pd.read_parquet(D/"v4_featB.parquet").merge(pd.read_parquet(D/"v4_textB.parquet"), on=["person_id", "encounter_id"], how="left")
F["enc_date"] = pd.to_datetime(F.enc_date)
DROP = {"person_id","encounter_id","enc_date","enroll_date","enroll_dt","zero_date","era","analysis","in_split","eligible_E1",
        "eligible_E2","covered_90","covered_180","covered_210","observed_90","occurred","E1","E1_goalsdef","E4_time","E4_event",
        "gap_days","grad_90","grad_goals_90","note_text","created_by_id","start_time","tier1"}
CATS = [c for c in ["state","market","entity","gender","race","contact_type","encounter_type","roles"] if c in F.columns]
num = [c for c in F.columns if c not in DROP and c not in CATS and pd.api.types.is_numeric_dtype(F[c])]
def design(f): return pd.get_dummies(f[num + CATS], columns=CATS, dummy_na=True).astype(float).to_numpy()

def run(f, y, label):
    dev = (f.enc_date < CUT).to_numpy(); X = design(f)
    m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1).fit(X[dev], y[dev])
    p = m.predict_proba(X[~dev])[:, 1]; yv = y[~dev]
    k = max(int(round(0.2*len(p))), 1); thr = np.sort(p)[::-1][k-1]; fl = p >= thr
    return {"label": label, "n": int(len(f)), "n_val": int((~dev).sum()), "event_rate": round(float(y.mean()), 4),
            "auroc": round(float(roc_auc_score(yv, p)), 4), "auprc": round(float(average_precision_score(yv, p)), 4),
            "sensitivity": round(float(yv[fl].sum()/max(yv.sum(), 1)), 4),
            "specificity": round(float((~fl & (yv == 0)).sum()/max((yv == 0).sum(), 1)), 4)}

A = F[F.analysis].copy()
out["base_lightgbm"] = run(A, A.E1.to_numpy(), "primary outcome (90-day, status-based completion)")
out["goal_based_completion"] = run(A, A.E1_goalsdef.to_numpy(), "program completion defined by two completed care-plan goals")

# 120-day disengagement window
enc = pd.read_parquet(D/"v4_encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date)
comp = enc[enc.occurred == "YES"]
by = {p: np.sort(day(g.enc_date)) for p, g in comp.groupby("person_id")}
sh = pd.read_parquet(D/"v4_status_history.parquet"); sh["updated_at"] = pd.to_datetime(sh.updated_at)
grad = sh[sh.new_status == "GRADUATED"].groupby("person_id").updated_at.min().dt.normalize()
def gap(pid, d0):
    a = by.get(pid)
    if a is None: return np.inf
    nx = a[a > d0]; return float(nx[0]-d0) if len(nx) else np.inf
A120 = A[A.enc_date <= pd.Timestamp("2026-06-07") - pd.Timedelta(days=120)].copy()
g = [gap(p, d) for p, d in zip(A120.person_id, day(A120.enc_date))]
gr = [(grad.get(p) is not None) and pd.notna(grad.get(p)) and (day(A120.enc_date)[i] < day(pd.Series([grad.get(p)]))[0] <= day(A120.enc_date)[i]+120) for i, p in enumerate(A120.person_id)]
y120 = ((np.array(g) > 120) & ~np.array(gr)).astype(int)
out["window_120_days"] = run(A120, y120, "120-day rather than 90-day disengagement window")

# exclude the enrollment-day contact
A0 = A[A.days_since_enroll > 0].copy()
out["excluding_enrollment_day_contact"] = run(A0, A0.E1.to_numpy(), "excluding the enrollment-day contact")
out["enrollment_day_share"] = {"share_of_contacts": round(float((A.days_since_enroll == 0).mean()), 4),
                               "disengagement_rate_day0": round(float(A[A.days_since_enroll == 0].E1.mean()), 4),
                               "disengagement_rate_later": round(float(A[A.days_since_enroll > 0].E1.mean()), 4),
                               "share_of_events_from_day0": round(float(A[A.days_since_enroll == 0].E1.sum()/A.E1.sum()), 4)}

# first 30 days after enrollment
out["first_30_days"] = {"share_of_contacts": round(float((A.days_since_enroll <= 30).mean()), 4),
                        "share_of_events": round(float(A[A.days_since_enroll <= 30].E1.sum()/A.E1.sum()), 4)}

# attempts after disengagement
att = enc[enc.occurred == "NO"]; bya = {p: np.sort(day(g.enc_date)) for p, g in att.groupby("person_id")}
def any_attempt(pid, d0, lo, hi):
    a = bya.get(pid)
    return 0 if a is None else int(np.searchsorted(a, d0+hi, "right") - np.searchsorted(a, d0+lo, "left"))
dis = A[A.E1 == 1]
out["attempts_after_disengagement"] = {"share_with_any_attempt_90d": round(float(np.mean([any_attempt(p, d, 1, 90) > 0 for p, d in zip(dis.person_id, day(dis.enc_date))])), 4),
                                       "n": int(len(dis))}
json.dump(out, open(R/"sensitivity_v4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
