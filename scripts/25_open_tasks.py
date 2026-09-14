"""Open tasks at the last contact, separated from care-plan goals.

A task is a discrete to-do a staff member wrote for themselves or a colleague, and most carry a
workflow tag (outreach, follow-up, referral, appointment reminder) rather than a clinical domain. An
open outreach or follow-up task at the moment a patient stops responding is a different construct
from unmet clinical need: it is work the team had scheduled and did not complete."""
import json, pathlib, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
num = lambda s: pd.to_numeric(s, errors="coerce")
out = {}

T = pd.read_parquet(D/"v4_tasks.parquet")
for c in ["created_at", "updated_at", "due_date"]: T[c] = pd.to_datetime(T[c], errors="coerce")
T = T[T.created_at.notna()].copy()
T["closed_at"] = pd.to_datetime(np.where(T.completed.astype(bool), T.updated_at.values, np.datetime64("NaT")))
tags = T.tags.astype(str).str.lower()
T["kind"] = np.where(tags.str.contains("outreach"), "outreach",
            np.where(tags.str.contains("follow-up|follow up"), "follow_up",
            np.where(tags.str.contains("referral"), "referral",
            np.where(tags.str.contains("appointment"), "appointment",
            np.where(tags.str.contains("assessment"), "assessment", "other")))))
out["task_inventory"] = {"tasks": int(len(T)), "patients": int(T.person_id.nunique()),
                         "by_tag": T.kind.value_counts().to_dict(),
                         "never_completed": int((~T.completed.astype(bool)).sum()),
                         "with_due_date": int(T.due_date.notna().sum())}

G = pd.read_parquet(D/"v4_goals.parquet")
for c in ["created_at", "updated_at"]: G[c] = pd.to_datetime(G[c], errors="coerce")
G = G[G.created_at.notna()].copy()
CLOSED = {"COMPLETED", "PROVISIONALLY_COMPLETED", "DISMISSED"}
G["closed_at"] = pd.to_datetime(np.where(G.status.isin(CLOSED), G.updated_at.values, np.datetime64("NaT")))

B = pd.read_parquet(D/"v4_outcomes_B.parquet"); B["enc_date"] = pd.to_datetime(B.enc_date); B = B[B.analysis]
last = B.sort_values("enc_date").groupby("person_id").tail(1)[["person_id", "encounter_id", "enc_date", "E1", "state"]]
F = pd.read_parquet(D/"v4_featB.parquet")[["person_id", "encounter_id", "age", "risk_percentile", "any_bh", "sud",
                                           "diabetes", "htn", "n_contact_prior_all", "days_since_enroll", "paid_prior_365"]]
L = last.merge(F, on=["person_id", "encounter_id"], how="left")

def opener(frame, keycol=None, keyval=None):
    f = frame if keycol is None else frame[frame[keycol] == keyval]
    by = {p: g for p, g in f.groupby("person_id")}
    def fn(pid, t):
        g = by.get(pid)
        if g is None: return 0
        return int(((g.created_at <= t) & (g.closed_at.isna() | (g.closed_at > t))).sum())
    return fn
def everer(frame, keycol=None, keyval=None):
    f = frame if keycol is None else frame[frame[keycol] == keyval]
    by = {p: g for p, g in f.groupby("person_id")}
    def fn(pid, t):
        g = by.get(pid)
        return 0 if g is None else int((g.created_at <= t).sum())
    return fn

cols = {}
cols["open_tasks_any"] = opener(T)
cols["open_tasks_outreach"] = opener(T, "kind", "outreach")
cols["open_tasks_followup"] = opener(T, "kind", "follow_up")
cols["open_tasks_referral"] = opener(T, "kind", "referral")
cols["open_tasks_appointment"] = opener(T, "kind", "appointment")
cols["open_goals_any"] = opener(G)
cols["tasks_ever"] = everer(T)
for k, fn in cols.items():
    L[k] = [fn(p, t) for p, t in zip(L.person_id, L.enc_date)]
L["overdue_task"] = [int(((T[T.person_id == p].created_at <= t) & (T[T.person_id == p].due_date.notna()) &
                          (T[T.person_id == p].due_date < t) &
                          (T[T.person_id == p].closed_at.isna() | (T[T.person_id == p].closed_at > t))).sum() > 0)
                     if p in set(T.person_id) else 0 for p, t in zip(L.person_id, L.enc_date)]

COV = ["age", "risk_percentile", "any_bh", "sud", "diabetes", "htn", "n_contact_prior_all", "days_since_enroll"]
X = L[COV].apply(num); X = X.fillna(X.median())
X["log_paid"] = np.log1p(num(L.paid_prior_365).fillna(0).clip(lower=0))
X = pd.concat([X, pd.get_dummies(L.state, drop_first=True).astype(float)], axis=1).astype(float)
Tr = L.E1.to_numpy()
Z = StandardScaler().fit_transform(X)
e = np.clip(LogisticRegression(max_iter=4000).fit(Z, Tr).predict_proba(Z)[:, 1], 1e-4, 1-1e-4)
w = np.where(Tr == 1, 1-e, e)
def smd(x, t, ww):
    m1, m0 = np.average(x[t == 1], weights=ww[t == 1]), np.average(x[t == 0], weights=ww[t == 0])
    v1 = np.average((x[t == 1]-m1)**2, weights=ww[t == 1]); v0 = np.average((x[t == 0]-m0)**2, weights=ww[t == 0])
    s = np.sqrt((v1+v0)/2); return 0.0 if s == 0 else float((m1-m0)/s)
out["overlap_balance_max_abs_smd"] = round(max(abs(smd(X[c].to_numpy(float), Tr, w)) for c in X.columns), 4)

rows = []
for col, lbl in [("open_tasks_any", "any open task"), ("open_tasks_outreach", "open outreach task"),
                 ("open_tasks_followup", "open follow-up task"), ("open_tasks_referral", "open referral task"),
                 ("open_tasks_appointment", "open appointment task"), ("overdue_task", "task past its due date"),
                 ("open_goals_any", "any open care-plan goal")]:
    y = (L[col] > 0).astype(float).to_numpy()
    if y.sum() < 25: 
        rows.append({"measure": lbl, "note": "too few patients"}); continue
    m1, m0 = float(np.average(y[Tr == 1], weights=w[Tr == 1])), float(np.average(y[Tr == 0], weights=w[Tr == 0]))
    mod = sm.GLM(y, sm.add_constant(Tr.astype(float)), family=sm.families.Binomial(), freq_weights=w).fit(cov_type="HC0")
    b, se = float(mod.params[1]), float(mod.bse[1])
    rows.append({"measure": lbl, "disengaged_pct": round(100*float(y[Tr == 1].mean()), 1),
                 "sustained_pct": round(100*float(y[Tr == 0].mean()), 1),
                 "weighted_disengaged_pct": round(100*m1, 1), "weighted_sustained_pct": round(100*m0, 1),
                 "odds_ratio": round(float(np.exp(b)), 3),
                 "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)]})
out["measures"] = rows
out["conditioning_on_having_tasks"] = {}
sub = L[L.tasks_ever > 0]
if len(sub) > 200:
    ys = (sub.open_tasks_any > 0).astype(float).to_numpy(); ts = sub.E1.to_numpy()
    out["conditioning_on_having_tasks"] = {"patients_with_any_task": int(len(sub)),
        "open_task_disengaged_pct": round(100*float(ys[ts == 1].mean()), 1),
        "open_task_sustained_pct": round(100*float(ys[ts == 0].mean()), 1)}
json.dump(out, open(R/"open_tasks_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
