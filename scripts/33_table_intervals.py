"""Patient-clustered bootstrap confidence intervals for table cells that were reported as point estimates.

Covers (a) the Youden-optimal confusion-matrix metrics in Table 4, (b) every operating point in Table 7, and
(c) the rank-agreement and cross-discrimination statistics in Table 5. Thresholds are held fixed at the values
that reproduce the full-sample flag counts, so point estimates match the existing tables exactly; the same seed
and resampling scheme as 06_ensemble_metrics.py are used (2,000 resamples of patients). Writes
results/table_intervals_v4.json.
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

P = pathlib.Path(__file__).resolve().parent.parent
D, R = P/"data_cache", P/"results"
SEED, B_BOOT = 20260911, 2000
rng = np.random.default_rng(SEED)

def ci(draws, nd=3):
    return [round(float(np.nanpercentile(draws, 2.5)), nd), round(float(np.nanpercentile(draws, 97.5)), nd)]

def rates(flag, yy):
    tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum(); fn = (~flag & (yy == 1)).sum(); tn = (~flag & (yy == 0)).sum()
    return {"sensitivity": tp/max(tp+fn, 1), "specificity": tn/max(tn+fp, 1), "ppv": tp/max(tp+fp, 1), "npv": tn/max(tn+fn, 1)}

out = {"n_bootstrap": B_BOOT, "seed": SEED}

# ---------------- contact task scores, with patient ids in the same row order ------------------------
sc = np.load(R/"val_scores_B_full.npy"); p_ens, p_risk, y = sc[:, 0], sc[:, 1], sc[:, 2]
V = pd.read_parquet(D/"v4_valscores_B_full.parquet")
assert len(V) == len(y) and np.allclose(V.p_ensemble.to_numpy(), p_ens) and (V.y.to_numpy() == y).all()
ids = V.person_id.to_numpy(); up = np.unique(ids); loc = {q: np.where(ids == q)[0] for q in up}
boot_idx = [np.concatenate([loc[q] for q in rng.choice(up, len(up))]) for _ in range(B_BOOT)]

def boot_rates(pp, thr):
    est = rates(pp >= thr, y); draws = {k: [] for k in est}
    for ix in boot_idx:
        r = rates(pp[ix] >= thr, y[ix])
        for k in est: draws[k].append(r[k])
    return {k: {"estimate": round(float(est[k]), 3), "ci_95": ci(draws[k])} for k in est}

# (a) Table 4, Youden-optimal rows: thresholds exactly as in 30_youden_confusion.py
YC = json.load(open(R/"youden_confusion_v4.json"))
thr_y = YC["threshold"]; k = YC["ensemble_stacked_calibrated"]["flagged"]
cut_risk = np.sort(p_risk)[::-1][k-1]
ens_y, risk_y = boot_rates(p_ens, thr_y), boot_rates(p_risk, cut_risk)
for name, res, ref in [("ensemble_stacked_calibrated", ens_y, YC["ensemble_stacked_calibrated"]), ("signal_risk_score", risk_y, YC["signal_risk_score"])]:
    for m in ("sensitivity", "specificity", "ppv"):
        assert res[m]["estimate"] == ref[m], (name, m, res[m]["estimate"], ref[m])
out["table4_youden"] = {"ensemble_stacked_calibrated": ens_y, "signal_risk_score": risk_y}

# (b) Table 7: one fixed threshold per operating point, reproducing the reported flag count
TH = json.load(open(R/"thresholds_v4.json"))["B"]["operating_points"]
rows = {}
for r in TH:
    thr = float(np.sort(p_ens)[::-1][r["flagged"]-1])
    res = boot_rates(p_ens, thr)
    assert int((p_ens >= thr).sum()) == r["flagged"], (r["operating_point"], (p_ens >= thr).sum(), r["flagged"])
    for m in ("sensitivity", "specificity", "ppv", "npv"):
        assert res[m]["estimate"] == r[m], (r["operating_point"], m, res[m]["estimate"], r[m])
    rows[r["operating_point"]] = res
out["table7"] = rows
print("table 4 and 7 intervals done", flush=True)

# (c) Table 5: rank agreement and cross-discrimination, patient-level resampling as in 08_concordance.py
A = pd.read_parquet(D/"v4_outcomes_A.parquet")
S = pd.read_parquet(D/"v4_score_all_A_full.parquet")[["person_id", "p_ensemble", "dev"]]
A = A.merge(S, on="person_id", how="inner")
A["risk_pct"] = pd.to_numeric(A.risk_percentile, errors="coerce"); A = A[A.risk_pct.notna()].reset_index(drop=True)
CO = json.load(open(R/"concordance_v4.json"))
def topfrac(x, f):
    kk = max(int(round(f*len(x))), 1); return set(np.argsort(-np.asarray(x))[:kk])
def rank_stats(df):
    pe, rp = df.p_ensemble.to_numpy(), df.risk_pct.to_numpy()
    o = {"spearman_rho": spearmanr(pe, rp)[0]}
    for f, lbl in [(0.10, "top_decile"), (0.20, "top_quintile")]:
        a, b = topfrac(pe, f), topfrac(rp, f); o[f"jaccard_{lbl}"] = len(a & b)/len(a | b)
    U = df[df.eligible_util]; acute = ((U.ed_1_180 + U.ip_1_180) > 0).astype(int).to_numpy()
    o["disengagement_score_for_acute_care"] = roc_auc_score(acute, U.p_ensemble)
    o["risk_score_for_acute_care"] = roc_auc_score(acute, U.risk_pct)
    o["risk_score_for_disengagement"] = roc_auc_score(df.E2_disengaged, df.risk_pct)
    return o
est = rank_stats(A)
assert round(est["spearman_rho"], 3) == CO["rank_agreement"]["spearman_rho"]
assert round(est["jaccard_top_decile"], 3) == CO["rank_agreement"]["top_decile"]["jaccard"]
assert round(est["disengagement_score_for_acute_care"], 3) == CO["cross_discrimination"]["disengagement_score_for_acute_care"]
assert round(est["risk_score_for_disengagement"], 3) == CO["cross_discrimination"]["risk_score_for_disengagement"]
draws = {k: [] for k in est}
rng2 = np.random.default_rng(SEED)
for _ in range(B_BOOT):
    smp = A.iloc[rng2.integers(0, len(A), len(A))]
    r = rank_stats(smp)
    for k in est: draws[k].append(r[k])
out["table5"] = {k: {"estimate": round(float(est[k]), 3), "ci_95": ci(draws[k])} for k in est}
json.dump(out, open(R/"table_intervals_v4.json", "w"), indent=1)
print(json.dumps(out["table5"], indent=1)); print(json.dumps(out["table4_youden"], indent=1))
