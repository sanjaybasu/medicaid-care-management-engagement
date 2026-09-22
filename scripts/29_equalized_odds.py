"""Equalized odds ratios with patient-clustered bootstrap confidence intervals.

The deployed threshold is the Youden-optimal operating point. For each protected attribute
(race and ethnicity, state, sex, and age band: under 35, 35 to 49, 50 to 64, 65 and older)
we report the ratio of lowest to highest true positive rate, the ratio of lowest to highest
false positive rate, and the equalized odds ratio (the smaller of the two), following the
standard definition of equalized odds. A ratio of 1 means the model errs identically across
groups. Writes results/equalized_odds_v4.json.
"""
import pandas as pd, numpy as np, pathlib, json

P = pathlib.Path(__file__).resolve().parent.parent
D, R = P/"data_cache", P/"results"
SEED, NBOOT = 20260916, 1000
rng = np.random.default_rng(SEED)

sc = np.load(R/"val_scores_B_full.npy")          # columns: ensemble, risk score, outcome
p_ens, y = sc[:, 0], sc[:, 2]
F = pd.read_parquet(D/"v4_featB.parquet")
F = F[F.analysis].copy()
val = (pd.to_datetime(F.enc_date) >= pd.Timestamp("2025-07-01")).to_numpy()
V = F.loc[val, ["person_id", "state", "race", "gender", "age"]].reset_index(drop=True)
V["age_band"] = pd.cut(pd.to_numeric(V.age, errors="coerce"), [-np.inf, 34, 49, 64, np.inf],
                       labels=["under 35", "35 to 49", "50 to 64", "65 and older"]).astype(str)
assert len(V) == len(y), (len(V), len(y))

THR = json.load(open(R/"thresholds_v4.json"))
row = next(r for r in THR["B"]["operating_points"] if r["operating_point"] == "Youden-optimal")
thr = row["threshold"]
flag = p_ens >= thr

def ratios(idx):
    """min/max TPR ratio, min/max FPR ratio, and their minimum, over groups in idx."""
    out = {}
    for var in ["race", "state", "gender", "age_band"]:
        lv = V[var].fillna("Unknown").to_numpy()[idx]
        yy, ff = y[idx], flag[idx]
        tprs, fprs = [], []
        for g in np.unique(lv):
            m = lv == g
            pos, neg = (yy[m] == 1), (yy[m] == 0)
            if m.sum() < 50 or pos.sum() < 5 or neg.sum() < 5:
                continue
            tprs.append(ff[m][pos].mean()); fprs.append(ff[m][neg].mean())
        if len(tprs) < 2:
            continue
        tpr_r = min(tprs)/max(max(tprs), 1e-9)
        fpr_r = min(fprs)/max(max(fprs), 1e-9)
        out[var] = {"tpr_ratio": tpr_r, "fpr_ratio": fpr_r, "equalized_odds_ratio": min(tpr_r, fpr_r)}
    return out

point = ratios(np.arange(len(y)))
pid = V.person_id.to_numpy()
patients = np.unique(pid)
index_of = {p: np.where(pid == p)[0] for p in patients}
boot = {v: {k: [] for k in ("tpr_ratio", "fpr_ratio", "equalized_odds_ratio")} for v in point}
for _ in range(NBOOT):
    samp = rng.choice(patients, size=len(patients), replace=True)
    idx = np.concatenate([index_of[s] for s in samp])
    r = ratios(idx)
    for v in boot:
        if v in r:
            for k in boot[v]:
                boot[v][k].append(r[v][k])

def group_rates(var):
    """point-estimate n, events, TPR, and FPR for every evaluated level of var."""
    lv = V[var].fillna("Unknown").to_numpy(); res = {}
    for g in np.unique(lv):
        m = lv == g; pos, neg = (y[m] == 1), (y[m] == 0)
        if m.sum() < 50 or pos.sum() < 5 or neg.sum() < 5:
            continue
        res[str(g)] = {"n": int(m.sum()), "events": int(pos.sum()),
                       "tpr": round(float(flag[m][pos].mean()), 3), "fpr": round(float(flag[m][neg].mean()), 3)}
    return res

out = {"threshold": round(float(thr), 4), "operating_point": "Youden-optimal", "n_bootstrap": NBOOT,
       "attributes": {}, "groups": {v: group_rates(v) for v in point}}
for v, m in point.items():
    out["attributes"][v] = {k: {"estimate": round(float(m[k]), 3),
                                "ci_95": [round(float(np.percentile(boot[v][k], 2.5)), 3),
                                          round(float(np.percentile(boot[v][k], 97.5)), 3)]}
                            for k in m}
json.dump(out, open(R/"equalized_odds_v4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
