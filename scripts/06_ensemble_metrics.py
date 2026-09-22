"""v4 step 6: stacked ensemble, calibration, and the Aim 1 metric set (Sections 8 and 9.1).

Stacking weights come from non-negative least squares on out-of-fold development predictions; a
rank-average ensemble is the pre-specified alternative. Metrics are computed on the temporal
validation set with patient-clustered bootstrap intervals.
Usage: 06_ensemble_metrics.py [B|A] [full|structured]"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

TASK = sys.argv[1] if len(sys.argv) > 1 else "B"
FEATSET = sys.argv[2] if len(sys.argv) > 2 else "full"
FLAG_FRAC, SEED, B_BOOT = 0.20, 20260911, 2000
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"

P = pd.read_parquet(D/f"v4_preds_{TASK}_{FEATSET}.parquet")
LEARNERS = [c[4:] for c in P.columns if c.startswith("oof_")]
dev, val = P.dev.to_numpy(), ~P.dev.to_numpy()
y = P.y.to_numpy(); ids = P.person_id.to_numpy()

# ---------------- ensembles ---------------------------------------------------------------------
Zd = np.column_stack([P[f"oof_{m}"].to_numpy() for m in LEARNERS])[dev]
Zv = np.column_stack([P[f"val_{m}"].to_numpy() for m in LEARNERS])[val]
w, _ = nnls(Zd, y[dev].astype(float))
w = w/w.sum() if w.sum() > 0 else np.ones(len(LEARNERS))/len(LEARNERS)
weights = {m: round(float(x), 4) for m, x in zip(LEARNERS, w)}
ens_dev, ens_val = Zd @ w, Zv @ w
rank_dev = np.column_stack([rankdata(Zd[:, j])/len(Zd) for j in range(Zd.shape[1])]).mean(1)
rank_val = np.column_stack([rankdata(Zv[:, j])/len(Zv) for j in range(Zv.shape[1])]).mean(1)

iso = IsotonicRegression(out_of_bounds="clip").fit(ens_dev, y[dev])
ens_val_cal = iso.predict(ens_val)

scores = {m: P[f"val_{m}"].to_numpy()[val] for m in LEARNERS}
scores["ensemble_stacked"] = ens_val
scores["ensemble_stacked_calibrated"] = ens_val_cal
scores["ensemble_rank_average"] = rank_val
rp = P.risk_percentile.to_numpy()[val]
scores["signal_risk_score"] = np.where(np.isnan(rp), np.nanmedian(rp), rp)/100.0
yv, idv = y[val], ids[val]

# ---------------- metrics ------------------------------------------------------------------------
def confusion(yy, pp, frac=FLAG_FRAC):
    k = max(int(round(frac*len(pp))), 1)
    thr = np.sort(pp)[::-1][k-1]
    f = pp >= thr
    tp, fp = int((f & (yy == 1)).sum()), int((f & (yy == 0)).sum())
    fn, tn = int((~f & (yy == 1)).sum()), int((~f & (yy == 0)).sum())
    return dict(threshold=float(thr), flagged=int(f.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                sensitivity=tp/max(tp+fn, 1), specificity=tn/max(tn+fp, 1), ppv=tp/max(tp+fp, 1), npv=tn/max(tn+fn, 1))

def calib(yy, pp):
    p = np.clip(pp, 1e-6, 1-1e-6); lp = np.log(p/(1-p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, yy)
    return float(m.coef_[0][0]), float(m.intercept_[0])

def youden(yy, pp):
    o = np.argsort(-pp); yo = yy[o]
    tp = np.cumsum(yo); fp = np.cumsum(1-yo)
    tpr, fpr = tp/max(yy.sum(), 1), fp/max((1-yy).sum(), 1)
    j = int(np.argmax(tpr-fpr))
    return float(pp[o][j]), float(tpr[j]), float(1-fpr[j])

def metric_pack(yy, pp):
    c = confusion(yy, pp); s, i = calib(yy, pp); yt, ytpr, ytnr = youden(yy, pp)
    return {"auroc": roc_auc_score(yy, pp), "auprc": average_precision_score(yy, pp), "brier": brier_score_loss(yy, np.clip(pp, 0, 1)),
            "sensitivity": c["sensitivity"], "specificity": c["specificity"], "ppv": c["ppv"], "npv": c["npv"],
            "calibration_slope": s, "calibration_intercept": i, "youden_threshold": yt, "youden_sensitivity": ytpr, "youden_specificity": ytnr}

rng = np.random.default_rng(SEED); up = np.unique(idv); loc = {q: np.where(idv == q)[0] for q in up}
boot_idx = [np.concatenate([loc[q] for q in rng.choice(up, len(up))]) for _ in range(B_BOOT)]
def with_ci(yy, pp):
    est = metric_pack(yy, pp); draws = {k: [] for k in est}
    for ix in boot_idx:
        if yy[ix].min() == yy[ix].max(): continue
        try:
            mp = metric_pack(yy[ix], pp[ix])
            for k, v in mp.items(): draws[k].append(v)
        except Exception: pass
    return {k: {"estimate": round(float(est[k]), 4),
                "ci_95": [round(float(np.nanpercentile(draws[k], 2.5)), 4), round(float(np.nanpercentile(draws[k], 97.5)), 4)] if draws[k] else None}
            for k in est}

res = {"task": TASK, "featset": FEATSET, "n_val": int(val.sum()), "n_val_patients": int(len(up)),
       "event_rate_val": round(float(yv.mean()), 4), "stack_weights": weights, "models": {}, "confusion": {}}
for name, pp in scores.items():
    res["models"][name] = with_ci(yv, pp)
    res["confusion"][name] = confusion(yv, pp)
    print(f"  {name:32s} AUROC {res['models'][name]['auroc']['estimate']:.3f} "
          f"AUPRC {res['models'][name]['auprc']['estimate']:.3f} sens {res['models'][name]['sensitivity']['estimate']:.3f} "
          f"spec {res['models'][name]['specificity']['estimate']:.3f}", flush=True)

# ---------------- primary comparisons -----------------------------------------------------------
def paired_diff(a, b):
    d = roc_auc_score(yv, a) - roc_auc_score(yv, b); dr = []
    for ix in boot_idx:
        if yv[ix].min() == yv[ix].max(): continue
        try: dr.append(roc_auc_score(yv[ix], a[ix]) - roc_auc_score(yv[ix], b[ix]))
        except Exception: pass
    return {"diff": round(float(d), 4), "ci_95": [round(float(np.nanpercentile(dr, 2.5)), 4), round(float(np.nanpercentile(dr, 97.5)), 4)]}
def paired_diff_pr(a, b):
    d = average_precision_score(yv, a) - average_precision_score(yv, b); dr = []
    for ix in boot_idx:
        if yv[ix].min() == yv[ix].max(): continue
        try: dr.append(average_precision_score(yv[ix], a[ix]) - average_precision_score(yv[ix], b[ix]))
        except Exception: pass
    return {"diff": round(float(d), 4), "ci_95": [round(float(np.nanpercentile(dr, 2.5)), 4), round(float(np.nanpercentile(dr, 97.5)), 4)]}

best_single = max(LEARNERS, key=lambda m: roc_auc_score(yv, scores[m]))
res["comparisons"] = {
    "ensemble_vs_risk_score_auroc": paired_diff(scores["ensemble_stacked"], scores["signal_risk_score"]),
    "ensemble_vs_risk_score_auprc": paired_diff_pr(scores["ensemble_stacked"], scores["signal_risk_score"]),
    "ensemble_vs_best_single_auroc": paired_diff(scores["ensemble_stacked"], scores[best_single]),
    "best_single_learner": best_single,
    "events_captured_ratio_ensemble_vs_risk": round(res["confusion"]["ensemble_stacked_calibrated"]["tp"]/max(res["confusion"]["signal_risk_score"]["tp"], 1), 2)}

# ---------------- decision curve ------------------------------------------------------------------
def net_benefit(pp, thr):
    f = pp >= thr; n = len(pp)
    tp = int((f & (yv == 1)).sum()); fp = int((f & (yv == 0)).sum())
    return (tp/n) - (fp/n)*(thr/(1-thr))
res["decision_curve"] = {f"{t:.2f}": {k: round(net_benefit(scores[k], t), 4) for k in ["ensemble_stacked_calibrated", "signal_risk_score"]} | {"treat_all": round(yv.mean()-(1-yv.mean())*(t/(1-t)), 4)}
                         for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]}

# ---------------- fairness (Section 10) -----------------------------------------------------------
F = pd.read_parquet(D/f"v4_feat{TASK}.parquet")
keycols = ["person_id", "encounter_id"] if TASK == "B" else ["person_id"]
att = F[keycols + ["state", "race", "gender", "age"]].drop_duplicates(keycols)
M = P.loc[val, keycols].merge(att, on=keycols, how="left")
pp = scores["ensemble_stacked_calibrated"]; thr = confusion(yv, pp)["threshold"]
fair = {}
for var in ["race", "state", "gender"]:
    g = {}
    for lvl, sub in M.groupby(M[var].fillna("Missing")):
        m = M[var].fillna("Missing").to_numpy() == lvl
        if m.sum() < 50 or yv[m].sum() < 5: continue
        f = pp[m] >= thr
        g[str(lvl)] = {"n": int(m.sum()), "event_rate": round(float(yv[m].mean()), 4), "flag_rate": round(float(f.mean()), 4),
                       "tpr": round(float(((f) & (yv[m] == 1)).sum()/max((yv[m] == 1).sum(), 1)), 4),
                       "fpr": round(float(((f) & (yv[m] == 0)).sum()/max((yv[m] == 0).sum(), 1)), 4),
                       "ppv": round(float(((f) & (yv[m] == 1)).sum()/max(f.sum(), 1)), 4)}
    if g:
        tprs = [v["tpr"] for v in g.values()]
        fair[var] = {"groups": g, "equalized_odds_ratio": round(min(tprs)/max(max(tprs), 1e-9), 3)}
res["fairness"] = fair

json.dump(res, open(R/f"metrics_{TASK}_{FEATSET}.json", "w"), indent=1)
np.save(R/f"val_scores_{TASK}_{FEATSET}.npy", np.column_stack([scores["ensemble_stacked_calibrated"], scores["signal_risk_score"], yv]))
P.loc[val, keycols].assign(p_ensemble=pp, y=yv).to_parquet(D/f"v4_valscores_{TASK}_{FEATSET}.parquet")
# cross-fitted score for every row (development rows use their out-of-fold prediction) so that
# Aims 2 and 3 can use a prediction that never saw the row it scores
all_raw = np.where(dev, Zd_full := np.column_stack([P[f"oof_{m}"].to_numpy() for m in LEARNERS]) @ w,
                   np.column_stack([P[f"val_{m}"].to_numpy() for m in LEARNERS]) @ w)
P[keycols].assign(p_ensemble=iso.predict(all_raw), y=y, dev=dev, index_date=P.index_date).to_parquet(D/f"v4_score_all_{TASK}_{FEATSET}.parquet")
print(json.dumps({k: res[k] for k in ["comparisons", "stack_weights"]}, indent=1))
