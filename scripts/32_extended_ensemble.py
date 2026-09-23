"""Sensitivity analysis: a twelve-learner stacked ensemble that adds TabPFN to the pre-registered library.

TabPFN receives the same five-fold patient-grouped cross-validation inside the development cohort as
the eleven pre-registered learners (identical GroupKFold splits to 05_models.py), so its out-of-fold
predictions can enter the non-negative least squares stack on equal terms. The twelve-learner ensemble
is scored on the temporal validation cohort and compared with the pre-registered eleven-learner
ensemble using the same 2,000 patient-clustered bootstrap resamples. TabFM is not stacked because a
single full-context fit takes about five hours on this hardware. Requires TABPFN_TOKEN.

Usage: 32_extended_ensemble.py [B|A] [n_estimators]
Writes results/extended_ensemble_{TASK}_v4.json; caches TabPFN out-of-fold predictions in results/preds_cache/.
"""
import sys, os, json, time, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

TASK = sys.argv[1] if len(sys.argv) > 1 else "B"
NEST = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FEATSET, SEED, NFOLD, B_BOOT = "full", 20260911, 5, 2000
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
CUT = pd.Timestamp("2025-07-01")
CACHE = R/"preds_cache"

# ---------------- feature matrix and splits, identical to 05_models.py ----------------------------
F = pd.read_parquet(D/f"v4_feat{TASK}.parquet")
T = pd.read_parquet(D/f"v4_text{TASK}.parquet")
if TASK == "B":
    F = F.merge(T, on=["person_id", "encounter_id"], how="left")
    F = F[F.analysis].copy(); Y = F.E1.to_numpy(); date = pd.to_datetime(F.enc_date)
else:
    F = F.merge(T, on=["person_id"], how="left")
    Y = F.E2_disengaged.to_numpy(); date = pd.to_datetime(F.enroll_date)
dev = (date < CUT).to_numpy(); val = ~dev
groups = F.person_id.to_numpy()
DROP = {"person_id","encounter_id","enc_date","enroll_date","enroll_dt","zero_date","era","analysis","in_split","eligible_E1",
        "eligible_E2","eligible_cost","eligible_util","covered_90","covered_180","covered_210","observed_90","occurred",
        "E1","E1_goalsdef","E4_time","E4_event","gap_days","grad_90","grad_goals_90","E2_disengaged","E3_beyond_enrollment",
        "graduated_ever","completed_goals_ever","cost_window_complete","cost_window_complete_31_210","util_window_complete",
        "note_text","created_by_id","start_time","tier1","index_date"}
DROP |= {c for c in F.columns if c.startswith(("ed_","ip_","med_paid_","rx_paid_","total_paid_","negctrl_","ed_claim_","ip_claim_"))}
CATS = [c for c in ["state","market","entity","gender","race","contact_type","encounter_type","roles"] if c in F.columns]
num = [c for c in F.columns if c not in DROP and c not in CATS and pd.api.types.is_numeric_dtype(F[c])]
X = pd.get_dummies(F[num + CATS], columns=CATS, dummy_na=True).astype(float)
X = X.loc[:, X.notna().any()]
Xv = X.to_numpy(dtype=np.float32)
meta = json.load(open(R/f"models_{TASK}_{FEATSET}.json"))
assert X.shape[1] == meta["n_features"] and int(dev.sum()) == meta["dev"] and int(val.sum()) == meta["val"]

P = pd.read_parquet(D/f"v4_preds_{TASK}_{FEATSET}.parquet")
assert (P.y.to_numpy() == Y).all() and (P.dev.to_numpy() == dev).all()
LEARNERS = [c[4:] for c in P.columns if c.startswith("oof_")]
dev_idx, val_idx = np.where(dev)[0], np.where(val)[0]
yv, idv = Y[val], groups[val]

# ---------------- TabPFN out-of-fold and validation predictions -------------------------------------
def tabpfn_fit_predict(tr, te):
    import torch
    from tabpfn import TabPFNClassifier
    device = os.environ.get("TABPFN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    clf = TabPFNClassifier(device=device, n_estimators=NEST, ignore_pretraining_limits=True, random_state=SEED)
    clf.fit(Xv[tr], Y[tr].astype(int))
    return clf.predict_proba(Xv[te])[:, 1]

oof_file = CACHE/f"{TASK}_{FEATSET}_tabpfn_oof.npz"
if oof_file.exists():
    oof_pfn = np.load(oof_file)["oof"]; print("  tabpfn oof cached", flush=True)
else:
    oof_pfn = np.full(len(Y), np.nan); t0 = time.time()
    gkf = GroupKFold(n_splits=NFOLD)
    for k, (tr_i, te_i) in enumerate(gkf.split(dev_idx, Y[dev_idx], groups[dev_idx])):
        tr, te = dev_idx[tr_i], dev_idx[te_i]
        oof_pfn[te] = tabpfn_fit_predict(tr, te)
        print(f"  tabpfn fold {k+1}/{NFOLD} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez(oof_file, oof=oof_pfn, seconds=np.array(round(time.time()-t0, 1)))
val_file = CACHE/f"{TASK}_{FEATSET}_tabpfn.npz"
val_pfn = np.load(val_file)["val"] if val_file.exists() else tabpfn_fit_predict(dev_idx, val_idx)
assert np.isfinite(oof_pfn[dev]).all() and len(val_pfn) == len(val_idx)
print(f"  tabpfn oof AUROC {roc_auc_score(Y[dev], oof_pfn[dev]):.3f}  val AUROC {roc_auc_score(yv, val_pfn):.3f}", flush=True)

# ---------------- stacks: pre-registered (11) and extended (12) -------------------------------------
def stack(names, extra_oof=None, extra_val=None):
    Zd = np.column_stack([P[f"oof_{m}"].to_numpy() for m in names])[dev]
    Zv = np.column_stack([P[f"val_{m}"].to_numpy() for m in names])[val]
    if extra_oof is not None:
        Zd = np.column_stack([Zd, extra_oof[dev]]); Zv = np.column_stack([Zv, extra_val])
    w, _ = nnls(Zd, Y[dev].astype(float)); w = w/w.sum()
    return Zv @ w, w

ens11, w11 = stack(LEARNERS)
ens12, w12 = stack(LEARNERS, oof_pfn, val_pfn)
M0 = json.load(open(R/f"metrics_{TASK}_{FEATSET}.json"))
assert abs(roc_auc_score(yv, ens11) - M0["models"]["ensemble_stacked"]["auroc"]["estimate"]) < 5e-4

# ---------------- metrics, as in 06_ensemble_metrics.py ---------------------------------------------
def calib(yy, pp):
    p = np.clip(pp, 1e-6, 1-1e-6); lp = np.log(p/(1-p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, yy); return float(m.coef_[0][0]), float(m.intercept_[0])
def metric_pack(yy, pp):
    s, i = calib(yy, pp)
    return {"auroc": roc_auc_score(yy, pp), "auprc": average_precision_score(yy, pp),
            "brier": brier_score_loss(yy, np.clip(pp, 0, 1)), "calibration_slope": s, "calibration_intercept": i}
rng = np.random.default_rng(SEED); up = np.unique(idv); loc = {q: np.where(idv == q)[0] for q in up}
boot_idx = [np.concatenate([loc[q] for q in rng.choice(up, len(up))]) for _ in range(B_BOOT)]
def with_ci(pp):
    est = metric_pack(yv, pp); draws = {k: [] for k in est}
    for ix in boot_idx:
        if yv[ix].min() == yv[ix].max(): continue
        try:
            for k, v in metric_pack(yv[ix], pp[ix]).items(): draws[k].append(v)
        except Exception: pass
    return {k: {"estimate": round(float(est[k]), 4), "ci_95": [round(float(np.nanpercentile(draws[k], 2.5)), 4), round(float(np.nanpercentile(draws[k], 97.5)), 4)]} for k in est}
def paired_diff(a, b):
    d = roc_auc_score(yv, a) - roc_auc_score(yv, b)
    dr = [roc_auc_score(yv[ix], a[ix]) - roc_auc_score(yv[ix], b[ix]) for ix in boot_idx if yv[ix].min() != yv[ix].max()]
    return {"diff": round(float(d), 4), "ci_95": [round(float(np.nanpercentile(dr, 2.5)), 4), round(float(np.nanpercentile(dr, 97.5)), 4)]}

out = {"task": TASK, "n_estimators": NEST, "n_dev": int(dev.sum()), "n_val": int(val.sum()), "n_val_patients": int(len(up)),
       "learners_extended": LEARNERS + ["tabpfn"],
       "stack_weights_11": {m: round(float(x), 4) for m, x in zip(LEARNERS, w11)},
       "stack_weights_12": {m: round(float(x), 4) for m, x in zip(LEARNERS + ["tabpfn"], w12)},
       "tabpfn_oof_auroc": round(float(roc_auc_score(Y[dev], oof_pfn[dev])), 4),
       "models": {"ensemble_stacked_11": with_ci(ens11), "ensemble_stacked_12_tabpfn": with_ci(ens12), "tabpfn": with_ci(val_pfn)},
       "comparisons": {"ensemble11_minus_ensemble12_auroc": paired_diff(ens11, ens12),
                       "ensemble12_minus_tabpfn_auroc": paired_diff(ens12, val_pfn)}}
json.dump(out, open(R/f"extended_ensemble_{TASK}_v4.json", "w"), indent=1)
for k, v in out["models"].items(): print(f"  {k:28s} AUROC {v['auroc']['estimate']:.3f} ({v['auroc']['ci_95'][0]:.3f} to {v['auroc']['ci_95'][1]:.3f})")
print(json.dumps(out["comparisons"], indent=1)); print("tabpfn stack weight:", out["stack_weights_12"]["tabpfn"])
