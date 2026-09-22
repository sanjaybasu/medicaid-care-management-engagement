"""Tabular foundation models as post hoc comparators (not in the pre-registered library).

TabPFN (Hollmann et al., Nature 2025) and TabFM (Google Research, 2026) are pretrained on synthetic
tables and predict by in-context learning, so they need no hyperparameter search. Each is shown the
full development cohort as context and scored on the temporal validation cohort, the same split and
feature matrix used by the pre-registered learners in 05_models.py. They are reported alongside the
stacked ensemble and the best single learner and are not added to the ensemble, whose weights and
calibration were locked before these comparators were fitted.

TabPFN weights require a one-time licence acceptance at https://ux.priorlabs.ai and the resulting
API key in TABPFN_TOKEN; when the variable is absent TabPFN is skipped and the JSON says so. TabFM
weights are released under a non-commercial, non-production licence, which this research use respects.

Usage: 31_foundation_models.py [B|A] [n_estimators]
Writes results/foundation_models_{TASK}_v4.json and caches predictions in results/preds_cache/.
"""
import sys, os, json, time, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

TASK = sys.argv[1] if len(sys.argv) > 1 else "B"
NEST = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FEATSET, SEED, B_BOOT = "full", 20260911, 2000
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
CUT = pd.Timestamp("2025-07-01")
CACHE = R/"preds_cache"; CACHE.mkdir(exist_ok=True)

# ---------------- feature matrix, identical to 05_models.py --------------------------------------
F = pd.read_parquet(D/f"v4_feat{TASK}.parquet")
T = pd.read_parquet(D/f"v4_text{TASK}.parquet")
if TASK == "B":
    F = F.merge(T, on=["person_id", "encounter_id"], how="left")
    F = F[F.analysis].copy(); Y = F.E1.to_numpy(); date = pd.to_datetime(F.enc_date)
else:
    F = F.merge(T, on=["person_id"], how="left")
    Y = F.E2_disengaged.to_numpy(); date = pd.to_datetime(F.enroll_date)
dev = (date < CUT).to_numpy(); val = ~dev
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
Xv = X.to_numpy(dtype=float)
meta = json.load(open(R/f"models_{TASK}_{FEATSET}.json"))
assert X.shape[1] == meta["n_features"] and int(dev.sum()) == meta["dev"] and int(val.sum()) == meta["val"], \
    (X.shape, dev.sum(), val.sum(), meta)

# the pre-registered learners' validation scores, in the same row order
P = pd.read_parquet(D/f"v4_preds_{TASK}_{FEATSET}.parquet")
assert (P.y.to_numpy() == Y).all() and (P.dev.to_numpy() == dev).all()
sc = np.load(R/f"val_scores_{TASK}_{FEATSET}.npy")
yv = Y[val]; assert (sc[:, 2] == yv).all()
idv = F.person_id.to_numpy()[val]
M0 = json.load(open(R/f"metrics_{TASK}_{FEATSET}.json"))
# the uncalibrated stacked ensemble reported in the manuscript, rebuilt exactly as in 06_ensemble_metrics.py
from scipy.optimize import nnls
LEARNERS = [c[4:] for c in P.columns if c.startswith("oof_")]
Zd = np.column_stack([P[f"oof_{m}"].to_numpy() for m in LEARNERS])[dev]
Zv = np.column_stack([P[f"val_{m}"].to_numpy() for m in LEARNERS])[val]
w, _ = nnls(Zd, Y[dev].astype(float)); w = w/w.sum()
ens_val = Zv @ w
assert abs(roc_auc_score(yv, ens_val) - M0["models"]["ensemble_stacked"]["auroc"]["estimate"]) < 5e-4
scores = {"ensemble_stacked": ens_val, "signal_risk_score": sc[:, 1]}
best = M0["comparisons"]["best_single_learner"]
scores[best] = P[f"val_{best}"].to_numpy()[val]
print(f"task {TASK}: X {X.shape}, dev {dev.sum()} val {val.sum()}, events val {yv.sum()}/{len(yv)}; best single = {best}", flush=True)

# ---------------- foundation models -----------------------------------------------------------------
Xtr, ytr, Xte = Xv[dev].astype(np.float32), Y[dev].astype(int), Xv[val].astype(np.float32)
# TabFM does not accept NaN; impute with development-cohort medians, as the regression learners did in 05_models.py
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median").fit(Xv[dev])
Xtr_imp, Xte_imp = imp.transform(Xv[dev]).astype(np.float32), imp.transform(Xv[val]).astype(np.float32)
timing, notes = {}, {}

def cached(name, fn):
    cf = CACHE/f"{TASK}_{FEATSET}_{name}.npz"
    if cf.exists():
        z = np.load(cf); timing[name] = float(z["seconds"]) if "seconds" in z else None
        print(f"  {name:10s} cached  val AUROC {roc_auc_score(yv, z['val']):.3f}", flush=True); return z["val"]
    t0 = time.time(); p = fn(); timing[name] = round(time.time()-t0, 1)
    np.savez(cf, val=p, seconds=np.array(timing[name])); print(f"  {name:10s} val AUROC {roc_auc_score(yv, p):.3f} ({timing[name]}s)", flush=True); return p

def fit_tabfm():
    import torch
    from tabfm import tabfm_v1_0_0_pytorch as tv, TabFMClassifier
    device = os.environ.get("TABFM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    max_rows = int(os.environ["TABFM_MAX_ROWS"]) if os.environ.get("TABFM_MAX_ROWS") else None   # context subsample fallback
    if max_rows: notes["tabfm"] = f"context subsampled to {max_rows} development rows"
    clf = TabFMClassifier(model=tv.load(device=device), n_estimators=NEST, max_num_rows=max_rows, random_state=SEED)
    clf.fit(Xtr_imp, ytr)
    return clf.predict_proba(Xte_imp)[:, 1]

def fit_tabpfn():
    import torch
    from tabpfn import TabPFNClassifier
    device = os.environ.get("TABPFN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    clf = TabPFNClassifier(device=device, n_estimators=NEST, ignore_pretraining_limits=True, random_state=SEED)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

ONLY = os.environ.get("FM_ONLY")            # fit one model to the cache and stop (lets the two run in parallel)
if ONLY == "tabpfn":
    cached("tabpfn", fit_tabpfn); sys.exit(0)
if ONLY == "tabfm":
    cached("tabfm", fit_tabfm); sys.exit(0)
scores["tabfm"] = cached("tabfm", fit_tabfm)
if os.environ.get("TABPFN_TOKEN") or (CACHE/f"{TASK}_{FEATSET}_tabpfn.npz").exists():
    scores["tabpfn"] = cached("tabpfn", fit_tabpfn)
else:
    notes["tabpfn"] = "skipped: TABPFN_TOKEN not set; weights require licence acceptance at https://ux.priorlabs.ai"
    print("  tabpfn     skipped (no TABPFN_TOKEN)", flush=True)

# ---------------- metrics with patient-clustered bootstrap, as in 06_ensemble_metrics.py ------------
def calib(yy, pp):
    p = np.clip(pp, 1e-6, 1-1e-6); lp = np.log(p/(1-p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, yy)
    return float(m.coef_[0][0]), float(m.intercept_[0])

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
    return {k: {"estimate": round(float(est[k]), 4),
                "ci_95": [round(float(np.nanpercentile(draws[k], 2.5)), 4), round(float(np.nanpercentile(draws[k], 97.5)), 4)]}
            for k in est}

def paired_diff(a, b):
    d = roc_auc_score(yv, a) - roc_auc_score(yv, b); dr = []
    for ix in boot_idx:
        if yv[ix].min() == yv[ix].max(): continue
        dr.append(roc_auc_score(yv[ix], a[ix]) - roc_auc_score(yv[ix], b[ix]))
    return {"diff": round(float(d), 4), "ci_95": [round(float(np.nanpercentile(dr, 2.5)), 4), round(float(np.nanpercentile(dr, 97.5)), 4)]}

out = {"task": TASK, "featset": FEATSET, "n_estimators": NEST, "n_dev": int(dev.sum()), "n_val": int(val.sum()),
       "n_val_patients": int(len(up)), "n_features": int(X.shape[1]), "seconds": timing, "notes": notes,
       "best_single_learner": best, "models": {}, "comparisons": {}}
for name, pp in scores.items():
    out["models"][name] = with_ci(pp)
    print(f"  {name:28s} AUROC {out['models'][name]['auroc']['estimate']:.3f} "
          f"({out['models'][name]['auroc']['ci_95'][0]:.3f} to {out['models'][name]['auroc']['ci_95'][1]:.3f})", flush=True)
for fm in [m for m in ("tabpfn", "tabfm") if m in scores]:
    out["comparisons"][f"ensemble_minus_{fm}_auroc"] = paired_diff(scores["ensemble_stacked"], scores[fm])
    out["comparisons"][f"{best}_minus_{fm}_auroc"] = paired_diff(scores[best], scores[fm])
json.dump(out, open(R/f"foundation_models_{TASK}_v4.json", "w"), indent=1)
print(json.dumps(out["comparisons"], indent=1))
