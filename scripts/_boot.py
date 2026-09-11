"""Shared member-bootstrap helper for percentile intervals of prediction metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6))); lr = LogisticRegression(C=1e6, max_iter=500).fit(lp.reshape(-1, 1), y); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def boot_ci(y, p, ids, fns, B=300, seed=20260908):
    """fns: dict name -> f(y, p) returning float. Returns {name: {"estimate", "ci_95"}} with member-cluster bootstrap."""
    rng = np.random.default_rng(seed); up = np.unique(ids); idx = {q: np.where(ids == q)[0] for q in up}; out = {k: [] for k in fns}
    est = {k: float(f(y, p)) for k, f in fns.items()}
    for _ in range(B):
        ix = np.concatenate([idx[q] for q in rng.choice(up, len(up))]); yy = y[ix]
        if yy.min() == yy.max(): continue
        for k, f in fns.items():
            try: out[k].append(float(f(yy, p[ix])))
            except Exception: pass
    return {k: {"estimate": round(est[k], 4), "ci_95": [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] if v else None} for k, v in out.items()}
def std_fns(thr=None):
    fns = {"auroc": roc_auc_score, "auprc": average_precision_score, "calibration_slope": lambda y, p: calib(y, p)[0], "calibration_intercept": lambda y, p: calib(y, p)[1]}
    if thr is not None:
        fns.update({"per_100_flags": lambda y, p: 100*y[p >= thr].mean() if (p >= thr).sum() else np.nan, "sensitivity": lambda y, p: y[p >= thr].sum()/max(y.sum(), 1), "flag_rate": lambda y, p: (p >= thr).mean()})
    return fns
