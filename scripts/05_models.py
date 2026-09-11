"""v4 step 5: the pre-registered learner library for Aim 1 (Section 9.1).

Fits every learner with five-fold patient-grouped cross-validation inside the development era, stores
out-of-fold development predictions and temporal-validation predictions, and writes them for the
ensemble and metric steps. Task B (E1) is primary; Task A (E2) uses the same library.
Usage: 05_models.py [B|A] [structured|full]"""
import sys, json, time, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nets import DeepSurv, FTClassifier

TASK = sys.argv[1] if len(sys.argv) > 1 else "B"
FEATSET = sys.argv[2] if len(sys.argv) > 2 else "full"
SEED, NFOLD = 20260911, 5
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
CUT = pd.Timestamp("2025-07-01")

# ---------------- data ------------------------------------------------------------------------
F = pd.read_parquet(D/f"v4_feat{TASK}.parquet")
T = pd.read_parquet(D/f"v4_text{TASK}.parquet")
if TASK == "B":
    F = F.merge(T, on=["person_id", "encounter_id"], how="left")
    F = F[F.analysis].copy(); Y = F.E1.to_numpy(); date = pd.to_datetime(F.enc_date)
    time_col, event_col = F.E4_time.to_numpy(float), F.E4_event.to_numpy(float)
else:
    F = F.merge(T, on=["person_id"], how="left")
    Y = F.E2_disengaged.to_numpy(); date = pd.to_datetime(F.enroll_date)
    time_col = np.minimum(F.gap_days.to_numpy(float), 90.0); event_col = (F.gap_days.to_numpy(float) <= 90).astype(float)
dev = (date < CUT).to_numpy(); val = ~dev
groups = F.person_id.to_numpy()

DROP = {"person_id","encounter_id","enc_date","enroll_date","enroll_dt","zero_date","era","analysis","in_split","eligible_E1",
        "eligible_E2","eligible_cost","eligible_util","covered_90","covered_180","covered_210","observed_90","occurred",
        "E1","E1_goalsdef","E4_time","E4_event","gap_days","grad_90","grad_goals_90","E2_disengaged","E3_beyond_enrollment",
        "graduated_ever","completed_goals_ever","cost_window_complete","cost_window_complete_31_210","util_window_complete",
        "note_text","created_by_id","start_time","tier1","index_date"}
DROP |= {c for c in F.columns if c.startswith(("ed_","ip_","med_paid_","rx_paid_","total_paid_","negctrl_","ed_claim_","ip_claim_"))}
CATS = [c for c in ["state","market","entity","gender","race","contact_type","encounter_type","roles"] if c in F.columns]
TEXTC = [c for c in F.columns if c.startswith(("svd", "emb", "lex_")) or c == "text_chars"]
num = [c for c in F.columns if c not in DROP and c not in CATS and pd.api.types.is_numeric_dtype(F[c])]
if FEATSET == "structured":
    num = [c for c in num if c not in TEXTC]
X = pd.get_dummies(F[num + CATS], columns=CATS, dummy_na=True).astype(float)
X = X.loc[:, X.notna().any()]
print(f"task {TASK} featset {FEATSET}: X {X.shape}, events {Y.sum()}/{len(Y)} ({Y.mean():.3f}), dev {dev.sum()} val {val.sum()}", flush=True)
Xv = X.to_numpy(dtype=float)
imp = SimpleImputer(strategy="median").fit(Xv[dev]); sc = StandardScaler().fit(imp.transform(Xv[dev]))
Xs = sc.transform(imp.transform(Xv))

# ---------------- learners --------------------------------------------------------------------
def fit_predict(name, tr, te):
    """Returns predicted probability of the event (disengagement) for rows `te`, trained on rows `tr`."""
    yb = Y[tr]
    if name == "logistic_elasticnet":
        m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", max_iter=3000, random_state=SEED).fit(Xs[tr], yb)
        return m.predict_proba(Xs[te])[:, 1]
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Xs[tr], yb)
        return m.predict_proba(Xs[te])[:, 1]
    if name == "hist_gbm":
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED).fit(Xv[tr], yb)
        return m.predict_proba(Xv[te])[:, 1]
    if name == "xgboost":
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xv[tr], yb)
        return m.predict_proba(Xv[te])[:, 1]
    if name == "lightgbm":
        m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1).fit(Xv[tr], yb)
        return m.predict_proba(Xv[te])[:, 1]
    if name == "catboost":
        m = CatBoostClassifier(iterations=700, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, random_seed=SEED, verbose=0).fit(Xv[tr], yb)
        return m.predict_proba(Xv[te])[:, 1]
    if name == "ft_transformer":
        keep = rank_features(tr, 128)          # token count capped for tractable attention
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            fin, fout = f"{td}/in.npz", f"{td}/out.npy"
            np.savez(fin, Xtr=Xs[tr][:, keep].astype("float32"), ytr=yb.astype("float32"),
                     Xte=Xs[te][:, keep].astype("float32"), seed=SEED, epochs=30, batch=1024)
            subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parent/"run_ft.py"), fin, fout], check=True)
            return np.load(fout)
    if name == "discrete_time_hazard":
        return discrete_hazard(tr, te)
    if name == "cox_ph":
        keep = rank_features(tr, 40)
        df = pd.DataFrame(Xs[tr][:, keep]); df["T"] = np.maximum(time_col[tr], 0.5); df["E"] = event_col[tr]
        cph = CoxPHFitter(penalizer=0.1).fit(df, "T", "E")
        lp = cph.predict_partial_hazard(pd.DataFrame(Xs[te][:, keep])).to_numpy()
        bs = cph.baseline_survival_
        s0 = float(bs.iloc[max(int(np.searchsorted(bs.index.values, 90, "right"))-1, 0), 0])
        return np.clip(s0**lp, 1e-6, 1-1e-6)                       # P(no contact within 90 days)
    if name == "random_survival_forest":
        keep = rank_features(tr, 40)
        sy = Surv.from_arrays(event=event_col[tr].astype(bool), time=np.maximum(time_col[tr], 0.5))
        m = RandomSurvivalForest(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=SEED).fit(Xs[tr][:, keep], sy)
        sf = m.predict_survival_function(Xs[te][:, keep], return_array=True)
        j = int(np.searchsorted(m.unique_times_, 90, "right") - 1)
        return np.clip(sf[:, max(j, 0)], 1e-6, 1-1e-6)
    if name == "deepsurv":
        m = DeepSurv(seed=SEED).fit(Xs[tr], np.maximum(time_col[tr], 0.5), event_col[tr])
        return np.clip(m.predict_surv(Xs[te], 90.0), 1e-6, 1-1e-6)
    raise ValueError(name)

def rank_features(tr, k):
    """Top-k features by absolute univariate correlation with the outcome in the training rows."""
    z = Xs[tr]; y = Y[tr]
    c = np.abs(np.nan_to_num(np.array([np.corrcoef(z[:, j], y)[0, 1] if z[:, j].std() > 0 else 0 for j in range(z.shape[1])])))
    return np.argsort(-c)[:k]

def discrete_hazard(tr, te, weeks=13):
    """Pooled logistic hazard of returning in each of 13 weekly intervals; P(disengage) = prod(1-h)."""
    rows, ys, wk = [], [], []
    t, e = time_col[tr], event_col[tr]
    for i in range(len(tr)):
        last = int(min(np.ceil(t[i]/7.0), weeks)); last = max(last, 1)
        for w in range(1, last+1):
            rows.append(i); ys.append(1 if (e[i] == 1 and w == last) else 0); wk.append(w)
    rows = np.asarray(rows); W = np.zeros((len(rows), weeks)); W[np.arange(len(rows)), np.asarray(wk)-1] = 1
    Z = np.hstack([Xs[tr][rows], W])
    m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=SEED).fit(Z, np.asarray(ys))
    surv = np.ones(len(te))
    for w in range(1, weeks+1):
        Ww = np.zeros((len(te), weeks)); Ww[:, w-1] = 1
        h = m.predict_proba(np.hstack([Xs[te], Ww]))[:, 1]
        surv *= (1-h)
    return np.clip(surv, 1e-6, 1-1e-6)

LEARNERS = ["logistic_elasticnet", "discrete_time_hazard", "cox_ph", "random_survival_forest", "random_forest",
            "hist_gbm", "xgboost", "lightgbm", "catboost", "deepsurv", "ft_transformer"]

# ---------------- cross-fitting and validation --------------------------------------------------
dev_idx = np.where(dev)[0]; val_idx = np.where(val)[0]
gkf = GroupKFold(n_splits=NFOLD)
oof = pd.DataFrame(index=F.index, columns=LEARNERS, dtype=float)
valp = pd.DataFrame(index=F.index, columns=LEARNERS, dtype=float)
timing = {}
CACHE = R/"preds_cache"; CACHE.mkdir(exist_ok=True)
for name in LEARNERS:
    t0 = time.time()
    cf = CACHE/f"{TASK}_{FEATSET}_{name}.npz"
    if cf.exists():
        z = np.load(cf)
        oof[name] = z["oof"]; valp[name] = z["val"]
        from sklearn.metrics import roc_auc_score
        print(f"  {name:24s} cached  val {roc_auc_score(Y[val_idx], valp[name].to_numpy()[val_idx]):.3f}", flush=True)
        continue
    for tr_i, te_i in gkf.split(dev_idx, Y[dev_idx], groups[dev_idx]):
        tr, te = dev_idx[tr_i], dev_idx[te_i]
        oof.iloc[te, oof.columns.get_loc(name)] = fit_predict(name, tr, te)
    valp.iloc[val_idx, valp.columns.get_loc(name)] = fit_predict(name, dev_idx, val_idx)
    np.savez(cf, oof=oof[name].to_numpy(dtype=float), val=valp[name].to_numpy(dtype=float))
    timing[name] = round(time.time()-t0, 1)
    from sklearn.metrics import roc_auc_score
    print(f"  {name:24s} oof {roc_auc_score(Y[dev_idx], oof[name].to_numpy()[dev_idx]):.3f} "
          f"val {roc_auc_score(Y[val_idx], valp[name].to_numpy()[val_idx]):.3f} ({timing[name]}s)", flush=True)

P = pd.concat([F[["person_id"] + (["encounter_id"] if TASK == "B" else [])].reset_index(drop=True),
               pd.Series(Y, name="y"), pd.Series(dev, name="dev"), pd.Series(date.to_numpy(), name="index_date"),
               oof.add_prefix("oof_").reset_index(drop=True), valp.add_prefix("val_").reset_index(drop=True)], axis=1)
P["risk_percentile"] = F.risk_percentile.to_numpy()
P.to_parquet(D/f"v4_preds_{TASK}_{FEATSET}.parquet")
json.dump({"task": TASK, "featset": FEATSET, "n_features": int(X.shape[1]), "learners": LEARNERS, "seconds": timing,
           "rows": int(len(F)), "dev": int(dev.sum()), "val": int(val.sum()), "event_rate": round(float(Y.mean()), 4)},
          open(R/f"models_{TASK}_{FEATSET}.json", "w"), indent=1)
print("saved", D/f"v4_preds_{TASK}_{FEATSET}.parquet")
