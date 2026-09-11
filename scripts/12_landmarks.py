"""v4 step 12: pseudo-prospective validation (pre-registration Section 8).

The ensemble is refit at each 2025 month-end landmark, frozen, and applied to every later eligible
contact. Landmark refits use the learners that carry non-trivial stacking weight and train in under a
minute (elastic-net logistic, random forest, histogram gradient boosting, XGBoost, LightGBM,
CatBoost); the transformer and survival learners are excluded for tractability."""
import json, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.optimize import nnls
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED, FLAG = 20260911, 0.20
F = pd.read_parquet(D/"v4_featB.parquet").merge(pd.read_parquet(D/"v4_textB.parquet"), on=["person_id", "encounter_id"], how="left")
F = F[F.eligible_E1].copy(); F["enc_date"] = pd.to_datetime(F.enc_date)
Y = F.E1.to_numpy()
DROP = {"person_id","encounter_id","enc_date","enroll_date","enroll_dt","zero_date","era","analysis","in_split","eligible_E1",
        "eligible_E2","covered_90","covered_180","covered_210","observed_90","occurred","E1","E1_goalsdef","E4_time","E4_event",
        "gap_days","grad_90","grad_goals_90","note_text","created_by_id","start_time","tier1"}
CATS = [c for c in ["state","market","entity","gender","race","contact_type","encounter_type","roles"] if c in F.columns]
num = [c for c in F.columns if c not in DROP and c not in CATS and pd.api.types.is_numeric_dtype(F[c])]
X = pd.get_dummies(F[num + CATS], columns=CATS, dummy_na=True).astype(float)
Xv = X.to_numpy(dtype=float)

def fit_ensemble(tr, te):
    imp = SimpleImputer(strategy="median").fit(Xv[tr]); sc = StandardScaler().fit(imp.transform(Xv[tr]))
    Zt, Ze = sc.transform(imp.transform(Xv[tr])), sc.transform(imp.transform(Xv[te]))
    preds_tr, preds_te = [], []
    for name in ["logreg", "rf", "hgb", "xgb", "lgb", "cat"]:
        if name == "logreg": m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", max_iter=2000, random_state=SEED).fit(Zt, Y[tr]); a, b = m.predict_proba(Zt)[:, 1], m.predict_proba(Ze)[:, 1]
        elif name == "rf": m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Zt, Y[tr]); a, b = m.predict_proba(Zt)[:, 1], m.predict_proba(Ze)[:, 1]
        elif name == "hgb": m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=SEED).fit(Xv[tr], Y[tr]); a, b = m.predict_proba(Xv[tr])[:, 1], m.predict_proba(Xv[te])[:, 1]
        elif name == "xgb": m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xv[tr], Y[tr]); a, b = m.predict_proba(Xv[tr])[:, 1], m.predict_proba(Xv[te])[:, 1]
        elif name == "lgb": m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1).fit(Xv[tr], Y[tr]); a, b = m.predict_proba(Xv[tr])[:, 1], m.predict_proba(Xv[te])[:, 1]
        else: m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0).fit(Xv[tr], Y[tr]); a, b = m.predict_proba(Xv[tr])[:, 1], m.predict_proba(Xv[te])[:, 1]
        preds_tr.append(a); preds_te.append(b)
    Ztr, Zte = np.column_stack(preds_tr), np.column_stack(preds_te)
    w, _ = nnls(Ztr, Y[tr].astype(float)); w = w/w.sum() if w.sum() else np.ones(Ztr.shape[1])/Ztr.shape[1]
    return Ztr @ w, Zte @ w

LAST_INDEX = pd.Timestamp("2026-03-09")
landmarks = [d for d in pd.date_range("2025-01-31", "2025-12-31", freq="ME") if d + pd.Timedelta(days=90) <= LAST_INDEX]
res = {"landmarks": {}}
for L in landmarks:
    tr = np.where((F.enc_date < L).to_numpy())[0]
    te = np.where((F.enc_date >= L).to_numpy())[0]
    if len(tr) < 1000 or len(te) < 200 or Y[te].sum() < 20: continue
    ptr, pte = fit_ensemble(tr, te)
    thr = float(np.quantile(ptr, 1-FLAG))
    f = pte >= thr
    key = str(L.date())
    seen = set(F.person_id.to_numpy()[tr]); unseen = np.array([p not in seen for p in F.person_id.to_numpy()[te]])
    by_month = {}
    dts = F.enc_date.to_numpy()[te]
    for mth, mask in pd.Series(pd.to_datetime(dts).to_period("M").astype(str)).groupby(lambda i: pd.to_datetime(dts[i]).to_period("M")):
        pass
    mser = pd.Series(pd.to_datetime(dts)).dt.to_period("M").astype(str)
    for mth in sorted(mser.unique()):
        m = (mser == mth).to_numpy()
        if m.sum() >= 100 and 0 < Y[te][m].sum() < m.sum():
            by_month[mth] = round(float(roc_auc_score(Y[te][m], pte[m])), 4)
    res["landmarks"][key] = {"train_n": int(len(tr)), "test_n": int(len(te)), "event_rate": round(float(Y[te].mean()), 4),
                             "auroc": round(float(roc_auc_score(Y[te], pte)), 4), "threshold": round(thr, 4),
                             "flag_rate": round(float(f.mean()), 4), "per_100_flags": round(float(100*Y[te][f].mean()), 1),
                             "sensitivity": round(float(Y[te][f].sum()/max(Y[te].sum(), 1)), 4),
                             "auroc_unseen_patients": round(float(roc_auc_score(Y[te][unseen], pte[unseen])), 4) if unseen.sum() > 100 and 0 < Y[te][unseen].sum() < unseen.sum() else None,
                             "n_unseen": int(unseen.sum()), "by_month": by_month}
    print(f"  {key}: train {len(tr)} test {len(te)} AUROC {res['landmarks'][key]['auroc']:.3f} per100 {res['landmarks'][key]['per_100_flags']}", flush=True)
res["headline_landmark"] = max(res["landmarks"], key=lambda k: k)
au = [v["auroc"] for v in res["landmarks"].values()]
res["auroc_range"] = [min(au), max(au)]
json.dump(res, open(R/"landmarks_v4.json", "w"), indent=1)
print(json.dumps({"headline": res["headline_landmark"], "range": res["auroc_range"]}, indent=1))
