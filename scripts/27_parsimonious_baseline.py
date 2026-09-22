"""Parsimonious contact-timing comparators for the contact task.

The acute care risk score is not a fair baseline for disengagement: it was built for a
different outcome. This fits logistic models on a handful of contact-timing variables,
using the same temporal split and validation cohort as the full learner library, so the
manuscript can say how much of the ensemble's discrimination is contact timing alone.
Writes results/parsimonious_baseline_v4.json (Appendix Table 12).
"""
import pandas as pd, numpy as np, pathlib, json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

P = pathlib.Path(__file__).resolve().parent.parent
D, R = P/"data_cache", P/"results"
CUT, SEED, NBOOT = pd.Timestamp("2025-07-01"), 20260915, 500

F = pd.read_parquet(D/"v4_featB.parquet")
F = F[F.analysis].copy()
y = F.E1.to_numpy()
dev = (pd.to_datetime(F.enc_date) < CUT).to_numpy()
pid = F.person_id.to_numpy()

SETS = {
    "1 variable: days since last contact": ["days_since_contact"],
    "2 variables: + prior contact count": ["days_since_contact", "n_contact_prior_all"],
    "4 variables: + days from enrollment, contacts in 30 d":
        ["days_since_contact", "n_contact_prior_all", "days_since_enroll", "n_contact_30d"],
}

rng = np.random.default_rng(SEED)
val_patients = np.unique(pid[~dev])
out = {}
for name, cols in SETS.items():
    X = F[cols].to_numpy(float)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000))
    model.fit(X[dev], y[dev])
    p = model.predict_proba(X[~dev])[:, 1]
    yv, pv, idv = y[~dev], p, pid[~dev]
    boot = []
    for _ in range(NBOOT):                      # patient-clustered bootstrap, as elsewhere
        samp = rng.choice(val_patients, size=len(val_patients), replace=True)
        idx = np.concatenate([np.where(idv == s)[0] for s in samp])
        if len(np.unique(yv[idx])) < 2:
            continue
        boot.append(roc_auc_score(yv[idx], pv[idx]))
    out[name] = {"auroc": round(float(roc_auc_score(yv, pv)), 3),
                 "ci": [round(float(np.percentile(boot, 2.5)), 3),
                        round(float(np.percentile(boot, 97.5)), 3)],
                 "n_features": len(cols)}
    print(f"{name:58s} {out[name]['auroc']:.3f} {out[name]['ci']}")

json.dump(out, open(R/"parsimonious_baseline_v4.json", "w"), indent=1)
print("wrote results/parsimonious_baseline_v4.json")
