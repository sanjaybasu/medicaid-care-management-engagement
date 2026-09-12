"""v4 step 20: quantities the TRIPOD+AI and STROBE checklists require."""
import json, pathlib
import numpy as np, pandas as pd
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
num = lambda s: pd.to_numeric(s, errors="coerce")
A = pd.read_parquet(D/"v4_outcomes_A.parquet"); A = A[A.eligible_E2].copy()
B = pd.read_parquet(D/"v4_outcomes_B.parquet"); B = B[B.analysis].copy()
B["enc_date"] = pd.to_datetime(B.enc_date); A["enroll_date"] = pd.to_datetime(A.enroll_date)
mod = json.load(open(R/"models_B_full.json"))
S = pd.read_parquet(D/"v4_score_all_A_full.parquet")
AS = A.merge(S[["person_id", "p_ensemble"]], on="person_id", how="left")
last = B.sort_values("enc_date").groupby("person_id").enc_date.max()
first = A.set_index("person_id").enroll_date
span = (last - first.reindex(last.index)).dt.days.dropna()
out = {
 "missing_data": {"race_missing_n": int(A.race.isna().sum() + (A.race.astype(str).str.lower().isin(["unknown", "none", ""])).sum()),
                  "race_missing_pct": round(100*float((A.race.isna() | A.race.astype(str).str.lower().isin(["unknown", "none", ""])).mean()), 1),
                  "age_missing_n": int(num(A.age).isna().sum()),
                  "risk_percentile_missing_n": int(num(A.risk_percentile).isna().sum()),
                  "patients_complete_on_adjustment_covariates": int((~A[["age", "risk_percentile"]].apply(num).isna().any(axis=1)).sum())},
 "study_size": {"patients": int(len(A)), "decision_points": int(len(B)),
                "validation_decision_points": int(mod["val"]), "validation_events": int(round(mod["val"]*0.2634)),
                "candidate_predictors": int(mod["n_features"]),
                "events_per_candidate_predictor": round(float(round(mod["val"]*0.2634)/mod["n_features"]), 2),
                "development_events": int(round(mod["dev"]*mod["event_rate"]))},
 "follow_up": {"median_decision_points_per_patient": float(B.groupby("person_id").size().median()),
               "iqr_decision_points": [float(B.groupby("person_id").size().quantile(.25)), float(B.groupby("person_id").size().quantile(.75))],
               "median_days_enrolment_to_last_contact": float(span.median()),
               "iqr_days": [float(span.quantile(.25)), float(span.quantile(.75))],
               "total_patient_years": round(float(span.sum()/365.25), 0)},
 "category_boundaries": {
    "predicted_risk_quintile_cuts": [round(float(x), 3) for x in np.nanquantile(AS.p_ensemble.astype(float), [.2, .4, .6, .8])],
    "age_bands": ["18-34", "35-49", "50-64", "65+"],
    "prior_spending_quartile_cuts_usd": [round(float(x), 0) for x in np.nanquantile(num(A.total_paid_pre365).astype(float), [.25, .5, .75])],
    "risk_decile_cuts": [round(float(x), 1) for x in np.nanquantile(num(A.risk_percentile).astype(float), np.arange(.1, 1.0, .1))]},
 "class_balance": {"event_rate_development": round(float(mod["event_rate"]), 4), "resampling_applied": False,
                   "recalibration": "isotonic regression on out-of-fold development predictions"},
}
json.dump(out, open(R/"reporting_gaps_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
