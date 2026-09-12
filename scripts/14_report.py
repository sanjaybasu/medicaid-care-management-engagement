"""v4 step 14: canonical numbers, tables, and figures for the manuscript.

Every number reported in the manuscript comes from results/canonical.json, which this script writes."""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

P = pathlib.Path(__file__).resolve().parent.parent
D, R = P/"data_cache", P/"results"
NB = P.parent.parent/"notebooks"/"care-management-engagement"
FIG = NB/"figures"; FIG.mkdir(parents=True, exist_ok=True)
J = lambda n: json.load(open(R/n))
flow, outc, mB, mA, conc, act, land, sens, phys = (J("flow_v4.json"), J("outcomes_v4.json"), J("metrics_B_full.json"),
    J("metrics_A_full.json"), J("concordance_v4.json"), J("actions_v4.json"), J("landmarks_v4.json"), J("sensitivity_v4.json"), J("physician_sample_v4.json"))
mod = J("models_B_full.json")
MAT, THR, ADV = J("matching_v4.json"), J("thresholds_v4.json"), J("actions_advanced_v4.json")
NEEDS, NM, RC, RG = J("needs_v4.json"), J("needs_matched_v4.json"), J("reviewer_checks_v4.json"), J("reporting_gaps_v4.json")
C = {"matching": MAT, "thresholds": THR, "actions_advanced": ADV, "needs": NEEDS, "needs_matched": NM, "reviewer_checks": RC, "reporting_gaps": RG, "flow": flow, "outcomes": outc, "metrics_taskB": mB, "metrics_taskA": mA, "concordance": conc,
     "actions": act, "landmarks": land, "sensitivity": sens, "physician_sample": phys, "models": mod,
     "text": J("text_v4.json"), "pull": J("pull_manifest_v4.json")}

A = pd.read_parquet(D/"v4_outcomes_A.parquet"); B = pd.read_parquet(D/"v4_outcomes_B.parquet")
EA = A[A.eligible_E2].copy(); EB = B[B.analysis].copy()
num = lambda s: pd.to_numeric(s, errors="coerce")

# ---------------- Table 1: cohort ------------------------------------------------------------------
def t1col(d):
    r = {}
    r["Patients, n"] = f"{len(d):,}"
    r["Age at enrollment, mean (SD), years"] = f"{num(d.age).mean():.1f} ({num(d.age).std():.1f})"
    r["Female, %"] = f"{100*(d.gender.astype(str).str.upper().str[0] == 'F').mean():.1f}"
    for lab, key in [("Black or African American", "Black or African American"), ("White", "White"), ("Hispanic", "Hispanic")]:
        r[f"{lab}, %"] = f"{100*(d.race.astype(str) == key).mean():.1f}"
    for st in ["OHIO", "VIRGINIA", "WASHINGTON"]:
        r[f"{st.title()}, %"] = f"{100*(d.state == st).mean():.1f}"
    for lab, c in [("Any behavioral health condition", "any_bh"), ("Substance use disorder", "sud"), ("Diabetes", "diabetes"),
                   ("Hypertension", "htn"), ("Chronic obstructive pulmonary disease", "copd"), ("Congestive heart failure", "chf")]:
        r[f"{lab}, %"] = f"{100*num(d[c]).fillna(0).mean():.1f}"
    r["Acute care risk percentile, median (IQR)"] = f"{num(d.risk_percentile).median():.0f} ({num(d.risk_percentile).quantile(.25):.0f} to {num(d.risk_percentile).quantile(.75):.0f})"
    r["Emergency department visits in prior year, mean"] = f"{num(d.ed_pre365).mean():.2f}"
    r["Inpatient admissions in prior year, mean"] = f"{num(d.ip_pre365).mean():.2f}"
    r["Total paid in prior year, median (IQR), $"] = f"{num(d.total_paid_pre365).median():,.0f} ({num(d.total_paid_pre365).quantile(.25):,.0f} to {num(d.total_paid_pre365).quantile(.75):,.0f})"
    r["Disengaged within 90 days of enrollment, %"] = f"{100*d.E2_disengaged.mean():.1f}"
    r["Completed the program (recorded), %"] = f"{100*d.graduated_ever.mean():.1f}"
    return r
T1 = pd.DataFrame({"All patients": t1col(EA), "Development (enrolled before July 2025)": t1col(EA[EA.era == "development"]),
                   "Validation (enrolled July 2025 or later)": t1col(EA[EA.era == "validation"])})
T1.index.name = "Characteristic"
C["table1"] = T1.to_dict()

# ---------------- Table 2: Aim 1 learners -----------------------------------------------------------
NAMES = {"logistic_elasticnet": "Penalized logistic regression", "discrete_time_hazard": "Discrete-time hazard model",
         "cox_ph": "Cox proportional hazards", "random_survival_forest": "Random survival forest", "random_forest": "Random forest",
         "hist_gbm": "Histogram gradient boosting", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
         "deepsurv": "DeepSurv", "ft_transformer": "FT-Transformer", "ensemble_stacked": "Stacked ensemble",
         "ensemble_stacked_calibrated": "Stacked ensemble, calibrated", "ensemble_rank_average": "Rank-average ensemble",
         "signal_risk_score": "Acute care risk score (comparator)"}
def fmt(v): return f"{v['estimate']:.3f} ({v['ci_95'][0]:.3f} to {v['ci_95'][1]:.3f})" if v.get("ci_95") else f"{v['estimate']:.3f}"
def perf_table(m):
    rows = []
    for k, lab in NAMES.items():
        if k not in m["models"]: continue
        v = m["models"][k]
        rows.append({"Model": lab, "AUROC (95% CI)": fmt(v["auroc"]), "AUPRC (95% CI)": fmt(v["auprc"]),
                     "Sensitivity": fmt(v["sensitivity"]), "Specificity": fmt(v["specificity"]),
                     "PPV": fmt(v["ppv"]), "NPV": fmt(v["npv"]), "Calibration slope": f"{v['calibration_slope']['estimate']:.2f}"})
    return pd.DataFrame(rows)
T2, T3 = perf_table(mB), perf_table(mA)
C["table2"] = T2.to_dict("records"); C["table3"] = T3.to_dict("records")

# ---------------- Table 4: confusion --------------------------------------------------------------
def conf_rows(m, label):
    rows = []
    for k in ["ensemble_stacked_calibrated", "signal_risk_score"]:
        c = m["confusion"][k]
        rows.append({"Population": label, "Model": NAMES[k], "Flagged": f"{c['flagged']:,}", "True positives": f"{c['tp']:,}",
                     "False positives": f"{c['fp']:,}", "False negatives": f"{c['fn']:,}", "True negatives": f"{c['tn']:,}",
                     "Sensitivity": f"{c['sensitivity']:.3f}", "Specificity": f"{c['specificity']:.3f}", "PPV": f"{c['ppv']:.3f}"})
    return rows
T4 = pd.DataFrame(conf_rows(mB, "At each contact") + conf_rows(mA, "At enrollment"))
C["table4"] = T4.to_dict("records")

# ---------------- Table 5: Aim 2 --------------------------------------------------------------------
rows = []
ra = conc["rank_agreement"]; cd = conc["cross_discrimination"]
rows.append({"Comparison": "Spearman correlation, predicted disengagement risk vs acute care risk percentile", "Value": f"{ra['spearman_rho']:.3f}"})
rows.append({"Comparison": "Overlap of the two top deciles (Jaccard index)", "Value": f"{ra['top_decile']['jaccard']:.3f}"})
rows.append({"Comparison": "Overlap of the two top quintiles (Jaccard index)", "Value": f"{ra['top_quintile']['jaccard']:.3f}"})
rows.append({"Comparison": "AUROC, disengagement score for any acute care event in 180 days", "Value": f"{cd['disengagement_score_for_acute_care']:.3f}"})
rows.append({"Comparison": "AUROC, acute care risk score for any acute care event in 180 days", "Value": f"{cd['risk_score_for_acute_care']:.3f}"})
rows.append({"Comparison": "AUROC, acute care risk score for disengagement within 90 days", "Value": f"{cd['risk_score_for_disengagement']:.3f}"})
for r in conc["predicted_risk_associations"]:
    if "error" in r: continue
    if "irr" in r:
        rows.append({"Comparison": f"Adjusted rate ratio per SD of predicted disengagement risk, {r['outcome'].replace('_1_180',' (days 1 to 180)').replace('ed','emergency department visits').replace('ip','inpatient admissions')}",
                     "Value": f"{r['irr']:.3f} ({r['ci_95'][0]:.3f} to {r['ci_95'][1]:.3f})"})
    else:
        rows.append({"Comparison": "Adjusted cost ratio per SD of predicted disengagement risk (days 1 to 180)",
                     "Value": f"{r.get('cost_ratio', r.get('cost_ratio_given_positive')):.3f} ({r.get('cost_ratio_ci', r.get('cost_ratio_ci'))[0]:.3f} to {r.get('cost_ratio_ci')[1]:.3f})"})
for r in conc["observed_disengagement_associations"]:
    rows.append({"Comparison": f"Observed disengagement, {r['outcome']} ({r['kind'].replace('_',' ')})",
                 "Value": f"{r['ratio']:.3f} ({r['ci_95'][0]:.3f} to {r['ci_95'][1]:.3f})"})
T5 = pd.DataFrame(rows); C["table5"] = T5.to_dict("records")

SPECL = {"unadjusted": "Unadjusted", "propensity_matched": "Exact match on acute care risk decile plus propensity match",
         "coarsened_exact_matching": "Coarsened exact matching", "overlap_weights": "Overlap weights"}
YL = {"ed_91_270": "Emergency department visits, days 91 to 270", "ip_91_270": "Inpatient admissions, days 91 to 270",
      "total_paid_91_270": "Total paid, days 91 to 270", "ed_pre365": "Negative control: emergency department visits in the prior year",
      "total_paid_pre180": "Negative control: total paid in the 180 days before enrolment",
      "negctrl_paid_1_180": "Negative control: dialysis spending"}
rows = []
for spec, lab in SPECL.items():
    v = MAT[spec]
    for y, yl in YL.items():
        e = v["estimates"].get(y)
        if not e or "ratio" not in e: continue
        rows.append({"Specification": lab, "Maximum absolute standardized difference": f"{v['balance']['max_abs_smd']:.3f}",
                     "Outcome": yl, "Ratio (95% CI)": f"{e['ratio']:.3f} ({e['ci_95'][0]:.3f} to {e['ci_95'][1]:.3f})",
                     "Disengaged": e["mean_exposed"], "Engaged": e["mean_unexposed"]})
T5b = pd.DataFrame(rows); C["table5b"] = T5b.to_dict("records")

rows = [{"Operating point": r["operating_point"], "Threshold": f"{r['threshold']:.3f}", "Contacts flagged": f"{100*r['flag_rate']:.1f}%",
         "Sensitivity": f"{r['sensitivity']:.3f}", "Specificity": f"{r['specificity']:.3f}", "PPV": f"{r['ppv']:.3f}",
         "NPV": f"{r['npv']:.3f}", "Youden J": f"{r['youden_j']:.3f}", "Number needed to flag": f"{r['number_needed_to_flag']:.2f}",
         "Contacts flagged per week": f"{r['flagged_per_week']:.1f}"} for r in THR["B"]["operating_points"]]
T7 = pd.DataFrame(rows); C["table7"] = T7.to_dict("records")

# ---------------- Table 6: Aim 3 --------------------------------------------------------------------
ALAB = {"A1_attempt_during_7day_lapse": "Logged outreach attempt during a 7-day lapse, versus none",
        "A2_inperson_chw_14d": "In-person visit by a community health worker within 14 days, versus none",
        "A3_therapy_30d": "First contact with a therapist within 30 days, versus none",
        "A4_pharmacist_30d": "First contact with a clinical pharmacist within 30 days, versus none",
        "A5_morning_weekday_call_14d": "Morning weekday telephone attempt within 14 days, versus none",
        "AC1_inperson_vs_phone_chw_14d": "Active comparator: in-person versus telephone contact by a community health worker",
        "AC2_therapy_vs_pharmacy_30d": "Active comparator: first therapist versus first clinical pharmacist contact",
        "AC3_call_vs_text_attempt_during_lapse": "Active comparator: telephone versus text attempt during a lapse"}
rows = []
for key, v in ADV.items():
    if key == "note": continue
    base, pop = key.rsplit("__", 1)
    lab = ALAB.get(base, base) + (" [high predicted risk]" if pop == "high_risk" else " [all contacts]")
    if "overlap_weighted" not in v:
        rows.append({"Contrast": lab, "Contacts": f"{v.get('n', 0):,}", "Exposed": f"{v.get('treated', 0) or 0:,}",
                     "Overlap-weighted risk difference (95% CI)": "not estimable", "Maximum absolute standardized difference": "",
                     "TMLE": "", "Causal forest": "", "Calibrated against negative controls": v.get("note", "")}); continue
    ow, tm, cf = v["overlap_weighted"], v["tmle"], v.get("causal_forest", {})
    ec = v.get("empirical_calibration")
    rows.append({"Contrast": lab, "Contacts": f"{v['n']:,}", "Exposed": f"{v['treated']:,}",
                 "Overlap-weighted risk difference (95% CI)": f"{ow['rd']:+.3f} ({ow['ci_95'][0]:+.3f} to {ow['ci_95'][1]:+.3f})",
                 "Maximum absolute standardized difference": f"{ow['max_abs_smd_weighted']:.3f}",
                 "TMLE": f"{tm['rd']:+.3f} ({tm['ci_95'][0]:+.3f} to {tm['ci_95'][1]:+.3f})",
                 "Causal forest": (f"{cf['ate']:+.3f} ({cf['ci_95'][0]:+.3f} to {cf['ci_95'][1]:+.3f})" if "ate" in cf else "not estimable"),
                 "Calibrated against negative controls": (f"{ec['calibrated_rd']:+.3f} ({ec['calibrated_ci_95'][0]:+.3f} to {ec['calibrated_ci_95'][1]:+.3f})" if ec else "")})
T6 = pd.DataFrame(rows); C["table6"] = T6.to_dict("records")

# ---------------- Table 8: documented open needs at the last contact -------------------------------
def needrows(block, label):
    return [{"Population": label, "Measure": r["measure"].capitalize(),
             "Disengaged, %": f"{r['disengaged_pct']:.1f}", "Sustained engagement, %": f"{r['sustained_pct']:.1f}",
             "Difference, percentage points": f"{r['difference_pp']:+.1f}",
             "Adjusted odds ratio (95% CI)": (f"{r['adjusted_or']:.3f} ({r['adjusted_ci_95'][0]:.3f} to {r['adjusted_ci_95'][1]:.3f})" if "adjusted_or" in r else "not estimable")}
            for r in block]
rows = (needrows(NEEDS["last_contact"], "All patients, last contact")
        + needrows(NEEDS["among_patients_with_a_care_plan"]["comparisons"], "Patients with a documented care plan")
        + needrows(NEEDS["late_contacts_with_a_care_plan"]["comparisons"], "Care plan and contact at least 30 days after enrolment"))
SPEC8 = {"unadjusted": "Unadjusted", "matched_clinical_risk_only": "Matched on acute care risk and clinical history",
         "matched_clinical_and_contact_depth": "Matched also on contact history and care-plan size",
         "overlap_weights_full": "Overlap weights on all covariates"}
rows = []
for k, lab in SPEC8.items():
    v = NM[k]
    bal = v["balance"]["max_abs_smd"] if "balance" in v else v["balance_full"]["max_abs_smd"]
    for r in v["estimates"]:
        eff = (f"{r['odds_ratio']:.3f} ({r['ci_95'][0]:.3f} to {r['ci_95'][1]:.3f})" if "odds_ratio" in r
               else f"{r['difference']:+.3f} ({r['ci_95'][0]:+.3f} to {r['ci_95'][1]:+.3f})")
        rows.append({"Specification": lab, "Maximum absolute standardized difference": f"{bal:.3f}",
                     "Measure": r["measure"].capitalize(), "Patients": f"{r['n']:,}",
                     "Disengaged": r["disengaged"], "Sustained engagement": r["sustained"],
                     "Odds ratio or difference (95% CI)": eff})
T8 = pd.DataFrame(rows); C["table8"] = T8.to_dict("records")

# ---------------- derived numbers for the text -------------------------------------------------------
vs = pd.read_parquet(D/"v4_valscores_B_full.parquet")
C["derived"] = {
 "patients_enrolled_in_period": flow["enrolled_in_study_period"],
 "taskA_patients": int(len(EA)), "taskA_dev": int((EA.era == "development").sum()), "taskA_val": int((EA.era == "validation").sum()),
 "taskB_contacts": int(len(EB)), "taskB_patients": int(EB.person_id.nunique()),
 "taskB_val_contacts": mB["n_val"], "taskB_val_patients": mB["n_val_patients"],
 "sustained_past_enrollment_pct": round(100*(1-outc["taskA"]["E2_disengaged_rate"]), 1),
 "contact_within_30d_pct": round(100*outc["taskA"]["E3_beyond_enrollment_rate"], 1),
 "completed_program_pct": round(100*outc["taskA"]["graduated_ever"], 1),
 "completion_concordance_pct": round(100*outc["completion_definition_concordance"]["graduated_with_2_goals"], 1),
 "E1_rate_pct": round(100*outc["taskB"]["E1_rate"], 1),
 "auroc_range_learners_taskB": [round(min(mB["models"][k]["auroc"]["estimate"] for k in NAMES if k in mB["models"] and not k.startswith("ensemble") and k != "signal_risk_score"), 3),
                                round(max(mB["models"][k]["auroc"]["estimate"] for k in NAMES if k in mB["models"] and not k.startswith("ensemble") and k != "signal_risk_score"), 3)],
 "sens_range_learners_taskB": [round(min(mB["models"][k]["sensitivity"]["estimate"] for k in NAMES if k in mB["models"] and not k.startswith("ensemble") and k != "signal_risk_score"), 3),
                               round(max(mB["models"][k]["sensitivity"]["estimate"] for k in NAMES if k in mB["models"] and not k.startswith("ensemble") and k != "signal_risk_score"), 3)],
 "spec_range_learners_taskB": [round(min(mB["models"][k]["specificity"]["estimate"] for k in NAMES if k in mB["models"] and not k.startswith("ensemble") and k != "signal_risk_score"), 3),
                               round(max(mB["models"][k]["specificity"]["estimate"] for k in NAMES if k in mB["models"] and not k.startswith("ensemble") and k != "signal_risk_score"), 3)],
 "events_captured_ratio": mB["comparisons"]["events_captured_ratio_ensemble_vs_risk"],
 "ensemble_vs_risk_auroc": mB["comparisons"]["ensemble_vs_risk_score_auroc"],
 "ensemble_vs_best_single": mB["comparisons"]["ensemble_vs_best_single_auroc"],
 "best_single_learner": NAMES[mB["comparisons"]["best_single_learner"]],
 "day0_share_pct": round(100*sens["enrollment_day_share"]["share_of_contacts"], 1),
 "day0_disengagement_pct": round(100*sens["enrollment_day_share"]["disengagement_rate_day0"], 1),
 "later_disengagement_pct": round(100*sens["enrollment_day_share"]["disengagement_rate_later"], 1),
 "day0_share_of_events_pct": round(100*sens["enrollment_day_share"]["share_of_events_from_day0"], 1),
 "first30_share_of_events_pct": round(100*sens["first_30_days"]["share_of_events"], 1),
 "no_attempt_after_disengagement_pct": round(100*(1-sens["attempts_after_disengagement"]["share_with_any_attempt_90d"]), 1),
 "landmark_auroc_range": land["auroc_range"], "headline_landmark": land["headline_landmark"],
 "landmark_headline": land["landmarks"][land["headline_landmark"]],
 "cost_q1": conc["mean_cost_by_quintile"]["q1"], "cost_q5": conc["mean_cost_by_quintile"]["q5"],
 "equalized_odds_race": mB["fairness"]["race"]["equalized_odds_ratio"],
 "equalized_odds_state": mB["fairness"]["state"]["equalized_odds_ratio"],
 "n_actions_passing_gates": int(sum(1 for v in act["actions"].values() if v.get("gates", {}).get("all_pass"))),
 "youden": {k: THR["B"]["operating_points"][-4][k] for k in ["threshold", "flag_rate", "sensitivity", "specificity", "ppv", "youden_j", "number_needed_to_flag", "flagged_per_week"]},
 "top20": {k: THR["B"]["operating_points"][2][k] for k in ["flag_rate", "sensitivity", "specificity", "ppv", "number_needed_to_flag", "flagged_per_week"]},
 "top30": {k: THR["B"]["operating_points"][3][k] for k in ["flag_rate", "sensitivity", "specificity", "ppv"]},
 "match_ed": MAT["propensity_matched"]["estimates"]["ed_91_270"], "match_ip": MAT["propensity_matched"]["estimates"]["ip_91_270"],
 "match_cost": MAT["propensity_matched"]["estimates"]["total_paid_91_270"],
 "match_negctrl_ed": MAT["propensity_matched"]["estimates"]["ed_pre365"],
 "match_negctrl_cost": MAT["propensity_matched"]["estimates"]["total_paid_pre180"],
 "match_pairs": MAT["propensity_matched"]["pairs"], "match_smd": MAT["propensity_matched"]["balance"]["max_abs_smd"],
 "unadj_cost": MAT["unadjusted"]["estimates"]["total_paid_91_270"], "unadj_negctrl_cost": MAT["unadjusted"]["estimates"]["total_paid_pre180"],
 "ow_ed": MAT["overlap_weights"]["estimates"]["ed_91_270"], "ow_cost": MAT["overlap_weights"]["estimates"]["total_paid_91_270"],
 "ow_smd": MAT["overlap_weights"]["balance"]["max_abs_smd"],
 "within_decile_ed_ratios": [v["ed_ratio"] for v in MAT["within_risk_decile"].values()],
 "within_decile_cost_ratios": [v["cost_ratio"] for v in MAT["within_risk_decile"].values()],
 "adv": {k: ADV[k] for k in ADV if k != "note"},
 "needs_last_contact": {r["measure"]: r for r in NEEDS["last_contact"]},
 "needs_with_plan": {r["measure"]: r for r in NEEDS["among_patients_with_a_care_plan"]["comparisons"]},
 "needs_late_plan": {r["measure"]: r for r in NEEDS["late_contacts_with_a_care_plan"]["comparisons"]},
 "care_plan_presence": NEEDS["care_plan_presence"],
 "needs_counts": NEEDS["last_contact_counts"],
 "needs_n_with_plan": NEEDS["among_patients_with_a_care_plan"]["n"],
 "needs_n_late_plan": NEEDS["late_contacts_with_a_care_plan"]["n"],
 "needs_text_kappa_medical": NEEDS["text_lexicon_vs_structured"]["medical"]["kappa"],
 "needs_text_kappa_social": NEEDS["text_lexicon_vs_structured"]["social"]["kappa"],
 "needs_top_categories": NEEDS["by_category"][:6],
 "needs_matched": {k: {"max_smd": (NM[k]["balance"]["max_abs_smd"] if "balance" in NM[k] else NM[k]["balance_full"]["max_abs_smd"]),
                       "n": NM[k].get("n", NM["n"]), "pairs": NM[k].get("pairs"),
                       "estimates": {r["measure"]: r for r in NM[k]["estimates"]}}
                   for k in ["unadjusted", "matched_clinical_risk_only", "matched_clinical_and_contact_depth", "overlap_weights_full"]},
 "needs_by_contacts": NM["by_number_of_prior_contacts"],
 "excluded_contacts_mean": RC["excluded_vs_included"]["mean_completed_contacts_excluded"],
 "included_contacts_mean": RC["excluded_vs_included"]["mean_completed_contacts_analysed"],
 "risk_iqr": RC["risk_score_range"]["iqr"], "risk_median": RC["risk_score_range"]["median"],
 "auroc_by_state": {k: v["auroc_ensemble"] for k, v in RC["discrimination_by_state"].items()},
 "risk_auroc_by_state": {k: v["auroc_risk_score"] for k, v in RC["discrimination_by_state"].items()},
 "ac1_mde": RC["active_comparator_power"]["AC1_inperson_vs_phone_chw_14d__all_risk"]["mde_risk_difference_80pct_power"],
 "ac2_mde": RC["active_comparator_power"]["AC2_therapy_vs_pharmacy_30d__all_risk"]["mde_risk_difference_80pct_power"],
 "race_missing_n": RG["missing_data"]["race_missing_n"], "race_missing_pct": RG["missing_data"]["race_missing_pct"],
 "candidate_predictors": RG["study_size"]["candidate_predictors"], "validation_events": RG["study_size"]["validation_events"],
 "development_events": RG["study_size"]["development_events"],
 "epp_development": round(RG["study_size"]["development_events"]/RG["study_size"]["candidate_predictors"], 1),
 "median_dp_per_patient": RG["follow_up"]["median_decision_points_per_patient"],
 "iqr_dp_per_patient": RG["follow_up"]["iqr_decision_points"],
 "median_days_to_last_contact": RG["follow_up"]["median_days_enrolment_to_last_contact"],
 "iqr_days_to_last_contact": RG["follow_up"]["iqr_days"],
 "patient_years": RG["follow_up"]["total_patient_years"],
 "quintile_cuts": RG["category_boundaries"]["predicted_risk_quintile_cuts"],
 "spending_quartile_cuts": RG["category_boundaries"]["prior_spending_quartile_cuts_usd"],
}
json.dump(C, open(R/"canonical.json", "w"), indent=1, default=str)

# ---------------- tables to markdown ------------------------------------------------------------------
def md(df, idx=False): return df.to_markdown(index=idx)
# main exhibits for a 5-exhibit journal: cohort, performance at both index points, operating points
KEEP = ["Stacked ensemble, calibrated", "Acute care risk score (comparator)"]
best_b = NAMES[mB["comparisons"]["best_single_learner"]]; best_a = NAMES[mA["comparisons"]["best_single_learner"]]
def slim(tbl, best, label):
    rows = [dict(r) for r in tbl.to_dict("records") if r["Model"] in KEEP + [best]]
    for r in rows: r["Index point"] = label
    return rows
T2main = pd.DataFrame(slim(T2, best_b, "At each contact") + slim(T3, best_a, "At enrolment"))
T2main = T2main[["Index point", "Model", "AUROC (95% CI)", "AUPRC (95% CI)", "Sensitivity", "Specificity", "PPV", "Calibration slope"]]
C["table2_main"] = T2main.to_dict("records")
main = {"Table 1": T1.reset_index(), "Table 2": T2main, "Table 3": T7}
appx = {"eTable 1. Full learner performance at each contact": T2,
        "eTable 2. Full learner performance at enrolment": T3,
        "eTable 3. Confusion matrices at a 20% flagging fraction": T4,
        "eTable 4. Agreement with acute care risk, and associations with utilization and cost": T5,
        "eTable 5. Utilization and cost among patients matched on baseline risk": T5b,
        "eTable 6. Care-team actions, estimators and active comparators": T6,
        "eTable 7. Open needs at the last contact, among comparable patients": T8}
md2 = lambda d: "\n\n".join(f"**{k}**\n\n{v.to_markdown(index=False)}" for k, v in d.items())
(NB/"tables_main_v4.md").write_text(md2(main))
(NB/"tables_appendix_v4.md").write_text(md2(appx))
tb = {"Table 1": T1.reset_index(), "Table 2": T2, "Table 3": T3, "Table 4": T4, "Table 5": T5,
      "Table 5b": T5b, "Table 6": T6, "Table 7": T7, "Table 8": T8}
(NB/"tables_v4.md").write_text("\n\n".join(f"**{k}**\n\n{md(v)}" for k, v in tb.items()))

# ---------------- figures -------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
sc = np.load(R/"val_scores_B_full.npy"); pe, pr, yv = sc[:, 0], sc[:, 1], sc[:, 2]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for p, lab in [(pe, "Stacked ensemble"), (pr, "Acute care risk score")]:
    f, t, _ = roc_curve(yv, p); ax[0].plot(f, t, label=lab)
    pc, rc, _ = precision_recall_curve(yv, p); ax[1].plot(rc, pc, label=lab)
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].legend(frameon=False, fontsize=7)
ax[1].axhline(yv.mean(), color="k", ls=":", lw=0.8); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
q = pd.qcut(pe, 10, labels=False, duplicates="drop")
ax[2].plot([pd.Series(pe)[q == i].mean() for i in range(10)], [yv[q == i].mean() for i in range(10)], "o-")
ax[2].plot([0, 1], [0, 1], "k:", lw=0.8); ax[2].set_xlabel("Predicted probability"); ax[2].set_ylabel("Observed proportion")
for a, t in zip(ax, "abc"): a.set_title(t, loc="left", fontweight="bold")
fig.savefig(FIG/"figure2_discrimination.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3))
ks = sorted(land["landmarks"]); au = [land["landmarks"][k]["auroc"] for k in ks]
ax.plot(range(len(ks)), au, "o-"); ax.set_xticks(range(len(ks))); ax.set_xticklabels([k[:7] for k in ks], rotation=45, ha="right")
ax.set_ylabel("AUROC on later contacts"); ax.set_ylim(0.75, 0.95); ax.set_xlabel("Model frozen at")
fig.savefig(FIG/"figure3_landmarks.png"); plt.close(fig)

A2 = A[A.eligible_cost].merge(pd.read_parquet(D/"v4_score_all_A_full.parquet")[["person_id", "p_ensemble"]], on="person_id")
A2["q"] = pd.qcut(A2.p_ensemble, 5, labels=False)
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
ax[0].bar(range(5), A2.groupby("q").total_paid_1_180.mean()); ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f"Q{i+1}" for i in range(5)])
ax[0].set_xlabel("Predicted disengagement risk quintile"); ax[0].set_ylabel("Total paid, days 1 to 180 ($)")
ax[1].bar(range(5), A2.groupby("q").ed_1_180.mean()); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f"Q{i+1}" for i in range(5)])
ax[1].set_xlabel("Predicted disengagement risk quintile"); ax[1].set_ylabel("Emergency department visits per patient")
ax[2].scatter(pd.to_numeric(A2.risk_percentile), A2.p_ensemble, s=2, alpha=0.15)
ax[2].set_xlabel("Acute care risk percentile"); ax[2].set_ylabel("Predicted disengagement risk")
for a, t in zip(ax, "abc"): a.set_title(t, loc="left", fontweight="bold")
fig.savefig(FIG/"figure4_concordance.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
items = [(ALAB[k.split("__")[0]] + (" (high risk)" if k.endswith("high_risk") else " (all)"), v) for k, v in act["actions"].items() if "aipw" in v]
ypos = range(len(items))
ax.errorbar([v["aipw"]["rd"] for _, v in items], list(ypos),
            xerr=[[v["aipw"]["rd"]-v["aipw"]["ci_95"][0] for _, v in items], [v["aipw"]["ci_95"][1]-v["aipw"]["rd"] for _, v in items]],
            fmt="o", capsize=2)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(list(ypos)); ax.set_yticklabels([l for l, _ in items], fontsize=7)
ax.set_xlabel("Difference in sustained engagement at 90 days (AIPW)")
fig.savefig(FIG/"figure5_actions.png"); plt.close(fig)
print("canonical.json, tables_v4.md, and 4 figures written")
print(json.dumps(C["derived"], indent=1, default=str))
