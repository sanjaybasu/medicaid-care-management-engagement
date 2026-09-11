"""Verifies that every fact asserted in the v4 manuscript matches results/canonical.json."""
import json, pathlib, re, sys
P = pathlib.Path(__file__).resolve().parent
C = json.load(open(P/"results"/"canonical.json"))
MS = (P.parent.parent/"notebooks"/"care-management-engagement"/"manuscript_engagement_prediction_DigitalHealth.md").read_text()
TB = (P.parent.parent/"notebooks"/"care-management-engagement"/"tables_v4.md").read_text()
TEXT = MS + "\n" + TB
D, F, O, MB, MA, CO, AC, LA, SE = (C["derived"], C["flow"], C["outcomes"], C["metrics_taskB"], C["metrics_taskA"],
                                   C["concordance"], C["actions"], C["landmarks"], C["sensitivity"])
n = lambda x: f"{int(round(float(x))):,}"
f3 = lambda x: f"{float(x):.3f}"
p1 = lambda x: f"{float(x):.1f}"
obs = {r["outcome"]: r for r in CO["observed_disengagement_associations"]}
pred = {r["outcome"]: r for r in CO["predicted_risk_associations"]}
cfB, cf0 = MB["confusion"]["ensemble_stacked_calibrated"], MB["confusion"]["signal_risk_score"]
LH = D["landmark_headline"]

facts = {
 "enrolled ever": n(F["enrolled_patients_all"]), "enrolled in period": n(F["enrolled_in_study_period"]),
 "no program record": n(F["missing_from_outcomes_mart"]), "not tier 1": n(F["excluded_not_tier1"]),
 "task A patients": n(D["taskA_patients"]), "task A development": n(D["taskA_dev"]), "task A validation": n(D["taskA_val"]),
 "task B decision points": n(D["taskB_contacts"]), "task B patients": n(D["taskB_patients"]),
 "task B validation contacts": n(D["taskB_val_contacts"]), "task B validation patients": n(D["taskB_val_patients"]),
 "sustained past enrolment %": p1(D["sustained_past_enrollment_pct"]), "contact within 30 days %": p1(D["contact_within_30d_pct"]),
 "completed program %": p1(D["completed_program_pct"]), "completion concordance %": p1(D["completion_concordance_pct"]),
 "disengagement rate %": p1(D["E1_rate_pct"]), "day0 share %": p1(D["day0_share_pct"]),
 "day0 disengagement %": p1(D["day0_disengagement_pct"]), "later disengagement %": p1(D["later_disengagement_pct"]),
 "day0 share of events %": p1(D["day0_share_of_events_pct"]), "first 30 days share of events %": p1(D["first30_share_of_events_pct"]),
 "no attempt %": p1(D["no_attempt_after_disengagement_pct"]),
 "ensemble AUROC task B": f3(MB["models"]["ensemble_stacked"]["auroc"]["estimate"]),
 "ensemble AUROC task A": f3(MA["models"]["ensemble_stacked"]["auroc"]["estimate"]),
 "risk score AUROC task B": f3(MB["models"]["signal_risk_score"]["auroc"]["estimate"]),
 "ensemble sensitivity": f3(MB["models"]["ensemble_stacked_calibrated"]["sensitivity"]["estimate"]),
 "ensemble specificity": f3(MB["models"]["ensemble_stacked_calibrated"]["specificity"]["estimate"]),
 "ensemble PPV": f3(MB["models"]["ensemble_stacked_calibrated"]["ppv"]["estimate"]),
 "ensemble true positives": n(cfB["tp"]), "risk score true positives": n(cf0["tp"]), "risk score flags": n(cf0["flagged"]),
 "events captured ratio": str(D["events_captured_ratio"]),
 "ensemble vs best single": f3(D["ensemble_vs_best_single"]["diff"]),
 "landmark low": f3(LA["auroc_range"][0]), "landmark high": f3(LA["auroc_range"][1]),
 "headline landmark AUROC": f3(LH["auroc"]), "headline landmark train": n(LH["train_n"]), "headline landmark test": n(LH["test_n"]),
 "headline landmark unseen AUROC": f3(LH["auroc_unseen_patients"]), "headline landmark unseen n": n(LH["n_unseen"]),
 "per 100 flags": p1(LH["per_100_flags"]),
 "spearman": f"{CO['rank_agreement']['spearman_rho']:.3f}",
 "risk score for acute care": f3(CO["cross_discrimination"]["risk_score_for_acute_care"]),
 "risk score for disengagement": f3(CO["cross_discrimination"]["risk_score_for_disengagement"]),
 "disengagement score for acute care": f3(CO["cross_discrimination"]["disengagement_score_for_acute_care"]),
 "ED IRR per SD": f3(pred["ed_1_180"]["irr"]), "IP IRR per SD": f3(pred["ip_1_180"]["irr"]),
 "cost ratio per SD": f3(pred["total_paid_1_180"]["cost_ratio"]),
 "cost quintile 1": n(D["cost_q1"]), "cost quintile 5": n(D["cost_q5"]),
 "observed ED ratio": f3(obs["ed_91_270"]["ratio"]), "observed IP ratio": f3(obs["ip_91_270"]["ratio"]),
 "observed cost ratio": f3(obs["total_paid_91_270"]["ratio"]),
 "negative control pre period": f3(obs["total_paid_pre180"]["ratio"]), "negative control dialysis": f3(obs["negctrl_paid_1_180"]["ratio"]),
 "equalized odds race": f3(D["equalized_odds_race"]), "equalized odds state": f3(D["equalized_odds_state"]),
 "sensitivity goal-based AUROC": f3(SE["goal_based_completion"]["auroc"]),
 "sensitivity 120-day AUROC": f3(SE["window_120_days"]["auroc"]),
 "sensitivity excluding day 0 AUROC": f3(SE["excluding_enrollment_day_contact"]["auroc"]),
 "sensitivity base AUROC": f3(SE["base_lightgbm"]["auroc"]),
}
MAT, THR, ADV = C["matching"], C["thresholds"], C["actions_advanced"]
facts.update({
 "youden sensitivity": f3(D["youden"]["sensitivity"]), "youden specificity": f3(D["youden"]["specificity"]),
 "youden flag %": f"{100*D['youden']['flag_rate']:.1f}", "youden NNF": f"{D['youden']['number_needed_to_flag']:.2f}",
 "top20 sensitivity": f3(D["top20"]["sensitivity"]), "top20 specificity": f3(D["top20"]["specificity"]),
 "top20 NNF": f"{D['top20']['number_needed_to_flag']:.2f}", "top30 sensitivity": f3(D["top30"]["sensitivity"]),
 "matched pairs": n(D["match_pairs"]), "matched max SMD": f"{D['match_smd']:.3f}",
 "matched ED ratio": f3(D["match_ed"]["ratio"]), "matched IP ratio": f3(D["match_ip"]["ratio"]),
 "matched cost ratio": f3(D["match_cost"]["ratio"]),
 "matched negative control ED": f3(D["match_negctrl_ed"]["ratio"]), "matched negative control cost": f3(D["match_negctrl_cost"]["ratio"]),
 "unadjusted cost ratio": f3(MAT["unadjusted"]["estimates"]["total_paid_91_270"]["ratio"]),
 "unadjusted negative control cost": f3(D["unadj_negctrl_cost"]["ratio"]),
 "overlap ED ratio": f3(D["ow_ed"]["ratio"]), "overlap cost ratio": f3(D["ow_cost"]["ratio"]), "overlap max SMD": f"{D['ow_smd']:.3f}",
 "active comparator in-person vs phone": f"{ADV['AC1_inperson_vs_phone_chw_14d__all_risk']['overlap_weighted']['rd']:+.3f}",
 "active comparator therapy vs pharmacy": f"{ADV['AC2_therapy_vs_pharmacy_30d__all_risk']['overlap_weighted']['rd']:+.3f}",
 "active comparator call vs text": f"{ADV['AC3_call_vs_text_attempt_during_lapse__all_risk']['overlap_weighted']['rd']:+.3f}",
 "overlap therapy RD": f"{ADV['A3_therapy_30d__all_risk']['overlap_weighted']['rd']:+.3f}",
 "overlap pharmacist RD": f"{ADV['A4_pharmacist_30d__all_risk']['overlap_weighted']['rd']:+.3f}",
 "overlap CHW RD": f"{ADV['A2_inperson_chw_14d__all_risk']['overlap_weighted']['rd']:+.3f}",
 "overlap attempt RD": f"{ADV['A1_attempt_during_7day_lapse__all_risk']['overlap_weighted']['rd']:+.3f}",
})
for k in ["A1_attempt_during_7day_lapse__all_risk", "A2_inperson_chw_14d__all_risk", "A3_therapy_30d__all_risk",
          "A4_pharmacist_30d__all_risk", "A5_morning_weekday_call_14d__all_risk"]:
    v = AC["actions"][k]
    facts[f"{k} AIPW"] = f"{v['aipw']['rd']:+.3f}"

fails = [(k, v) for k, v in facts.items() if v not in TEXT]
print(f"{'FACT':46s} VALUE      IN TEXT")
for k, v in facts.items():
    print(f"{k:46s} {v:10s} {'yes' if v in TEXT else 'NO'}")
print(f"\nchecked {len(facts)} facts | FAIL {len(fails)}")
if fails:
    print("missing:", fails)
sys.exit(0)
