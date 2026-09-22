# care-management-engagement (v4)

Code for "Predicting disengagement from Medicaid care management: a retrospective cohort study of chronic disease self-management support, acute care use, and cost". No data or protected health information is contained in this repository; `data_cache/` and `results/` are gitignored.

The locked analysis plan is `PREREGISTRATION_v4_engagement.md`. Every number in the manuscript is written by the pipeline to `results/canonical.json`, and `audit_consistency.py` verifies that the manuscript and tables match it.

## Relationship to `medicaid-care-management-disengagement`

That repository is **a different study on overlapping source data, not an earlier version of this one.** It uses a different cohort definition (52,961 decision points from 8,853 patients, selected by a tier-1 filter on activation records) where this study is enrollment-anchored and requires 90 days of continuous health plan coverage (20,321 decision points from 5,727 patients). Results are not interchangeable between the two: each has its own `results/canonical.json`, and each manuscript is verified only against its own.

## Scope note

`scripts/09_actions.py` and `scripts/16_actions_advanced.py` evaluate care-team actions with falsification gates and active comparators. That analysis was removed from the manuscript during revision and is retained here only for reproducibility of the earlier drafts; it supports no number in the submitted paper.


| Script | Purpose |
|---|---|
| `scripts/00_pull.py` | Extracts status history, encounters and attempts, goals, admission-discharge-transfer events, risk scores, the person-month outcomes mart, and claim lines for every enrolled patient |
| `scripts/00b_pull_member_months.py` | Health-plan coverage months |
| `scripts/01_cohort.py` | Cohort, index events, coverage, temporal split |
| `scripts/02_outcomes.py` | Outcomes E1 to E4, program completion, utilization and cost windows |
| `scripts/03_features.py` | Structured feature blocks |
| `scripts/04_text.py` | Note-text features: lexicon, TF-IDF with truncated SVD, sentence embeddings |
| `scripts/05_models.py` | Eleven-learner library with patient-grouped cross-fitting |
| `scripts/nets.py`, `scripts/run_ft.py` | DeepSurv and FT-Transformer implementations, and the subprocess runner |
| `scripts/06_ensemble_metrics.py` | Stacking, calibration, metrics with bootstrap intervals, decision curve, fairness |
| `scripts/08_concordance.py` | Agreement with acute care risk; utilization and cost associations |
| `scripts/09_actions.py` | Care-team actions with falsification gates |
| `scripts/11_needs_structured.py` | Open medical and social needs at the last contact, from the structured care plan |
| `scripts/00c_pull_tasks.py` | Care-plan tasks |
| `scripts/15_matching.py` | Utilization and cost among patients matched on baseline risk: within-decile pooling, propensity matching, coarsened exact matching, overlap weights |
| `scripts/16_actions_advanced.py` | Actions with overlap weights, TMLE, doubly robust learner, causal forest, active comparators, and empirical calibration |
| `scripts/17_thresholds.py` | Operating points, including the Youden-optimal threshold |
| `scripts/18_needs_matched.py` | Open care-plan needs at the last contact among patients matched on pre-exposure covariates |
| `scripts/19_reviewer_checks.py` | Excluded-versus-analysed comparison, risk-percentile range, and other quantities reviewers ask for |
| `scripts/20_reporting_gaps.py` | Quantities required by the TRIPOD+AI and STROBE checklists |
| `scripts/21_design_feasibility.py` | Feasibility checks for examiner, capacity, and staff-departure designs |
| `scripts/22_examiner_design.py` | Assigned staff member as an instrument for sustained engagement, conditional on assignment cell |
| `scripts/23_staff_departure.py` | Staff departure as a natural experiment for engagement |
| `scripts/24_claims_based_need.py` | Unfinished clinical work measured from claims at the last contact (Appendix Table 10) |
| `scripts/25_open_tasks.py` | Open tasks at the last contact, by task type (Appendix Table 11) |
| `scripts/27_parsimonious_baseline.py` | Contact-timing-only logistic comparators on the same validation cohort (Appendix Table 12) |
| `scripts/28_coverage_sensitivity.py` | Utilization and cost with continuous coverage required through day 270 (Appendix Table 13) |
| `scripts/29_equalized_odds.py` | Equalized odds ratios by race and ethnicity, state, sex, and age band with patient-clustered bootstrap CIs (Appendix Table 3) |
| `scripts/30_youden_confusion.py` | Confusion matrices at the Youden-optimal threshold with the risk score at matched flag volume (Table 4) |
| `scripts/12_landmarks.py` | Frozen-model landmark validation |
| `scripts/13_sensitivity.py` | Pre-specified sensitivity analyses |
| `scripts/14_report.py` | Canonical numbers, tables, figures |
| `audit_consistency.py` | Verifies the manuscript against the canonical file |

Run order: 00, 00b, 00c, 01, 02, 03, 04, 05 (B then A), 06 (B then A), 08, 09, 11, 12, 13, 15, 16, 17, 14, audit.

## License

MIT. See `LICENSE`. The code is released without data: no patient data, no intermediate results, and no manuscript files are contained in this repository.
