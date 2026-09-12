# care-management-engagement (v4)

Code for "Predicting engagement after enrolment in Medicaid care management, and what it does and does not predict about acute care use and cost". No data or protected health information is contained in this repository; `data_cache/` and `results/` are gitignored.

The locked analysis plan is `PREREGISTRATION_v4_engagement.md`. Every number in the manuscript is written by the pipeline to `results/canonical.json`, and `audit_consistency.py` verifies that the manuscript and tables match it.

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
| `scripts/12_landmarks.py` | Frozen-model landmark validation |
| `scripts/13_sensitivity.py` | Pre-specified sensitivity analyses |
| `scripts/14_report.py` | Canonical numbers, tables, figures |
| `audit_consistency.py` | Verifies the manuscript against the canonical file |

Run order: 00, 00b, 00c, 01, 02, 03, 04, 05 (B then A), 06 (B then A), 08, 09, 11, 12, 13, 15, 16, 17, 14, audit.

## License

MIT. See `LICENSE`. The code is released without data: no patient data, no intermediate results, and no manuscript files are contained in this repository.
