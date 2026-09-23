#!/usr/bin/env bash
set -euo pipefail
PY=/opt/anaconda3/bin/python3
$PY scripts/00_pull.py
$PY scripts/00b_pull_member_months.py
$PY scripts/00c_pull_tasks.py
$PY scripts/01_cohort.py
$PY scripts/02_outcomes.py
$PY scripts/03_features.py
$PY scripts/04_text.py
$PY scripts/05_models.py B full
$PY scripts/06_ensemble_metrics.py B full
$PY scripts/05_models.py A full
$PY scripts/06_ensemble_metrics.py A full
$PY scripts/08_concordance.py
$PY scripts/09_actions.py
$PY scripts/11_needs_structured.py
$PY scripts/12_landmarks.py
$PY scripts/13_sensitivity.py
$PY scripts/15_matching.py
$PY scripts/16_actions_advanced.py
$PY scripts/17_thresholds.py
$PY scripts/18_needs_matched.py
$PY scripts/19_reviewer_checks.py
$PY scripts/20_reporting_gaps.py
$PY scripts/21_design_feasibility.py
$PY scripts/22_examiner_design.py
$PY scripts/23_staff_departure.py
$PY scripts/24_claims_based_need.py
$PY scripts/25_open_tasks.py
$PY scripts/27_parsimonious_baseline.py
$PY scripts/28_coverage_sensitivity.py
$PY scripts/29_equalized_odds.py
$PY scripts/30_youden_confusion.py
$PY scripts/31_foundation_models.py A 8   # post hoc comparators; TabPFN needs TABPFN_TOKEN
$PY scripts/31_foundation_models.py B 8
$PY scripts/32_extended_ensemble.py A 8   # sensitivity: TabPFN added to the stack
$PY scripts/32_extended_ensemble.py B 8
$PY scripts/33_table_intervals.py
$PY scripts/14_report.py
$PY audit_consistency.py SageOpenMed
