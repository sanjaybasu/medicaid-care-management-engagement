#!/usr/bin/env bash
set -euo pipefail
PY=/opt/anaconda3/bin/python3
$PY scripts/00_pull.py
$PY scripts/00b_pull_member_months.py
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
$PY scripts/10_physician_sample.py
$PY scripts/12_landmarks.py
$PY scripts/13_sensitivity.py
$PY scripts/14_report.py
$PY audit_consistency.py
