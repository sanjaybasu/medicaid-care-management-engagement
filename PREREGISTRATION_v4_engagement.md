# Pre-registration: predicting engagement after enrollment in Medicaid care management, its relation to acute care use and cost, and the care-team actions that sustain it

Version 4.0. Locked 2026-09-11. Sanjay Basu (analysis lead), John Morgan, Christian Moher, Parth Sheth, Namrata Elamaran, Aakriti Kinra, Benjamin Huynh, Rajaie Batniji. WCG IRB protocol 20253751 (waiver of consent and HIPAA authorization). Target journal: Digital Health (SAGE), Original Research Article. Reporting guidelines: TRIPOD+AI (Aim 1), STROBE (Aims 2 and 3).

## 0. Provenance and status of this plan

This plan replaces the v3 series (v3.0 with amendments v3.1 to v3.7) in full. It is a re-specification written after those analyses of an overlapping dataset, not a blinded registration, and the manuscript will describe the study as a retrospective cohort analysis of routinely collected data. The cohort definition, the index events, the outcome set, the model library, and two of the three aims differ from the v3 series. Numbers from the v3 series are not carried forward; every number in the new manuscript will be re-derived by the v4 pipeline in `packaging/care-management-engagement/` and written to `results/canonical.json`. This document is locked before the v4 pipeline is run and before any v4 outcome is examined. Changes after that point are recorded in Section 16 with the date, the reason, and whether the change was made before or after the affected outcome was seen.

## 1. Background and rationale

Medicaid care management programs identify patients at high risk of emergency department visits, hospital admissions, and high health care spending, and assign a multidisciplinary team to them. The benefit of these programs is concentrated among patients who stay engaged: patients who stop responding receive fewer services and, in trial and observational evidence, show smaller changes in utilization. Programs therefore order outreach by an acute care risk score, which answers the question of who is at risk of a hospital event but not the question of who will still be reachable. Whether the risk of disengagement can be predicted from data a program already holds, whether that risk identifies the same patients as the acute care risk score, and which care-team actions restore engagement among patients at high predicted risk of disengagement have not been established.

## 2. Objectives

**Aim 1 (prediction).** Predict, at enrollment and at each subsequent completed contact, whether a patient will sustain engagement with the care management program. Compare a pre-specified library of learners and their ensemble against the acute care risk score the program uses to order outreach.

**Aim 2 (concordance and consequence).** Quantify how far predicted disengagement risk agrees with predicted and observed risk of emergency department visits, inpatient admissions, and total paid cost, and estimate the association between disengagement and subsequent utilization and cost.

**Aim 3 (actions).** Among patients at high predicted risk of disengagement, estimate which care-team actions are followed by sustained engagement, using within-patient and weighted designs with pre-specified falsification gates. Describe, with natural language processing of care notes validated against blinded physician adjudication, how often unresolved medical or social needs are documented at the last contact before disengagement compared with contacts of patients who sustain engagement.

### Hypotheses

H1. An ensemble of learners trained on the care record discriminates disengagement better than the acute care risk score (pre-specified superiority in AUROC on the temporal validation set).

H2. Predicted disengagement risk is weakly correlated with the acute care risk score (pre-specified interpretation: |Spearman rho| < 0.30 indicates the two scores rank different patients), and is associated with subsequent acute care use and cost after adjustment for the acute care risk score.

H3. At least one care-team action is followed by a higher probability of sustained engagement among high-risk patients, with the effect estimate surviving all pre-specified falsification gates.

## 3. Setting, data sources, and study period

Waymark delivers community-based care management to Medicaid managed-care members in Ohio, Virginia, and Washington through local teams of community health workers, care coordinators, clinical pharmacists, and therapists, under capitated contracts with health plans. Data sources:

1. `lighthouse` (care-team application database): patient status history, completed encounters and logged attempts with note text, timestamps, disciplines and channels, care-plan goals and tasks, staff assignment, and the admission-discharge-transfer feed, which is the source of truth for emergency department and inpatient events.
2. `coredb` schema `dbt_tuva_core` (claims mart: `medical_claim`, `pharmacy_claim`, `member_months`, `condition`) and schema `dbt` (`cost_util_long_format`, person-month paid amounts and encounter counts with enrolled days; `mco_hedis_all_files`).
3. The deployed Signal acute care risk model output (percentile), which predicts avoidable acute care utilization and which the program uses to order outreach lists.

Study period: 2023-05-23 (first enrollment in the care-team application) to 2026-06-07 (extraction date for contacts). Claims are complete through 2026-04-30 by the completeness rule in Section 6.3; analyses that use cost are restricted accordingly.

## 4. Cohort

**Unit of enrollment.** A patient enters the cohort at enrollment, defined as the first transition to the `ACTIVATED` status in the patient status history. This event is called *enrollment* throughout the study; the term *activation* is not used.

**Inclusion.** (a) Enrollment between 2023-05-23 and 2026-06-07; (b) designated rising-risk (program tier 1) at enrollment, the tier for which longitudinal care management is intended, because patients in the health-plan quality-measure tier may require only a single contact to schedule an appointment; (c) continuous health-plan enrollment for all 90 days after the index event (Section 5), so that loss of coverage cannot be recorded as disengagement.

**Exclusion.** Death before the end of the outcome window (patients are censored at death in time-to-event analyses and excluded from binary-outcome analyses whose window they do not complete); enrollment records with no completed contact at any time; contacts of development-era patients dated on or after the temporal cutoff (Section 8).

**Index events.** Aim 1 has two prediction tasks with different index events.

- *Task A, enrollment-time prediction.* One row per patient. Index is the enrollment date. Predictors use only information timestamped before the index instant.
- *Task B, dynamic prediction.* One row per completed care-team contact within 365 days of enrollment (decision point). Index is the contact. Predictors use the record as it stood at that contact, including all prior contacts.

## 5. Outcomes

### 5.1 Engagement outcomes (Aim 1, and the outcome for Aim 3)

**E1 (primary, Task B).** Disengagement after a completed contact: no further completed care-team contact within 90 days of the index contact and no program completion (Section 5.2) within those 90 days. Patients must have 90 days of continuous plan enrollment after the contact to be eligible.

**E2 (primary, Task A).** Sustained engagement through day 90 after enrollment: at least one completed contact in days 1 to 90 after enrollment with no 90-day contact-free interval, or program completion by day 90. Its complement is disengagement within 90 days of enrollment.

**E3 (secondary, Task A).** Engagement beyond enrollment: at least one completed contact after the enrollment contact and within 30 days of it.

**E4 (time-to-event, Task B).** Time from the index contact to the start of the first 90-day contact-free interval, censored at loss of plan enrollment, death, program completion, or 2026-06-07, whichever comes first. This is the outcome for the survival learners in Section 7.

### 5.2 Program completion

Program completion (the program's term is graduation) is recorded as the `GRADUATED` patient status. The study's primary definition is that recorded status. A secondary definition, pre-specified because the framework for this study defines completion by goal attainment, is the completion of at least two care-plan goals, where care-plan goals are goals of type `STANDARD` or `SIGNAL` (excluding the health-plan quality-measure goals of type `HEDIS`, which are program-set rather than set with the patient) and completion is the status `COMPLETED` or `PROVISIONALLY_COMPLETED`. Concordance between the two definitions will be reported; in a feasibility count made before this plan was locked, 89.6% of patients with a recorded `GRADUATED` status had at least two completed care-plan goals, against 1.7% of other patients. Analyses use the recorded status; the goal-based definition is a pre-specified sensitivity analysis for every outcome in which program completion appears.

### 5.3 Utilization and cost outcomes (Aim 2)

Measured over days 1 to 180 after the index event, and, for the analyses in which observed disengagement is the exposure, over days 31 to 210 with the first 30 days blanked (Section 9.3).

- **U1.** Emergency department visits, from the admission-discharge-transfer feed, de-duplicated to stays (events with the same patient and an admission within 24 hours of a prior discharge are one stay).
- **U2.** Inpatient admissions, from the same feed, de-duplicated to stays.
- **U3.** Total paid amount, from claims (`dbt.cost_util_long_format` person-month paid, cross-checked against the sum of `dbt_tuva_core.medical_claim.paid_amount` plus pharmacy paid), expressed per member per month using enrolled days as the denominator, winsorized at the 99th percentile, and reported in nominal dollars with a sensitivity analysis adjusting to 2026 dollars by the Consumer Price Index medical care component.
- **U4 (secondary).** Acute care events from claims (emergency department and acute inpatient encounter types), reported alongside U1 and U2 because claims capture only a fraction of admission-discharge-transfer events.
- **U5 (negative control outcome).** Dental and vision paid amount, chosen because it should not respond to care management engagement; used to detect residual confounding in Section 9.3.

### 5.4 Documented unresolved needs (Aim 3b)

For each patient's last completed contact, five binary categories judged from the note text and the record available at that contact: medication gap, pending referral or appointment, uncontrolled condition, active social crisis, and behavioral health need, plus their disjunction (any open need). The reference standard is blinded physician adjudication of 200 sampled cases (Section 11).

## 6. Eligibility windows and data completeness

6.1 **Engagement outcomes** require 90 days of continuous plan enrollment after the index and an index date on or before 2026-03-09, so that the 90-day window is observed in full.

6.2 **Utilization outcomes** (U1, U2, admission-discharge-transfer based) require an index on or before 2025-12-09 for the 180-day window, and continuous plan enrollment through that window.

6.3 **Cost outcomes** (U3, U4, U5) require the 180-day window to close on or before 2026-04-30, the claims completeness date. That date is fixed by the pre-specified rule that a month is complete when its claim-line count is within 15% of the trailing six-month mean; applied at extraction, the last complete month is April 2026. The cost cohort is therefore enrollments and contacts on or before 2025-10-31. A sensitivity analysis extends to 2026-05-31.

## 7. Predictors

All predictors use information timestamped strictly before the index instant. Blocks:

- **B1 Demographics and program context.** Age, sex, race and ethnicity, preferred language, state, health plan, rural-urban commuting area of residence, days since enrollment (Task B), calendar quarter.
- **B2 Clinical history from claims.** Chronic condition flags (CMS chronic conditions), counts of emergency department visits and admissions in the prior 365 days, prior-year total paid amount, medication counts and classes, behavioral health and substance use flags.
- **B3 Risk score.** Signal acute care risk percentile at the index, and its available components.
- **B4 Program and contact history.** Counts and rates of completed contacts and logged attempts overall and in trailing 7, 30, and 90 days; days since last completed contact and since last attempt; channel mix (phone, text, video, in person), discipline mix, time of day and day of week of contacts and attempts, contact duration, number of distinct staff, continuity, and the sequence of gaps between contacts.
- **B5 Goals and tasks.** Counts of active, completed, and incomplete care-plan goals by category, open tasks, and time since the last goal update.
- **B6 Note text.** Term frequency-inverse document frequency features over the notes of the trailing 90 days, a nine-pattern clinical lexicon, and sentence embeddings (bge-base) reduced by principal components. Text features in the development set are restricted to notes written before the temporal cutoff.
- **B7 Staff and panel context.** Assigned staff role and discipline, panel size, and market, coded so that no staff identifier enters the model.

Leakage rules: no feature may be derived from any event at or after the index instant; no feature may use the outcome window; features derived from the same note as the index contact are permitted only for Task B and only from text available at the time of the contact.

## 8. Data splitting and validation

**Temporal split.** Development set: patients enrolled before 2025-07-01, using only their contacts dated before 2025-07-01. Temporal validation set: patients enrolled on or after 2025-07-01, all of their eligible contacts. The sets are disjoint in patients and forward in time in both enrollment and contact date.

**Tuning.** Five-fold patient-grouped cross-validation inside the development set. Hyperparameters are selected on the pre-specified grids in Section 7 of the analysis code; no hyperparameter is chosen using the validation set.

**Stacking.** Ensemble weights are estimated by non-negative least squares on out-of-fold development predictions (a cross-validated super learner), with a rank-average ensemble as a pre-specified alternative.

**Pseudo-prospective validation.** The ensemble is refit at each month-end landmark of 2025, frozen, and applied to all later contacts. The headline landmark is fixed by the rule that it is the last landmark with at least 90 days of subsequent follow-up.

## 9. Statistical analysis

### 9.1 Aim 1: prediction

The pre-specified learner library, all fit on the same feature blocks:

1. Elastic-net penalized logistic regression.
2. Discrete-time hazard logistic regression in person-period form with time-varying covariates.
3. Cox proportional hazards with time-varying covariates (E4).
4. Random survival forest (E4).
5. Random forest.
6. Gradient boosting: XGBoost, LightGBM, CatBoost, and histogram gradient boosting.
7. DeepSurv, a multilayer perceptron trained on the Cox partial likelihood, implemented in PyTorch following Katzman and colleagues.
8. FT-Transformer, a tabular transformer with feature tokenizer, implemented in PyTorch following Gorishniy and colleagues.
9. The stacked ensemble of the above.

Comparator: the Signal acute care risk percentile used as a score without refitting.

Primary metric: area under the receiver operating characteristic curve for E1 on the temporal validation set, comparing the ensemble with the risk score. Co-primary: area under the precision-recall curve. Secondary: sensitivity, specificity, positive and negative predictive value at a flagging fraction of 20% of contacts (the program's stated outreach capacity) and at the Youden-optimal threshold; calibration slope and intercept; time-dependent concordance and integrated Brier score for the survival learners; decision curve analysis over threshold probabilities of 0.1 to 0.5. All intervals are 95% percentile intervals from 2,000 bootstrap resamples clustered on patients. Model comparisons use the bootstrap distribution of the paired difference.

Task A is reported with the same metric set for E2 and E3.

### 9.2 Aim 2: concordance

Three analyses, all in the temporal validation set.

1. **Rank agreement.** Spearman correlation between the predicted disengagement probability and the acute care risk percentile; the overlap (Jaccard index) of their top deciles and top quintiles; a five-by-five cross-tabulation of quintiles; and the AUROC of each score for the other's outcome.
2. **Prospective association of predicted disengagement risk with utilization and cost.** Negative binomial regression for U1, U2, and U4 with log enrolled days as offset, and a two-part model (logistic for any spending, gamma with log link for positive spending) for U3, with predicted disengagement risk entered per standard deviation and by quintile, adjusted for age, sex, state, plan, the acute care risk percentile, prior-year utilization and cost, chronic condition count, and behavioral health and substance use flags. Standard errors clustered on patients.
3. **Association of observed disengagement with utilization and cost.** The same models with observed disengagement (E1 or E2) as the exposure, estimated with inverse probability of treatment weighting on the Section 7 covariates, with outcomes measured in days 31 to 210 to blank the period in which an acute event mechanically triggers a care-team contact through the admission-discharge-transfer feed, and with the negative control outcome U5 estimated in the same specification. A negative control estimate whose confidence interval excludes the null indicates residual confounding and will be reported as such, and the primary estimate will then be described as an association that does not support a causal reading.

### 9.3 Aim 3: care-team actions

Population: contacts in the top quintile of predicted disengagement risk from the frozen ensemble (high risk), with the full range reported as a secondary analysis.

Pre-specified actions, each a binary exposure defined on a fixed window after the index contact:

- **A1.** Any completed contact or logged attempt within 7 days after a risk spike, where a spike is an increase of at least one decile in predicted disengagement risk from the patient's previous contact.
- **A2.** An in-person contact by a community health worker within 14 days.
- **A3.** A first completed contact with a therapist (behavioral health) within 30 days.
- **A4.** A first completed contact with a clinical pharmacist within 30 days.
- **A5.** A morning (08:00 to 12:00 local) weekday telephone attempt, versus attempts at other times or by text, within 14 days.

Estimand: the difference in the probability of sustained engagement at 90 days (E1 complement) under the action versus no action, among high-risk contacts in the equipoise region.

Estimators, in order of preference: (i) within-patient conditional logistic regression with patient fixed effects, for actions that vary within patient; (ii) a marginal structural model with stabilized inverse probability weights; (iii) augmented inverse probability weighting with cross-fitting. Equipoise trimming removes contacts with a propensity outside 0.05 to 0.95.

Falsification gates, all pre-specified, each action passing or failing on its own:

- **G1 Pre-trend.** The action must not be associated with engagement in the 30 days before it (absolute standardized difference below 0.10).
- **G2 Balance.** After weighting, every covariate standardized mean difference below 0.10 and the joint test p above 0.05.
- **G3 Negative control.** No association with an outcome the action should not affect (completion of a health-plan quality-measure goal unrelated to the action's domain).
- **G4 Sensitivity.** The E-value for the point estimate and for the confidence limit nearest the null is reported; an estimate whose E-value is below the strength of measured confounding is reported as an association only.

An action that fails any gate is reported as an association, in the same table, with the failed gate named. No action is described as causal.

Heterogeneity: a causal forest and a T-learner estimate individualized effects for the actions that pass all gates; effects are fit on one random half of patients and evaluated on the other half, with the pre-specified success criterion that the top-half-by-predicted-benefit shows a larger effect than the bottom half in the held-out half.

### 9.4 Aim 3b: documented unresolved needs

A classifier for each of the five need categories (Section 5.4) is trained on the physician-adjudicated cases with regularized logistic regression over lexicon, term frequency-inverse document frequency, and embedding features, and evaluated by patient-grouped cross-validation against the physician reference standard (sensitivity, specificity, F1, and agreement with the adjudicated majority). If F1 for a category is at least 0.60, the classifier is applied to all last contacts and the prevalence of that need is compared between patients who disengaged and patients who sustained engagement or completed the program, by logistic regression adjusted for the Section 9.2 covariates. Categories below that threshold are reported only from the adjudicated sample.

## 10. Fairness

For the ensemble at the 20% flagging threshold: sensitivity, specificity, flag rate, and positive predictive value by race and ethnicity, preferred language, sex, age band, and state; the equalized odds ratio (the minimum-to-maximum ratio of true positive rates across groups) with bootstrap intervals; and the thresholds that would equalize true positive rates. A ratio below 0.80 is pre-specified as the threshold for reporting the score as requiring group-specific thresholds in deployment.

## 11. Blinded physician adjudication

Three physicians, blinded to outcome, each review approximately 80 cases drawn from 200 sampled contacts (100 whose patient disengaged after the contact and 100 whose patient sustained engagement or completed the program), with 20% of cases double-read. For each case the reviewer records which of the five needs (Section 5.4) were open at that contact and how much it would matter clinically if the patient had no further contact for 90 days (low, moderate, high). Sampling seed 20260909. The adjudication serves two purposes: the reference standard for Section 9.4, and the clinical-stakes comparison between disengaged and sustained contacts, tested by a two-sided test of proportions. With 100 per group, the design has 81.5% power to detect a difference of 15 percentage points in the proportion with any open need at alpha 0.05. No form is filled by anyone other than the reviewing physician, and no automated process writes to the adjudication files.

## 12. Sample size

The cohort is fixed by the program's history; no sample size is chosen. Precision, calculated from the v3 extraction and reported here so that the plan can be judged: approximately 6,900 rising-risk patients with approximately 38,600 eligible contacts, of which approximately 14,800 contacts from 2,900 patients fall in the temporal validation set with an event fraction near 18%. The half-width of a 95% interval for AUROC at that size is approximately 0.01. For Aim 2, approximately 5,000 enrollments meet the cost completeness rule, giving 80% power to detect a 0.13 standard deviation difference in log cost between disengagement quintiles. For Aim 3, the minimum detectable risk difference at 80% power is approximately 4 percentage points for the most common action (A1) and is reported for each action alongside its estimate.

## 13. Missing data

Structured predictors are missing not at random by design (a patient with no claims history has no claims features); the gradient-boosted and transformer learners take missingness natively, and for the regression learners missing indicators are added with median imputation within the development set. Race and ethnicity are missing for a minority of patients; fairness analyses report the missing group as its own stratum rather than imputing. No outcome is imputed; records that do not complete the outcome window are excluded by Section 6 rather than imputed.

## 14. Multiplicity

One primary outcome per aim (E1 for Aim 1, the rank agreement and the adjusted association of predicted risk with total cost for Aim 2, and the maximum across the five actions for Aim 3). Within each secondary family, p values are adjusted by the Hochberg procedure. The five actions in Aim 3 constitute one family.

## 15. Software, seeds, and reproducibility

Python 3.12 with scikit-learn 1.6, XGBoost 3.1, LightGBM 4.6, CatBoost 1.2, lifelines 0.30, scikit-survival 0.26, PyTorch 2.12, econml 0.16, statsmodels 0.14, and sentence-transformers 5.1. DeepSurv and FT-Transformer are implemented in the repository rather than taken from an external package, with the implementations and their unit tests in `scripts/`. All seeds are fixed and recorded in `results/canonical.json`. Every number in the manuscript is written by the pipeline to that file, and `audit_consistency.py` verifies that every number in the manuscript, supplement, tables, and figure legends matches it before submission. Code is released without data.

## 16. Deviations

Deviations from this plan are recorded in a table in the supplement with the date, the change, the reason, and whether the change was made before or after the affected outcome was examined. The pipeline writes the table from a machine-readable log so that it cannot be edited silently.

## 17. Analysis sequence

1. `01_cohort.py` enrollment cohort, tier, coverage spans, index events.
2. `02_outcomes.py` E1 to E4, program completion by both definitions, utilization and cost outcomes with completeness rules.
3. `03_features.py` blocks B1 to B5 and B7; `04_text.py` block B6.
4. `05_models.py` learner library and tuning; `06_survival.py` E4 learners; `07_ensemble.py` stacking and calibration.
5. `08_metrics.py` metrics, intervals, decision curves, fairness.
6. `09_concordance.py` Aim 2 analyses 1 to 3.
7. `10_actions.py` Aim 3 actions, gates, heterogeneity.
8. `11_needs_nlp.py` Aim 3b classifier and comparison.
9. `12_landmarks.py` frozen-model validation; `13_sensitivity.py` pre-specified sensitivity analyses.
10. `14_report.py` canonical numbers, tables, figures; `audit_consistency.py`.

## 18. Pre-specified sensitivity analyses

(a) Goal-based rather than status-based program completion; (b) all tiers rather than rising-risk only; (c) 60 rather than 90 days of required plan enrollment; (d) a 120-day rather than 90-day disengagement window; (e) cost window extended to 2026-05-31; (f) exclusion of the enrollment-day contact from Task B; (g) text features excluded; (h) quarterly refitting instead of a single frozen model.
