"""v4 step 1: cohort, index events, coverage, and the temporal split (pre-registration Sections 4, 6, 8).

Task A = one row per enrolled patient (index = enrollment date).
Task B = one row per completed contact within 365 days of enrollment (index = contact).
Writes data_cache/v4_taskA.parquet, data_cache/v4_taskB.parquet, results/flow_v4.json."""
import json, pathlib
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
STUDY_START, STUDY_END = pd.Timestamp("2023-05-23"), pd.Timestamp("2026-06-07")
CUT = pd.Timestamp("2025-07-01")           # temporal split (Section 8)
E1_LAST_INDEX = STUDY_END - pd.Timedelta(days=90)   # 90-day outcome observed in full

flow = {}
enr = pd.read_parquet(D/"v4_enrollment.parquet")
enr["enroll_date"] = pd.to_datetime(enr.enroll_date)
flow["enrolled_patients_all"] = len(enr)

enr = enr[(enr.enroll_date >= STUDY_START) & (enr.enroll_date <= STUDY_END)].copy()
flow["enrolled_in_study_period"] = len(enr)

# ---- tier, demographics, and program context from the person-month mart -------------------
panel = pd.read_parquet(D/"v4_panel.parquet")
panel["zero_date"] = pd.to_datetime(panel.zero_date)
num = lambda s: pd.to_numeric(s, errors="coerce")
attrs = (panel.sort_values(["person_id", "zero_date", "ms"]).groupby("person_id")
         .agg(zero_date=("zero_date", "first"), tier1=("tier1_flg", "first"), risk_percentile=("risk_percentile", "first"),
              age=("age", "first"), gender=("gender", "first"), race=("race", "first"), state=("state", "first"),
              market=("market", "first"), entity=("entity", "first"), postal_code=("postal_code", "first"),
              alcohol_use_disorder=("alcohol_use_disorder", "first"), any_bh=("any_bh", "first"), asthma=("asthma", "first"),
              chf=("chf", "first"), copd=("copd", "first"), diabetes=("diabetes", "first"), gad=("gad", "first"),
              htn=("htn", "first"), high_ed_ip=("high_ed_ip", "first"), increasing_ed_ip=("increasing_ed_ip", "first"),
              mdd=("mdd", "first"), no_pcp_last_10mo=("no_pcp_last_10mo", "first"), polypharmacy=("polypharmacy", "first"),
              postpartum=("postpartum", "first"), prenatal=("prenatal", "first"), psychosis=("psychosis", "first"),
              sud=("sud", "first")).reset_index())
for c in ["tier1", "risk_percentile", "age"] + ["alcohol_use_disorder","any_bh","asthma","chf","copd","diabetes","gad","htn",
          "high_ed_ip","increasing_ed_ip","mdd","no_pcp_last_10mo","polypharmacy","postpartum","prenatal","psychosis","sud"]:
    attrs[c] = num(attrs[c])
A = enr.merge(attrs, on="person_id", how="left")
flow["missing_from_outcomes_mart"] = int(A.tier1.isna().sum())
A = A[A.tier1.notna()].copy()
flow["with_program_attributes"] = len(A)

STATES = ["OHIO", "VIRGINIA", "WASHINGTON"]
flow["excluded_outside_three_states"] = int((~A.state.isin(STATES)).sum())
A = A[A.state.isin(STATES)].copy()
flow["in_three_states"] = len(A)

A["tier1"] = (A.tier1 == 1)
flow["excluded_not_tier1"] = int((~A.tier1).sum())
A = A[A.tier1].copy()
flow["tier1_patients"] = len(A)

# ---- health-plan coverage (month level) ---------------------------------------------------
# Coverage months come from the union of the eligibility file and the program's enrollment-anchored
# mart. Neither alone is complete: the eligibility file was restated at the 2024/2025 contract-year
# boundary for two plans (ABHVA, UHCWA), which would otherwise exclude Virginia and Washington
# contacts whose 90-day window crosses that boundary.
mm = pd.read_parquet(D/"v4_member_months.parquet")
mm["m"] = pd.PeriodIndex(pd.to_datetime(mm.year_month.astype(str), format="%Y%m"), freq="M")
pmm = panel.copy(); pmm["ms"] = pmm.ms.astype(int); pmm["ed_days"] = num(pmm.enrolled_days).fillna(0)
pmm = pmm[pmm.ed_days > 0]; pmm["m"] = pmm.zero_date.dt.to_period("M") + pmm.ms
cov_src = pd.concat([mm[["person_id", "m"]], pmm[["person_id", "m"]]], ignore_index=True).drop_duplicates()
cov = {k: set(v) for k, v in cov_src.groupby("person_id").m.apply(set).items()}
def covered(pid, start, days):
    """True when every calendar month touched by [start, start+days] is a covered month."""
    months = pd.period_range(pd.Period(start, freq="M"), pd.Period(start + pd.Timedelta(days=days), freq="M"), freq="M")
    have = cov.get(pid, set())
    return all(m in have for m in months)

A["covered_90"] = [covered(p, d, 90) for p, d in zip(A.person_id, A.enroll_date)]
A["covered_180"] = [covered(p, d, 180) for p, d in zip(A.person_id, A.enroll_date)]
A["covered_210"] = [covered(p, d, 210) for p, d in zip(A.person_id, A.enroll_date)]
flow["taskA_covered_90"] = int(A.covered_90.sum())

A["era"] = np.where(A.enroll_date < CUT, "development", "validation")
A["eligible_E2"] = A.covered_90 & (A.enroll_date <= E1_LAST_INDEX)
flow["taskA_eligible_E2"] = int(A.eligible_E2.sum())
flow["taskA_eligible_E2_by_era"] = A[A.eligible_E2].era.value_counts().to_dict()

# ---- Task B: completed contacts within 365 days of enrollment ------------------------------
enc = pd.read_parquet(D/"v4_encounters.parquet")
enc["enc_date"] = pd.to_datetime(enc.enc_date)
enc = enc[enc.person_id.isin(set(A.person_id))].copy()
flow["encounter_rows_cohort"] = len(enc)
comp = enc[enc.occurred == "YES"].copy()
flow["completed_contacts_cohort"] = len(comp)
flow["logged_attempts_cohort"] = int((enc.occurred == "NO").sum())

comp = comp.merge(A[["person_id", "enroll_date", "era"]], on="person_id", how="inner")
comp["days_since_enroll"] = (comp.enc_date - comp.enroll_date).dt.days
B = comp[(comp.days_since_enroll >= 0) & (comp.days_since_enroll <= 365) & (comp.enc_date <= STUDY_END)].copy()
flow["taskB_contacts_in_window"] = len(B)
B = B.sort_values(["person_id", "enc_date", "encounter_id"]).drop_duplicates(["person_id", "enc_date"], keep="first")
flow["taskB_decision_points"] = len(B)
flow["taskB_patients"] = int(B.person_id.nunique())

B["covered_90"] = [covered(p, d, 90) for p, d in zip(B.person_id, B.enc_date)]
B["observed_90"] = B.enc_date <= E1_LAST_INDEX
B["eligible_E1"] = B.covered_90 & B.observed_90
flow["taskB_excluded_no_coverage"] = int((~B.covered_90).sum())
flow["taskB_excluded_window_open"] = int((B.covered_90 & ~B.observed_90).sum())
flow["taskB_eligible_E1"] = int(B.eligible_E1.sum())

# temporal split: development-era patients contribute only contacts before the cutoff
B["era"] = np.where(B.enroll_date < CUT, "development", "validation")
B["in_split"] = np.where(B.era == "development", B.enc_date < CUT, True)
flow["taskB_excluded_development_overlap"] = int((B.eligible_E1 & ~B.in_split).sum())
B["analysis"] = B.eligible_E1 & B.in_split
flow["taskB_analysis"] = int(B.analysis.sum())
flow["taskB_analysis_patients"] = int(B[B.analysis].person_id.nunique())
flow["taskB_analysis_by_era"] = B[B.analysis].era.value_counts().to_dict()
flow["taskB_analysis_patients_by_era"] = B[B.analysis].groupby("era").person_id.nunique().to_dict()

B = B.merge(A.drop(columns=["enroll_dt", "enroll_date", "era", "covered_90", "covered_180", "covered_210", "eligible_E2", "zero_date"]), on="person_id", how="left")
A.to_parquet(D/"v4_taskA.parquet"); B.to_parquet(D/"v4_taskB.parquet")
flow["windows"] = {"study_start": str(STUDY_START.date()), "study_end": str(STUDY_END.date()), "split_cut": str(CUT.date()),
                   "last_index_for_90d": str(E1_LAST_INDEX.date())}
json.dump(flow, open(R/"flow_v4.json", "w"), indent=1)
print(json.dumps(flow, indent=1))
