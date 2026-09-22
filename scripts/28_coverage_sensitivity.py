"""Does differential loss of Medicaid coverage explain the inverse utilization and cost finding?

Claims accrue only while a beneficiary is enrolled in the health plan, so if patients who
disengage lose coverage more often, their observed cost and utilization would fall for a
reason that has nothing to do with engagement. The primary analysis requires continuous
coverage through day 210 while outcomes run to day 270; this recomputes coverage through
day 270 and repeats the overlap-weighted estimates in that subgroup.
Writes results/coverage_270_check.json and results/coverage_270_sensitivity.json (Appendix Table 13).
"""
import pandas as pd, numpy as np, pathlib, json

P = pathlib.Path(__file__).resolve().parent.parent
D, R = P/"data_cache", P/"results"
num = lambda s: pd.to_numeric(s, errors="coerce")

# coverage months, from the union of the eligibility file and the program mart (as in 01_cohort)
mm = pd.read_parquet(D/"v4_member_months.parquet")
mm["m"] = pd.PeriodIndex(pd.to_datetime(mm.year_month.astype(str), format="%Y%m"), freq="M")
pmm = pd.read_parquet(D/"monthly_panel.parquet")
pmm["ms"] = pmm.ms.astype(int)
pmm["zero_date"] = pd.to_datetime(pmm.zero_date)
pmm = pmm[num(pmm.enrolled_days).fillna(0) > 0]
pmm["m"] = pmm.zero_date.dt.to_period("M") + pmm.ms
cov_src = pd.concat([mm[["person_id", "m"]], pmm[["person_id", "m"]]], ignore_index=True).drop_duplicates()
cov = {k: set(v) for k, v in cov_src.groupby("person_id").m.apply(set).items()}

def covered(pid, start, days):
    months = pd.period_range(pd.Period(start, freq="M"), pd.Period(start + pd.Timedelta(days=days), freq="M"), freq="M")
    return all(m in cov.get(pid, set()) for m in months)

A0 = pd.read_parquet(D/"v4_featA.parquet")
A0["enroll_date"] = pd.to_datetime(A0.enroll_date)
sub = A0[A0.eligible_E2 & A0.covered_210].copy()
sub["cov270_true"] = [covered(p, d, 270) for p, d in zip(sub.person_id, sub.enroll_date)]
e = sub.E2_disengaged.astype(bool)
check = {"n_covered_210": int(len(sub)), "n_covered_270": int(sub.cov270_true.sum()),
         "pct_covered_270_disengaged": round(100*float(sub.cov270_true[e].mean()), 1),
         "pct_covered_270_sustained": round(100*float(sub.cov270_true[~e].mean()), 1)}
json.dump(check, open(R/"coverage_270_check.json", "w"), indent=1)
sub[["person_id", "cov270_true"]].to_parquet(D/"covered_270.parquet", index=False)
print(json.dumps(check, indent=1))

# reuse the matching script's cohort, propensity model, and estimator
src = open(P/"scripts"/"15_matching.py").read().split("# ---- 1.")[0]
ns = {"__file__": str(P/"scripts"/"15_matching.py")}
exec(compile(src, "15_matching_head", "exec"), ns)
A, estimate, balance = ns["A"], ns["estimate"], ns["balance"]
A = A.merge(sub[["person_id", "cov270_true"]], on="person_id", how="left")
A["cov270_true"] = A.cov270_true.fillna(False)

res = {}
for label, df in [("primary (coverage through day 210)", A),
                  ("restricted to coverage through day 270", A[A.cov270_true].copy())]:
    t = df.E2_disengaged.to_numpy(int)
    w = np.where(t == 1, 1 - df.ps.to_numpy(float), df.ps.to_numpy(float))   # overlap weights
    est, bal = estimate(df, w, label), balance(df, w)
    res[label] = {"n": int(len(df)), "exposed": int(t.sum()), "max_abs_smd": bal["max_abs_smd"],
                  "estimates": {k: {"ratio": v.get("ratio"), "ci_95": v.get("ci_95")} for k, v in est.items()}}
    print(f"\n{label}  n={len(df):,}  max |SMD| {bal['max_abs_smd']}")
    for k, v in est.items():
        if v.get("ratio") is not None:
            print(f"   {k:22s} {v['ratio']:.3f} ({v['ci_95'][0]:.3f} to {v['ci_95'][1]:.3f})")

json.dump(res, open(R/"coverage_270_sensitivity.json", "w"), indent=1, default=str)
print("\nwrote results/coverage_270_sensitivity.json")
