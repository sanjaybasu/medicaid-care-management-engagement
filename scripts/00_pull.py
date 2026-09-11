"""v4 step 0: extract every table the v4 plan needs, for all patients ever enrolled (ACTIVATED).

Writes data_cache/v4_*.parquet (PHI; local only) and results/pull_manifest_v4.json.
Sources: lighthouse (status history, encounters incl. attempts, goals, ADT, SignalRisk),
coredb ops_data_mart (person-month cost/utilization/demographics), coredb dbt_tuva_core (claim lines)."""
import sys, json, time, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path.home()/".claude/skills/waymark-data-access/scripts"))
from wm_conn import lighthouse, coredb, query

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"; R.mkdir(exist_ok=True)
LH, CO = lighthouse("prod"), coredb("prod")
MAN, T0 = {}, time.time()
def save(df, name, **extra):
    df.to_parquet(D/f"v4_{name}.parquet")
    MAN[name] = {"rows": len(df), "persons": int(df.person_id.nunique()) if "person_id" in df else None, **extra}
    print(f"  {name}: {len(df):,} rows ({time.time()-T0:.0f}s)", flush=True)

print("1/7 status history", flush=True)
sh = query(LH, """SELECT p."waymarkPatientNumber" AS person_id, s.updated_at, s.old_status::text AS old_status, s.new_status::text AS new_status
                  FROM public.patient_status_history s JOIN public."Patient" p ON p.id = s.patient_id""")
sh["updated_at"] = pd.to_datetime(sh.updated_at, errors="coerce"); sh = sh.dropna(subset=["updated_at"]); save(sh, "status_history")
ENR = sh[sh.new_status == "ACTIVATED"].groupby("person_id").updated_at.min().rename("enroll_dt").reset_index()
ENR["enroll_date"] = ENR.enroll_dt.dt.normalize()
save(ENR, "enrollment", date_min=str(ENR.enroll_date.min())[:10], date_max=str(ENR.enroll_date.max())[:10])
IDS = tuple(ENR.person_id.tolist())

print("2/7 encounters and attempts (with timestamps)", flush=True)
chunks = []
for a, b in [("2022-01-01","2024-01-01"),("2024-01-01","2025-01-01"),("2025-01-01","2025-07-01"),("2025-07-01","2026-01-01"),("2026-01-01","2027-01-01")]:
    raw = CO.raw_connection() if False else None
    sql = f"""SELECT p."waymarkPatientNumber" AS person_id, e.id AS encounter_id, e."dateOfEncounter"::date AS enc_date,
                     e."startTime" AS start_time, e."contactType"::text AS contact_type, e."encounterType"::text AS encounter_type,
                     e."encounterOccurred"::text AS occurred, e."createdByWaymarkerId" AS created_by_id,
                     COALESCE(e."noteV2", e.note, '') AS note_text
              FROM public."EncounterNote" e JOIN public."Patient" p ON p.id = e."patientId"
              WHERE e."dateOfEncounter" >= '{a}' AND e."dateOfEncounter" < '{b}'
                AND COALESCE(e.deleted, false) = false AND e."encounterType" IS NOT NULL"""
    c = query(LH, sql); chunks.append(c); print(f"   {a[:7]}: {len(c):,}", flush=True)
enc = pd.concat(chunks, ignore_index=True)
enc = enc[enc.person_id.isin(set(ENR.person_id))].copy()
enc["enc_date"] = pd.to_datetime(enc.enc_date, errors="coerce"); enc["start_time"] = pd.to_datetime(enc.start_time, errors="coerce")
enc = enc.dropna(subset=["enc_date"])
roles = query(LH, """SELECT wr.waymarker_id AS created_by_id, STRING_AGG(DISTINCT r.identifier, '|') AS roles
                     FROM public.waymarker_role wr JOIN public.rbac_role r ON r.id = wr.role_id GROUP BY 1""")
enc = enc.merge(roles, on="created_by_id", how="left")
save(enc, "encounters", occurred=enc.occurred.value_counts().to_dict())

print("3/7 goals", flush=True)
g = query(LH, """SELECT p."waymarkPatientNumber" AS person_id, g.id AS goal_id, g.category::text AS category, g.type::text AS goal_type,
                        g.status::text AS status, g.title, g."createdAt" AS created_at, g."updatedAt" AS updated_at
                 FROM public."Goal" g JOIN public."Patient" p ON p.id = g."patientId" WHERE COALESCE(g.deleted,false)=false""")
for c in ["created_at","updated_at"]: g[c] = pd.to_datetime(g[c], errors="coerce")
g = g[g.person_id.isin(set(ENR.person_id))].copy(); save(g, "goals", types=g.goal_type.value_counts().to_dict())

print("4/7 ADT", flush=True)
adt = query(LH, """SELECT p."waymarkPatientNumber" AS person_id, a.id AS adt_id, a."admitDate" AS admit_date,
                          a."dischargedDate" AS discharged_date, a."patientClassCode"::text AS class_code
                   FROM public."AdmissionDischargeTransfer" a JOIN public."Patient" p ON p.id = a."patientId" """)
adt = adt[adt.person_id.isin(set(ENR.person_id))].copy()
adt["admit_date"] = pd.to_datetime(adt.admit_date, errors="coerce"); save(adt, "adt", classes=adt.class_code.value_counts().head(8).to_dict())

print("5/7 SignalRisk", flush=True)
sr = query(LH, """SELECT p."waymarkPatientNumber" AS person_id, r."appliedDate"::date AS applied_date, r."risingRiskScore"::float AS rr_score,
                         r."risingRiskScorePercentile"::float AS rr_pctile, r."riskTier"::int AS risk_tier
                  FROM public."SignalRisk" r JOIN public."Patient" p ON p.id = r."patientId" WHERE r."appliedDate" IS NOT NULL""")
sr = sr[sr.person_id.isin(set(ENR.person_id))].copy(); save(sr, "signal_risk")

print("6/7 person-month outcomes mart (no matching filter)", flush=True)
cols = ("person_id, zero_date, person_zero_id, zero_date_rank, months_since_zero_date::int AS ms, enrolled_days::float AS enrolled_days, "
        "COALESCE(emergency_department_ct::float,0) AS ed, COALESCE(acute_inpatient_ct::float,0) AS ip, "
        "COALESCE(adt_emergency_department_ct::float,0) AS adt_ed, COALESCE(adt_acute_inpatient_ct::float,0) AS adt_ip, "
        "COALESCE(total_paid::float,0) AS total_paid, COALESCE(medical_paid::float,0) AS medical_paid, COALESCE(pharmacy_paid::float,0) AS pharmacy_paid, "
        "COALESCE(emergency_department_paid::float,0) AS ed_paid, COALESCE(acute_inpatient_paid::float,0) AS ip_paid, "
        "COALESCE(ed_nyu::float,0) AS ed_nyu, COALESCE(pqi::float,0) AS pqi, "
        "ever_targeted, ever_activated, rr_flag, tier1_flg, risk_percentile::float AS risk_percentile, "
        "alcohol_use_disorder, any_bh, asthma, chf, copd, diabetes, gad, htn, high_ed_ip, increasing_ed_ip, mdd, no_pcp_last_10mo, polypharmacy, "
        "postpartum, prenatal, psychosis, sud, gender, race, age::float AS age, state, market, entity, tin, postal_code, "
        "last_rr_date_before_zero_date, first_eligible_date, last_eligible_date, "
        "total_in_person, total_therapy, total_pharmacy, total_chw, total_completed_goals")
mp = query(CO, f"""SELECT {cols} FROM ops_data_mart.outcomes_with_enrollment__months_since
                   WHERE ever_activated = '1' AND months_since_zero_date::int BETWEEN -13 AND 13""")
save(mp, "panel", months=[int(mp.ms.min()), int(mp.ms.max())], zero_dates=[str(mp.zero_date.min()), str(mp.zero_date.max())])

print("7/7 claim lines", flush=True)
raw = CO.raw_connection()
try:
    med = pd.read_sql("""SELECT person_id, claim_start_date, claim_type, encounter_type, COALESCE(paid_amount,0) AS paid
                         FROM dbt_tuva_core.medical_claim WHERE person_id IN %s AND claim_start_date >= '2022-01-01'""", raw, params=(IDS,))
    med["claim_start_date"] = pd.to_datetime(med.claim_start_date, errors="coerce"); save(med, "medical_claim_lines", types=med.claim_type.value_counts().head(5).to_dict())
    ph = pd.read_sql("""SELECT person_id, dispensing_date AS claim_start_date, COALESCE(paid_amount,0) AS paid
                        FROM dbt_tuva_core.pharmacy_claim WHERE person_id IN %s AND dispensing_date >= '2022-01-01'""", raw, params=(IDS,))
    ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date, errors="coerce"); save(ph, "pharmacy_claim_lines")
finally:
    raw.close()

json.dump(MAN, open(R/"pull_manifest_v4.json", "w"), indent=1)
print(json.dumps(MAN, indent=1, default=str)[:1500])
