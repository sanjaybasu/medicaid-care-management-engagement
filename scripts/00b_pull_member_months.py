"""v4 step 0b: health-plan coverage months for the enrolled cohort (pre-registration Section 6)."""
import sys, pathlib, json
import pandas as pd
sys.path.insert(0, str(pathlib.Path.home()/".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
CO = coredb("prod")
print(query(CO, "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='dbt_tuva_core' AND table_name='member_months' ORDER BY ordinal_position").to_string())
ids = tuple(pd.read_parquet(D/"v4_enrollment.parquet").person_id.tolist())
raw = CO.raw_connection()
try:
    mm = pd.read_sql("""SELECT DISTINCT person_id, year_month FROM dbt_tuva_core.member_months
                        WHERE person_id IN %s AND year_month >= '202201'""", raw, params=(ids,))
finally:
    raw.close()
mm.to_parquet(D/"v4_member_months.parquet")
print("member months", len(mm), "persons", mm.person_id.nunique(), "range", mm.year_month.min(), mm.year_month.max())
