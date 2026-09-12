"""v4 step 0c: care-plan tasks for the enrolled cohort, for the structured open-need measure."""
import sys, pathlib, json
import pandas as pd
sys.path.insert(0, str(pathlib.Path.home()/".claude/skills/waymark-data-access/scripts"))
from wm_conn import lighthouse, query
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
LH = lighthouse("prod")
ids = set(pd.read_parquet(D/"v4_enrollment.parquet").person_id)
t = query(LH, '''SELECT p."waymarkPatientNumber" AS person_id, t.id AS task_id, t."taskTitle" AS title,
                        ARRAY_TO_STRING(t."taskTag", '|') AS tags, t."dueDate" AS due_date, t.completed,
                        t."goalId" AS goal_id, t."createdAt" AS created_at, t."updatedAt" AS updated_at
                 FROM public."Task" t JOIN public."Patient" p ON p.id = t."patientId"
                 WHERE COALESCE(t.deleted, false) = false''')
t = t[t.person_id.isin(ids)].copy()
for c in ["due_date", "created_at", "updated_at"]:
    t[c] = pd.to_datetime(t[c], errors="coerce")
t.to_parquet(D/"v4_tasks.parquet")
meta = {"rows": len(t), "persons": int(t.person_id.nunique()), "completed": int(t.completed.sum()),
        "open": int((~t.completed.astype(bool)).sum()), "linked_to_goal": int(t.goal_id.notna().sum())}
json.dump(meta, open(R/"tasks_pull_v4.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
