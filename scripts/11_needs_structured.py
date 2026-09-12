"""v4 step 11: open medical and social needs at the last contact, from the structured care plan.

Replaces the blinded physician panel. A care-plan goal or task is open at a contact if it was created
on or before that contact and had not yet been closed by then; closure is reconstructed from the
current status and the last update time, symmetrically for patients who did and did not disengage.
Goals carry a category, which maps to a medical or a social domain; tasks inherit the domain of their
linked goal or, when unlinked, of their title. A note-text lexicon is scored against the structured
measure so that the text classifier is validated without a physician reference standard."""
import json, pathlib, re, html, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, cohen_kappa_score
warnings.filterwarnings("ignore")

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
out = {}

MEDICAL = {"MENTAL_HEALTH", "MEDICATION_ADHERENCE", "HYPERTENSION", "ASTHMA_COPD", "SUBSTANCE_USE", "DIABETES",
           "MATERNITY", "PCP_APPOINTMENT", "HEART_FAILURE", "DENTAL", "SMOKING_CESSATION", "EYE_CARE",
           "MEDICATION_OPTIMIZATION", "POSTPARTUM_CARE", "CARE_FOR_MH_BH", "DEPRESSION", "ANXIETY",
           "WEIGHT_MANAGEMENT", "PRENATAL_CARE", "ALCOHOL_USE", "OTHER_MENTAL_BEHAVIORAL", "FOOD_DIET_NUTRITION",
           "ACTIVITY", "CARE"}
SOCIAL = {"INSURANCE_COVERAGE", "TRANSPORTATION", "HOUSING_INSECURITY", "FINANCIAL", "FOOD_INSECURITY",
          "UTILITIES", "CHILDCARE", "EMPLOYMENT", "SOCIAL_CONNECTION", "TECHNOLOGY", "LEGAL",
          "HOUSING_QUALITY_SAFETY", "VIOLENCE", "EDUCATION"}
CLOSED = {"COMPLETED", "PROVISIONALLY_COMPLETED", "DISMISSED"}
def domain(cat, goal_type):
    if goal_type == "HEDIS": return "quality_measure"
    if cat in MEDICAL: return "medical"
    if cat in SOCIAL: return "social"
    return "other"

B = pd.read_parquet(D/"v4_outcomes_B.parquet"); B["enc_date"] = pd.to_datetime(B.enc_date)
B = B[B.analysis].copy()
FB = pd.read_parquet(D/"v4_featB.parquet")[["person_id", "encounter_id", "n_contact_90d", "days_since_contact"]]
B = B.merge(FB, on=["person_id", "encounter_id"], how="left")
G = pd.read_parquet(D/"v4_goals.parquet")
for c in ["created_at", "updated_at"]: G[c] = pd.to_datetime(G[c], errors="coerce")
G = G[G.created_at.notna()].copy()
G["domain"] = [domain(c, t) for c, t in zip(G.category, G.goal_type)]
G["closed_at"] = np.where(G.status.isin(CLOSED), G.updated_at.values, np.datetime64("NaT"))
G["closed_at"] = pd.to_datetime(G.closed_at)
out["goal_domains"] = G.domain.value_counts().to_dict()
out["goal_status"] = G.status.value_counts().to_dict()

T = pd.read_parquet(D/"v4_tasks.parquet")
gmap = G.set_index("goal_id").domain.to_dict() if "goal_id" in G.columns else {}
TITLE_MED = re.compile(r"medicat|refill|pharmac|appointment|pcp|provider|specialist|lab|screen|a1c|blood pressure|refer|therap|counsel|immuniz|vaccin|wound|diabet|asthma|copd|prenatal|postpartum", re.I)
TITLE_SOC = re.compile(r"hous|food|transport|ride|utilit|insur|medicaid renew|benefit|employ|job|childcare|legal|shelter|snap|wic|bill|rent", re.I)
def task_domain(row):
    d = gmap.get(row.goal_id)
    if d in ("medical", "social"): return d
    ttl = f"{row.title} {row.tags}"
    if TITLE_SOC.search(ttl): return "social"
    if TITLE_MED.search(ttl): return "medical"
    return "other"
T["domain"] = [task_domain(r) for r in T.itertuples()]
T["closed_at"] = np.where(T.completed.astype(bool), T.updated_at.values, np.datetime64("NaT"))
T["closed_at"] = pd.to_datetime(T.closed_at)
out["task_domains"] = T.domain.value_counts().to_dict()

def open_counts(frame, key_domains):
    """per (person, index date) counts of items created on or before the index and not yet closed."""
    by = {p: g for p, g in frame.groupby("person_id")}
    def f(pid, t):
        g = by.get(pid)
        if g is None: return {d: 0 for d in key_domains}
        created = g.created_at <= t
        openrows = created & (g.closed_at.isna() | (g.closed_at > t))
        sub = g[openrows]
        return {d: int((sub.domain == d).sum()) for d in key_domains}
    return f

DOMS = ["medical", "social", "quality_measure", "other"]
gf, tf = open_counts(G, DOMS), open_counts(T, DOMS)
rows = []
for pid, t in zip(B.person_id, B.enc_date):
    gg, tt = gf(pid, t), tf(pid, t)
    rows.append({"open_goal_medical": gg["medical"], "open_goal_social": gg["social"],
                 "open_goal_quality": gg["quality_measure"], "open_task_medical": tt["medical"],
                 "open_task_social": tt["social"]})
N = pd.DataFrame(rows, index=B.index)
B = pd.concat([B, N], axis=1)
B["any_open_medical"] = ((B.open_goal_medical + B.open_task_medical) > 0).astype(int)
B["any_open_social"] = ((B.open_goal_social + B.open_task_social) > 0).astype(int)
B["any_open_need"] = ((B.any_open_medical + B.any_open_social) > 0).astype(int)
B["n_open_needs"] = B.open_goal_medical + B.open_goal_social + B.open_task_medical + B.open_task_social

# items ever created before the index, so that "no open need" can be separated from "no care plan"
def ever_counts(frame):
    by = {p: g for p, g in frame.groupby("person_id")}
    def f(pid, t):
        g = by.get(pid)
        return 0 if g is None else int((g.created_at <= t).sum())
    return f
gev, tev = ever_counts(G), ever_counts(T)
B["n_goals_before"] = [gev(p, t) for p, t in zip(B.person_id, B.enc_date)]
B["n_tasks_before"] = [tev(p, t) for p, t in zip(B.person_id, B.enc_date)]
B["has_care_plan"] = ((B.n_goals_before + B.n_tasks_before) > 0).astype(int)

# ---------------- primary comparison: each patient's last contact -------------------------------
last = B.sort_values("enc_date").groupby("person_id").tail(1).copy()
out["design"] = {"contacts": int(len(B)), "patients_last_contact": int(len(last)),
                 "disengaged_at_last_contact": int(last.E1.sum()), "sustained_at_last_contact": int((1-last.E1).sum())}

COV = ["age", "risk_percentile", "any_bh", "sud", "diabetes", "htn", "days_since_enroll", "n_contact_90d"]
def compare(df, ycol, label):
    a = df[df.E1 == 1][ycol].astype(float); b = df[df.E1 == 0][ycol].astype(float)
    X = df[COV].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = pd.concat([df[["E1"]].astype(float).reset_index(drop=True),
                   X.reset_index(drop=True),
                   pd.get_dummies(df.state, prefix="st", drop_first=True).astype(float).reset_index(drop=True)], axis=1)
    X = sm.add_constant(X.astype(float), has_constant="add")
    y = df[ycol].astype(float).to_numpy()
    res = {"measure": label, "disengaged_pct": round(100*float(a.mean()), 1), "sustained_pct": round(100*float(b.mean()), 1),
           "difference_pp": round(100*float(a.mean()-b.mean()), 1), "n_disengaged": int(len(a)), "n_sustained": int(len(b))}
    try:
        m = sm.Logit(y, X.to_numpy()).fit(disp=0)
        k = list(X.columns).index("E1"); bb, se = float(m.params[k]), float(m.bse[k])
        res |= {"adjusted_or": round(float(np.exp(bb)), 3),
                "adjusted_ci_95": [round(float(np.exp(bb-1.96*se)), 3), round(float(np.exp(bb+1.96*se)), 3)],
                "p": round(float(m.pvalues[k]), 4)}
    except Exception as e:
        res |= {"error": str(e)[:70]}
    return res

prim = [compare(last, c, l) for c, l in [("any_open_medical", "any open medical need"),
                                         ("any_open_social", "any open social need"),
                                         ("any_open_need", "any open need")]]
out["last_contact"] = prim
out["last_contact_counts"] = {
 "mean_open_medical_disengaged": round(float(last[last.E1 == 1].open_goal_medical.mean()), 2),
 "mean_open_medical_sustained": round(float(last[last.E1 == 0].open_goal_medical.mean()), 2),
 "mean_open_social_disengaged": round(float(last[last.E1 == 1].open_goal_social.mean()), 2),
 "mean_open_social_sustained": round(float(last[last.E1 == 0].open_goal_social.mean()), 2),
 "mean_open_needs_disengaged": round(float(last[last.E1 == 1].n_open_needs.mean()), 2),
 "mean_open_needs_sustained": round(float(last[last.E1 == 0].n_open_needs.mean()), 2)}
# separating "no documented care plan" from "a plan with nothing left open"
out["care_plan_presence"] = {
 "with_plan_disengaged_pct": round(100*float(last[last.E1 == 1].has_care_plan.mean()), 1),
 "with_plan_sustained_pct": round(100*float(last[last.E1 == 0].has_care_plan.mean()), 1),
 "mean_goals_before_disengaged": round(float(last[last.E1 == 1].n_goals_before.mean()), 2),
 "mean_goals_before_sustained": round(float(last[last.E1 == 0].n_goals_before.mean()), 2)}
planned = last[last.has_care_plan == 1]
out["among_patients_with_a_care_plan"] = {
 "n": int(len(planned)), "n_disengaged": int(planned.E1.sum()),
 "comparisons": [compare(planned, c, l) for c, l in [("any_open_medical", "any open medical need"),
                                                     ("any_open_social", "any open social need"),
                                                     ("any_open_need", "any open need")]]}
late = last[last.days_since_enroll >= 30]
out["contacts_at_least_30_days_after_enrolment"] = {
 "n": int(len(late)), "n_disengaged": int(late.E1.sum()),
 "with_plan_disengaged_pct": round(100*float(late[late.E1 == 1].has_care_plan.mean()), 1),
 "with_plan_sustained_pct": round(100*float(late[late.E1 == 0].has_care_plan.mean()), 1),
 "comparisons": [compare(late, c, l) for c, l in [("any_open_medical", "any open medical need"),
                                                  ("any_open_social", "any open social need")]]}
latep = late[late.has_care_plan == 1]
out["late_contacts_with_a_care_plan"] = {
 "n": int(len(latep)), "n_disengaged": int(latep.E1.sum()),
 "comparisons": [compare(latep, c, l) for c, l in [("any_open_medical", "any open medical need"),
                                                   ("any_open_social", "any open social need"),
                                                   ("any_open_need", "any open need")]]}
out["all_contacts"] = [compare(B, c, l) for c, l in [("any_open_medical", "any open medical need"),
                                                     ("any_open_social", "any open social need")]]
# by specific category, at the last contact
cat_rows = []
for cat in sorted(set(G.category.dropna()) - {"DEFAULT"}):
    sub = G[G.category == cat]
    if len(sub) < 200: continue
    f = open_counts(sub, ["medical", "social", "quality_measure", "other"])
    vals = np.array([sum(f(p, t).values()) > 0 for p, t in zip(last.person_id, last.enc_date)], float)
    if vals.sum() < 25: continue
    cat_rows.append({"category": cat, "domain": domain(cat, "STANDARD"),
                     "open_at_last_contact_pct": round(100*float(vals.mean()), 1),
                     "disengaged_pct": round(100*float(vals[last.E1.to_numpy() == 1].mean()), 1),
                     "sustained_pct": round(100*float(vals[last.E1.to_numpy() == 0].mean()), 1)})
out["by_category"] = sorted(cat_rows, key=lambda r: -r["open_at_last_contact_pct"])

# ---------------- note-text lexicon scored against the structured measure ------------------------
notes = pd.read_parquet(D/"v4_notesB.parquet"); notes["enc_date"] = pd.to_datetime(notes.enc_date)
nb = notes.set_index(["person_id", "encounter_id"]).note_text
TAG = re.compile(r"<[^>]+>"); clean = lambda t: html.unescape(TAG.sub(" ", str(t)))
LEX = {"medical": r"medication|refill|pharmac|appointment|specialist|referral|lab|a1c|blood pressure|uncontrolled|symptom|pain|follow[- ]up with (the )?(doctor|provider|pcp)|therap|counsel",
       "social": r"hous|homeless|evict|food|snap|wic|transport|ride|utilit|insurance|medicaid renewal|benefit|employ|job|childcare|legal|shelter|rent"}
txt = [clean(nb.get((p, e), ""))[:8000] for p, e in zip(last.person_id, last.encounter_id)]
val = {}
for dom, pat in LEX.items():
    rx = re.compile(pat, re.I)
    flag = np.array([1 if rx.search(t) else 0 for t in txt])
    ref = last[f"any_open_{dom}"].to_numpy()
    tp = int(((flag == 1) & (ref == 1)).sum()); fp = int(((flag == 1) & (ref == 0)).sum())
    fn = int(((flag == 0) & (ref == 1)).sum()); tn = int(((flag == 0) & (ref == 0)).sum())
    val[dom] = {"text_flag_rate": round(float(flag.mean()), 3), "structured_rate": round(float(ref.mean()), 3),
                "sensitivity": round(tp/max(tp+fn, 1), 3), "specificity": round(tn/max(tn+fp, 1), 3),
                "ppv": round(tp/max(tp+fp, 1), 3), "kappa": round(float(cohen_kappa_score(flag, ref)), 3),
                "f1": round(2*tp/max(2*tp+fp+fn, 1), 3)}
out["text_lexicon_vs_structured"] = val

# ---------------- do open needs carry predictive signal? ------------------------------------------
try:
    out["discrimination_of_open_needs_alone"] = {
        "auroc_any_open_medical": round(float(roc_auc_score(last.E1, last.any_open_medical)), 3),
        "auroc_n_open_needs": round(float(roc_auc_score(last.E1, last.n_open_needs)), 3)}
except Exception as e:
    out["discrimination_of_open_needs_alone"] = {"error": str(e)[:60]}

B[["person_id", "encounter_id", "enc_date", "E1", "has_care_plan", "n_goals_before", "open_goal_medical", "open_goal_social", "open_goal_quality",
   "open_task_medical", "open_task_social", "any_open_medical", "any_open_social", "any_open_need", "n_open_needs"]].to_parquet(D/"v4_needs.parquet")
json.dump(out, open(R/"needs_v4.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ["design", "last_contact", "care_plan_presence", "among_patients_with_a_care_plan",
                                      "contacts_at_least_30_days_after_enrolment", "late_contacts_with_a_care_plan",
                                      "text_lexicon_vs_structured"]}, indent=1, default=str))
