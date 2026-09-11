"""v4 step 3: structured feature blocks B1-B5 and B7 (pre-registration Section 7).

Every feature uses only information timestamped strictly before the index instant.
Writes data_cache/v4_featB.parquet (Task B) and data_cache/v4_featA.parquet (Task A)."""
import json, pathlib
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)

B = pd.read_parquet(D/"v4_outcomes_B.parquet"); A = pd.read_parquet(D/"v4_outcomes_A.parquet")
enc = pd.read_parquet(D/"v4_encounters.parquet")
enc["enc_date"] = pd.to_datetime(enc.enc_date); enc["start_time"] = pd.to_datetime(enc.start_time)
enc = enc[enc.person_id.isin(set(A.person_id))].copy()
TZ = {"OHIO": "America/New_York", "VIRGINIA": "America/New_York", "WASHINGTON": "America/Los_Angeles"}
_st = A.set_index("person_id").state.to_dict()
_tz = enc.person_id.map(lambda p: TZ.get(_st.get(p), "America/New_York"))
enc["local_hour"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(z).hour if pd.notna(t) else np.nan
                     for t, z in zip(enc.start_time, _tz)]
adt = pd.read_parquet(D/"v4_adt.parquet"); adt["admit_date"] = pd.to_datetime(adt.admit_date)
med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["claim_start_date"] = pd.to_datetime(med.claim_start_date)
ph = pd.read_parquet(D/"v4_pharmacy_claim_lines.parquet"); ph["claim_start_date"] = pd.to_datetime(ph.claim_start_date)
gl = pd.read_parquet(D/"v4_goals.parquet"); gl["created_at"] = pd.to_datetime(gl.created_at); gl["updated_at"] = pd.to_datetime(gl.updated_at)

class Stream:
    """Per-patient sorted event days with optional weights, for windowed counts and sums."""
    def __init__(self, df, datecol, weightcol=None):
        d = df.dropna(subset=[datecol]).sort_values(["person_id", datecol])
        self.days = {p: day(g[datecol]) for p, g in d.groupby("person_id")}
        self.wts = {p: g[weightcol].to_numpy(dtype=float) for p, g in d.groupby("person_id")} if weightcol else None
    def count(self, pid, d0, lo, hi):
        a = self.days.get(pid)
        if a is None: return 0
        return int(np.searchsorted(a, d0+hi, "right") - np.searchsorted(a, d0+lo, "left"))
    def total(self, pid, d0, lo, hi):
        a = self.days.get(pid)
        if a is None: return 0.0
        i, j = np.searchsorted(a, d0+lo, "left"), np.searchsorted(a, d0+hi, "right")
        return float(self.wts[pid][i:j].sum()) if j > i else 0.0
    def days_since_last(self, pid, d0):
        a = self.days.get(pid)
        if a is None: return np.nan
        i = np.searchsorted(a, d0, "left")            # strictly before d0
        return float(d0 - a[i-1]) if i > 0 else np.nan

completed = enc[enc.occurred == "YES"]
attempts = enc[enc.occurred == "NO"]
S = {"contact": Stream(completed, "enc_date"), "attempt": Stream(attempts, "enc_date"),
     "adt": Stream(adt, "admit_date"), "goal_new": Stream(gl, "created_at"), "goal_upd": Stream(gl, "updated_at"),
     "rx": Stream(ph, "claim_start_date", "paid"), "med": Stream(med, "claim_start_date", "paid")}
for lbl, sub in [("inperson", completed[completed.contact_type.astype(str).str.contains("HOME_VISIT|HOSPITAL|OTHER_INPERSON|CBO|IN_COMMUNITY|PROVIDER_OFFICE", case=False, na=False)]),
                 ("phone", completed[completed.contact_type.astype(str).str.contains("PHONE", case=False, na=False)]),
                 ("text", completed[completed.contact_type.astype(str).str.contains("SMS|TEXT", case=False, na=False)]),
                 ("chw", completed[completed.roles.astype(str).str.contains("CHW", case=False, na=False)]),
                 ("therapy", completed[completed.roles.astype(str).str.contains("THERAP|BHS|LCSW", case=False, na=False)]),
                 ("pharm", completed[completed.roles.astype(str).str.contains("PHARM", case=False, na=False)]),
                 ("outreach", completed[completed.encounter_type.astype(str).str.contains("OUTREACH", case=False, na=False)]),
                 ("provider", completed[completed.encounter_type.astype(str).str.contains("PROVIDER", case=False, na=False)])]:
    S[lbl] = Stream(sub, "enc_date")
ed_adt = adt[adt.class_code.astype(str).str.upper().str.startswith("E")]
ip_adt = adt[adt.class_code.astype(str).str.upper().str.startswith("I")]
S["adt_ed"], S["adt_ip"] = Stream(ed_adt, "admit_date"), Stream(ip_adt, "admit_date")

STAFF = {p: g.sort_values("enc_date")[["enc_date", "created_by_id"]] for p, g in completed.groupby("person_id")}
HOUR = {p: g.sort_values("enc_date")[["enc_date", "local_hour"]] for p, g in completed.groupby("person_id")}

def build(idx, datecol, taskB):
    d0s = day(idx[datecol]); rows = []
    for k, (pid, d0) in enumerate(zip(idx.person_id.to_numpy(), d0s)):
        r = {}
        for nm in ["contact", "attempt", "inperson", "phone", "text", "chw", "therapy", "pharm", "outreach", "provider", "adt", "adt_ed", "adt_ip", "goal_new", "goal_upd"]:
            for w in (7, 30, 90, 365):
                r[f"n_{nm}_{w}d"] = S[nm].count(pid, d0, -w, -1)
            r[f"days_since_{nm}"] = S[nm].days_since_last(pid, d0)
        r["n_contact_prior_all"] = S["contact"].count(pid, d0, -20000, -1)
        r["n_attempt_prior_all"] = S["attempt"].count(pid, d0, -20000, -1)
        r["paid_prior_365"] = S["med"].total(pid, d0, -365, -1) + S["rx"].total(pid, d0, -365, -1)
        r["paid_prior_90"] = S["med"].total(pid, d0, -90, -1) + S["rx"].total(pid, d0, -90, -1)
        h = HOUR.get(pid)
        if h is not None:
            prior = h[day(h.enc_date) < d0]
            r["n_staff_distinct"] = int(STAFF[pid][day(STAFF[pid].enc_date) < d0].created_by_id.nunique()) if pid in STAFF else 0
            st = prior.local_hour.dropna()
            r["mean_hour_prior"] = float(st.mean()) if len(st) else np.nan
            r["share_morning_prior"] = float(st.between(8, 11).mean()) if len(st) else np.nan
            r["share_weekend_prior"] = float((prior.enc_date.dt.dayofweek >= 5).mean()) if len(prior) else np.nan
            gaps = np.diff(day(prior.enc_date)) if len(prior) > 1 else np.array([])
            r["mean_gap_prior"] = float(gaps.mean()) if len(gaps) else np.nan
            r["sd_gap_prior"] = float(gaps.std()) if len(gaps) > 1 else np.nan
            r["max_gap_prior"] = float(gaps.max()) if len(gaps) else np.nan
            r["last_gap_prior"] = float(gaps[-1]) if len(gaps) else np.nan
        rows.append(r)
    F = pd.DataFrame(rows, index=idx.index)
    F["contact_rate_30d"] = F.n_contact_30d/30.0
    F["attempt_success_90d"] = F.n_contact_90d/(F.n_contact_90d + F.n_attempt_90d).replace(0, np.nan)
    F["inperson_share_90d"] = F.n_inperson_90d/F.n_contact_90d.replace(0, np.nan)
    F["phone_share_90d"] = F.n_phone_90d/F.n_contact_90d.replace(0, np.nan)
    return F

def assemble(idx, datecol, taskB):
    base = idx.copy()
    F = build(base, datecol, taskB)
    out = pd.concat([base.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    d = pd.to_datetime(out[datecol])
    out["index_month"] = d.dt.month; out["index_dow"] = d.dt.dayofweek; out["index_quarter"] = d.dt.quarter
    out["index_year_frac"] = d.dt.year + (d.dt.dayofyear/366.0)
    if taskB:
        out["days_since_enroll"] = (d - pd.to_datetime(out.enroll_date)).dt.days
        out["contact_index"] = out.groupby("person_id").cumcount()
        out["index_hour"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ.get(st, "America/New_York")).hour
                             if pd.notna(t) else np.nan for t, st in zip(out.start_time, out.state)]
        out["index_is_outreach"] = out.encounter_type.astype(str).str.contains("OUTREACH", case=False, na=False).astype(int)
        out["index_is_inperson"] = out.contact_type.astype(str).str.contains("HOME_VISIT|HOSPITAL|OTHER_INPERSON|CBO|IN_COMMUNITY|PROVIDER_OFFICE", case=False, na=False).astype(int)
        out["index_is_phone"] = out.contact_type.astype(str).str.contains("PHONE", case=False, na=False).astype(int)
        out["index_is_chw"] = out.roles.astype(str).str.contains("CHW", case=False, na=False).astype(int)
        out["index_note_chars"] = out.note_text.astype(str).str.len()
    return out

print("building Task B features", flush=True)
FB = assemble(B[B.eligible_E1].copy(), "enc_date", True)
FB.drop(columns=["note_text"]).to_parquet(D/"v4_featB.parquet")
FB[["person_id", "encounter_id", "enc_date", "note_text"]].to_parquet(D/"v4_notesB.parquet")
print("Task B features", FB.shape, flush=True)
print("building Task A features", flush=True)
FA = assemble(A[A.eligible_E2].copy(), "enroll_date", False)
FA.to_parquet(D/"v4_featA.parquet")
print("Task A features", FA.shape)
json.dump({"taskB_rows": int(FB.shape[0]), "taskB_cols": int(FB.shape[1]), "taskA_rows": int(FA.shape[0]), "taskA_cols": int(FA.shape[1])},
          open(R/"features_v4.json", "w"), indent=1)
