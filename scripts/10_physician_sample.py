"""v4 step 10: blinded physician adjudication sample (pre-registration Section 11).

200 validation-era contacts: 100 whose patient disengaged after the contact and 100 whose patient
sustained engagement or completed the program. Three reviewers, 20% double-read, seed 20260909.
Writes case packets and blank forms to data_cache/physician_review_v4/ (PHI; local only) and
counts to results/physician_sample_v4.json. Forms are filled only by the reviewing physician."""
import json, pathlib, re, html
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
OUT = D/"physician_review_v4"; (OUT/"cases").mkdir(parents=True, exist_ok=True); (OUT/"forms").mkdir(exist_ok=True)
SEED, N_PER_GROUP, DOUBLE = 20260909, 100, 0.20
TAG = re.compile(r"<[^>]+>"); clean = lambda t: re.sub(r"\n{3,}", "\n\n", html.unescape(TAG.sub(" ", str(t))))

B = pd.read_parquet(D/"v4_outcomes_B.parquet"); B["enc_date"] = pd.to_datetime(B.enc_date)
B = B[B.analysis & (B.era == "validation")].copy()
notes = pd.read_parquet(D/"v4_notesB.parquet"); notes["enc_date"] = pd.to_datetime(notes.enc_date)
enc = pd.read_parquet(D/"v4_encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date)

last = B.sort_values("enc_date").groupby("person_id").tail(1)
dis = last[last.E1 == 1]
sus = last[(last.E1 == 0)]
rng = np.random.default_rng(SEED)
pick = lambda d, lab: d.sample(n=min(N_PER_GROUP, len(d)), random_state=int(rng.integers(1e9))).assign(group=lab)
S = pd.concat([pick(dis, "disengaged"), pick(sus, "sustained")], ignore_index=True)
S["case_id"] = [f"V{i:04d}" for i in rng.permutation(len(S))]
S = S.sort_values("case_id").reset_index(drop=True)

rev = np.array(["A", "B", "C"])[np.arange(len(S)) % 3]
S["reviewer"] = rev
dbl = S.groupby("group", group_keys=False).apply(lambda g: g.sample(frac=DOUBLE, random_state=SEED)).case_id
S["second_reviewer"] = [({"A": "B", "B": "C", "C": "A"}[r] if c in set(dbl) else "") for r, c in zip(S.reviewer, S.case_id)]

for _, r in S.iterrows():
    hist = enc[(enc.person_id == r.person_id) & (enc.enc_date <= r.enc_date) & (enc.enc_date >= r.enc_date - pd.Timedelta(days=90))].sort_values("enc_date")
    lines = [f"# Case {r.case_id}", "", f"Index contact date: {r.enc_date.date()}", f"State: {r.state}",
             f"Age: {int(r.age) if pd.notna(r.age) else 'unknown'}", "", "## Contacts and attempts in the 90 days up to and including the index contact", ""]
    for _, e in hist.iterrows():
        lines += [f"### {e.enc_date.date()} - {e.contact_type} - {e.encounter_type} - occurred: {e.occurred} - staff role: {e.roles}", clean(e.note_text)[:6000], ""]
    (OUT/"cases"/f"case_{r.case_id}.md").write_text("\n".join(lines))

cols = ["case_id", "medication_gap", "pending_referral_or_appointment", "uncontrolled_condition", "active_social_crisis",
        "behavioral_health_need", "no_open_need", "harm_if_no_contact_90d", "comment"]
for rv in ["A", "B", "C"]:
    ids = sorted(set(S[S.reviewer == rv].case_id) | set(S[S.second_reviewer == rv].case_id))
    f = pd.DataFrame({"case_id": ids})
    for c in cols[1:]: f[c] = ""
    f.to_csv(OUT/"forms"/f"reviewer_{rv}_form.csv", index=False)
    with open(OUT/"cases"/f"reviewer_{rv}_all_cases.md", "w") as fh:
        fh.write("\n\n---\n\n".join((OUT/"cases"/f"case_{i}.md").read_text() for i in ids))
S[["case_id", "person_id", "encounter_id", "enc_date", "group", "reviewer", "second_reviewer"]].to_csv(OUT/"ANSWER_KEY_do_not_share.csv", index=False)
S[["case_id", "reviewer", "second_reviewer"]].to_csv(OUT/"assignment.csv", index=False)

meta = {"design": "v4_single_phase", "seed": SEED, "cases": int(len(S)), "per_group": {k: int(v) for k, v in S.group.value_counts().items()},
        "candidates": {"disengaged": int(len(dis)), "sustained": int(len(sus))},
        "forms_per_reviewer": {rv: int(len(set(S[S.reviewer == rv].case_id) | set(S[S.second_reviewer == rv].case_id))) for rv in ["A", "B", "C"]},
        "double_read_cases": int(len(set(dbl))),
        "power": {"alpha": 0.05, "two_sided": True, "p_disengaged": 0.90, "p_sustained": 0.75, "n_per_group": N_PER_GROUP, "power": 0.815}}
json.dump(meta, open(R/"physician_sample_v4.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
