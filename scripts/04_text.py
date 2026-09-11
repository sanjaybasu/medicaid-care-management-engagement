"""v4 step 4: note-text features (block B6) for Task B and Task A.

Trailing-90-day note text at each index, as (a) a nine-pattern clinical lexicon, (b) TF-IDF reduced by
truncated SVD, and (c) bge-base sentence embeddings reduced by PCA. All transforms are fit on the
development era only. Writes data_cache/v4_textB.parquet and v4_textA.parquet."""
import json, pathlib, re, html
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED = 20260911
CUT = pd.Timestamp("2025-07-01")
TAG = re.compile(r"<[^>]+>")
clean = lambda t: html.unescape(TAG.sub(" ", str(t))).replace("\xa0", " ")
LEX = {"unable_to_reach": r"unable to reach|could not reach|no answer|left (a )?(voice ?mail|vm|message)|did not answer",
       "declined": r"declin\w+|refus\w+|not interested|no longer wants|opt(ed)? out",
       "incarcerated": r"incarcerat\w+|in jail|in prison|detention",
       "hospitalized": r"hospitaliz\w+|admitted to|in the hospital|inpatient stay",
       "moved": r"moved (out|away|to)|relocat\w+|new address|out of state",
       "phone_problem": r"disconnect\w+|number (is )?(not|no longer) (in service|working)|wrong number|phone (is )?off",
       "housing_crisis": r"homeless|evict\w+|shelter|unstable housing|couch surf",
       "coverage_issue": r"lost (medicaid|coverage|insurance)|redetermin\w+|renewal|no longer eligible|coverage (ended|lapsed)",
       "engaged_positive": r"agreed to|scheduled (an? )?(appointment|visit|follow)|confirmed|will call back|looking forward"}

notesB = pd.read_parquet(D/"v4_notesB.parquet"); notesB["enc_date"] = pd.to_datetime(notesB.enc_date)
allnotes = pd.read_parquet(D/"v4_encounters.parquet", columns=["person_id", "enc_date", "occurred", "note_text"])
allnotes["enc_date"] = pd.to_datetime(allnotes.enc_date)
allnotes = allnotes[allnotes.note_text.astype(str).str.len() > 0].sort_values(["person_id", "enc_date"])
byp = {p: g for p, g in allnotes.groupby("person_id")}

def trailing_text(pid, d0, days=90, maxchars=8000):
    g = byp.get(pid)
    if g is None: return ""
    w = g[(g.enc_date < d0) & (g.enc_date >= d0 - pd.Timedelta(days=days))]
    if not len(w): return ""
    return clean(" \n ".join(w.note_text.astype(str).tolist()))[-maxchars:]

def build(idx, datecol, index_note=None):
    d = pd.to_datetime(idx[datecol])
    prior = [trailing_text(p, x) for p, x in zip(idx.person_id, d)]
    if index_note is not None:
        txt = [clean(a)[-4000:] + " \n " + b for a, b in zip(index_note, prior)]
    else:
        txt = prior
    return txt

FB = pd.read_parquet(D/"v4_featB.parquet"); FA = pd.read_parquet(D/"v4_featA.parquet")
nb = notesB.set_index(["person_id", "enc_date"]).note_text
idx_note = [nb.get((p, d), "") for p, d in zip(FB.person_id, pd.to_datetime(FB.enc_date))]
txtB = build(FB, "enc_date", idx_note)
txtA = build(FA, "enroll_date")
print("text built", len(txtB), len(txtA), "median chars B", int(np.median([len(t) for t in txtB])), flush=True)

devB = (pd.to_datetime(FB.enc_date) < CUT).to_numpy()
devA = (pd.to_datetime(FA.enroll_date) < CUT).to_numpy()

def lex_frame(txt, pref):
    out = {}
    for k, pat in LEX.items():
        rx = re.compile(pat, re.I)
        out[f"{pref}lex_{k}"] = [len(rx.findall(t)) for t in txt]
    out[f"{pref}text_chars"] = [len(t) for t in txt]
    return pd.DataFrame(out)

def tfidf_svd(txt, dev, pref, ncomp=100):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, stop_words="english")
    vec.fit([t for t, m in zip(txt, dev) if m])
    X = vec.transform(txt)
    svd = TruncatedSVD(n_components=ncomp, random_state=SEED).fit(X[dev])
    Z = svd.transform(X)
    return pd.DataFrame(Z, columns=[f"{pref}svd{i:03d}" for i in range(ncomp)]), float(svd.explained_variance_ratio_.sum())

def embed(txt, dev, pref, ncomp=64):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")
    E = m.encode([t[-2000:] if t else "" for t in txt], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    p = PCA(n_components=ncomp, random_state=SEED).fit(E[dev])
    return pd.DataFrame(p.transform(E), columns=[f"{pref}emb{i:02d}" for i in range(ncomp)]), float(p.explained_variance_ratio_.sum())

meta = {}
for name, txt, dev, F, key in [("B", txtB, devB, FB, ["person_id", "encounter_id"]), ("A", txtA, devA, FA, ["person_id"])]:
    L = lex_frame(txt, "")
    S, ev = tfidf_svd(txt, dev, "", 100 if name == "B" else 60)
    E, ev2 = embed(txt, dev, "", 64 if name == "B" else 48)
    out = pd.concat([F[key].reset_index(drop=True), L, S, E], axis=1)
    out.to_parquet(D/f"v4_text{name}.parquet")
    meta[name] = {"rows": len(out), "cols": int(out.shape[1]), "svd_explained": round(ev, 3), "pca_explained": round(ev2, 3),
                  "lexicon_hit_rate": {k: round(float((L[f"lex_{k}"] > 0).mean()), 3) for k in LEX}}
    print(name, meta[name]["rows"], meta[name]["cols"], flush=True)
json.dump(meta, open(R/"text_v4.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
