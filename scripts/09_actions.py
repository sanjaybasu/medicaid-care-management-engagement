"""v4 step 9: Aim 3 - care-team actions among patients at high predicted risk of disengagement
(pre-registration Section 9.3): within-patient, marginal structural, and cross-fitted AIPW estimates
with pre-specified falsification gates."""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.model_selection import KFold
import lightgbm as lgb
warnings.filterwarnings("ignore")

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED = 20260911
HIGH_RISK_FRAC = 0.20
out = {"high_risk_fraction": HIGH_RISK_FRAC}

F = pd.read_parquet(D/"v4_featB.parquet")
S = pd.read_parquet(D/"v4_score_all_B_full.parquet")
F = F.merge(S[["person_id", "encounter_id", "p_ensemble"]], on=["person_id", "encounter_id"], how="inner")
F["enc_date"] = pd.to_datetime(F.enc_date)
enc = pd.read_parquet(D/"v4_encounters.parquet")
enc["enc_date"] = pd.to_datetime(enc.enc_date); enc["start_time"] = pd.to_datetime(enc.start_time)
enc = enc[enc.person_id.isin(set(F.person_id))]
comp, att = enc[enc.occurred == "YES"], enc[enc.occurred == "NO"]
sh = pd.read_parquet(D/"v4_status_history.parquet"); sh["updated_at"] = pd.to_datetime(sh.updated_at)
grad = sh[sh.new_status == "GRADUATED"].groupby("person_id").updated_at.min().dt.normalize()
gl = pd.read_parquet(D/"v4_goals.parquet"); gl["updated_at"] = pd.to_datetime(gl.updated_at)
hedis_done = gl[(gl.goal_type == "HEDIS") & gl.status.isin(["COMPLETED", "PROVISIONALLY_COMPLETED"])]

day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
def streams(df, col="enc_date"):
    d = df.dropna(subset=[col]).sort_values(["person_id", col])
    return {p: day(g[col]) for p, g in d.groupby("person_id")}
SC, SA = streams(comp), streams(att)
def in_win(store, pid, d0, lo, hi):
    a = store.get(pid)
    return 0 if a is None else int(np.searchsorted(a, d0+hi, "right") - np.searchsorted(a, d0+lo, "left"))

sub = lambda d, pat, col: d[d[col].astype(str).str.contains(pat, case=False, na=False)]
S_inperson_chw = streams(sub(sub(comp, "HOME_VISIT|HOSPITAL|OTHER_INPERSON|CBO|IN_COMMUNITY|PROVIDER_OFFICE", "contact_type"), "CHW", "roles"))
S_therapy = streams(sub(comp, "THERAP|BHS|LCSW", "roles"))
S_pharm = streams(sub(comp, "PHARM", "roles"))
TZ = {"OHIO": "America/New_York", "VIRGINIA": "America/New_York", "WASHINGTON": "America/Los_Angeles"}
_st = F.drop_duplicates("person_id").set_index("person_id").state.to_dict()
att = att.copy()
att["local_hour"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ.get(_st.get(p), "America/New_York")).hour if pd.notna(t) else np.nan
                     for t, p in zip(att.start_time, att.person_id)]
am = att[(att.local_hour.between(8, 11)) & (att.enc_date.dt.dayofweek < 5) & att.contact_type.astype(str).str.contains("PHONE", case=False, na=False)]
S_morning_call = streams(am)
S_hedis = streams(hedis_done.rename(columns={"updated_at": "enc_date"}))

F["d0"] = day(F.enc_date)
F["prev_p"] = F.sort_values(["person_id", "enc_date"]).groupby("person_id").p_ensemble.shift()
F["spike"] = ((F.p_ensemble.rank(pct=True)*10).astype(int) - (F.prev_p.rank(pct=True)*10).fillna(-99).astype(int) >= 1) & F.prev_p.notna()
thr = F.p_ensemble.quantile(1-HIGH_RISK_FRAC)
F["high_risk"] = F.p_ensemble >= thr
out["high_risk_threshold"] = round(float(thr), 4)

def outcome_after(pid, d0, wend):
    """sustained engagement: a completed contact in (wend, wend+90] or program completion in that period."""
    if in_win(SC, pid, d0, wend+1, wend+90) > 0: return 1
    g = grad.get(pid)
    return int(g is not None and pd.notna(g) and d0 + wend < day(pd.Series([g]))[0] <= d0 + wend + 90)

COVS = ["p_ensemble", "age", "risk_percentile", "days_since_enroll", "n_contact_30d", "n_contact_90d", "n_attempt_30d",
        "days_since_contact", "days_since_attempt", "n_adt_90d", "any_bh", "sud", "diabetes", "htn", "contact_index",
        "inperson_share_90d", "phone_share_90d", "mean_gap_prior", "n_goal_upd_90d"]
def covmat(d):
    X = d[COVS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = pd.concat([X, pd.get_dummies(d.state, prefix="st", drop_first=True).astype(float)], axis=1)
    X = X.astype(float)
    return X.loc[:, X.std() > 0]

def smd(x, t, w=None):
    w = np.ones(len(x)) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1]-m1)**2, weights=w[t == 1]); v0 = np.average((x[t == 0]-m0)**2, weights=w[t == 0])
    s = np.sqrt((v1+v0)/2)
    return 0.0 if s == 0 else float((m1-m0)/s)

def evalue(rr):
    rr = rr if rr >= 1 else 1/rr
    return round(float(rr + np.sqrt(rr*(rr-1))), 2)

def aipw(X, t, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    n = len(y); mu1 = np.zeros(n); mu0 = np.zeros(n); ps = np.zeros(n)
    for tr, te in kf.split(X):
        g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=SEED)
        ps[te] = g.fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[te])[:, 1]
        for a, mu in [(1, mu1), (0, mu0)]:
            m = tr[t[tr] == a]
            if len(m) < 30 or len(np.unique(y[m])) < 2: mu[te] = y[m].mean() if len(m) else 0.0; continue
            mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=SEED)
            mu[te] = mdl.fit(X.iloc[m], y[m]).predict_proba(X.iloc[te])[:, 1]
    ps = np.clip(ps, 0.05, 0.95)
    psi = mu1 - mu0 + t*(y-mu1)/ps - (1-t)*(y-mu0)/(1-ps)
    return float(psi.mean()), float(psi.std(ddof=1)/np.sqrt(len(psi))), ps

ACTIONS = {
 "A1_attempt_during_7day_lapse": dict(window=7, restrict="lapse",
    expose=lambda pid, d0: int(in_win(SA, pid, d0, 1, 7) > 0)),
 "A2_inperson_chw_14d": dict(window=14, restrict=None,
    expose=lambda pid, d0: int(in_win(S_inperson_chw, pid, d0, 1, 14) > 0)),
 "A3_therapy_30d": dict(window=30, restrict="no_prior_therapy",
    expose=lambda pid, d0: int(in_win(S_therapy, pid, d0, 1, 30) > 0)),
 "A4_pharmacist_30d": dict(window=30, restrict="no_prior_pharm",
    expose=lambda pid, d0: int(in_win(S_pharm, pid, d0, 1, 30) > 0)),
 "A5_morning_weekday_call_14d": dict(window=14, restrict="any_attempt_14d",
    expose=lambda pid, d0: int(in_win(S_morning_call, pid, d0, 1, 14) > 0)),
}

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
results = {}
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def analyse(d, spec, label):
    w = spec["window"]
    d = d.copy()
    d["T"] = [spec["expose"](p, x) for p, x in zip(d.person_id, d.d0)]
    d["Y"] = [outcome_after(p, x, w) for p, x in zip(d.person_id, d.d0)]
    d["pre_engage"] = [in_win(SC, p, x, -30, -1) for p, x in zip(d.person_id, d.d0)]
    d["neg_ctrl"] = [int(in_win(S_hedis, p, x, w+1, w+90) > 0) for p, x in zip(d.person_id, d.d0)]
    X = covmat(d); t = d["T"].to_numpy(); y = d["Y"].to_numpy()
    if t.sum() < 50 or (1-t).sum() < 50:
        return {"n": int(len(d)), "treated": int(t.sum()), "note": "insufficient variation"}
    Xz = StandardScaler().fit_transform(X)
    ps = LogisticRegression(max_iter=3000, C=1.0).fit(Xz, t).predict_proba(Xz)[:, 1]
    keep = (ps > 0.05) & (ps < 0.95)
    if keep.sum() < 100 or t[keep].sum() < 25:
        return {"n": int(len(d)), "treated": int(t.sum()), "n_trimmed": int(keep.sum()), "note": "equipoise region too small"}
    Xk, tk, yk, psk, dk = X[keep], t[keep], y[keep], np.clip(ps[keep], 0.05, 0.95), d[keep]
    sw = np.where(tk == 1, tk.mean()/psk, (1-tk.mean())/(1-psk))
    m = sm.GLM(yk, sm.add_constant(np.column_stack([tk])), family=sm.families.Binomial(), freq_weights=sw).fit(cov_type="HC0")
    inv = sm.families.links.Logit().inverse
    p1, p0 = float(inv(m.params[0]+m.params[1])), float(inv(m.params[0]))
    b, se = float(m.params[1]), float(np.sqrt(np.diag(m.cov_params())[1]))
    msm = {"rd": round(p1-p0, 4), "or": round(float(np.exp(b)), 3),
           "or_ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)]}
    ate, se_ate, _ = aipw(Xk, tk, yk)
    aipw_res = {"rd": round(ate, 4), "ci_95": [round(ate-1.96*se_ate, 4), round(ate+1.96*se_ate, 4)]}
    grp = dk.groupby("person_id").T.nunique(); disc = set(grp[grp > 1].index)
    wp = {"n_discordant_patients": int(len(disc))}
    if len(disc) >= 50:
        dd = dk[dk.person_id.isin(disc)]
        try:
            cl = ConditionalLogit(dd.Y.to_numpy(), dd[["T"]].astype(float).to_numpy(), groups=dd.person_id.to_numpy()).fit(disp=0)
            bb, ss = float(cl.params[0]), float(cl.bse[0])
            wp |= {"or": round(float(np.exp(bb)), 3), "ci_95": [round(float(np.exp(bb-1.96*ss)), 3), round(float(np.exp(bb+1.96*ss)), 3)]}
        except Exception as e:
            wp |= {"error": str(e)[:80]}
    g1 = abs(smd(dk.pre_engage.to_numpy(float), tk))
    smds = {c: abs(smd(Xk[c].to_numpy(float), tk, sw)) for c in Xk.columns}
    try:
        joint_p = round(float(sm.Logit(tk, sm.add_constant(Xk.to_numpy(), has_constant="add")).fit(disp=0).llr_pvalue), 4)
    except Exception:
        joint_p = None
    g3 = sm.GLM(dk.neg_ctrl.to_numpy(float), sm.add_constant(np.column_stack([tk])), family=sm.families.Binomial(), freq_weights=sw).fit(cov_type="HC0")
    bn, sn = float(g3.params[1]), float(np.sqrt(np.diag(g3.cov_params())[1]))
    rr = p1/max(p0, 1e-6)
    gates = {"G1_pretrend_smd": round(g1, 3), "G1_pass": bool(g1 < 0.10),
             "G2_max_smd_weighted": round(max(smds.values()), 3), "G2_joint_p": joint_p,
             "G2_pass": bool(max(smds.values()) < 0.10),
             "G3_negctrl_or": round(float(np.exp(bn)), 3),
             "G3_negctrl_ci": [round(float(np.exp(bn-1.96*sn)), 3), round(float(np.exp(bn+1.96*sn)), 3)],
             "G3_pass": bool(np.exp(bn-1.96*sn) < 1 < np.exp(bn+1.96*sn)),
             "G4_evalue_point": evalue(rr)}
    gates["all_pass"] = bool(gates["G1_pass"] and gates["G2_pass"] and gates["G3_pass"])
    return {"n": int(len(d)), "n_trimmed": int(keep.sum()), "treated": int(tk.sum()), "outcome_rate": round(float(yk.mean()), 4),
            "outcome_window": f"days {w+1} to {w+90}", "msm": msm, "aipw": aipw_res, "within_patient": wp, "gates": gates,
            "mdc_rd_80pct": round(float(2.8*np.sqrt(yk.mean()*(1-yk.mean())*(1/max(tk.sum(), 1)+1/max((1-tk).sum(), 1)))), 4)}

for POP in ["high_risk", "all_risk"]:
    H = F[F.high_risk].copy() if POP == "high_risk" else F.copy()
    for name0, spec in ACTIONS.items():
        d = H
        if spec["restrict"] == "lapse": d = d[[in_win(SC, p, x, 1, 7) == 0 for p, x in zip(d.person_id, d.d0)]]
        if spec["restrict"] == "no_prior_therapy": d = d[[in_win(S_therapy, p, x, -20000, 0) == 0 for p, x in zip(d.person_id, d.d0)]]
        if spec["restrict"] == "no_prior_pharm": d = d[[in_win(S_pharm, p, x, -20000, 0) == 0 for p, x in zip(d.person_id, d.d0)]]
        if spec["restrict"] == "any_attempt_14d": d = d[[in_win(SA, p, x, 1, 14) > 0 for p, x in zip(d.person_id, d.d0)]]
        key = f"{name0}__{POP}"
        results[key] = {"n": int(len(d)), "note": "too few eligible contacts"} if len(d) < 200 else analyse(d, spec, key)
        r = results[key]
        print(f"  {key}: n={r.get('n')} treated={r.get('treated')} " +
              (f"MSM RD={r['msm']['rd']:+.3f} AIPW RD={r['aipw']['rd']:+.3f} within-patient={r['within_patient'].get('or')} gates={r['gates']['all_pass']}"
               if "msm" in r else r.get("note", "")), flush=True)

out["actions"] = results
json.dump(out, open(R/"actions_v4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:2500])
