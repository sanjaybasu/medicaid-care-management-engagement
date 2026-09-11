"""v4 step 16: care-team actions with estimators that handle selection and confounding by indication
better than the trimmed marginal structural model of step 9.

Adds (a) overlap weights, which target the population with genuine equipoise without discarding rows;
(b) targeted maximum likelihood estimation; (c) cross-fitted doubly robust learners and a causal
forest with honest splitting; (d) active-comparator contrasts, in which both arms required the team to
act, so the indication to act is held fixed; and (e) empirical calibration of each estimate against a
null distribution built from six negative-control outcomes."""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
warnings.filterwarnings("ignore")
from econml.dml import CausalForestDML
from econml.dr import LinearDRLearner

D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
R = pathlib.Path(__file__).resolve().parent.parent/"results"
SEED = 20260911
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)

F = pd.read_parquet(D/"v4_featB.parquet").merge(
    pd.read_parquet(D/"v4_score_all_B_full.parquet")[["person_id", "encounter_id", "p_ensemble"]],
    on=["person_id", "encounter_id"], how="inner")
F["enc_date"] = pd.to_datetime(F.enc_date); F["d0"] = day(F.enc_date)
enc = pd.read_parquet(D/"v4_encounters.parquet")
enc["enc_date"] = pd.to_datetime(enc.enc_date); enc["start_time"] = pd.to_datetime(enc.start_time)
enc = enc[enc.person_id.isin(set(F.person_id))]
comp, att = enc[enc.occurred == "YES"], enc[enc.occurred == "NO"]
sh = pd.read_parquet(D/"v4_status_history.parquet"); sh["updated_at"] = pd.to_datetime(sh.updated_at)
grad = sh[sh.new_status == "GRADUATED"].groupby("person_id").updated_at.min().dt.normalize()
gl = pd.read_parquet(D/"v4_goals.parquet"); gl["updated_at"] = pd.to_datetime(gl.updated_at)
hedis = gl[(gl.goal_type == "HEDIS") & gl.status.isin(["COMPLETED", "PROVISIONALLY_COMPLETED"])].rename(columns={"updated_at": "enc_date"})
med = pd.read_parquet(D/"v4_medical_claim_lines.parquet"); med["enc_date"] = pd.to_datetime(med.claim_start_date)
dial = med[med.encounter_type.astype(str).str.lower().str.contains("dialysis")]
edcl = med[med.encounter_type.astype(str).str.lower().str.contains("emergency")]

def streams(df, col="enc_date"):
    d = df.dropna(subset=[col]).sort_values(["person_id", col])
    return {p: day(g[col]) for p, g in d.groupby("person_id")}
SC, SA, SH, SDI, SED = streams(comp), streams(att), streams(hedis), streams(dial), streams(edcl)
INP = "HOME_VISIT|HOSPITAL|OTHER_INPERSON|CBO|IN_COMMUNITY|PROVIDER_OFFICE"
sub = lambda d, pat, col: d[d[col].astype(str).str.contains(pat, case=False, na=False)]
chw = sub(comp, "CHW", "roles")
S_chw_inperson = streams(sub(chw, INP, "contact_type")); S_chw_phone = streams(sub(chw, "PHONE", "contact_type"))
S_ther = streams(sub(comp, "THERAP|BHS|LCSW", "roles")); S_pharm = streams(sub(comp, "PHARM", "roles"))
TZ = {"OHIO": "America/New_York", "VIRGINIA": "America/New_York", "WASHINGTON": "America/Los_Angeles"}
_st = F.drop_duplicates("person_id").set_index("person_id").state.to_dict()
att = att.copy()
att["local_hour"] = [pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ.get(_st.get(p), "America/New_York")).hour if pd.notna(t) else np.nan
                     for t, p in zip(att.start_time, att.person_id)]
S_call = streams(att[att.contact_type.astype(str).str.contains("PHONE", case=False, na=False)])
S_text = streams(att[att.contact_type.astype(str).str.contains("SMS|TEXT", case=False, na=False)])

def win(store, pid, d0, lo, hi):
    a = store.get(pid)
    return 0 if a is None else int(np.searchsorted(a, d0+hi, "right") - np.searchsorted(a, d0+lo, "left"))
def sustained(pid, d0, wend):
    if win(SC, pid, d0, wend+1, wend+90) > 0: return 1
    g = grad.get(pid)
    return int(g is not None and pd.notna(g) and d0 + wend < day(pd.Series([g]))[0] <= d0 + wend + 90)

COVS = ["p_ensemble", "age", "risk_percentile", "days_since_enroll", "n_contact_30d", "n_contact_90d", "n_attempt_30d",
        "days_since_contact", "days_since_attempt", "n_adt_90d", "any_bh", "sud", "diabetes", "htn", "contact_index",
        "inperson_share_90d", "phone_share_90d", "mean_gap_prior", "n_goal_upd_90d", "n_inperson_90d", "n_chw_90d"]
def covmat(d):
    X = d[[c for c in COVS if c in d.columns]].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = pd.concat([X, pd.get_dummies(d.state, prefix="st", drop_first=True).astype(float)], axis=1).astype(float)
    return X.loc[:, X.std() > 0]

def smd(x, t, w):
    m1, m0 = np.average(x[t == 1], weights=w[t == 1]), np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1]-m1)**2, weights=w[t == 1]); v0 = np.average((x[t == 0]-m0)**2, weights=w[t == 0])
    s = np.sqrt((v1+v0)/2)
    return 0.0 if s == 0 else float((m1-m0)/s)

def ow_estimate(X, t, y):
    """Overlap-weighted risk difference: weights t*(1-e)+(1-t)*e, which need no trimming."""
    Xz = StandardScaler().fit_transform(X)
    e = np.clip(LogisticRegression(max_iter=3000, C=1.0).fit(Xz, t).predict_proba(Xz)[:, 1], 1e-4, 1-1e-4)
    w = np.where(t == 1, 1-e, e)
    m = sm.GLM(y, sm.add_constant(t.astype(float)), family=sm.families.Binomial(), freq_weights=w).fit(cov_type="HC0")
    inv = sm.families.links.Logit().inverse
    p1, p0 = float(inv(m.params[0]+m.params[1])), float(inv(m.params[0]))
    b, se = float(m.params[1]), float(np.sqrt(np.diag(m.cov_params())[1]))
    lo, hi = inv(m.params[0]+b-1.96*se)-p0, inv(m.params[0]+b+1.96*se)-p0
    bal = max(abs(smd(X[c].to_numpy(float), t, w)) for c in X.columns)
    return {"rd": round(p1-p0, 4), "ci_95": [round(float(lo), 4), round(float(hi), 4)],
            "max_abs_smd_weighted": round(bal, 3), "ess": round(float(w.sum()**2/np.sum(w**2)), 0)}, e, w

def tmle(X, t, y, e):
    """Targeted maximum likelihood estimate of the risk difference with a logistic fluctuation."""
    kf = KFold(5, shuffle=True, random_state=SEED); n = len(y); q1 = np.zeros(n); q0 = np.zeros(n); qa = np.zeros(n)
    for tr, te in kf.split(X):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=SEED)
        Xtr = np.column_stack([X.iloc[tr].to_numpy(), t[tr]])
        m.fit(Xtr, y[tr])
        q1[te] = m.predict_proba(np.column_stack([X.iloc[te].to_numpy(), np.ones(len(te))]))[:, 1]
        q0[te] = m.predict_proba(np.column_stack([X.iloc[te].to_numpy(), np.zeros(len(te))]))[:, 1]
        qa[te] = m.predict_proba(np.column_stack([X.iloc[te].to_numpy(), t[te]]))[:, 1]
    q1, q0, qa = np.clip(q1, 1e-4, 1-1e-4), np.clip(q0, 1e-4, 1-1e-4), np.clip(qa, 1e-4, 1-1e-4)
    h = t/e - (1-t)/(1-e)
    off = np.log(qa/(1-qa))
    eps = sm.GLM(y, h.reshape(-1, 1), family=sm.families.Binomial(), offset=off).fit().params[0]
    inv = lambda z: 1/(1+np.exp(-z))
    q1s = inv(np.log(q1/(1-q1)) + eps/e); q0s = inv(np.log(q0/(1-q0)) - eps/(1-e))
    psi = q1s - q0s
    ic = (t/e - (1-t)/(1-e))*(y-np.where(t == 1, q1s, q0s)) + psi - psi.mean()
    se = float(np.std(ic, ddof=1)/np.sqrt(len(y)))
    return {"rd": round(float(psi.mean()), 4), "ci_95": [round(float(psi.mean()-1.96*se), 4), round(float(psi.mean()+1.96*se), 4)]}

def dr_and_forest(X, t, y):
    res = {}
    try:
        dr = LinearDRLearner(model_propensity=lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1, random_state=SEED),
                             model_regression=lgb.LGBMRegressor(n_estimators=200, num_leaves=15, verbose=-1, random_state=SEED),
                             cv=5, random_state=SEED).fit(y, t, X=X.to_numpy())
        inf = dr.ate_inference(X=X.to_numpy(), T0=0, T1=1)
        res["dr_learner"] = {"rd": round(float(inf.mean_point), 4), "ci_95": [round(float(inf.conf_int_mean()[0]), 4), round(float(inf.conf_int_mean()[1]), 4)]}
    except Exception as ex:
        res["dr_learner"] = {"error": str(ex)[:90]}
    try:
        cf = CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=200, num_leaves=15, verbose=-1, random_state=SEED),
                             model_t=lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1, random_state=SEED),
                             discrete_treatment=True, n_estimators=500, min_samples_leaf=20, cv=5,
                             honest=True, random_state=SEED).fit(y, t, X=X.to_numpy())
        ate = cf.ate(X.to_numpy()); lo, hi = cf.ate_interval(X.to_numpy())
        cate = cf.effect(X.to_numpy())
        half = np.random.default_rng(SEED).permutation(len(y)) < len(y)//2
        cf2 = CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=200, num_leaves=15, verbose=-1, random_state=SEED),
                              model_t=lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1, random_state=SEED),
                              discrete_treatment=True, n_estimators=300, min_samples_leaf=20, cv=5,
                              honest=True, random_state=SEED).fit(y[half], t[half], X=X[half].to_numpy())
        pred_other = cf2.effect(X[~half].to_numpy())
        top = pred_other >= np.median(pred_other)
        Xo, to, yo = X[~half], t[~half], y[~half]
        def naive_rd(mask):
            tt, yy = to[mask], yo[mask]
            return float(yy[tt == 1].mean() - yy[tt == 0].mean()) if tt.sum() > 10 and (1-tt).sum() > 10 else np.nan
        res["causal_forest"] = {"ate": round(float(ate), 4), "ci_95": [round(float(lo), 4), round(float(hi), 4)],
                                "cate_iqr": [round(float(np.percentile(cate, 25)), 4), round(float(np.percentile(cate, 75)), 4)],
                                "split_sample_validation": {"rd_top_half_by_predicted_benefit": round(naive_rd(top), 4),
                                                            "rd_bottom_half": round(naive_rd(~top), 4)}}
    except Exception as ex:
        res["causal_forest"] = {"error": str(ex)[:90]}
    return res

NEG = [("prior_30d_contacts", lambda p, d0, w: int(win(SC, p, d0, -30, -1) > 0)),
       ("prior_90d_attempts", lambda p, d0, w: int(win(SA, p, d0, -90, -1) > 0)),
       ("prior_365d_ed_claims", lambda p, d0, w: int(win(SED, p, d0, -365, -1) > 0)),
       ("prior_90d_adt", lambda p, d0, w: int(win(SED, p, d0, -90, -1) > 0)),
       ("dialysis_in_outcome_window", lambda p, d0, w: int(win(SDI, p, d0, w+1, w+90) > 0)),
       ("hedis_goal_in_outcome_window", lambda p, d0, w: int(win(SH, p, d0, w+1, w+90) > 0))]

def calibrate(primary_rd, primary_se, nulls):
    """Empirical calibration: widen the interval by the dispersion of the negative-control estimates."""
    arr = np.array([x for x in nulls if np.isfinite(x)])
    if len(arr) < 3: return None
    mu, sd = float(arr.mean()), float(arr.std(ddof=1))
    se_cal = float(np.sqrt(primary_se**2 + sd**2))
    return {"null_mean_rd": round(mu, 4), "null_sd_rd": round(sd, 4),
            "calibrated_rd": round(primary_rd - mu, 4),
            "calibrated_ci_95": [round(primary_rd - mu - 1.96*se_cal, 4), round(primary_rd - mu + 1.96*se_cal, 4)],
            "crosses_null": bool((primary_rd - mu - 1.96*se_cal) < 0 < (primary_rd - mu + 1.96*se_cal))}

ACTIONS = {
 "A1_attempt_during_7day_lapse": dict(w=7, restrict=lambda d: d[[win(SC, p, x, 1, 7) == 0 for p, x in zip(d.person_id, d.d0)]],
   expose=lambda p, x: int(win(SA, p, x, 1, 7) > 0)),
 "A2_inperson_chw_14d": dict(w=14, restrict=lambda d: d, expose=lambda p, x: int(win(S_chw_inperson, p, x, 1, 14) > 0)),
 "A3_therapy_30d": dict(w=30, restrict=lambda d: d[[win(S_ther, p, x, -20000, 0) == 0 for p, x in zip(d.person_id, d.d0)]],
   expose=lambda p, x: int(win(S_ther, p, x, 1, 30) > 0)),
 "A4_pharmacist_30d": dict(w=30, restrict=lambda d: d[[win(S_pharm, p, x, -20000, 0) == 0 for p, x in zip(d.person_id, d.d0)]],
   expose=lambda p, x: int(win(S_pharm, p, x, 1, 30) > 0)),
}
ACTIVE = {
 "AC1_inperson_vs_phone_chw_14d": dict(w=14,
   restrict=lambda d: d[[(win(S_chw_inperson, p, x, 1, 14) + win(S_chw_phone, p, x, 1, 14)) > 0 for p, x in zip(d.person_id, d.d0)]],
   expose=lambda p, x: int(win(S_chw_inperson, p, x, 1, 14) > 0)),
 "AC2_therapy_vs_pharmacy_30d": dict(w=30,
   restrict=lambda d: d[[(win(S_ther, p, x, 1, 30) + win(S_pharm, p, x, 1, 30)) > 0 and win(S_ther, p, x, -20000, 0) == 0 and win(S_pharm, p, x, -20000, 0) == 0 for p, x in zip(d.person_id, d.d0)]],
   expose=lambda p, x: int(win(S_ther, p, x, 1, 30) > 0)),
 "AC3_call_vs_text_attempt_during_lapse": dict(w=7,
   restrict=lambda d: d[[win(SC, p, x, 1, 7) == 0 and (win(S_call, p, x, 1, 7) + win(S_text, p, x, 1, 7)) > 0 for p, x in zip(d.person_id, d.d0)]],
   expose=lambda p, x: int(win(S_call, p, x, 1, 7) > 0)),
}

def run(name, spec, pop_df, pop):
    d = spec["restrict"](pop_df).copy()
    if len(d) < 200: return {"n": int(len(d)), "note": "too few eligible contacts"}
    w = spec["w"]
    d["T"] = [spec["expose"](p, x) for p, x in zip(d.person_id, d.d0)]
    d["Y"] = [sustained(p, x, w) for p, x in zip(d.person_id, d.d0)]
    t, y = d["T"].to_numpy(), d["Y"].to_numpy()
    if t.sum() < 40 or (1-t).sum() < 40: return {"n": int(len(d)), "treated": int(t.sum()), "note": "insufficient variation"}
    X = covmat(d)
    ow, e, wt = ow_estimate(X, t, y)
    res = {"n": int(len(d)), "treated": int(t.sum()), "outcome_rate": round(float(y.mean()), 4),
           "outcome_window": f"days {w+1} to {w+90}", "overlap_weighted": ow, "tmle": tmle(X, t, y, np.clip(e, 0.01, 0.99))}
    res |= dr_and_forest(X, t, y)
    nulls = []
    for nm, fn in NEG:
        yn = np.array([fn(p, x, w) for p, x in zip(d.person_id, d.d0)], float)
        if yn.std() == 0: continue
        try:
            m = sm.GLM(yn, sm.add_constant(t.astype(float)), family=sm.families.Binomial(), freq_weights=wt).fit(cov_type="HC0")
            inv = sm.families.links.Logit().inverse
            nulls.append(float(inv(m.params[0]+m.params[1]) - inv(m.params[0])))
        except Exception: pass
    se = (ow["ci_95"][1]-ow["ci_95"][0])/3.92
    res["negative_control_rds"] = [round(x, 4) for x in nulls]
    res["empirical_calibration"] = calibrate(ow["rd"], se, nulls)
    print(f"  {name} [{pop}]: n={res['n']} treated={res['treated']} OW RD={ow['rd']:+.3f} (maxSMD {ow['max_abs_smd_weighted']}) "
          f"TMLE={res['tmle']['rd']:+.3f} CF={res.get('causal_forest', {}).get('ate')} "
          f"calibrated={res['empirical_calibration']['calibrated_rd'] if res['empirical_calibration'] else None}", flush=True)
    return res

out = {"note": "overlap weights, TMLE, DR learner, causal forest, active comparators, empirical calibration"}
thr = F.p_ensemble.quantile(0.80)
for pop, pop_df in [("high_risk", F[F.p_ensemble >= thr]), ("all_risk", F)]:
    for name, spec in {**ACTIONS, **ACTIVE}.items():
        out[f"{name}__{pop}"] = run(name, spec, pop_df, pop)
json.dump(out, open(R/"actions_advanced_v4.json", "w"), indent=1, default=str)
print("written")
