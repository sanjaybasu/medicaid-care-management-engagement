"""Confusion matrices for the ensemble and the acute care risk score at the Youden-optimal
operating point, with the risk score held at a comparable flag volume so the comparison is fair.

The Youden threshold is recomputed here rather than read back from thresholds_v4.json: the
isotonic-calibrated scores contain large blocks of tied values, so a threshold rounded to four
decimal places can fall on the wrong side of a tie block and shift the flag count materially.
Writes results/youden_confusion_v4.json.
"""
import numpy as np, json, pathlib
P = pathlib.Path(__file__).resolve().parent.parent
R = P/"results"
sc = np.load(R/"val_scores_B_full.npy")
p_ens, p_risk, y = sc[:, 0], sc[:, 1], sc[:, 2]

o = np.argsort(-p_ens); yo = y[o]
tp_c, fp_c = np.cumsum(yo), np.cumsum(1-yo)
tpr, fpr = tp_c/max(y.sum(), 1), fp_c/max((1-y).sum(), 1)
thr = float(p_ens[o][int(np.argmax(tpr-fpr))])          # unrounded

def cm(flag):
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    return {"flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": round(tp/max(tp+fn, 1), 3), "specificity": round(tn/max(tn+fp, 1), 3),
            "ppv": round(tp/max(tp+fp, 1), 3), "npv": round(tn/max(tn+fn, 1), 3)}

ens = cm(p_ens >= thr)
k = ens["flagged"]
cut = np.sort(p_risk)[::-1][k-1]
risk = cm(p_risk >= cut)
out = {"threshold": thr, "flag_rate": round(ens["flagged"]/len(y), 4), "events": int(y.sum()), "n": int(len(y)),
       "ensemble_stacked_calibrated": ens, "signal_risk_score": risk,
       "events_captured_ratio": round(ens["tp"]/max(risk["tp"], 1), 2)}
json.dump(out, open(R/"youden_confusion_v4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
