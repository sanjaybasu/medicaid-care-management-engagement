"""v4 step 17: operating points for the deployed score.

Reports the sensitivity-specificity trade-off across flagging fractions, at the Youden-optimal
threshold, at the F1-optimal threshold, and at thresholds that hit target sensitivities, with the
capacity implication (contacts flagged per week) at each."""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = pathlib.Path(__file__).resolve().parent.parent/"results"
D = pathlib.Path(__file__).resolve().parent.parent/"data_cache"
out = {}
for TASK in ["B", "A"]:
    sc = np.load(R/f"val_scores_{TASK}_full.npy")
    p, y = sc[:, 0], sc[:, 2]
    ids = pd.read_parquet(D/f"v4_valscores_{TASK}_full.parquet").person_id.to_numpy()
    days = 341 if TASK == "B" else 341      # validation window length in days (1 July 2025 to 7 June 2026)
    def pack(thr, label):
        f = p >= thr
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
        sens, spec = tp/max(tp+fn, 1), tn/max(tn+fp, 1)
        return {"operating_point": label, "threshold": round(float(thr), 4), "flag_rate": round(float(f.mean()), 4),
                "flagged": int(f.sum()), "flagged_per_week": round(float(f.sum()/(days/7)), 1),
                "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                "ppv": round(tp/max(tp+fp, 1), 3), "npv": round(tn/max(tn+fn, 1), 3),
                "youden_j": round(sens+spec-1, 3), "f1": round(2*tp/max(2*tp+fp+fn, 1), 3),
                "number_needed_to_flag": round((tp+fp)/max(tp, 1), 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    rows = []
    for frac in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        k = max(int(round(frac*len(p))), 1); rows.append(pack(np.sort(p)[::-1][k-1], f"top {int(frac*100)}% of contacts"))
    o = np.argsort(-p); yo = y[o]; tp = np.cumsum(yo); fp = np.cumsum(1-yo)
    tpr, fpr = tp/max(y.sum(), 1), fp/max((1-y).sum(), 1)
    rows.append(pack(p[o][int(np.argmax(tpr-fpr))], "Youden-optimal"))
    f1 = 2*tp/np.maximum(2*tp+fp+(y.sum()-tp), 1)
    rows.append(pack(p[o][int(np.argmax(f1))], "F1-optimal"))
    for target in [0.70, 0.80, 0.90]:
        j = int(np.searchsorted(tpr, target))
        if j < len(p): rows.append(pack(p[o][min(j, len(p)-1)], f"sensitivity {target:.2f} target"))
    out[TASK] = {"auroc": round(float(roc_auc_score(y, p)), 4), "n": int(len(p)), "events": int(y.sum()),
                 "event_rate": round(float(y.mean()), 4), "validation_days": days,
                 "patients": int(len(np.unique(ids))), "operating_points": rows}
    print(f"\n=== Task {TASK} (AUROC {out[TASK]['auroc']}, {out[TASK]['events']} events of {out[TASK]['n']})")
    print(pd.DataFrame(rows)[["operating_point", "flag_rate", "sensitivity", "specificity", "ppv", "youden_j",
                              "number_needed_to_flag", "flagged_per_week"]].to_string(index=False))
json.dump(out, open(R/"thresholds_v4.json", "w"), indent=1)
