"""Subprocess runner for the FT-Transformer: torch on Metal deadlocks when it shares a process with
the OpenMP runtimes loaded by LightGBM and CatBoost, so the transformer is fit in a clean process."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nets import FTClassifier
z = np.load(sys.argv[1])
m = FTClassifier(seed=int(z["seed"]), epochs=int(z["epochs"]), batch=int(z["batch"])).fit(z["Xtr"], z["ytr"])
np.save(sys.argv[2], m.predict_proba(z["Xte"])[:, 1])
