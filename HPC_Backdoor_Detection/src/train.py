"""Fit the detector on the single clean trace and calibrate per-length thresholds for all candidates.
Usage: python src/train.py --data path/to/trace.csv [--out models/detector.pkl] [--lengths 200 400 ...] [--n-cal 3000]
Lengths other than the standard six are only calibrated when requested explicitly here."""
import argparse, hashlib, os, pickle, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy, sklearn
import config as C, features as F, model as M, calibration as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the organisers' clean trace.csv (not bundled)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models" / "detector.pkl"))
    ap.add_argument("--lengths", type=int, nargs="+", default=list(C.LENGTHS))
    ap.add_argument("--n-cal", type=int, default=C.N_CAL)
    a = ap.parse_args(); t0 = time.time()
    df = pd.read_csv(a.data); X = F.validate_trace(df)
    bad = [L for L in a.lengths if L < C.MIN_LENGTH]
    if bad: sys.exit("lengths below minimum %d: %s" % (C.MIN_LENGTH, bad))
    model = M.fit_model(X)
    cal = {}
    for L in sorted(set(a.lengths)):
        cal[L] = K.calibrate_length(model, X, L, n_cal=a.n_cal)
        print("L=%4d windows=%3d  " % (L, cal[L]["n_windows"]) +
              "  ".join("%s q%.3f=%.4f" % (m, q, t) for m in ("cen", "unc") for q, t in cal[L]["thr"][m].items()), flush=True)
    det = dict(format_version=1, model=model, calibration=cal,
               meta=dict(trace_sha256_16=hashlib.sha256(X.tobytes()).hexdigest()[:16], n_rows=len(X), n_cal=a.n_cal,
                         lw_shrinkage={m: model["lw"][m]["shrinkage"] for m in model["lw"]},
                         versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                       sklearn=sklearn.__version__, pandas=pd.__version__),
                         runtime_s=round(time.time() - t0, 1)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f: pickle.dump(det, f)
    print("saved", a.out, "| trace sha", det["meta"]["trace_sha256_16"], "| LW shrinkage", det["meta"]["lw_shrinkage"], "|", det["meta"]["runtime_s"], "s")


if __name__ == "__main__":
    main()
