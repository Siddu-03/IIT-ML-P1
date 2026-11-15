"""Per-length threshold calibration (researcher's final.py procedure generalised to length L):
N_CAL World-W + N_CAL World-B20 simulated clean traces of length L are drawn from the training trace, scored
with the deployed model, pooled, and thr = np.quantile(pooled, q, method='higher'). Decision is strict '>'.
A threshold is NEVER reused across lengths and never interpolated."""
import numpy as np
import config as C
import model as M


def thr_from(cal, q):
    return float(np.quantile(cal, q, method="higher"))


def calibrate_length(model, X, L, n_cal=C.N_CAL, seeds=None, quantiles=C.QUANTILES):
    """Returns {'n_windows', 'seeds', 'n_cal', 'thr': {mode: {q: thr}}}."""
    sW, sB = seeds if seeds is not None else C.cal_seeds(L)
    trW, trB = M.sim_W(X, n_cal, L, sW), M.sim_B(X, n_cal, L, sB)
    out = dict(seeds=(sW, sB), n_cal=n_cal, thr={})
    for mode in ("cen", "unc"):
        sc = []
        for tr in (trW, trB):
            s, _, nw = M.trace_scores(model, tr, mode); sc.append(s); out["n_windows"] = nw
        pooled = np.concatenate(sc)
        out["thr"][mode] = {q: thr_from(pooled, q) for q in quantiles}
    return out


def lookup(detector, L, mode, q):
    """Explicit lookup; raises (no fallback, no interpolation) if the length or quantile was not calibrated."""
    cal = detector["calibration"]
    if L not in cal:
        raise KeyError("no calibrated threshold for trace length %d; calibrated lengths: %s. "
                       "Add it explicitly with: python train.py --lengths %s" %
                       (L, sorted(cal), " ".join(map(str, sorted(set(cal) | {L})))))
    if q not in cal[L]["thr"][mode]:
        raise KeyError("quantile %s not calibrated; available: %s" % (q, sorted(cal[L]["thr"][mode])))
    return cal[L]["thr"][mode][q]
