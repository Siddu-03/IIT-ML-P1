"""Inference. CLI:  python src/predict.py --input TRACE.csv [--candidate A|B|U] [--q 0.99] [--model PATH]   (TRACE.csv may also be given positionally)
Default model: <submission root>/models/detector.pkl, resolved relative to this file (works from any working directory).
API: predict(df, candidate='A', q=0.99, detector=load_detector()) -> dict. Exit code 2 on malformed input (never a silent 'CLEAN')."""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import config as C, features as F, model as M, calibration as K


DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "detector.pkl"


def load_detector(path=None):
    with open(DEFAULT_MODEL if path is None else path, "rb") as f: det = pickle.load(f)
    if det.get("format_version") != 1: raise ValueError("unsupported detector format")
    return det


NON_FINITE = "non_finite_fail_closed"


def _checked_scores(model, X, mode, need_level):
    """Same computation as model.trace_scores (window scores -> top-6 mean, level statistic) plus a finiteness check on the
    intermediate (per-window) and final outputs. Returns (score, level, n_windows, problem); problem is None for every scorable
    trace, in which case score/level are numerically identical to model.trace_scores. Audit F-01: `NaN > threshold` is False, so a
    NaN score used to be reported CLEAN. Windows are checked individually because np.sort places NaN last, so a NaN window could
    otherwise be silently dropped from the top-6 aggregate."""
    with np.errstate(all="ignore"):               # overflow/invalid are reported through `problem`, not as numpy warnings
        S = M.window_scores(model, X[None], mode)
        bad = ~np.isfinite(S)
        score = M.aggregate(S); lvl = M.level_statistic(model, X[None])
    problem = None
    if bad.any(): problem = "non-finite (NaN/inf) score for %d of %d windows" % (int(bad.sum()), S.size)
    elif not np.isfinite(score).all(): problem = "non-finite aggregate score"
    elif need_level and not np.isfinite(lvl).all(): problem = "non-finite level statistic"
    return score, lvl, S.shape[1], problem


def predict(df, candidate="A", q=C.PRIMARY_Q, detector=None):
    """Malformed input raises (CLI exit 2). A trace that passes validation but cannot be scored numerically (NaN/+-inf window score,
    aggregate score, or - for candidate B - level statistic) is FAIL-CLOSED: final_decision = "ANOMALY", the invalid number is never
    compared, never replaced by zero and never reported as a score (mahalanobis_score = None), and score_status/warning say why."""
    if candidate not in C.CANDIDATES: raise ValueError("candidate must be one of %s" % list(C.CANDIDATES))
    det = detector if detector is not None else load_detector()
    X = F.validate_trace(df); L = len(X); cfg = C.CANDIDATES[candidate]
    thr = K.lookup(det, L, cfg["mode"], q)                     # raises for uncalibrated lengths; no fallback
    score, lvl, nw, problem = _checked_scores(det["model"], X, cfg["mode"], cfg["level"]); score, lvl = float(score[0]), float(lvl[0])
    if problem is not None:
        lvl_fin = cfg["level"] and bool(np.isfinite(lvl))
        return dict(prediction=1, candidate=candidate, trace_length=L, window_count=nw, quantile=q,
                    mahalanobis_score=None, mahalanobis_threshold=thr, mahalanobis_flag=NON_FINITE,
                    level_test_used=cfg["level"], level_statistic=(lvl if lvl_fin else None), level_threshold=(C.LEVEL_TOL if cfg["level"] else None),
                    level_flag=(bool(lvl > C.LEVEL_TOL) if lvl_fin else (NON_FINITE if cfg["level"] else "not_used")),
                    final_decision="ANOMALY", score_status="non_finite",
                    warning="%s: the trace cannot be scored numerically; reported as ANOMALY (fail-closed)" % problem)
    mflag = score > thr                                         # strict
    if cfg["level"]:
        lflag = lvl > C.LEVEL_TOL; lvl_out, lthr_out, lflag_out = lvl, C.LEVEL_TOL, bool(lflag)
    else:
        lflag, lvl_out, lthr_out, lflag_out = False, None, None, "not_used"
    anomalous = bool(mflag or lflag)
    return dict(prediction=int(anomalous), candidate=candidate, trace_length=L, window_count=nw, quantile=q,
                mahalanobis_score=score, mahalanobis_threshold=thr, mahalanobis_flag=bool(mflag),
                level_test_used=cfg["level"], level_statistic=lvl_out, level_threshold=lthr_out, level_flag=lflag_out,
                final_decision="ANOMALY" if anomalous else "CLEAN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?"); ap.add_argument("--input", dest="input_csv"); ap.add_argument("--candidate", default="A", choices=list(C.CANDIDATES))
    ap.add_argument("--q", type=float, default=C.PRIMARY_Q); ap.add_argument("--model", default=None)
    a = ap.parse_args()
    path = a.input_csv or a.trace
    if path is None or (a.input_csv and a.trace): ap.error("give exactly one input CSV: --input TRACE.csv (or positional TRACE.csv)")
    try:
        res = predict(pd.read_csv(path), a.candidate, a.q, load_detector(a.model))
        if res.get("score_status") == "non_finite": print("WARNING: %s" % res["warning"], file=sys.stderr)
        print(json.dumps(res, indent=2))
    except (F.TraceError, KeyError, ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print("ERROR: %s" % (e.args[0] if isinstance(e, KeyError) else e), file=sys.stderr); sys.exit(2)


if __name__ == "__main__":
    main()
