"""Input validation + robust 15-feature extraction. Formulas are copied from the researcher's
p1lib_v3.window_feats_rob / to_z (FK='rob', no clip, no spike filter, no med3); do not reinterpret."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
import config as C


class TraceError(ValueError):
    """Raised for malformed input. Malformed traces are never classified."""


def validate_trace(obj):
    """Return a float64 (L,3) array in canonical column order, or raise TraceError.
    Accepts a DataFrame (columns selected BY NAME, so column order cannot change meaning)."""
    if not isinstance(obj, pd.DataFrame):
        raise TraceError("input must be a pandas DataFrame with named columns %s" % (list(C.COLUMNS),))
    missing = [c for c in C.COLUMNS if c not in obj.columns]
    if missing:
        raise TraceError("missing required column(s): %s (found: %s)" % (missing, list(obj.columns)))
    if obj.columns.duplicated().any():
        raise TraceError("duplicate column names are not allowed")
    sub = obj[list(C.COLUMNS)]
    for c in C.COLUMNS:
        dt = sub[c].dtype
        if dt == bool or not (pd.api.types.is_integer_dtype(dt) or pd.api.types.is_float_dtype(dt)):
            raise TraceError("column '%s' must be numeric (int/float), got dtype %s" % (c, dt))
    X = sub.to_numpy(dtype=np.float64)
    if np.isnan(X).any():
        raise TraceError("NaN values found (%d)" % int(np.isnan(X).sum()))
    if np.isinf(X).any():
        raise TraceError("infinite values found (%d)" % int(np.isinf(X).sum()))
    if (X <= 0).any():
        raise TraceError("zero or negative counter values found (%d); counters must be > 0" % int((X <= 0).sum()))
    if len(X) < C.MIN_LENGTH:
        raise TraceError("trace has %d rows; minimum supported length is %d" % (len(X), C.MIN_LENGTH))
    return X


def window_starts(L):
    st = list(range(0, L - C.WINDOW + 1, C.STRIDE))
    if st[-1] != L - C.WINDOW:               # research: final window aligned to last row if a tail remains
        st.append(L - C.WINDOW)
    return st


def to_z(tr, mode, sd, mu):
    """tr: (N,L,3) raw. 'cen': per-trace median centring / training sd. 'unc': (x-mu_train)/sd_train."""
    if mode == "cen":
        return (tr - np.median(tr, axis=1, keepdims=True)) / sd
    if mode == "unc":
        return (tr - mu) / sd
    raise ValueError(mode)


def window_features(Z):
    """Z: (N,L,3) z-units -> (N,nw,15), ordering = config.FEATURE_NAMES."""
    st = window_starts(Z.shape[1])
    w = swv(Z, C.WINDOW, axis=1)[:, st]                              # (N,nw,3,W)
    m = np.median(w, -1); a = np.abs(w - m[..., None])
    mad = np.median(a, -1) * C.MAD_SCALE + C.MAD_EPS
    mdiff = np.median(np.abs(np.diff(w, axis=-1)), -1)
    r = np.argsort(np.argsort(w, -1), -1).astype(np.float64); r -= r.mean(-1, keepdims=True)
    nr = np.sqrt((r ** 2).sum(-1))
    def c(i, j):
        return np.clip((r[..., i, :] * r[..., j, :]).sum(-1) / (nr[..., i] * nr[..., j]), -C.CORR_CLIP, C.CORR_CLIP)
    fz = np.stack([np.arctanh(c(0, 1)), np.arctanh(c(0, 2)), np.arctanh(c(1, 2))], -1)
    q90 = np.sort(a, -1)[..., int(0.9 * (w.shape[-1] - 1))]
    return np.concatenate([m, np.log(mad), mdiff, fz, q90], -1)
