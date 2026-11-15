"""Scoring logic. Fit = researcher's fit_ctx(Xtr, seed=0, sim_seed=3000, modes=('unc','cen')) 'TSIM' Ledoit-Wolf models:
the LW covariance is fitted on ALL windows of N_SIM_TRAIN iid-row-resampled copies of the training trace.
Inference needs only numpy: the stored LW location/precision are used directly (same arithmetic as research score_win)."""
import numpy as np
from sklearn.covariance import LedoitWolf
import config as C
import features as F


def sim_W(pool, n, L, seed):
    """World W: iid row resampling (copied from p1lib_v3.sim_W; RNG stream must stay identical)."""
    r = np.random.default_rng(seed)
    return pool[r.integers(0, len(pool), (n, L))]


def sim_B(pool, n, L, seed, b=C.BLOCK_B):
    """World B20: circular blocks of 20 rows (copied from p1lib_v3.sim_B)."""
    r = np.random.default_rng(seed); nb = -(-L // b)
    st = r.integers(0, len(pool), (n, nb))
    idx = ((st[:, :, None] + np.arange(b)) % len(pool)).reshape(n, nb * b)[:, :L]
    return pool[idx]


def fit_model(X, sim_seed=C.SIM_SEED_TRAIN):
    """X: validated (L,3) clean training trace. Returns a plain-dict model (no sklearn objects)."""
    sd = X.std(0, ddof=1); mu = X.mean(0); ref_median = np.median(X, 0)
    sims = sim_W(X, C.N_SIM_TRAIN, len(X), sim_seed)
    model = dict(mu=mu, sd=sd, ref_median=ref_median, train_length=len(X), lw={})
    for mode in ("unc", "cen"):
        Fm = F.window_features(F.to_z(sims, mode, sd, mu)).reshape(-1, len(C.FEATURE_NAMES))
        lw = LedoitWolf().fit(Fm)
        model["lw"][mode] = dict(location=lw.location_.copy(), precision=lw.precision_.copy(), shrinkage=float(lw.shrinkage_))
    return model


def window_scores(model, tr, mode, chunk=400):
    """tr: (N,L,3) raw -> (N,nw) float32 squared Mahalanobis distances per window."""
    lw = model["lw"][mode]; out = []
    for i in range(0, len(tr), chunk):
        A = F.window_features(F.to_z(tr[i:i + chunk], mode, model["sd"], model["mu"]))
        n, nw, p = A.shape; A = A.reshape(-1, p); D = A - lw["location"]
        out.append(np.einsum("ij,jk,ik->i", D, lw["precision"], D).reshape(n, nw).astype(np.float32))
    return np.concatenate(out)


def aggregate(S, k=C.TOP_K):
    """Mean of the k highest window scores per trace (research agg 'top6')."""
    return (-np.sort(-S, axis=1))[:, :min(k, S.shape[1])].mean(1)


def level_statistic(model, tr):
    """max over counters of |trace_median / training_median - 1|. tr: (N,L,3)."""
    return np.abs(np.median(tr, axis=1) / model["ref_median"] - 1).max(1)


def trace_scores(model, tr, mode):
    S = window_scores(model, tr, mode)
    return aggregate(S), level_statistic(model, tr), S.shape[1]
