"""Fixed parameters for the P1 HPC detector. Values mirror the researcher's v3 harness
(p1lib_v3.py with FK='rob') and final.py. Nothing here is tuned against evaluation data.
All numbers are SYNTHETIC-evaluation settings derived from the single supplied clean trace."""

COLUMNS = ("cache-references", "cycles", "LLC-loads")   # canonical order; features are bound to these NAMES

WINDOW, STRIDE = 40, 10          # rows
TOP_K = 6                        # mean of the 6 highest window scores
MAD_SCALE = 1.4826
MAD_EPS = 1e-9                   # added after MAD scaling, before log (research: mad*1.4826 + 1e-9)
CORR_CLIP = 0.999                # Spearman clip before Fisher-z
FEATURE_NAMES = (
    [f"median_z_{c}" for c in COLUMNS] +
    [f"log_mad_{c}" for c in COLUMNS] +
    [f"median_abs_diff_{c}" for c in COLUMNS] +
    ["fisher_z_spearman_cache_cycles", "fisher_z_spearman_cache_llc", "fisher_z_spearman_cycles_llc"] +
    [f"p90_abs_dev_{c}" for c in COLUMNS]
)

# --- model fit (research fit_ctx defaults, final.py call: seed=0, sim_seed=3000, n_sim_train=300) ---
N_SIM_TRAIN = 300
SIM_SEED_TRAIN = 3000

# --- calibration (final.py: NCAL=3000 per world, pooled W+B20, np.quantile(method='higher'), strict '>') ---
N_CAL = 3000
BLOCK_B = 20                     # World B20 block length
QUANTILES = (0.99, 0.995, 0.999)
PRIMARY_Q = 0.99
LENGTHS = (200, 400, 800, 1600, 2400, 3200)
MIN_LENGTH = min(LENGTHS)        # shorter traces are rejected
# (seedW, seedB) per length. L=800 equals final.py (calW=91000, calB=91001); other lengths are an
# implementation choice (research 03_length.py used a different seeding scheme) -> documented in README.
CAL_SEEDS = {800: (91000, 91001), 200: (91100, 91101), 400: (91200, 91201),
             1600: (91300, 91301), 2400: (91400, 91401), 3200: (91500, 91501)}

def cal_seeds(L):
    """Seeds for length L: table above for the six standard lengths; deterministic, collision-free otherwise."""
    return CAL_SEEDS.get(L, (1_000_000 + 2 * L, 1_000_001 + 2 * L))

# --- candidates ---
LEVEL_TOL = 0.01                 # 1% level tolerance (experimental; research: unverifiable assumptions)
CANDIDATES = {
    "A": dict(mode="cen", level=False, desc="Centered Mahalanobis (LW), top-6, per-length threshold"),
    "B": dict(mode="cen", level=True,  desc="Candidate A + separate 1% absolute-level test"),
    "U": dict(mode="unc", level=False, desc="Uncentered Mahalanobis comparator (research only)"),
}
