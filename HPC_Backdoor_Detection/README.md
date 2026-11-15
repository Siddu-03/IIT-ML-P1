# HPC Backdoor Detection — Problem 1 (anomaly detector for HPC traces)

## 1. Overview
This project detects anomalous (potentially backdoored) model inferences from Hardware Performance Counter (HPC)
traces with columns `cache-references`, `cycles` and `LLC-loads`. Only **one clean trace** (800 rows x 3 counters) was
available for training and no backdoor traces exist, so the detector is a one-class / novelty detector: it models normal
behaviour and flags traces that deviate from it.

**Important:** no performance on real backdoor traces has been measured. All calibration is simulation-based and derived
from that single clean trace (see Section 9).

## 2. Detector architecture
```
trace (L x 3) -> input validation -> centring + scaling -> sliding windows (40 rows, stride 10)
  -> 15 robust features per window -> Mahalanobis distance (Ledoit-Wolf covariance) per window
  -> mean of the 6 highest window scores -> compare with a per-length threshold -> CLEAN / ANOMALY
```
* **Features (15 per window):** median, log-MAD, median absolute first difference and 90th-percentile absolute deviation
  for each of the 3 counters (12 features), plus the Fisher-z of the Spearman correlation for each counter pair (3 features).
* **Normalisation:** each trace is centred by its own median and divided by the training trace's per-counter standard deviation.
* **Model:** Ledoit-Wolf mean/precision fitted on all windows of 300 row-resampled copies of the training trace
  (stored as plain arrays in `models/detector.pkl`).
* **Trace score:** mean of the 6 largest per-window squared Mahalanobis distances.
* **Threshold:** one per supported trace length, the 99th percentile of scores of 3000 + 3000 simulated clean traces
  (iid row resampling + 20-row block resampling of the training trace). Decision is strict `score > threshold`.
* **Candidates (selectable with `--candidate`):** `A` (default) = the detector above. `B` = A plus a separate level test,
  `max_c |median(trace_c)/median(train_c) - 1| > 0.01` (flag if either fires). `U` = uncentred comparator kept from the
  research stage. Candidate A is the default; B and U are provided but their use is a judgement call (see Section 9).

## 3. Main components
| File | Purpose |
|---|---|
| `src/config.py` | All fixed parameters (window, stride, top-k, lengths, seeds, quantiles, level tolerance) |
| `src/features.py` | Input validation and the 15-feature window extraction |
| `src/model.py` | Ledoit-Wolf fit, window scoring, top-6 aggregation, level statistic |
| `src/calibration.py` | Per-length threshold calibration and threshold lookup |
| `src/predict.py` | Command-line and Python prediction interface |
| `src/train.py` | Re-creates `models/detector.pkl` from the clean trace (optional; needs the organisers' `trace.csv`, not bundled) |
| `models/detector.pkl` | Saved detector: model parameters + calibrated thresholds |

## 4. Requirements
Python 3.12 or newer (developed on 3.13; verified in packaging on 3.12.3). Pinned packages in `requirements.txt`:
`numpy==2.4.4`, `scipy==1.17.1`, `scikit-learn==1.8.0`, `pandas==3.0.2`.

## 5. Installation
```bash
cd HPC_Backdoor_Detection_Submission
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip ...
```

## 6. Command-line usage
```bash
.venv/bin/python src/predict.py --input path/to/trace.csv            # candidate A, q = 0.99
.venv/bin/python src/predict.py path/to/trace.csv --candidate B      # trace may also be positional
```
Options: `--candidate {A,B,U}` (default A), `--q {0.99,0.995,0.999}` (default 0.99), `--model PATH` (default: `models/detector.pkl`
next to `src/`, found automatically from any working directory).

Optional retraining (about 3-4 minutes, deterministic):
```bash
.venv/bin/python src/train.py --data path/to/clean/trace.csv --out models/detector.pkl
```

## 7. Expected input
A CSV with a header row containing the columns `cache-references`, `cycles` and `LLC-loads` (selected by name; order does not
matter; extra columns are ignored). Values must be numeric, finite and > 0. **The number of rows must be one of the calibrated
lengths: 200, 400, 800, 1600, 2400 or 3200.** Each row is one measurement of the same inference trace.

## 8. Expected output
A JSON object printed to stdout, e.g. (the clean training trace itself, candidate A):
```json
{"prediction": 0, "candidate": "A", "trace_length": 800, "window_count": 77, "quantile": 0.99,
 "mahalanobis_score": 27.05, "mahalanobis_threshold": 42.64, "mahalanobis_flag": false,
 "level_test_used": false, "level_statistic": null, "level_threshold": null, "level_flag": "not_used",
 "final_decision": "CLEAN"}
```
`prediction` is 0 (CLEAN) or 1 (BACKDOOR/ANOMALY); `final_decision` is the same in words; `mahalanobis_score` is the anomaly score.
* Malformed input (missing column, NaN/inf, value <= 0, non-numeric data, fewer than 200 rows, an uncalibrated length,
  unreadable file) prints `ERROR: ...` to stderr and exits with code 2; it never returns CLEAN.
* A valid trace that cannot be scored numerically (float overflow) is reported fail-closed as `ANOMALY`, with
  `mahalanobis_score: null` and a `warning`.

Python API: `import predict; predict.predict(pandas_dataframe, candidate="A", q=0.99)` (run from `src/` or add it to `sys.path`).

## 9. Important limitations
* **No real backdoor data was available.** Detection performance on backdoor traces is unknown. Thresholds are calibrated on
  simulated clean traces built from the single supplied clean trace, so the nominal 1% false-positive level holds only if
  unseen clean traces resemble that trace (same machine, noise level, level, temporal structure).
* Unsupported trace lengths are rejected (error, no prediction); thresholds are not interpolated.
* Internal synthetic stress tests (not competition results) showed that candidate A is structurally blind to whole-trace additive
  level shifts, isolated 1-2-row events, sparse spikes (spacing of 20 or more rows), small changes in counter correlations and
  very small sustained shifts; and that it can flag clean-class variation such as a >= 20% change in noise level or a >= 25% change
  in absolute counter scale. Candidate B adds sensitivity to level shifts above 1% but flags benign level offsets of that size.
* Temporal order is only weakly used (most features are order-free within a 40-row window).
* `detector.pkl` is a Python pickle: load only trusted files. Results were checked with the pinned library versions above.
