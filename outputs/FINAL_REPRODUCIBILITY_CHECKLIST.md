# Systematic Reproducibility Checklist

This checklist documents the reproducibility and scientific audit status of the Backblaze HDD Failure Prediction research project repository. Every item is evaluated as **PASS**, **PARTIAL**, **FAIL**, or **NOT VERIFIED**, supported by verified file artifacts and code evidence.

---

## Reproducibility Evaluation Matrix

| Checklist Item | Status | Verified Evidence & Artifact Location | Concise Explanation |
| :--- | :---: | :--- | :--- |
| **1. Python Environment & Dependency Specification** | **PASS** | `README.md`, `requirements.txt`, `.venv` (Python 3.11.16) | Core dependencies pinned: PyTorch 2.6.0+cu124, Scikit-Learn 1.9.1, Pandas 3.0.6, NumPy 2.4.6, Joblib 1.6.0, PyArrow 26.0.0. Environment executes without missing dependency errors. |
| **2. Non-Destructive Raw Data Streaming** | **PASS** | `src/census.py`, `src/dataset.py`, `Backblaze_All_Data.zip` | All 546 daily CSV files across 6 quarters were streamed directly from in-memory zip buffers. The raw zip archive was never extracted, duplicated, or modified on disk. |
| **3. Device Type Isolation (SSD / Non-Drive Removal)** | **PASS** | `src/census.py` (lines 40–55, 107–110), `outputs/logs/pass1_census.log` | Solid-state drives and BOSS controllers were filtered via regex (`SSD_PATTERN` and `NON_DRIVE_PATTERN`), ensuring pure mechanical HDD telemetry (178.5M HDD rows from 180M raw). |
| **4. Leakage-Free Preprocessing & Standardization** | **PASS** | `src/dataset.py` (lines 175–190), `outputs/data/simple_imputer.joblib`, `outputs/data/standard_scaler.joblib` | `SimpleImputer(median)` and `StandardScaler` were fitted exclusively on 6,399,714 observations from the 2025 training period. Scalers were persisted and applied downstream to Val and Test. |
| **5. Strict Temporal Windowing & Calendar Continuity** | **PASS** | `src/dataset.py` (lines 211–227), `outputs/logs/scientific_verification_audit.json` | 30-day sequences require exactly 30 consecutive calendar days; missing days invalidate the sequence. Target lookahead is strictly 7 calendar days with a mandatory 1-day isolation buffer. |
| **6. Deterministic Phase-Anchored Weekly Sampling** | **PASS** | `src/dataset.py` (lines 220–222) | Window sampling phase is fixed deterministically per drive using `MD5(serial_number) % 7`, preventing label-dependent window selection while reducing redundancy. |
| **7. Chronological Partitioning & Boundary Isolation** | **PASS** | `src/dataset.py` (lines 273–289), `tests/scientific_verification_audit.py` (Check 1b) | Train (target end $\le$ 2025-12-31), Val (target 2026-01-01 to 2026-03-31), Test (target 2026-04-01 to 2026-06-30). Target ranges are strictly disjoint with 1-day boundary buffers. |
| **8. Boundary Filter & Window Count Reconciliation** | **PASS** | `outputs/logs/pass2_dataset.log`, `tests/scientific_verification_audit.py` (Check 3) | 1,164,052 generated $\to$ 1,137,870 retained $\to$ 26,182 excluded. All 26,182 excluded windows independently enumerated from split metadata (13,423 Train-Val straddles, 12,627 Val-Test straddles, 132 edge gaps) and confirmed as boundary target isolation. |
| **9. Model Weight Serialization & Checkpoint Persistence** | **PASS** | `outputs/models/lstm_best.pt`, `outputs/models/transformer_best.pt`, `outputs/models/isolation_forest.joblib` | Best weights saved during training via validation checkpointing and verified loadable for evaluation without retraining. |
| **10. Validation-Only Decision Threshold Tuning** | **PASS** | `src/evaluate.py` (lines 65–85), `outputs/metrics/model_comparison.csv` | Decision thresholds ($\tau^*_{\text{TF}}=0.4365, \tau^*_{\text{LSTM}}=0.5669, \tau^*_{\text{IF}}=0.1981$) were tuned exclusively on Q1 2026 validation via quantile search and frozen prior to test evaluation. Zero test label tuning. |
| **11. Test Label Ground-Truth Integrity** | **PASS** | `tests/scientific_verification_audit.py` (Check 1e), `outputs/data/failure_catalog.csv` | Train (2,799/2,799) and Test (809/809) positives 100% verified against failure catalog. 1 validation window (Drive 2BJD03RD) verified against raw telemetry as an authentic re-failure following drive re-deployment. |
| **12. Uncertainty Treatment of Disappeared Drives** | **PASS** | `tests/scientific_verification_audit.py` (Check 1g), `outputs/FINAL_RESEARCH_REPORT.md` (Section 15) | 323 cohort drives stopped reporting without failure flag. 303 (93.81%) showed normal SMART (routine decommissions); 20 (6.19%) showed elevated errors (possible proactive swaps). Exactly 0 suspect drives appear in test split; test set is 100% insulated. |
| **13. Statistical Uncertainty via Cluster Bootstrap** | **PASS** | `tests/scientific_verification_audit.py` (Check 2), `outputs/logs/scientific_verification_audit.json` | 5,000-repetition drive-cluster bootstrap resamples drive IDs with replacement, preserving intra-drive temporal autocorrelation. Vectorized sufficiency formulation matches iterative resampling with $0.00$ error. |
| **14. Array, Label, and Prediction Alignment** | **PASS** | `outputs/data/X_test.npy`, `outputs/data/y_test.npy`, `outputs/data/test_predictions.parquet` | Row-by-row identity verified across all 177,023 test sequences. `y_test == test_predictions['y_true']` confirmed `True`. Prediction bounds valid; zero NaNs or Infs. |
| **15. Local Benchmark Latency Disclosure** | **PASS** | `outputs/FINAL_RESEARCH_REPORT.md` (Section 14), `outputs/logs/scientific_verification_audit.json` | Disclosed as local hardware benchmark measurements (RTX 3050 Laptop GPU, Ryzen 5 5600H) excluding disk I/O and network overhead; explicitly not claimed as cloud service guarantees. |
| **16. Claim Modesty & Scientific Qualification** | **PASS** | `outputs/FINAL_RESEARCH_REPORT.md` (Sections 13, 15, 17) | Paper explicitly notes low recall (11%–21%), substantial false alarms (507–757), CI overlap between LSTM and Transformer, and necessity of layered storage defenses rather than standalone deployment. |

---

## Overall Reproducibility Verdict: **PASS**

All 16 reproducibility and scientific criteria are fully met with empirical proof, exact array alignment, and verified file artifacts in the repository. No checks failed or remain unverified.
