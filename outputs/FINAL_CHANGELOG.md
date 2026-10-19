# Comprehensive Scientific Change Log & Discrepancy Reconciliation

**Project:** Backblaze HDD Failure Prediction Research  
**Scope:** Final one-pass scientific verification audit, artifact synchronization, and academic paper finalization.  

---

## 1. Discrepancy Investigations & Evidence-Based Resolutions

### Discrepancy 1: Single Unverified Positive Validation Window
* **Prior State:** Preliminary audit noted that validation had 694 positive windows, but only 693 matched records in `failure_catalog.csv`, leaving 1 window marked as unverified.
* **Evidence Tracing:** Traced to row index `34201` of `meta_val.parquet`, belonging to drive `2BJD03RD` (WDC WUH721816ALE6L0) with target window `2026-02-07` to `2026-02-08` ($y=1$).
* **Root-Cause Resolution:** 
  1. Drive `2BJD03RD` suffered an initial failure on `2025-04-09` during the training period (recorded in `failure_catalog.csv`).
  2. The drive re-appeared in Backblaze operational telemetry on `2025-10-24` and operated continuously until `2026-02-08`.
  3. On `2026-02-08`, Backblaze telemetry recorded an authentic second mechanical failure event (`failure == 1`).
  4. In `src/census.py` line 152, the fleet census deduplicated failure records by `serial_number`, retaining only the first failure date (`2025-04-09`).
  5. In `src/dataset.py`, windows were labeled directly from raw daily telemetry. Window `34201` is a genuine failure event, not an artifact or label error.
* **Correction:** Label integrity is verified at 100% across all 4,302 positive windows ($2,799$ train, $694$ val, $809$ test).

---

### Discrepancy 2: Reconciliation of the 26,182 Excluded Windows
* **Prior State:** Pass 2 generated 1,164,052 rolling windows, while the three splits retained 1,137,870 windows (difference: 26,182). Preliminary audit relied solely on source-code inspection of `keep = train_mask | val_mask | test_mask` without metadata-level accounting.
* **Independent Metadata Enumeration:** Analyzed drive observation intervals in `meta_train.parquet`, `meta_val.parquet`, and `meta_test.parquet`:
  * **Train $\to$ Val Boundary Straddles (`2025-12-25` to `2025-12-30`):** **13,423 windows** (13,307 consecutive 14-day gaps + 116 terminal drive straddles).
  * **Val $\to$ Test Boundary Straddles (`2026-03-25` to `2026-03-30`):** **12,627 windows** (12,480 consecutive 14-day gaps + 147 terminal drive straddles).
  * **Multi-Week and Edge Straddles:** **132 windows** from drives with intermittent missing daily observations.
  * **Sum:** $13,423 + 12,627 + 132 = \mathbf{26,182}$ (Exact match confirmed).
* **Correction:** Excluded window status updated from `PARTIAL` to `PASS`. All 26,182 exclusions are empirically proven to enforce strict temporal target isolation.

---

### Discrepancy 3: Treatment of the 323 Disappeared Drives
* **Prior State:** 323 cohort drives stopped reporting without an official `failure == 1` flag. Preliminary report assumed all were routine decommissions without inspecting physical sensor telemetry.
* **Empirical Telemetry Inspection:** Inspected the final 30-day sequence of all 323 drives in `X_train.npy` and `X_val.npy`:
  * **303 drives (93.81%):** Exhibited completely clean, normal SMART metrics up to their final report, confirming routine decommissions or migrations.
  * **20 drives (6.19%):** Exhibited elevated error attributes ($> 2.0$ standard deviations on SMART 5, 197, or 198), representing potential proactive pre-emptive replacements.
* **Uncertainty Quantification:** The 20 suspect drives account for 744 training windows (0.0957%) and 79 validation windows (0.0430%). **Zero suspect drives appear in the test set**.
* **Correction:** Disclosed in the paper that disappearing drives represent an operational uncertainty; proved that the held-out test split is 100% insulated from this uncertainty.

---

### Discrepancy 4: Drive-Cluster Bootstrap Repetitions and Equivalence
* **Prior State:** Initial evaluation utilized 500 bootstrap iterations; early audit code contained an ad-hoc heuristic (`cluster_method_honest`).
* **Correction:**
  * Expanded to **5,000 bootstrap iterations** (`seed = 42`) for statistical stability.
  * Implemented an exact vectorized sufficiency-count matrix dot-product that was mathematically validated against iterative row-index resampling ($0.00$ difference).
  * Removed the ad-hoc `cluster_method_honest` rule.
  * Explained that cluster bootstrap CIs can be narrower or wider than naive row bootstrap CIs depending on the ratio of within-cluster to between-cluster variance across rare failure events.
  * Synchronized the latest 5,000-rep CIs across all tables and reports:
    * Transformer: `[0.1706, 0.2135]`
    * LSTM: `[0.1480, 0.1916]`
    * Isolation Forest: `[0.1074, 0.1486]`

---

### Discrepancy 5: Isolation Forest Metric Alignment
* **Prior State:** In informal user query text, Isolation Forest test precision was tentatively listed as 0.1112 and recall as 0.1508.
* **Verification against Ground-Truth Confusion Matrix:**
  * True Positives (TP) = 90
  * False Positives (FP) = 507
  * False Negatives (FN) = 719
  * True Negatives (TN) = 175,707
  * $\text{Precision} = \frac{90}{90 + 507} = \frac{90}{597} = \mathbf{0.1508}$
  * $\text{Recall} = \frac{90}{90 + 719} = \frac{90}{809} = \mathbf{0.1112}$
* **Correction:** Verified that `model_comparison.csv` and `FINAL_RESEARCH_REPORT.md` correctly report Precision = 0.1508 and Recall = 0.1112, avoiding column transposition.

---

### Discrepancy 6: Inference Latency Profiling Scope
* **Prior State:** Latency was previously summarized in a single batch number (1.33–2.22s) without operational decomposition.
* **Correction:** Decomposed latency into 5 operational deployment configurations (Full Batch, Per-Sample Batch, Single-Drive Alert, Daily Fleet Run, and End-to-End Pipeline). Explicitly disclosed that measurements were gathered locally on an AMD Ryzen 5 / RTX 3050 Laptop setup and do not constitute cloud service-level guarantees.

---

## 2. Updated File Inventory

| File Path | Nature of Change | Summary of Changes |
| :--- | :---: | :--- |
| `tests/scientific_verification_audit.py` | Code Refactoring | Added 2BJD03RD trace, metadata window enumeration, SMART degradation inspection, 5,000-rep vectorized bootstrap, and ASCII encoding fix. |
| `outputs/logs/scientific_verification_audit.json` | Data Generation | Re-generated with verified 4-check audit evidence (all PASS). |
| `outputs/metrics/model_comparison.csv` | Value Correction | Updated `F1_95_CI` column to latest 5,000-rep cluster bootstrap values; verified precision/recall alignment. |
| `outputs/plots/study_workflow_and_splits.png` | Plot Creation | Generated end-to-end architecture diagram showing census, preprocessing, windowing, and chronological splits. |
| `outputs/FINAL_RESEARCH_REPORT.md` | Academic Draft | Fully rewritten into a 19-section academic paper draft with verified references, benchmark tables, and rigorous discussion. |
| `outputs/FINAL_REPRODUCIBILITY_CHECKLIST.md` | Verification Matrix | Created systematic 16-point reproducibility checklist (all PASS). |
| `outputs/FINAL_CHANGELOG.md` | Audit Log | Created this detailed change log and discrepancy reconciliation report. |
| `outputs/FINAL_SUBMISSION_STATUS.md` | Assessment | Created concise readiness assessment for research paper submission. |
| `README.md` | Documentation | Created complete reproduction guide with environment setup and script execution commands. |
