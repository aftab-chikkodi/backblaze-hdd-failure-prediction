# ML-Based Predictive Server Failure Detection Using Sensor Data in
Datacenter Environments

This repository contains the complete experimental codebase, evaluation artifacts, publication-ready figures, reproducibility checklist, and research report for our comparative study predicting mechanical hard disk drive (HDD) failures from longitudinal SMART telemetry across six calendar quarters (Q1 2025 – Q2 2026).

---

## 1. Project Overview & Methodology

* **Task Formulation:** Predict a recorded mechanical drive failure within the **next 7 calendar days** ($T+1$ to $T+7$) using the preceding **30 calendar days** of SMART sensor history ($T-29$ to $T$).
* **Corpus Scope:** 546 daily fleet snapshots containing 179,959,674 raw telemetry observations.
* **Evaluated Paradigms:**
  1. **Long Short-Term Memory (LSTM):** Recurrent neural network ($64 \to 32$, dropout 0.25).
  2. **Multi-Head Transformer Encoder:** Self-attention network ($d_{\text{model}}=64$, 4 heads, feedforward dimension 128).
  3. **Isolation Forest:** Unsupervised anomaly detector operating on 28-dimensional trajectory summaries ($[\text{last}, \text{mean}, \text{max}, \text{slope}]$).
* **Experimental Controls:**
  * Strict exclusion of solid-state drives (SSDs) and non-drive RAID/BOSS controllers.
  * Leak-free chronological splitting:
    * **Train:** 2025-01-01 to 2025-12-31 (4 quarters, 804,809 sequences, 2,799 failures)
    * **Validation:** 2026-01-01 to 2026-03-31 (Q1 2026, 156,038 sequences, 694 failures)
    * **Held-out Test:** 2026-04-01 to 2026-06-30 (Q2 2026, 177,023 sequences, 809 failures)
  * Feature imputation (median) and standardization strictly fitted on 2025 training data and frozen.
  * Decision thresholds ($\tau^*$) tuned exclusively on Q1 2026 validation data via 1,001-point quantile search and frozen prior to test evaluation.
  * Statistical uncertainty quantified using 5,000-repetition drive-cluster bootstrap resampling.

---

## 2. Seven SMART Features

Our pipeline tracks seven critical SMART telemetry attributes selected for predictive power and physical fault correlation:

| SMART Attribute | Raw Column Name | Physical Description & Failure Rationale | Preprocessing |
| :--- | :--- | :--- | :--- |
| **SMART 5** | `smart_5_raw` | **Reallocated Sectors Count:** Count of damaged physical sectors retired and remapped to spare sectors. Primary physical indicator of degrading magnetic media. | $\log(1 + x)$, imputed, scaled |
| **SMART 9** | `smart_9_raw` | **Power-On Hours:** Total cumulative operational lifetime in hours. Captures operational aging and wear progression. | $\log(1 + x)$, imputed, scaled |
| **SMART 187** | `smart_187_raw` | **Reported Uncorrectable Errors:** Read/write operations that could not be recovered through hardware Error Correction Codes (ECC). | $\log(1 + x)$, imputed, scaled |
| **SMART 188** | `smart_188_raw` | **Command Timeout:** Aborted read/write commands caused by internal bus or controller interface communication failures. | $\log(1 + x)$, imputed, scaled |
| **SMART 194** | `smart_194_raw` | **Internal Temperature:** Internal drive thermal reading in degrees Celsius. Tracks thermal stress and cooling degradation. | Imputed, scaled (linear) |
| **SMART 197** | `smart_197_raw` | **Current Pending Sector Count:** Unstable physical sectors awaiting remapping upon subsequent write failure. Critical warning of imminent corruption. | $\log(1 + x)$, imputed, scaled |
| **SMART 198** | `smart_198_raw` | **Offline / Uncorrectable Sectors:** Irrecoverable physical sectors discovered during background diagnostic sweeps. | $\log(1 + x)$, imputed, scaled |

> **Feature Engineering Note:** Count attributes exhibit heavy-tailed distributions and undergo a monotonic transformation $\log(1 + \max(0, x))$. Imputation medians and scaling parameters are computed strictly from the 2025 training period and serialized to prevent future-feature leakage.

---

## 3. Directory Structure

```
Backblaze_Project/
├── README.md                          # Reproduction guide and execution instructions
├── requirements.txt                   # Pinned dependency environment
├── .gitignore                         # Excludes virtual environments, raw data, and checkpoints
├── src/
│   ├── census.py                      # Pass 1: Streaming fleet census & whitelist creation
│   ├── dataset.py                     # Pass 2: Feature pipeline, windowing, and chronological splits
│   ├── models.py                      # LSTM, Transformer, and Isolation Forest architectures
│   ├── train.py                       # PyTorch GPU training with AMP and BCE pos_weight=20.0
│   ├── evaluate.py                    # Quantile threshold tuning on Val and frozen Test evaluation
│   └── main.py                        # Pipeline orchestrator
├── tests/
│   ├── test_pilot.py                  # 5-day ingestion and pre-flight memory test
│   └── scientific_verification_audit.py # Comprehensive 4-check scientific audit
└── outputs/
    ├── FINAL_RESEARCH_REPORT.md       # Complete 19-section academic research paper draft
    ├── FINAL_REPRODUCIBILITY_CHECKLIST.md # Systematic 16-item reproducibility checklist
    ├── FINAL_CHANGELOG.md             # Detailed change log and audit reconciliation notes
    ├── FINAL_SUBMISSION_STATUS.md     # Readiness assessment for paper submission
    ├── data/                          # Failure catalog, cohort whitelist, split metadata
    ├── metrics/                       # model_comparison.csv, confusion_matrices.json
    ├── plots/                         # PR curves, confusion matrices, bar charts, workflow diagram
    └── logs/                          # Verified audit evidence JSON and test logs
```

---

## 4. Computational Environment & Prerequisites

### Hardware Requirements
* **CPU:** 6-core processor or higher (Benchmarked on AMD Ryzen 5 5600H).
* **System RAM:** Minimum 16 GB DDR4 RAM.
* **GPU (Recommended):** NVIDIA GPU with CUDA support and at least 4 GB VRAM (Benchmarked on NVIDIA GeForce RTX 3050 Laptop GPU).
* **Storage:** Fast NVMe SSD recommended for streaming compressed archives.

### Software Specifications
* **Operating System:** Windows 10/11 or Linux (Ubuntu 22.04+).
* **Python Version:** Python 3.11 (CPython 3.11.16).

---

## 5. Setup and Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/<username>/Backblaze_Project.git
cd Backblaze_Project
```

### Step 2: Initialize Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
# source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies from requirements.txt
pip install -r requirements.txt
```

---

## 6. Data Acquisition & Pipeline Reproduction

### Raw Telemetry Acquisition
Raw telemetry is publicly hosted by Backblaze:
1. Download the quarterly telemetry archives for **Q1 2025 through Q2 2026** from the [Backblaze Hard Drive Test Data repository](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).
2. Store the archives or combine them into a multi-quarter zip container (e.g., `Backblaze_All_Data.zip`).
3. Set the environment variable to point to your archive:
   ```powershell
   # Windows PowerShell
   $env:BACKBLAZE_ARCHIVE_PATH = "C:\path\to\Backblaze_All_Data.zip"

   # Linux/macOS
   # export BACKBLAZE_ARCHIVE_PATH="/path/to/Backblaze_All_Data.zip"
   ```

> [!NOTE]
> Large preprocessed array files (`outputs/data/*.npy`, ~950 MB total) and serialized model weights (`outputs/models/*.pt`, `*.joblib`) are excluded from GitHub to adhere to repository size limits. The repository includes all lightweight summary data, failure catalogs, whitelists, trained evaluation metrics, audit evidence, and plots. You can regenerate the preprocessed arrays and checkpoints using the documented stages below.

### Pre-Flight Verification (5-Day Pilot Test)
To verify memory limits and compressed streaming without modifying data:
```bash
python tests/test_pilot.py
```

### Stage 1: Fleet Census & Cohort Whitelist (Pass 1)
Streams 546 daily CSVs across 6 quarters directly from compressed memory streams to catalog all 6,861 confirmed failed HDDs and select 15,000 stable healthy drives:
```bash
python -m src.census
```
*Outputs: `outputs/data/failure_catalog.csv`, `outputs/data/cohort_whitelist.csv` (Runtime: ~18.5 min).*

### Stage 2: Feature Pipeline & Chronological Windowing (Pass 2)
Extracts 7 SMART features, applies $\log(1+x)$ transformation, fits median imputer and standard scaler exclusively on 2025 training data, builds 30-day continuous sequences with weekly phase anchoring, and partitions into train/val/test splits:
```bash
python -m src.dataset
```
*Outputs: `outputs/data/X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, `simple_imputer.joblib`, `standard_scaler.joblib` (Runtime: ~50 min).*

### Stage 3: Model Training
Trains LSTM ($64 \to 32$) and Transformer Encoder ($d_{\text{model}}=64$, 4 heads) for 15 epochs with AMP and `pos_weight=20.0` on GPU; fits Isolation Forest on CPU:
```bash
python -m src.train
```
*Outputs: `outputs/models/lstm_best.pt`, `outputs/models/transformer_best.pt`, `outputs/models/isolation_forest.joblib` (Runtime: ~10 min).*

### Stage 4: Threshold Selection & Evaluation
Conducts a 1,001-point quantile search on validation predictions to select optimal F1 thresholds ($\tau^*$), freezes them, evaluates once on the held-out test set, and generates PR curves, confusion matrices, and metrics:
```bash
python -m src.evaluate
```
*Outputs: `outputs/metrics/model_comparison.csv`, `outputs/metrics/confusion_matrices.json`, `outputs/plots/` (Runtime: ~2 min).*

### Stage 5: Scientific Verification Audit
Executes four automated verification checks (1-day gap verification, 5,000-repetition drive-cluster bootstrap, metadata-level exclusion enumeration, and latency decomposition):
```bash
python tests/scientific_verification_audit.py
```
*Outputs: `outputs/logs/scientific_verification_audit.json` (Runtime: ~1 min).*

---

## 7. Published Benchmark Results (Held-Out Test Set: Q2 2026)

All models are evaluated on the strictly held-out Q2 2026 test set (177,023 sequences, 809 failures; failure prevalence = 0.4570%). Decision thresholds are frozen from the validation set:

| Model | Val Threshold ($\tau^*$) | Test Precision | Test Recall | Test F1-Score | 95% Drive-Cluster CI | Average Precision (PR-AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | `0.5669` | 0.1779 | 0.1632 | 0.1702 | `[0.1480, 0.1916]` | 0.0961 |
| **Transformer Encoder** | `0.4365` | **0.1798** | **0.2052** | **0.1917** | **`[0.1706, 0.2135]`** | **0.1044** |
| **Isolation Forest** | `0.1981` | 0.1508 | 0.1112 | 0.1280 | `[0.1074, 0.1486]` | 0.0756 |
| **Random Guess Baseline** | `0.5000` | 0.0046 | 0.0046 | 0.0046 | — | 0.00457 |

### Confusion Matrices (Held-Out Test Set: $N = 177,023$)
* **Transformer Encoder:** TP = 166, FP = 757, FN = 643, TN = 175,457
* **LSTM:** TP = 132, FP = 610, FN = 677, TN = 175,604
* **Isolation Forest:** TP = 90, FP = 507, FN = 719, TN = 175,707

---

## 8. Limitations & Operational Scope

1. **Local Benchmark Latency:** Latency figures represent in-memory benchmark measurements on an NVIDIA RTX 3050 Laptop GPU and AMD Ryzen 5 CPU; they do not include distributed network queries or real-time database lookups.
2. **Disappeared Drive Uncertainty:** 20 cohort drives that stopped reporting exhibited elevated error metrics (> 2 std) prior to departure, representing possible proactive drive swaps. These account for 744 train and 79 validation windows, but **0 test windows**. The test partition is 100% insulated from this uncertainty.
3. **Cohort Representation:** While sequence-level test failure rarity (0.4570%) reflects realistic operational conditions, fleet-wide failure proportions should not be inferred directly from the cohort design.
4. **Standalone Warning:** Machine learning models should not be used as standalone data protection mechanisms. Predictive alerts should trigger non-disruptive mitigation (proactive RAID scrubbing, migration priority) rather than immediate destructive replacement.
