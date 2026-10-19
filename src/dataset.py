"""
Pass 2: Feature Ingestion, Scaling, and Calendar-Strict Windowing
=================================================================
Implements the expanded 7-feature pipeline matching the user's Colab notebook:
- Features: SMART 5, 187, 188, 194, 197, 198, 9
- Non-drive / SSD pattern filtering
- Log-transformation on count attributes: log1p(clip(lower=0))
- SimpleImputer(median) + StandardScaler fitted strictly on training data
- Calendar-strict 30-day history with weekly-phase candidate sampling
- 7-day lookahead target validation (confirms all 7 future days or complete span up to failure)
- Chronological train (2025), val (Q1 2026), and test (Q2 2026) splits
"""

import os
import io
import gc
import json
import time
import zipfile
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.environ.get(
    "BACKBLAZE_ARCHIVE_PATH",
    os.path.join(os.path.expanduser("~"), "Documents", "Backblaze_All_Data.zip")
)
OUTPUT_DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", os.path.join(PROJECT_ROOT, "outputs", "data"))
OUTPUT_LOG_DIR = os.environ.get("OUTPUT_LOG_DIR", os.path.join(PROJECT_ROOT, "outputs", "logs"))

FEATURES = [
    "smart_5_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw", "smart_9_raw"
]
LOG_FEATURES = [
    "smart_5_raw", "smart_187_raw", "smart_188_raw",
    "smart_197_raw", "smart_198_raw", "smart_9_raw"
]

WINDOW = 30
HORIZON = 7
WEEKLY_SAMPLING = True

# Chronological split boundaries across 2025 and 2026
TRAIN_TARGET_END = pd.Timestamp("2025-12-31")
VAL_TARGET_START = pd.Timestamp("2026-01-01")
VAL_TARGET_END = pd.Timestamp("2026-03-31")
TEST_TARGET_START = pd.Timestamp("2026-04-01")
TEST_TARGET_END = pd.Timestamp("2026-06-30")

QUARTERS = [
    "data_Q1_2025.zip",
    "data_Q2_2025.zip",
    "data_Q3_2025.zip",
    "data_Q4_2025.zip",
    "data_Q1_2026.zip",
    "data_Q2_2026.zip"
]

REQUIRED_COLS = ["date", "serial_number", "model", "failure"]


def setup_logger():
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_LOG_DIR, "pass2_dataset.log")
    
    logger = logging.getLogger("Dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_dataset():
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("PASS 2: 7-FEATURE INGESTION, CALENDAR WINDOWING & CHRONOLOGICAL SPLITS")
    logger.info(f"SMART Features: {FEATURES}")
    logger.info("=" * 70)
    t0 = time.time()

    cohort_path = os.path.join(OUTPUT_DATA_DIR, "cohort_whitelist.csv")
    if not os.path.isfile(cohort_path):
        raise FileNotFoundError(f"Cohort whitelist not found at {cohort_path}. Run Pass 1 first!")

    cohort_df = pd.read_csv(cohort_path)
    whitelist_serials = set(cohort_df["serial_number"].astype(str))
    logger.info(f"Loaded whitelist containing {len(whitelist_serials):,} target HDDs.")

    # Step 1: Stream daily CSVs and collect cohort rows for the 7 SMART features
    frames = []
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as outer_zip:
        outer_names = set(outer_zip.namelist())
        
        for q_idx, q_zip_name in enumerate(QUARTERS, 1):
            if q_zip_name not in outer_names:
                logger.warning(f"Quarter {q_zip_name} not found in outer zip!")
                continue
            
            logger.info(f"[{q_idx}/6] Streaming features from {q_zip_name}...")
            q_bytes = outer_zip.read(q_zip_name)
            
            with zipfile.ZipFile(io.BytesIO(q_bytes), "r") as inner_zip:
                csv_files = sorted([f for f in inner_zip.namelist() if f.endswith(".csv")])
                
                for idx, csv_name in enumerate(csv_files):
                    with inner_zip.open(csv_name) as f:
                        header = pd.read_csv(f, nrows=0).columns.tolist()
                        usecols = [c for c in REQUIRED_COLS + FEATURES if c in header]
                        
                        # Re-open stream for full read with determined usecols
                        f.seek(0)
                        df = pd.read_csv(f, usecols=usecols, low_memory=False)
                        
                        df["serial_number"] = df["serial_number"].astype(str)
                        # Filter strictly for whitelist HDDs
                        df = df[df["serial_number"].isin(whitelist_serials)].copy()
                        
                        if not df.empty:
                            for ft in FEATURES:
                                if ft not in df.columns:
                                    df[ft] = np.nan
                            
                            df["date"] = pd.to_datetime(df["date"], errors="coerce")
                            df["failure"] = pd.to_numeric(df["failure"], errors="coerce").fillna(0).astype("int8")
                            for ft in FEATURES:
                                df[ft] = pd.to_numeric(df[ft], errors="coerce").replace([np.inf, -np.inf], np.nan)
                            
                            df = df.dropna(subset=["date", "serial_number"])
                            frames.append(df[REQUIRED_COLS + FEATURES])
                    
                    if (idx + 1) % 30 == 0 or (idx + 1) == len(csv_files):
                        logger.info(f"   [{idx + 1}/{len(csv_files)}] daily CSVs processed.")
            
            del q_bytes
            gc.collect()

    logger.info("Consolidating daily cohort records into DataFrame...")
    daily = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    logger.info(f"Total cohort records ingested: {len(daily):,}")
    daily = daily.sort_values(["serial_number", "date"], kind="mergesort")
    
    # Deduplicate dates
    dup_count = int(daily.duplicated(["serial_number", "date"]).sum())
    if dup_count > 0:
        logger.info(f"Dropping {dup_count:,} duplicate drive-day rows (keeping first).")
        daily = daily.drop_duplicates(["serial_number", "date"], keep="first")

    logger.info(f"Ingested records date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    # Keep only rows through the final test target end
    daily = daily.loc[daily["date"] <= TEST_TARGET_END].copy()

    # Step 2: Log transform count attributes
    logger.info("Applying log1p transformation to count features...")
    for f in LOG_FEATURES:
        daily[f] = np.log1p(daily[f].clip(lower=0))

    # Step 3: Leak-free imputation & scaling (fitted ONLY on training period rows)
    logger.info("Fitting SimpleImputer(median) and StandardScaler EXCLUSIVELY on training-period rows...")
    train_rows_mask = daily["date"] <= TRAIN_TARGET_END
    logger.info(f"Training observations used for fitting: {int(train_rows_mask.sum()):,}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputer.fit(daily.loc[train_rows_mask, FEATURES])
    train_imputed = imputer.transform(daily.loc[train_rows_mask, FEATURES])
    scaler.fit(train_imputed)

    # Save fitted imputer and scaler
    joblib.dump(imputer, os.path.join(OUTPUT_DATA_DIR, "simple_imputer.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DATA_DIR, "standard_scaler.joblib"))
    logger.info("Saved fitted imputer and scaler to outputs/data/")

    all_imputed = imputer.transform(daily[FEATURES])
    all_scaled = scaler.transform(all_imputed).astype(np.float32)
    daily.loc[:, FEATURES] = all_scaled

    # Step 4: Calendar-Strict Sliding Window Generation
    logger.info("Constructing 30-day history windows and 7-day lookahead binary targets...")
    
    X_parts, y_parts, meta_parts = [], [], []
    daily = daily.sort_values(["serial_number", "date"], kind="mergesort")

    for drive_id, g in daily.groupby("serial_number", sort=False):
        g = g.sort_values("date")
        g = g.drop_duplicates("date", keep="first")
        if len(g) < WINDOW + 1:
            continue

        dates = g["date"].to_numpy(dtype="datetime64[D]")
        vals = g[FEATURES].to_numpy(dtype=np.float32)
        failures = g["failure"].to_numpy(dtype=np.int8)

        date_to_idx = {d: j for j, d in enumerate(dates)}
        candidate_indices = range(WINDOW - 1, len(g) - 1)
        drive_phase = (sum(ord(ch) for ch in str(drive_id)) % 7) if WEEKLY_SAMPLING else 0

        x_drive, y_drive, m_drive = [], [], []
        for end_idx in candidate_indices:
            hist_start = dates[end_idx] - np.timedelta64(WINDOW - 1, "D")
            history_dates = hist_start + np.arange(WINDOW).astype("timedelta64[D]")
            hist_idx = [date_to_idx.get(d) for d in history_dates]
            # Require all 30 consecutive calendar days
            if any(j is None for j in hist_idx):
                continue

            # Fixed weekly phase per drive avoids selecting windows based on the label
            if WEEKLY_SAMPLING and ((end_idx - (WINDOW - 1) - drive_phase) % 7 != 0):
                continue

            target_start = dates[end_idx] + np.timedelta64(1, "D")
            target_dates = target_start + np.arange(HORIZON).astype("timedelta64[D]")
            target_idx = [date_to_idx.get(d) for d in target_dates]

            fail_candidates = [
                j for j, d in enumerate(dates)
                if target_start <= d <= target_dates[-1] and failures[j] == 1
            ]
            
            if fail_candidates:
                first_fail_date = dates[fail_candidates[0]]
                required_dates = target_start + np.arange(
                    int((first_fail_date - target_start).astype("timedelta64[D]").astype(int)) + 1
                ).astype("timedelta64[D]")
                if all(d in date_to_idx for d in required_dates):
                    label = 1
                    target_end_used = first_fail_date
                else:
                    continue
            else:
                if any(j is None for j in target_idx):
                    continue
                if any(failures[j] == 1 for j in target_idx):
                    continue
                label = 0
                target_end_used = target_dates[-1]

            x_drive.append(vals[hist_idx])
            y_drive.append(label)
            m_drive.append((
                str(drive_id),
                pd.Timestamp(dates[end_idx]),
                pd.Timestamp(target_start),
                pd.Timestamp(target_end_used)
            ))

        if x_drive:
            X_parts.append(np.asarray(x_drive, dtype=np.float32))
            y_parts.append(np.asarray(y_drive, dtype=np.int8))
            meta_parts.extend(m_drive)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    meta = pd.DataFrame(meta_parts, columns=["serial_number", "history_end", "target_start", "target_end"])
    del X_parts, y_parts, meta_parts, daily
    gc.collect()

    logger.info(f"Total calendar-strict windows generated: {len(X):,} | Positives: {int(y.sum()):,} ({y.mean():.4%})")

    # Step 5: Chronological Splits
    train_mask = meta["target_end"] <= TRAIN_TARGET_END
    val_mask = (meta["target_start"] >= VAL_TARGET_START) & (meta["target_end"] <= VAL_TARGET_END)
    test_mask = (meta["target_start"] >= TEST_TARGET_START) & (meta["target_end"] <= TEST_TARGET_END)

    keep = train_mask | val_mask | test_mask
    X, y, meta = X[keep.to_numpy()], y[keep.to_numpy()], meta.loc[keep].reset_index(drop=True)

    train_mask = (meta["target_end"] <= TRAIN_TARGET_END).to_numpy()
    val_mask = ((meta["target_start"] >= VAL_TARGET_START) & (meta["target_end"] <= VAL_TARGET_END)).to_numpy()
    test_mask = ((meta["target_start"] >= TEST_TARGET_START) & (meta["target_end"] <= TEST_TARGET_END)).to_numpy()

    split_idx = {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask)
    }

    for name, idx in split_idx.items():
        pos = int(y[idx].sum())
        logger.info(f"{name.upper()} split: {len(idx):,} windows | Positives: {pos:,} ({pos/len(idx):.4%})")
        logger.info(f"   Target date span: {meta.iloc[idx]['target_start'].min().date()} to {meta.iloc[idx]['target_end'].max().date()}")

    # Assert integrity
    assert len(split_idx["train"]) and len(split_idx["val"]) and len(split_idx["test"])
    assert y[split_idx["train"]].sum() > 0, "No positive training windows."
    assert y[split_idx["val"]].sum() > 0, "No positive validation windows."
    assert y[split_idx["test"]].sum() > 0, "No positive test windows."
    assert meta.iloc[split_idx["train"]]["target_end"].max() < meta.iloc[split_idx["val"]]["target_start"].min()
    assert meta.iloc[split_idx["val"]]["target_end"].max() < meta.iloc[split_idx["test"]]["target_start"].min()
    logger.info("Chronological boundary assertion checks passed successfully.")

    tr, va, te = split_idx["train"], split_idx["val"], split_idx["test"]
    X_train, y_train, meta_train = X[tr], y[tr], meta.iloc[tr].reset_index(drop=True)
    X_val, y_val, meta_val = X[va], y[va], meta.iloc[va].reset_index(drop=True)
    X_test, y_test, meta_test = X[te], y[te], meta.iloc[te].reset_index(drop=True)

    # Save arrays
    logger.info("Saving preprocessed arrays to outputs/data/...")
    np.save(os.path.join(OUTPUT_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y_train.npy"), y_train)
    meta_train.to_parquet(os.path.join(OUTPUT_DATA_DIR, "meta_train.parquet"), index=False)

    np.save(os.path.join(OUTPUT_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y_val.npy"), y_val)
    meta_val.to_parquet(os.path.join(OUTPUT_DATA_DIR, "meta_val.parquet"), index=False)

    np.save(os.path.join(OUTPUT_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y_test.npy"), y_test)
    meta_test.to_parquet(os.path.join(OUTPUT_DATA_DIR, "meta_test.parquet"), index=False)

    elapsed = round(time.time() - t0, 2)
    split_summary = {
        "features": FEATURES,
        "window": WINDOW,
        "horizon": HORIZON,
        "train": {"windows": len(y_train), "positives": int(y_train.sum()), "prevalence": float(y_train.mean())},
        "validation": {"windows": len(y_val), "positives": int(y_val.sum()), "prevalence": float(y_val.mean())},
        "test": {"windows": len(y_test), "positives": int(y_test.sum()), "prevalence": float(y_test.mean())},
        "elapsed_seconds": elapsed
    }
    with open(os.path.join(OUTPUT_DATA_DIR, "split_summary.json"), "w") as f:
        json.dump(split_summary, f, indent=2)

    logger.info(f"Pass 2 completed in {elapsed} s ({elapsed/60:.2f} min).")
    return split_summary


if __name__ == "__main__":
    build_dataset()
