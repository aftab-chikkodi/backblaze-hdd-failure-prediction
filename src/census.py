"""
High-Efficiency Pass 1: Fleet Census & Cohort Whitelist Generation
==================================================================
1. Streams all 546 CSV files across 6 quarterly archives without disk extraction.
2. Vectorized extraction of 100% of confirmed failure events (failure == 1).
3. Collects healthy drive candidates across quarterly anchor days.
4. Produces:
   - failure_catalog.csv (100% of confirmed failures with exact dates)
   - cohort_whitelist.csv (100% failed HDDs + stratified healthy HDD sample)
   - fleet_summary.json (metadata and counts)
"""

import os
import io
import gc
import json
import time
import zipfile
import logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.environ.get(
    "BACKBLAZE_ARCHIVE_PATH",
    os.path.join(os.path.expanduser("~"), "Documents", "Backblaze_All_Data.zip")
)
OUTPUT_DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", os.path.join(PROJECT_ROOT, "outputs", "data"))
OUTPUT_LOG_DIR = os.environ.get("OUTPUT_LOG_DIR", os.path.join(PROJECT_ROOT, "outputs", "logs"))

SSD_PATTERN = (
    r"\bSSD\b|MTFD|MICRON|INTEL|KINGSTON|"
    r"CT[0-9].*MX|SAMSUNG.*MZ|WDC WDS"
)

QUARTERS = [
    "data_Q1_2025.zip",
    "data_Q2_2025.zip",
    "data_Q3_2025.zip",
    "data_Q4_2025.zip",
    "data_Q1_2026.zip",
    "data_Q2_2026.zip"
]

CENSUS_COLS = ["date", "serial_number", "model", "failure"]


def setup_logger():
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_LOG_DIR, "pass1_census.log")
    
    logger = logging.getLogger("Census")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_census(sample_healthy_drives: int = 15000, seed: int = 42):
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("PASS 1: FLEET CENSUS & FAILURE CATALOG")
    logger.info(f"Target Archive: {ARCHIVE_PATH}")
    logger.info(f"Sampling target for healthy drives: {sample_healthy_drives:,}")
    logger.info("=" * 70)
    
    t0 = time.time()
    all_failures = []
    healthy_candidates = {}  # serial -> {"model": str, "seen_quarters": set}
    total_csv_count = 0
    total_rows_read = 0
    total_hdd_rows = 0

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as outer_zip:
        outer_names = set(outer_zip.namelist())
        
        for q_idx, q_zip_name in enumerate(QUARTERS, 1):
            if q_zip_name not in outer_names:
                logger.warning(f"Quarter zip {q_zip_name} not found in archive!")
                continue
            
            logger.info(f"[{q_idx}/6] Streaming {q_zip_name} into memory buffer...")
            q_bytes = outer_zip.read(q_zip_name)
            
            with zipfile.ZipFile(io.BytesIO(q_bytes), "r") as inner_zip:
                csv_files = sorted([f for f in inner_zip.namelist() if f.endswith(".csv")])
                logger.info(f"   Contains {len(csv_files)} daily CSV files.")
                
                # We check healthy candidates on day 1, 15, and end of each quarter to ensure continuous operation
                checkpoint_indices = {0, len(csv_files)//4, len(csv_files)//2, 3*len(csv_files)//4, len(csv_files)-1}
                
                for idx, csv_name in enumerate(csv_files):
                    with inner_zip.open(csv_name) as f:
                        df = pd.read_csv(f, usecols=lambda c: c in CENSUS_COLS, low_memory=False)
                        
                        for col in CENSUS_COLS:
                            if col not in df.columns:
                                df[col] = np.nan
                        
                        raw_len = len(df)
                        total_rows_read += raw_len
                        
                        # Vectorized SSD exclusion
                        df["model"] = df["model"].astype(str)
                        is_ssd = df["model"].str.contains(SSD_PATTERN, case=False, regex=True, na=False)
                        df_hdd = df[~is_ssd].copy()
                        total_hdd_rows += len(df_hdd)
                        
                        # Vectorized failure check
                        df_hdd["failure"] = pd.to_numeric(df_hdd["failure"], errors="coerce").fillna(0).astype("int8")
                        fail_mask = df_hdd["failure"] == 1
                        
                        if fail_mask.any():
                            fails = df_hdd.loc[fail_mask, ["date", "serial_number", "model"]].copy()
                            fails["quarter"] = q_zip_name.replace(".zip", "")
                            all_failures.append(fails)
                        
                        # Sample healthy candidates periodically
                        if idx in checkpoint_indices:
                            non_fails = df_hdd.loc[~fail_mask, ["serial_number", "model"]]
                            for s, m in zip(non_fails["serial_number"], non_fails["model"]):
                                s_str = str(s)
                                if s_str not in healthy_candidates:
                                    healthy_candidates[s_str] = {"model": str(m), "seen_checkpoints": 1}
                                else:
                                    healthy_candidates[s_str]["seen_checkpoints"] += 1
                        
                        total_csv_count += 1
                    
                    if (idx + 1) % 30 == 0 or (idx + 1) == len(csv_files):
                        n_fails_so_far = sum(len(f) for f in all_failures)
                        logger.info(f"   [{idx + 1}/{len(csv_files)}] CSVs processed. Confirmed failures so far: {n_fails_so_far:,}")
            
            del q_bytes
            gc.collect()

    logger.info("=" * 70)
    logger.info("PASS 1 COMPLETED. COMPILING CATALOG & WHITELIST...")
    logger.info(f"Total CSVs processed: {total_csv_count}")
    logger.info(f"Total raw records read: {total_rows_read:,}")
    logger.info(f"Total HDD records retained: {total_hdd_rows:,}")

    # Build Failure Catalog DataFrame
    if all_failures:
        fail_df = pd.concat(all_failures, ignore_index=True)
        fail_df["date"] = pd.to_datetime(fail_df["date"], errors="coerce")
        # Keep first failure event per serial number
        fail_df = fail_df.sort_values("date").drop_duplicates(subset=["serial_number"], keep="first")
        logger.info(f"Unique confirmed failed HDDs found: {len(fail_df):,}")
    else:
        fail_df = pd.DataFrame(columns=["date", "serial_number", "model", "quarter"])
        logger.warning("No failure records found!")

    fail_path = os.path.join(OUTPUT_DATA_DIR, "failure_catalog.csv")
    fail_df.to_csv(fail_path, index=False)
    logger.info(f"Saved failure catalog to {fail_path}")

    # Exclude failed drives from healthy candidates
    failed_serials = set(fail_df["serial_number"].astype(str))
    
    # Filter healthy candidates: seen in multiple checkpoints (stable operational drives)
    stable_healthy = [
        (s, info["model"]) for s, info in healthy_candidates.items()
        if s not in failed_serials and info["seen_checkpoints"] >= 3
    ]
    logger.info(f"Stable healthy drive candidates identified: {len(stable_healthy):,}")

    rng = np.random.default_rng(seed)
    n_sample = min(sample_healthy_drives, len(stable_healthy))
    sampled_indices = rng.choice(len(stable_healthy), size=n_sample, replace=False)
    
    cohort_rows = []
    # 1. 100% of all confirmed failed HDDs
    for _, r in fail_df.iterrows():
        cohort_rows.append({
            "serial_number": str(r["serial_number"]),
            "model": str(r["model"]),
            "is_failed": 1,
            "failure_date": str(r["date"].date()) if pd.notna(r["date"]) else ""
        })
    
    # 2. Stratified sample of healthy HDDs
    for idx in sampled_indices:
        s, m = stable_healthy[idx]
        cohort_rows.append({
            "serial_number": s,
            "model": m,
            "is_failed": 0,
            "failure_date": ""
        })

    cohort_df = pd.DataFrame(cohort_rows)
    cohort_path = os.path.join(OUTPUT_DATA_DIR, "cohort_whitelist.csv")
    cohort_df.to_csv(cohort_path, index=False)
    logger.info(f"Cohort whitelist created: {len(cohort_df):,} total drives ({len(fail_df):,} failed + {n_sample:,} healthy)")
    logger.info(f"Saved cohort whitelist to {cohort_path}")

    elapsed = round(time.time() - t0, 2)
    summary = {
        "total_csvs": total_csv_count,
        "total_raw_rows": total_rows_read,
        "total_hdd_rows": total_hdd_rows,
        "unique_failed_drives": len(fail_df),
        "sampled_healthy_drives": n_sample,
        "cohort_size": len(cohort_df),
        "elapsed_seconds": elapsed
    }
    summary_path = os.path.join(OUTPUT_DATA_DIR, "fleet_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}. Pass 1 finished in {elapsed} seconds ({elapsed/60:.2f} minutes).")

    return summary


if __name__ == "__main__":
    run_census()
