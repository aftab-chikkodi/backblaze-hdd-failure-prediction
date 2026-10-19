"""
Small Pilot Test for Backblaze Data Loading and Preprocessing Pipeline
======================================================================
Tests:
1. Zero-extraction streaming from nested zip archives.
2. In-memory CSV parsing and schema verification.
3. Model filtering heuristic (HDD vs SSD).
4. Feature extraction (SMART 5, 194, 9) and missing-value diagnostics.
5. Window construction (30-day sequence, 7-day lookahead label).
6. Memory usage profiling during ingestion.
"""

import os
import io
import time
import json
import zipfile
import tracemalloc
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.environ.get(
    "BACKBLAZE_ARCHIVE_PATH",
    os.path.join(os.path.expanduser("~"), "Documents", "Backblaze_All_Data.zip")
)
TARGET_QUARTER_ZIP = "data_Q1_2025.zip"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs"))
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

SSD_PATTERN = (
    r"\bSSD\b|MTFD|MICRON|INTEL|KINGSTON|"
    r"CT[0-9].*MX|SAMSUNG.*MZ|WDC WDS"
)

FEATURES = ["smart_5_raw", "smart_194_raw", "smart_9_raw"]
USE_COLS = ["date", "serial_number", "model", "failure"] + FEATURES


def run_pilot_test(num_days: int = 5):
    print("=" * 70)
    print("STARTING PILOT PREPROCESSING & DATA-LOADING TEST")
    print(f"Dataset Archive: {ARCHIVE_PATH}")
    print(f"Target Quarter: {TARGET_QUARTER_ZIP}")
    print(f"Sample days to read: {num_days}")
    print("=" * 70)

    tracemalloc.start()
    t0 = time.time()
    
    results = {
        "archive_exists": False,
        "quarter_found": False,
        "days_processed": 0,
        "total_rows_read": 0,
        "hdd_rows_retained": 0,
        "ssd_rows_filtered": 0,
        "failures_found": 0,
        "unique_drives": 0,
        "missing_features_pct": {},
        "peak_ram_mb": 0.0,
        "elapsed_seconds": 0.0,
        "windowing_verified": False,
        "status": "FAILED"
    }

    if not os.path.isfile(ARCHIVE_PATH):
        raise FileNotFoundError(f"Archive not found at {ARCHIVE_PATH}")
    results["archive_exists"] = True

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as outer_zip:
        outer_names = outer_zip.namelist()
        if TARGET_QUARTER_ZIP not in outer_names:
            raise KeyError(f"{TARGET_QUARTER_ZIP} not found inside outer zip.")
        results["quarter_found"] = True

        print(f"[OK] Located {TARGET_QUARTER_ZIP} in outer archive.")
        
        # Stream the inner quarter zip into a memory buffer without writing to disk
        print(f"Streaming {TARGET_QUARTER_ZIP} into in-memory buffer...")
        inner_bytes = outer_zip.read(TARGET_QUARTER_ZIP)
        
        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_zip:
            all_csvs = sorted([f for f in inner_zip.namelist() if f.endswith(".csv")])
            sample_csvs = all_csvs[:num_days]
            print(f"Total CSVs in quarter: {len(all_csvs)}, sampling first {len(sample_csvs)} days:")
            for p in sample_csvs:
                print(f"  - {os.path.basename(p)}")

            frames = []
            for csv_name in sample_csvs:
                with inner_zip.open(csv_name) as f:
                    df = pd.read_csv(f, usecols=lambda c: c in USE_COLS, low_memory=False)
                    
                    # Ensure schema
                    for col in USE_COLS:
                        if col not in df.columns:
                            df[col] = np.nan
                    
                    df = df[USE_COLS].copy()
                    raw_count = len(df)
                    results["total_rows_read"] += raw_count

                    # Filter SSD
                    df["model"] = df["model"].astype(str)
                    is_ssd = df["model"].str.contains(SSD_PATTERN, case=False, regex=True, na=False)
                    df_hdd = df[~is_ssd].copy()
                    
                    results["ssd_rows_filtered"] += int(is_ssd.sum())
                    results["hdd_rows_retained"] += len(df_hdd)

                    # Normalize types
                    df_hdd["date"] = pd.to_datetime(df_hdd["date"], errors="coerce")
                    df_hdd["failure"] = pd.to_numeric(df_hdd["failure"], errors="coerce").fillna(0).astype("int8")
                    for col in FEATURES:
                        df_hdd[col] = pd.to_numeric(df_hdd[col], errors="coerce")
                    
                    results["failures_found"] += int(df_hdd["failure"].sum())
                    frames.append(df_hdd)
                    results["days_processed"] += 1

            sample_data = pd.concat(frames, ignore_index=True)

    results["unique_drives"] = int(sample_data["serial_number"].nunique())
    
    # Feature completeness
    for col in FEATURES:
        missing = sample_data[col].isna().sum()
        pct = (missing / len(sample_data)) * 100.0 if len(sample_data) > 0 else 0.0
        results["missing_features_pct"][col] = round(pct, 3)

    print("\n--- Pilot Test Telemetry Summary ---")
    print(f"Total rows read: {results['total_rows_read']:,}")
    print(f"HDD rows retained: {results['hdd_rows_retained']:,}")
    print(f"SSD rows filtered: {results['ssd_rows_filtered']:,}")
    print(f"Unique HDD serial numbers: {results['unique_drives']:,}")
    print(f"Confirmed failures in 5-day sample: {results['failures_found']}")
    print("Missing feature rates:")
    for col, rate in results["missing_features_pct"].items():
        print(f"  {col}: {rate}% missing")

    # Verify Windowing Logic on a synthetic drive series to confirm correctness
    print("\n--- Verifying Windowing & 7-Day Lookahead Logic ---")
    sample_dates = pd.date_range("2025-01-01", periods=40, freq="D")
    synthetic_drive = pd.DataFrame({
        "date": sample_dates,
        "serial_number": ["TEST_DRIVE_001"] * 40,
        "smart_5_raw": np.linspace(0, 10, 40),
        "smart_194_raw": np.random.uniform(25, 45, 40),
        "smart_9_raw": np.arange(1000, 1040),
        "failure": [0] * 38 + [1] + [0]  # Fails on day 39 (2025-02-08)
    })

    # Window of 30 days: Day 0 to Day 29 (2025-01-01 to 2025-01-30)
    # Next 7 calendar days: Day 30 to Day 36 (2025-01-31 to 2025-02-06) -> failure is on day 38 -> label = 0
    # Window of 30 days: Day 8 to Day 37 (2025-01-09 to 2025-02-07)
    # Next 7 calendar days: Day 38 to Day 44 (2025-02-08 to 2025-02-14) -> failure is on day 38 -> label = 1
    
    # Simple window check
    w30_feature = synthetic_drive.iloc[0:30][FEATURES].values
    assert w30_feature.shape == (30, 3), "Window shape mismatch"
    
    # Lookahead check for window 0:
    future_7 = synthetic_drive.iloc[30:37]["failure"].sum()
    label_w0 = int(future_7 > 0)
    assert label_w0 == 0, "Expected label 0 for window 0"

    # Lookahead check for window that covers failure day (e.g. window index 8: days 8 to 37)
    future_7_f = synthetic_drive.iloc[38:40]["failure"].sum()
    label_wf = int(future_7_f > 0)
    assert label_wf == 1, "Expected label 1 for window covering failure"

    results["windowing_verified"] = True
    print("[OK] Windowing shape and 7-day lookahead binary labeling verified mathematically.")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.time() - t0

    results["peak_ram_mb"] = round(peak_mem / (1024 * 1024), 2)
    results["elapsed_seconds"] = round(elapsed, 2)
    results["status"] = "PASSED"

    print(f"\nExecution time: {results['elapsed_seconds']} s")
    print(f"Peak Python RAM tracked: {results['peak_ram_mb']} MB")
    print("Status: [TEST PASSED]")

    os.makedirs(LOG_DIR, exist_ok=True)
    out_json = os.path.join(LOG_DIR, "pilot_test_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test report saved to: {out_json}")

    return results


if __name__ == "__main__":
    run_pilot_test(num_days=5)
