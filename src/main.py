"""
Master Pipeline Runner: End-to-End Orchestration
=================================================
Orchestrates:
1. Pass 1: Fleet Census & Cohort Whitelist (src/census.py)
2. Pass 2: Feature Extraction, Windowing, & Leak-Free Splits (src/dataset.py)
3. Step 3: PyTorch GPU Training for LSTM & Transformer + Isolation Forest Baseline (src/train.py)
4. Step 4: Scientific Evaluation, Validation-Only Threshold Tuning, PR Curves, & Confusion Matrices (src/evaluate.py)
"""

import sys
import os
import time
import logging

from src.census import run_census
from src.dataset import build_dataset
from src.train import run_training_pipeline
from src.evaluate import run_evaluation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", os.path.join(PROJECT_ROOT, "outputs", "data"))


def main():
    print("=" * 80)
    print("STARTING END-TO-END RESEARCH PIPELINE FOR HDD FAILURE PREDICTION")
    print("=" * 80)
    total_start = time.time()

    # Step 1: Pass 1 Census
    cohort_path = os.path.join(OUTPUT_DATA_DIR, "cohort_whitelist.csv")
    if not os.path.isfile(cohort_path):
        print("\n>>> STEP 1: Executing Pass 1 (Fleet Census & Cohort Whitelist)...")
        run_census()
    else:
        print("\n>>> STEP 1: Pass 1 Census already completed. Found existing cohort_whitelist.csv.")

    # Step 2: Pass 2 Dataset Building
    split_summary_path = os.path.join(OUTPUT_DATA_DIR, "split_summary.json")
    if not os.path.isfile(split_summary_path):
        print("\n>>> STEP 2: Executing Pass 2 (Feature Extraction & Sliding Window Generation)...")
        build_dataset()
    else:
        print("\n>>> STEP 2: Pass 2 Dataset already generated. Found existing split_summary.json.")

    # Step 3: Model Training
    val_pred_path = os.path.join(OUTPUT_DATA_DIR, "val_predictions.parquet")
    if not os.path.isfile(val_pred_path):
        print("\n>>> STEP 3: Executing Model Training (LSTM, Transformer, Isolation Forest)...")
        run_training_pipeline()
    else:
        print("\n>>> STEP 3: Model training already executed. Found existing prediction parquets.")

    # Step 4: Scientific Evaluation
    print("\n>>> STEP 4: Executing Scientific Evaluation, PR Curves & Confusion Matrices...")
    results_df = run_evaluation()
    
    elapsed_total = round(time.time() - total_start, 2)
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_total} s ({elapsed_total/60:.2f} min).")
    print("=" * 80)
    print("\nFinal Comparative Model Performance (Test Set: Q2 2026):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
