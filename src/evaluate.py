"""
Comprehensive Evaluation Engine: Validation-Only Threshold Tuning & Test Benchmarking
======================================================================================
1. Tunes decision thresholds EXCLUSIVELY on validation set by maximizing F1 score.
2. Evaluates held-out test predictions using frozen validation thresholds (Zero test leakage).
3. Computes Precision, Recall, F1, Average Precision (PR-AUC), and Confusion Matrices.
4. Calculates 95% Drive-Cluster Bootstrap Confidence Intervals for F1.
5. Produces publication-quality plots: PR curves, Confusion Matrices, and Metric Comparisons.
6. Saves all metrics and plots into outputs/metrics/ and outputs/plots/.
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

OUTPUT_DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", os.path.join(PROJECT_ROOT, "outputs", "data"))
OUTPUT_METRICS_DIR = os.environ.get("OUTPUT_METRICS_DIR", os.path.join(PROJECT_ROOT, "outputs", "metrics"))
OUTPUT_PLOTS_DIR = os.environ.get("OUTPUT_PLOTS_DIR", os.path.join(PROJECT_ROOT, "outputs", "plots"))
OUTPUT_LOG_DIR = os.environ.get("OUTPUT_LOG_DIR", os.path.join(PROJECT_ROOT, "outputs", "logs"))


def setup_logger():
    os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    
    log_path = os.path.join(OUTPUT_LOG_DIR, "evaluate.log")
    logger = logging.getLogger("Evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray):
    """Finds threshold maximizing F1 on validation data using quantile candidate grid."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = scores >= t
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_t, best_f1 = t, f
    return float(best_t), float(best_f1)


def cluster_bootstrap_f1(y_true: np.ndarray, scores: np.ndarray, threshold: float, drive_serials: np.ndarray, n_boot: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_drives = np.unique(drive_serials)
    n_drives = len(unique_drives)
    
    # Pre-group indices per drive for fast resampling
    drive_to_indices = {}
    for idx, d in enumerate(drive_serials):
        drive_to_indices.setdefault(d, []).append(idx)
        
    boot_f1s = []
    preds = (scores >= threshold).astype(int)
    
    for _ in range(n_boot):
        sample_drives = rng.choice(unique_drives, size=n_drives, replace=True)
        boot_idx = []
        for d in sample_drives:
            boot_idx.extend(drive_to_indices[d])
            
        b_true = y_true[boot_idx]
        b_pred = preds[boot_idx]
        
        if b_true.sum() > 0 and b_pred.sum() > 0:
            score = f1_score(b_true, b_pred, zero_division=0)
        else:
            score = 0.0
        boot_f1s.append(score)
        
    boot_f1s = np.array(boot_f1s)
    ci_lower = float(np.percentile(boot_f1s, 2.5))
    ci_upper = float(np.percentile(boot_f1s, 97.5))
    return ci_lower, ci_upper


def run_evaluation():
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("STARTING MODEL EVALUATION & SCIENTIFIC BENCHMARKING")
    logger.info("=" * 70)
    
    val_df = pd.read_parquet(os.path.join(OUTPUT_DATA_DIR, "val_predictions.parquet"))
    test_df = pd.read_parquet(os.path.join(OUTPUT_DATA_DIR, "test_predictions.parquet"))
    meta_test = pd.read_parquet(os.path.join(OUTPUT_DATA_DIR, "meta_test.parquet"))
    
    with open(os.path.join(OUTPUT_DATA_DIR, "timing_benchmarks.json"), "r") as f:
        timing_info = json.load(f)
        
    y_val = val_df["y_true"].values
    y_test = test_df["y_true"].values
    test_drives = meta_test["serial_number"].values
    
    models = ["LSTM", "Transformer", "Isolation_Forest"]
    results = []
    confusion_matrices = {}
    curves_data = {}
    
    logger.info("Selecting decision thresholds on VALIDATION split (F1-optimal)...")
    
    for m in models:
        val_scores = val_df[m].values
        test_scores = test_df[m].values
        
        # 1. Validation threshold selection
        best_thresh, val_f1 = find_optimal_threshold(y_val, val_scores)
        logger.info(f"[{m}] Validation F1: {val_f1:.4f} at optimal threshold: {best_thresh:.4f}")
        
        # 2. Held-out test evaluation using frozen validation threshold
        test_preds = (test_scores >= best_thresh).astype(int)
        
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        ap = average_precision_score(y_test, test_scores) if y_test.sum() > 0 else 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        confusion_matrices[m] = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        
        # Bootstrap 95% CI
        ci_low, ci_high = cluster_bootstrap_f1(y_test, test_scores, best_thresh, test_drives)
        
        # PR Curve data
        p_curve, r_curve, _ = precision_recall_curve(y_test, test_scores)
        curves_data[m] = {"precision": p_curve, "recall": r_curve, "ap": ap}
        
        train_time = timing_info["training_times_seconds"].get(m, 0.0)
        infer_lat = timing_info["test_inference_latency_seconds"].get(m, 0.0)
        
        results.append({
            "Model": m.replace("_", " "),
            "Val_Threshold": round(best_thresh, 4),
            "Test_Precision": round(prec, 4),
            "Test_Recall": round(rec, 4),
            "Test_F1": round(f1, 4),
            "F1_95_CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
            "Test_AP_PR_AUC": round(ap, 4),
            "Inference_Latency_s": round(infer_lat, 3),
            "Training_Time_s": round(train_time, 2),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })

    res_df = pd.DataFrame(results)
    res_path = os.path.join(OUTPUT_METRICS_DIR, "model_comparison.csv")
    res_df.to_csv(res_path, index=False)
    logger.info(f"Saved model comparison table to {res_path}")

    with open(os.path.join(OUTPUT_METRICS_DIR, "confusion_matrices.json"), "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # ==========================================
    # Plot 1: Precision-Recall Curves
    # ==========================================
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {"LSTM": "#1f77b4", "Transformer": "#ff7f0e", "Isolation_Forest": "#2ca02c"}
    
    for m in models:
        p_c = curves_data[m]["precision"]
        r_c = curves_data[m]["recall"]
        ap_val = curves_data[m]["ap"]
        plt.plot(r_c, p_c, label=f"{m.replace('_', ' ')} (AP = {ap_val:.4f})", color=colors[m], linewidth=2.2)
        
    prevalence = y_test.mean()
    plt.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.7, label=f"Random Prevalence ({prevalence:.2%})")
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves on Untouched Chronological Test Set (Q2 2026)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    pr_plot_path = os.path.join(OUTPUT_PLOTS_DIR, "precision_recall_curves.png")
    plt.savefig(pr_plot_path)
    plt.close()
    logger.info(f"Saved PR curve plot to {pr_plot_path}")

    # ==========================================
    # Plot 2: Confusion Matrices
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for idx, m in enumerate(models):
        cm = np.array([
            [confusion_matrices[m]["TN"], confusion_matrices[m]["FP"]],
            [confusion_matrices[m]["FN"], confusion_matrices[m]["TP"]]
        ])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{m.replace('_', ' ')}", fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Normal", "Predicted Failure"])
        ax.set_yticklabels(["Actual Normal", "Actual Failure"])
        
        # Annotate text
        thresh_val = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        color="white" if val > thresh_val else "black", fontsize=11, fontweight="bold")
    
    plt.suptitle("Confusion Matrices on Chronological Test Set (Q2 2026)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUT_PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Confusion Matrices plot to {cm_plot_path}")

    # ==========================================
    # Plot 3: Comparative Bar Chart
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bar_width = 0.22
    x = np.arange(len(models))
    
    prec_vals = res_df["Test_Precision"].values
    rec_vals = res_df["Test_Recall"].values
    f1_vals = res_df["Test_F1"].values
    
    ax.bar(x - bar_width, prec_vals, width=bar_width, label="Precision", color="#4C72B0")
    ax.bar(x, rec_vals, width=bar_width, label="Recall", color="#55A868")
    ax.bar(x + bar_width, f1_vals, width=bar_width, label="F1-Score", color="#C44E52")
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, max(max(prec_vals), max(rec_vals), max(f1_vals)) * 1.25)
    ax.set_title("Test Set Performance Comparison (Precision, Recall, F1)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Add values on top of bars
    for i in range(len(models)):
        ax.text(i - bar_width, prec_vals[i] + 0.005, f"{prec_vals[i]:.3f}", ha="center", fontsize=9)
        ax.text(i, rec_vals[i] + 0.005, f"{rec_vals[i]:.3f}", ha="center", fontsize=9)
        ax.text(i + bar_width, f1_vals[i] + 0.005, f"{f1_vals[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    bar_plot_path = os.path.join(OUTPUT_PLOTS_DIR, "model_comparison_bar.png")
    plt.savefig(bar_plot_path)
    plt.close()
    logger.info(f"Saved metric bar chart to {bar_plot_path}")

    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    return res_df


if __name__ == "__main__":
    run_evaluation()
