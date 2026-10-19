"""
Model Training Engine: LSTM, Transformer Encoder, and Isolation Forest
======================================================================
Implements multi-model training matching the user's Colab notebook:
- LSTM & Transformer trained using PyTorch CUDA with AMP on RTX 3050.
- Class weighting applied to binary cross-entropy.
- Isolation Forest fitted on 28-feature trajectory summaries (last, mean, max, slope) of normal sequences.
- Validation and test scores exported to Parquet for rigorous test evaluation.
"""

import os
import sys
import time
import json
import joblib
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score

from src.models import LSTMClassifier, TransformerClassifier, IsolationForestBaseline, summarize_windows

OUTPUT_DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", os.path.join(PROJECT_ROOT, "outputs", "data"))
OUTPUT_MODEL_DIR = os.environ.get("OUTPUT_MODEL_DIR", os.path.join(PROJECT_ROOT, "outputs", "models"))
OUTPUT_LOG_DIR = os.environ.get("OUTPUT_LOG_DIR", os.path.join(PROJECT_ROOT, "outputs", "logs"))


def setup_logger():
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_LOG_DIR, "train.log")
    
    logger = logging.getLogger("Train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train_deep_model(model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                     pos_weight: float, epochs: int = 15, lr: float = 1e-3, device: torch.device = None, logger=None):
    logger.info(f"--- Training {model_name} on {device} ---")
    model.to(device)
    
    pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    best_val_ap = -1.0
    best_weights_path = os.path.join(OUTPUT_MODEL_DIR, f"{model_name.lower()}_best.pt")
    history = {"train_loss": [], "val_ap": [], "epoch_time": []}
    
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_train_loss = total_loss / max(n_batches, 1)
        
        # Validation evaluation
        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())
                val_targets.append(batch_y.numpy())
                
        val_probs = np.concatenate(val_probs)
        val_targets = np.concatenate(val_targets)
        val_ap = average_precision_score(val_targets, val_probs) if val_targets.sum() > 0 else 0.0
        ep_duration = round(time.time() - t_ep, 2)
        
        history["train_loss"].append(avg_train_loss)
        history["val_ap"].append(val_ap)
        history["epoch_time"].append(ep_duration)
        
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Average Precision (PR-AUC): {val_ap:.4f} | Duration: {ep_duration}s")
        
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"   [*] Best checkpoint saved (Val AP: {val_ap:.4f})")
            
    # Load best weights
    model.load_state_dict(torch.load(best_weights_path))
    logger.info(f"{model_name} training complete. Best Val AP: {best_val_ap:.4f}")
    return model, history


def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs = []
    t0 = time.time()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(batch_x)
                p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy())
    latency = time.time() - t0
    return np.concatenate(probs), latency


def run_training_pipeline():
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("STARTING MULTI-MODEL TRAINING (LSTM, TRANSFORMER, ISOLATION FOREST)")
    logger.info("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # Load processed arrays
    logger.info("Loading preprocessed arrays from outputs/data/...")
    X_train = np.load(os.path.join(OUTPUT_DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(OUTPUT_DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(OUTPUT_DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(OUTPUT_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DATA_DIR, "y_test.npy"))
    
    n_features = X_train.shape[2]
    logger.info(f"Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape} with {n_features} features.")
    logger.info(f"Train Positives: {int(y_train.sum()):,} / {len(y_train):,} ({y_train.mean():.4%})")
    logger.info(f"Val Positives:   {int(y_val.sum()):,} / {len(y_val):,} ({y_val.mean():.4%})")
    logger.info(f"Test Positives:  {int(y_test.sum()):,} / {len(y_test):,} ({y_test.mean():.4%})")

    # PyTorch DataLoaders
    batch_size = 512
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Class weighting
    natural_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    relaxed_weight = min(20.0, float(natural_ratio))
    logger.info(f"Class imbalance ratio: {natural_ratio:.1f}:1. Using positive weight: {relaxed_weight:.1f}")

    predictions = {
        "val": {"y_true": y_val},
        "test": {"y_true": y_test},
        "latencies": {},
        "training_times": {}
    }

    # 1. Train LSTM
    t0_lstm = time.time()
    lstm = LSTMClassifier(input_dim=n_features, hidden_dim1=64, hidden_dim2=32, dropout=0.25)
    lstm, lstm_hist = train_deep_model("LSTM", lstm, train_loader, val_loader, pos_weight=relaxed_weight, epochs=15, lr=1e-3, device=device, logger=logger)
    predictions["training_times"]["LSTM"] = round(time.time() - t0_lstm, 2)
    
    val_lstm_scores, _ = predict_scores(lstm, val_loader, device)
    test_lstm_scores, lstm_lat = predict_scores(lstm, test_loader, device)
    predictions["val"]["LSTM"] = val_lstm_scores
    predictions["test"]["LSTM"] = test_lstm_scores
    predictions["latencies"]["LSTM"] = round(lstm_lat, 3)

    # 2. Train Transformer Encoder
    t0_tf = time.time()
    transformer = TransformerClassifier(input_dim=n_features, seq_len=30, d_model=64, nhead=4, ff_dim=128, dropout=0.2)
    transformer, tf_hist = train_deep_model("Transformer", transformer, train_loader, val_loader, pos_weight=relaxed_weight, epochs=15, lr=5e-4, device=device, logger=logger)
    predictions["training_times"]["Transformer"] = round(time.time() - t0_tf, 2)
    
    val_tf_scores, _ = predict_scores(transformer, val_loader, device)
    test_tf_scores, tf_lat = predict_scores(transformer, test_loader, device)
    predictions["val"]["Transformer"] = val_tf_scores
    predictions["test"]["Transformer"] = test_tf_scores
    predictions["latencies"]["Transformer"] = round(tf_lat, 3)

    # 3. Fit Isolation Forest Baseline
    logger.info("--- Fitting Isolation Forest Baseline (28 Trajectory Features on Normal Sequences) ---")
    t0_iso = time.time()
    X_train_summary = summarize_windows(X_train)
    X_val_summary = summarize_windows(X_val)
    X_test_summary = summarize_windows(X_test)
    
    normal_train = X_train_summary[y_train == 0]
    rng = np.random.default_rng(42)
    if len(normal_train) > 200_000:
        normal_train = normal_train[rng.choice(len(normal_train), 200_000, replace=False)]
    
    iso = IsolationForestBaseline(n_estimators=300, max_samples="auto", contamination="auto", random_state=42)
    iso.fit(normal_train)
    predictions["training_times"]["Isolation_Forest"] = round(time.time() - t0_iso, 2)
    logger.info(f"Isolation Forest fitted in {predictions['training_times']['Isolation_Forest']}s")
    
    val_iso_scores = iso.predict_anomaly_score(X_val_summary)
    t0_iso_test = time.time()
    test_iso_scores = iso.predict_anomaly_score(X_test_summary)
    iso_lat = round(time.time() - t0_iso_test, 3)
    
    predictions["val"]["Isolation_Forest"] = val_iso_scores
    predictions["test"]["Isolation_Forest"] = test_iso_scores
    predictions["latencies"]["Isolation_Forest"] = iso_lat
    
    joblib.dump(iso, os.path.join(OUTPUT_MODEL_DIR, "isolation_forest.joblib"))

    # Save predictions
    val_df = pd.DataFrame(predictions["val"])
    test_df = pd.DataFrame(predictions["test"])
    val_df.to_parquet(os.path.join(OUTPUT_DATA_DIR, "val_predictions.parquet"), index=False)
    test_df.to_parquet(os.path.join(OUTPUT_DATA_DIR, "test_predictions.parquet"), index=False)
    
    timing_info = {
        "training_times_seconds": predictions["training_times"],
        "test_inference_latency_seconds": predictions["latencies"]
    }
    with open(os.path.join(OUTPUT_DATA_DIR, "timing_benchmarks.json"), "w") as f:
        json.dump(timing_info, f, indent=2)
        
    logger.info("All model scores, checkpoints, and timing benchmarks saved successfully.")
    return predictions


if __name__ == "__main__":
    run_training_pipeline()
