"""
Model Definitions: LSTM, Transformer Encoder, and Isolation Forest
===================================================================
1. LSTMClassifier: 2-layer stacked LSTM (64 -> 32) matching the user's architecture.
2. TransformerClassifier: Multi-head attention encoder (d_model=64, 4 heads, ff_dim=128) with learned position embeddings.
3. IsolationForestBaseline: Trajectory summary representation (last, mean, max, slope) with decision_function anomaly scoring.
"""

import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import numpy as np


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.25):
        super().__init__()
        # Layer 1: LSTM with return_sequences=True
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        # Layer 2: LSTM with return_sequences=False
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True
        )
        # Dense head
        self.dense = nn.Linear(hidden_dim2, 32)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, 7)
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)
        out2, (hn, cn) = self.lstm2(out1)
        # Last time step of lstm2
        h_last = out2[:, -1, :]
        d = self.drop2(self.relu(self.dense(h_last)))
        logits = self.out(d)
        return logits.squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int = 7, seq_len: int = 30, d_model: int = 64, nhead: int = 4, ff_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, 7)
        h = self.input_proj(x) + self.pos_embedding
        h_enc = self.encoder(h)
        # Global Average Pooling across the 30 days
        pooled = torch.mean(h_enc, dim=1)
        logits = self.head(pooled)
        return logits.squeeze(-1)


def summarize_windows(arr: np.ndarray) -> np.ndarray:
    """Summarizes 30-day sequence windows into trajectory features: [last, mean, max, slope]."""
    arr = np.asarray(arr, dtype=np.float32)
    last = arr[:, -1, :]
    mean = arr.mean(axis=1)
    maximum = arr.max(axis=1)
    slope = (arr[:, -1, :] - arr[:, 0, :]) / max(arr.shape[1] - 1, 1)
    return np.concatenate([last, mean, maximum, slope], axis=1)


class IsolationForestBaseline:
    def __init__(self, n_estimators: int = 300, max_samples: str = "auto", contamination: str = "auto", random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_summary_normal: np.ndarray):
        self.model.fit(X_summary_normal)

    def predict_anomaly_score(self, X_summary: np.ndarray) -> np.ndarray:
        # decision_function gives negative values for anomalies, positive for normal
        # -decision_function gives higher scores for anomalies (failure risk)
        return -self.model.decision_function(X_summary)
