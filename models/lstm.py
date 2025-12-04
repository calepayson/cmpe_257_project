from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


_STEP_PATTERN = re.compile(r"_t-(\d+)$")


@dataclass
class _SequenceStructure:
    ordered_columns: List[str]
    steps: List[int]
    features_per_step: int


class _LSTMForecaster(nn.Module):
    """Simple LSTM head that returns the prediction for the final step."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output, _ = self.lstm(x)
        final_step = output[:, -1, :]
        return self.regressor(final_step).squeeze(-1)


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-style wrapper around a PyTorch LSTM for sequence regression."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        if torch is None:  # pragma: no cover - executed only when torch missing
            raise ImportError(
                "LSTMRegressor requires the 'torch' package. Install it via pip install torch."
            )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self._structure: _SequenceStructure | None = None
        self._model: _LSTMForecaster | None = None

    def _infer_structure(self, columns: Sequence[str]) -> _SequenceStructure:
        step_map: Dict[int, List[str]] = {}
        for col in columns:
            match = _STEP_PATTERN.search(col)
            if not match:
                raise ValueError(
                    "LSTMRegressor expects feature columns to follow the '<feature>_t-<lag>' naming pattern."
                )
            step = int(match.group(1))
            step_map.setdefault(step, []).append(col)

        steps = sorted(step_map.keys(), reverse=True)  # oldest -> newest
        ordered_columns: List[str] = []
        features_per_step = None
        for step in steps:
            cols = sorted(step_map[step])
            if features_per_step is None:
                features_per_step = len(cols)
            elif len(cols) != features_per_step:
                raise ValueError("Uneven feature counts per time step detected; cannot reshape into sequences.")
            ordered_columns.extend(cols)

        if features_per_step is None:
            raise ValueError("Could not infer feature structure for LSTM input.")

        return _SequenceStructure(ordered_columns, steps, features_per_step)

    def _prepare_sequences(self, X: pd.DataFrame, fit_structure: bool = False) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LSTMRegressor expects a pandas DataFrame as input.")

        if fit_structure or self._structure is None:
            self._structure = self._infer_structure(X.columns)

        missing = set(self._structure.ordered_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing expected columns needed for LSTM inference: {sorted(missing)}")

        values = X[self._structure.ordered_columns].to_numpy(dtype=np.float32)
        seq_len = len(self._structure.steps)
        features_per_step = self._structure.features_per_step
        return values.reshape(len(X), seq_len, features_per_step)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LSTMRegressor":  # type: ignore[override]
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting LSTM training: epochs={self.epochs}, batch_size={self.batch_size}, device={self.device}")
        
        X_seq = self._prepare_sequences(X, fit_structure=True)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)

        dataset = TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_array),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = self._structure.features_per_step  # type: ignore[union-attr]
        self._model = _LSTMForecaster(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self._model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            avg_loss = epoch_loss / len(loader.dataset)
            logger.info(f"LSTM Epoch {epoch+1}/{self.epochs} completed - Avg Loss: {avg_loss:.6f}")

        self._model.eval()
        logger.info("LSTM training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        if self._model is None or self._structure is None:
            raise RuntimeError("The model must be fitted before calling predict().")

        X_seq = self._prepare_sequences(X)
        tensor = torch.from_numpy(X_seq).to(self.device)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(tensor).cpu().numpy()
        return preds