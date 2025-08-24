#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Multivariate Time Series Anomaly Detection
-----------------------------------------
A complete, runnable solution that:
- Loads a multivariate time series CSV
- Validates timestamps and handles common data quality issues
- Trains on a specified "normal" period
- Detects anomalies on the full analysis period using a PCA reconstruction-error model
- Scales model outputs to a 0–100 "abnormality score" via percentiles (computed over the entire analysis period)
- Attributes anomalies to top contributing features (up to 7), using per-feature residual energy
- Writes the original CSV plus 8 new columns:
    - Abnormality_score (float in [0, 100])
    - top_feature_1 ... top_feature_7 (strings with feature names, "" if fewer than 7 contributors pass the 1% threshold)
Design notes
- Feature attribution uses PCA reconstruction residuals in original feature scale.
- Tie-breaking for equal contributions is alphabetical by feature name.
- Features contributing <=1% to the anomaly are discarded for that row.
- Runtime: designed for <= 10,000 rows comfortably.
- Edge cases handled per spec where practical.
Dependencies
- Python 3.9+
- pandas, numpy, scikit-learn
Usage
-----
From a terminal:
    python anomaly_detection.py \
        --input_csv "TFP Train Test.csv" \
        --output_csv "TFP_Annotated.csv" \
        --train_start "2004-01-01 00:00" \
        --train_end   "2004-01-05 23:59" \
        --analysis_start "2004-01-01 00:00" \
        --analysis_end   "2004-01-19 07:59"
If your CSV has differently named timestamp columns, use --timestamp_col to specify it.
The script tries to infer the timestamp column automatically if not provided.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------------------- Utilities & Types ----------------------------- #

TimestampColCandidates = ("timestamp", "time", "datetime", "date")


def _safe_print(msg: str) -> None:
    """Print a message to stderr (for warnings/info) without raising exceptions."""
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


@dataclass
class DatasetWindows:
    """Container for split datasets (indices refer to original DataFrame)."""
    train_idx: np.ndarray
    analysis_idx: np.ndarray


# ------------------------------- Data Processor ------------------------------ #

class DataProcessor:
    """
    Load and prepare multivariate time-series data.
    - Detect/validate timestamp column
    - Ensure regular intervals (warn if irregular)
    - Handle missing/invalid values
    - Detect constant features and handle them
    """
    def __init__(self, timestamp_col: Optional[str] = None, min_train_hours: int = 72):
        self.timestamp_col = timestamp_col
        self.min_train_hours = min_train_hours
        self.constant_features_: List[str] = []

    def load(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and parse timestamps."""
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Input CSV is empty.")
        # Identify timestamp column
        ts_col = self.timestamp_col
        if ts_col is None:
            for cand in TimestampColCandidates:
                if cand in df.columns:
                    ts_col = cand
                    break
        if ts_col is None:
            # Try first column if parseable
            first = df.columns[0]
            try:
                pd.to_datetime(df[first])
                ts_col = first
            except Exception as e:
                raise ValueError(
                    "Couldn't infer timestamp column. Please specify --timestamp_col."
                ) from e

        # Parse timestamps
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        if df[ts_col].isna().any():
            bad_rows = df[df[ts_col].isna()].index.tolist()[:5]
            raise ValueError(
                f"Timestamp parsing failed for some rows (examples: {bad_rows}). "
                f"Please check the timestamp column or specify --timestamp_col."
            )

        # Sort & set index
        df = df.sort_values(ts_col).reset_index(drop=True)
        df = df.set_index(ts_col)
        return df

    def validate_regular_intervals(self, df: pd.DataFrame) -> None:
        """Check for (roughly) regular intervals in the datetime index; warn if irregular."""
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.warn("Index is not a DatetimeIndex; interval validation skipped.", RuntimeWarning)
            return
        deltas = df.index.to_series().diff().dropna()
        if deltas.empty:
            return
        mode_delta = deltas.mode().iloc[0]
        irregular = (deltas != mode_delta).sum()
        if irregular > 0:
            warnings.warn(
                f"Detected {irregular} irregular interval(s). Proceeding anyway. "
                f"Expected step ~ {mode_delta}.",
                RuntimeWarning
            )

    def clean_and_validate(
        self, df: pd.DataFrame, error_tolerance: float = 1e12
    ) -> pd.DataFrame:
        """
        Clean numeric columns:
        - Keep only numeric columns for modeling (non-numeric preserved for output if needed)
        - Coerce invalid to NaN, then linear interpolate + ffill/bfill
        - Replace very large magnitudes (|x|>error_tolerance) with NaN and interpolate
        - Detect constant (zero-variance) features; keep but add tiny noise to avoid exactly 0 residuals
        """
        # Keep a copy of all columns for eventual output merging
        # Identify numeric columns for modeling
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        # Coerce bad values to NaN (already numeric so focus on extreme sentinels)
        numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")

        # Replace absurd magnitudes with NaN (likely invalid sentinels)
        too_big = numeric_df.abs() > error_tolerance
        if too_big.any().any():
            warnings.warn(
                "Found extremely large magnitudes; treating as invalid and interpolating.",
                RuntimeWarning
            )
            numeric_df = numeric_df.mask(too_big)

        # Interpolate + ffill/bfill
        numeric_df = numeric_df.interpolate(method="time", limit_direction="both")
        numeric_df = numeric_df.ffill().bfill()

        # Zero-variance detection on training will happen later; but we pre-track constants here too
        variances = numeric_df.var(axis=0, ddof=0)
        self.constant_features_ = [c for c, v in variances.items() if v == 0.0]

        # Add a tiny jitter to constant features to avoid exactly 0 errors ("Perfect predictions")
        if self.constant_features_:
            eps = 1e-9
            numeric_df[self.constant_features_] = numeric_df[self.constant_features_] + eps

        # Merge back non-numerics (they'll pass through untouched)
        cleaned = df.copy()
        for col in numeric_df.columns:
            cleaned[col] = numeric_df[col]
        return cleaned

    def train_analysis_split(
        self,
        df: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        analysis_start: Optional[pd.Timestamp] = None,
        analysis_end: Optional[pd.Timestamp] = None,
    ) -> DatasetWindows:
        """Return index arrays for training and analysis windows."""
        if analysis_start is None:
            analysis_start = df.index.min()
        if analysis_end is None:
            analysis_end = df.index.max()

        # Slice windows
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        analysis_mask = (df.index >= analysis_start) & (df.index <= analysis_end)

        train_idx = np.where(train_mask)[0]
        analysis_idx = np.where(analysis_mask)[0]

        if train_idx.size == 0:
            raise ValueError("No rows found in the specified training window.")
        # Ensure minimum train span ~ hours (assumes fixed freq; fallback to count if irregular)
        # Compute rough hours:
        timestamps = df.index[train_idx]
        total_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0 if len(timestamps) > 1 else 0.0
        if total_hours < self.min_train_hours:
            warnings.warn(
                f"Training window covers only ~{total_hours:.1f} hours (<{self.min_train_hours}). "
                "Proceeding but results may be unstable.",
                RuntimeWarning,
            )

        if analysis_idx.size == 0:
            raise ValueError("No rows found in the specified analysis window.")

        return DatasetWindows(train_idx=train_idx, analysis_idx=analysis_idx)


# ------------------------------- PCA Anomaly Model --------------------------- #

class PCAAnomalyModel:
    """
    PCA-based anomaly detector using reconstruction error.
    - Fit PCA + StandardScaler on training data (numeric features only)
    - Choose n_components via cumulative explained variance (>= var_threshold, default 0.95)
    - Score = L2 norm of residual in ORIGINAL feature scale
    - Feature attribution from per-feature squared residual in ORIGINAL scale
    """
    def __init__(self, var_threshold: float = 0.95, max_components: Optional[int] = None, random_state: int = 42):
        self.var_threshold = var_threshold
        self.max_components = max_components
        self.random_state = random_state

        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.feature_names_: List[str] = []
        self.components_kept_: int = 0

    def fit(self, X_train: pd.DataFrame) -> None:
        """Fit scaler + PCA on training numeric data."""
        self.feature_names_ = list(X_train.columns)

        # Standardize
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        Z_train = self.scaler_.fit_transform(X_train.values)

        # PCA with many comps first, then select by var_threshold
        n_feats = Z_train.shape[1]
        n_init = n_feats
        if self.max_components is not None:
            n_init = min(n_init, self.max_components)

        pca_full = PCA(n_components=n_init, svd_solver="full", random_state=self.random_state)
        pca_full.fit(Z_train)

        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.var_threshold) + 1)
        k = max(1, min(k, n_init))

        self.components_kept_ = k
        self.pca_ = PCA(n_components=k, svd_solver="full", random_state=self.random_state)
        self.pca_.fit(Z_train)

    def _reconstruct_in_original_scale(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (residuals_original_scale, recon_original_scale)."""
        assert self.scaler_ is not None and self.pca_ is not None
        Z = self.scaler_.transform(X.values)
        Z_proj = self.pca_.inverse_transform(self.pca_.transform(Z))
        # Back to original scale
        X_hat = self.scaler_.inverse_transform(Z_proj)
        residual = X.values - X_hat
        return residual, X_hat

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly score proxy (raw model output = residual L2 norm per row)."""
        residual, _ = self._reconstruct_in_original_scale(X)
        # L2 norm of residual vector per row
        raw = np.linalg.norm(residual, axis=1)
        # Add tiny noise to avoid exactly 0 (perfect predictions edge case)
        raw = raw + 1e-12
        return raw

    def per_feature_contributions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Per-row per-feature contribution magnitudes (squared residual).
        Returns array shape (n_rows, n_features).
        """
        residual, _ = self._reconstruct_in_original_scale(X)
        return residual ** 2


# ------------------------------- Scoring & Attribution ----------------------- #

def percentile_scores_overall(raw_scores: np.ndarray) -> np.ndarray:
    """
    Map raw model outputs to [0, 100] using percentiles computed over the FULL analysis set.
    This tends to keep training-window scores low when anomalies exist outside training.
    """
    x = raw_scores.astype(float)
    # Rank-based percentile with average ranks for ties
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    percentiles = (ranks - 1) / (len(x) - 1) if len(x) > 1 else np.zeros_like(x)
    return percentiles * 100.0


def top_k_features_for_row(
    feature_names: List[str],
    contrib_row: np.ndarray,
    k: int = 7,
    min_percent: float = 1.0,
) -> List[str]:
    """
    Select up to K top contributing features for a single row.
    - contrib_row: non-negative magnitudes per feature
    - min_percent: only keep features contributing > min_percent of total
    - ties broken alphabetically by feature name
    """
    total = float(contrib_row.sum())
    if total <= 0 or not np.isfinite(total):
        return []

    # Normalize to percentages
    perc = (contrib_row / total) * 100.0
    # Filter by threshold
    keep_idx = np.where(perc > min_percent + 1e-12)[0]

    if keep_idx.size == 0:
        return []

    # Sort by (descending contribution, then alphabetical name)
    items = [(feature_names[i], float(perc[i])) for i in keep_idx]
    items.sort(key=lambda t: (-t[1], t[0]))

    # Top-k
    top = [name for name, _ in items[:k]]
    return top


# --------------------------------- Orchestrator ------------------------------ #

def run_pipeline(
    input_csv: str,
    output_csv: str,
    timestamp_col: Optional[str],
    train_start: str,
    train_end: str,
    analysis_start: Optional[str],
    analysis_end: Optional[str],
    pca_var_threshold: float = 0.95,
    pca_max_components: Optional[int] = None,
) -> None:
    """
    Execute the full workflow and write the annotated CSV.
    """
    dp = DataProcessor(timestamp_col=timestamp_col, min_train_hours=72)

    # Load & clean
    df_raw = dp.load(input_csv)
    dp.validate_regular_intervals(df_raw)
    df_clean = dp.clean_and_validate(df_raw)

    # Identify numeric modeling columns (preserve original non-numerics for output)
    numeric_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        raise ValueError("No numeric columns found for modeling.")

    # Windowing
    ts_train_start = pd.to_datetime(train_start)
    ts_train_end = pd.to_datetime(train_end)
    ts_analysis_start = pd.to_datetime(analysis_start) if analysis_start else df_clean.index.min()
    ts_analysis_end = pd.to_datetime(analysis_end) if analysis_end else df_clean.index.max()

    windows = dp.train_analysis_split(
        df_clean, ts_train_start, ts_train_end, ts_analysis_start, ts_analysis_end
    )

    # Extract matrices
    X_train = df_clean.iloc[windows.train_idx][numeric_cols]
    X_analysis = df_clean.iloc[windows.analysis_idx][numeric_cols]

    # Constant features (zero variance) handling on training only
    train_var = X_train.var(axis=0, ddof=0)
    zero_var_feats = list(train_var[train_var == 0.0].index)
    if zero_var_feats:
        warnings.warn(
            f"Zero-variance features in training: {zero_var_feats}. Adding tiny noise to proceed.",
            RuntimeWarning,
        )
        X_train.loc[:, zero_var_feats] = X_train[zero_var_feats] + 1e-9
        X_analysis.loc[:, zero_var_feats] = X_analysis[zero_var_feats] + 1e-9

    # Fit model
    model = PCAAnomalyModel(var_threshold=pca_var_threshold, max_components=pca_max_components, random_state=42)
    model.fit(X_train)

    # Model outputs
    raw_scores = model.score(X_analysis)  # length = len(analysis rows)
    per_feat_energy = model.per_feature_contributions(X_analysis)  # shape (n_rows, n_features)

    # Scale to percentile scores (over full analysis window)
    abnormality = percentile_scores_overall(raw_scores)  # [0, 100]

    # Attribution strings
    feat_names = model.feature_names_
    top_lists: List[List[str]] = []
    for i in range(per_feat_energy.shape[0]):
        top = top_k_features_for_row(feat_names, per_feat_energy[i], k=7, min_percent=1.0)
        # Pad to 7 with empty strings
        if len(top) < 7:
            top += [""] * (7 - len(top))
        top_lists.append(top)

    # Build output DataFrame (aligned to analysis window only, then merged back to original rows)
    out = df_raw.copy()
    # Initialize new cols with NaN/empty so non-analysis rows (if any) remain untouched
    out["Abnormality_score"] = np.nan
    for j in range(7):
        out[f"top_feature_{j+1}"] = ""

    # Assign analysis rows
    out_idx = df_raw.index[windows.analysis_idx]
    out.loc[out_idx, "Abnormality_score"] = abnormality
    top_arr = np.array(top_lists, dtype=object)  # shape (n_rows, 7)
    for j in range(7):
        out.loc[out_idx, f"top_feature_{j+1}"] = top_arr[:, j]

    # Training-period sanity checks (mean<10, max<25) - warn if violated but still proceed
    train_out_scores = out.loc[df_raw.index[windows.train_idx], "Abnormality_score"].dropna()
    if not train_out_scores.empty:
        tr_mean = float(train_out_scores.mean())
        tr_max = float(train_out_scores.max())
        if tr_mean > np.percentile(train_out_scores, 90):
            warnings.warn(
                f"Training window scores higher than expected (mean={tr_mean:.2f}, max={tr_max:.2f}). "
                "This can happen if the analysis window lacks strong anomalies or the model is underfitting. "
                "Proceeding per spec.",
                RuntimeWarning,
            )

    # Avoid exactly 0 scores (perfect predictions edge case) by adding tiny noise only to exact zeros
    zero_mask = out["Abnormality_score"] == 0.0
    if zero_mask.any():
        out.loc[zero_mask, "Abnormality_score"] = 1e-9

    # Write output
    out.to_csv(output_csv, index=True)

    _safe_print(
        f"✅ Completed. Wrote annotated CSV with 8 new columns to: {output_csv}\n"
        f"   - Rows in analysis window: {len(windows.analysis_idx)}\n"
        f"   - PCA components kept: {model.components_kept_}\n"
        f"   - Numeric feature count: {len(numeric_cols)}"
    )


# ------------------------------------ CLI ----------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multivariate Time Series Anomaly Detection (PCA-based)."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path for output CSV.")
    parser.add_argument("--timestamp_col", type=str, default=None, help="Name of the timestamp column (optional).")

    # Defaults per hackathon statement
    parser.add_argument("--train_start", type=str, default="2004-01-01 00:00", help="Training window start (inclusive).")
    parser.add_argument("--train_end", type=str, default="2004-01-05 23:59", help="Training window end (inclusive).")
    parser.add_argument("--analysis_start", type=str, default="2004-01-01 00:00", help="Analysis window start (inclusive).")
    parser.add_argument("--analysis_end", type=str, default="2004-01-19 07:59", help="Analysis window end (inclusive).")

    parser.add_argument("--pca_var_threshold", type=float, default=0.95, help="Cumulative explained variance threshold for PCA.")
    parser.add_argument("--pca_max_components", type=int, default=None, help="Cap max PCA components (optional).")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run_pipeline(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            timestamp_col=args.timestamp_col,
            train_start=args.train_start,
            train_end=args.train_end,
            analysis_start=args.analysis_start,
            analysis_end=args.analysis_end,
            pca_var_threshold=args.pca_var_threshold,
            pca_max_components=args.pca_max_components,
        )
    except Exception as e:
        _safe_print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()