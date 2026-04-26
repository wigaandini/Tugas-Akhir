# Pipeline per trial:
#     1. Load raw CSV (8 channels, 200 Hz)
#     2. Skip first 2 seconds (rest→gesture transition)
#     3. Band-pass Butterworth filter (20–95 Hz)
#     4. Notch filter (50 Hz) for power line interference
#     5. Sliding window segmentation (250 ms, stride 50 ms)
#     6. Save windows as .npz to WINDOWS_DIR

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from config import (
    SAMPLING_RATE,
    N_CHANNELS,
    REST_TRANSITION_SEC,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    BANDPASS_ORDER,
    NOTCH_FREQ,
    NOTCH_Q,
    WINDOW_SIZE,
    STRIDE_SIZE,
    GESTURE_TO_LABEL,
    METADATA_PATH,
)


# FILTER DESIGN (computed once at import time for efficiency)

def _design_bandpass_sos(
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
    fs: float = SAMPLING_RATE,
) -> np.ndarray:
    # Design a Butterworth band-pass filter in second-order-sections form
    nyquist = fs / 2.0
    return butter(order, [low / nyquist, high / nyquist],
                  btype="bandpass", output="sos")


def _design_notch_sos(
    freq: float = NOTCH_FREQ,
    q: float = NOTCH_Q,
    fs: float = SAMPLING_RATE,
) -> np.ndarray:
    # Design a notch filter and convert to SOS form
    b, a = iirnotch(freq, q, fs=fs)
    return tf2sos(b, a)


# Pre-computed filter coefficients - reused across all trials
_BANDPASS_SOS = _design_bandpass_sos()
_NOTCH_SOS = _design_notch_sos()


# LOADING AND FILTERING

def load_raw_trial(csv_path: str | Path) -> np.ndarray:
    # Load one trial CSV file.

    # Returns:
    #     Array of shape (n_samples, N_CHANNELS). The first N_CHANNELS
    #     columns are the sEMG channels. Any extra columns are ignored.

    df = pd.read_csv(csv_path, header=None)
    signal = df.iloc[:, :N_CHANNELS].to_numpy(dtype=np.float32)
    return signal


def skip_rest_transition(
    signal: np.ndarray,
    rest_sec: float = REST_TRANSITION_SEC,
    fs: float = SAMPLING_RATE,
) -> np.ndarray:
    # Discard the first `rest_sec` seconds of the trial
    n_skip = int(rest_sec * fs)
    if signal.shape[0] <= n_skip:
        # Trial too short - return empty array, caller should handle
        return np.empty((0, signal.shape[1]), dtype=signal.dtype)
    return signal[n_skip:]


def apply_bandpass(signal: np.ndarray) -> np.ndarray:
    # Apply band-pass filter along axis 0 (time), zero-phase
    return sosfiltfilt(_BANDPASS_SOS, signal, axis=0).astype(np.float32)


def apply_notch(signal: np.ndarray) -> np.ndarray:
    # Apply notch filter along axis 0 (time), zero-phase
    return sosfiltfilt(_NOTCH_SOS, signal, axis=0).astype(np.float32)


def filter_signal(signal: np.ndarray) -> np.ndarray:
    # Apply the full filter chain: band-pass then notch
    signal = apply_bandpass(signal)
    signal = apply_notch(signal)
    return signal


# SLIDING WINDOW SEGMENTATION

def sliding_window(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE_SIZE,
) -> np.ndarray:
    # Segment a (n_samples, n_channels) signal into overlapping windows.

    # Returns:
    #     Array of shape (n_windows, n_channels, window_size).
    #     Format (n_channels, window_size) per window matches the image
    #     shape (8, 50) used by CNN/SCNN. Note: channels-first layout.
     
    n_samples = signal.shape[0]
    if n_samples < window_size:
        # Not enough samples for even one window
        return np.empty((0, signal.shape[1], window_size), dtype=signal.dtype)

    n_windows = (n_samples - window_size) // stride + 1
    # Shape: (n_windows, window_size, n_channels)
    windows = np.stack(
        [signal[i * stride:i * stride + window_size] for i in range(n_windows)],
        axis=0,
    )
    # Transpose to (n_windows, n_channels, window_size) - channels-first
    windows = windows.transpose(0, 2, 1)
    return windows.astype(np.float32)


# PER-TRIAL PIPELINE

def preprocess_trial(
    csv_path: str | Path,
    gesture_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Run the full preprocessing pipeline on a single trial file.

    # Returns:
    #     windows: array (n_windows, 8, 50), dtype float32
    #     labels: array (n_windows,), dtype int64 - all equal to gesture_label

    signal = load_raw_trial(csv_path)
    signal = skip_rest_transition(signal)

    if signal.shape[0] < WINDOW_SIZE:
        return (
            np.empty((0, N_CHANNELS, WINDOW_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    signal = filter_signal(signal)
    windows = sliding_window(signal)
    labels = np.full(windows.shape[0], gesture_label, dtype=np.int64)
    return windows, labels


def save_trial_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    # Save preprocessed windows and labels to a compressed .npz file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, windows=windows, labels=labels)


def load_trial_windows(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    # Load preprocessed windows and labels from a .npz file
    data = np.load(npz_path)
    return data["windows"], data["labels"]


# BATCH PROCESSING: APPLY TO ENTIRE METADATA

def process_all_trials(
    metadata: Optional[pd.DataFrame] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    # Run preprocessing for every trial listed in the metadata, saving each trial's windows to its `window_file_path`.

    # Updates the `n_windows` column with the actual number of windows produced per trial and re-saves the metadata CSV.

    # Args:
    #     metadata: DataFrame to process. If None, loads from METADATA_PATH.
    #     overwrite: If False, skips trials whose output file already exists.
    #     verbose: If True, prints progress every 500 trials.

    # Returns:
    #     Updated metadata DataFrame with filled `n_windows` column.

    if metadata is None:
        metadata = pd.read_csv(METADATA_PATH)

    metadata = metadata.copy()
    n_total = len(metadata)
    n_processed = 0
    n_skipped = 0
    n_empty = 0
    failures = []

    for idx, row in metadata.iterrows():
        output_path = Path(row["window_file_path"])

        # Skip if already processed and not overwriting
        if not overwrite and output_path.exists():
            try:
                data = np.load(output_path)
                metadata.at[idx, "n_windows"] = int(data["windows"].shape[0])
                n_skipped += 1
                continue
            except Exception:
                # Corrupted file - reprocess
                pass

        try:
            windows, labels = preprocess_trial(
                csv_path=row["raw_file_path"],
                gesture_label=int(row["gesture_label"]),
            )
        except Exception as exc:
            failures.append((row["raw_file_path"], str(exc)))
            continue

        if windows.shape[0] == 0:
            n_empty += 1
            metadata.at[idx, "n_windows"] = 0
            continue

        save_trial_windows(windows, labels, output_path)
        metadata.at[idx, "n_windows"] = int(windows.shape[0])
        n_processed += 1

        if verbose and (n_processed + n_skipped) % 500 == 0:
            print(f"  [{n_processed + n_skipped}/{n_total}] "
                  f"processed={n_processed} skipped={n_skipped} empty={n_empty}")

    # Save updated metadata (with n_windows filled in)
    metadata.to_csv(METADATA_PATH, index=False)

    if verbose:
        print()
        print("=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total trials in metadata: {n_total}")
        print(f"Newly processed: {n_processed}")
        print(f"Skipped (already exists): {n_skipped}")
        print(f"Empty (trial too short): {n_empty}")
        print(f"Failures: {len(failures)}")
        if failures:
            print("First 5 failures:")
            for path, err in failures[:5]:
                print(f"  - {path}: {err}")
        total_windows = int(metadata["n_windows"].sum())
        print(f"Total windows produced: {total_windows:,}")
        print("=" * 60)

    return metadata


if __name__ == "__main__":
    process_all_trials(overwrite=False, verbose=True)