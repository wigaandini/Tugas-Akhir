import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import (
    METADATA_PATH, N_CHANNELS, WINDOW_SIZE, RANDOM_SEED,
    SUBJECTS_INTERDAY_LONG, SUBJECTS_INTERDAY_SHORT,
    SUBJECTS_NON_FATIGUE, SUBJECTS_FATIGUE, SUBJECTS_ALL,
)
from src.preprocessing import load_trial_windows


# Loading helpers

def load_metadata(valid_only=True):
    df = pd.read_csv(METADATA_PATH)
    if valid_only and "is_valid" in df.columns:
        df = df[df["is_valid"]].reset_index(drop=True)
    return df


def load_windows_from_metadata(df, verbose=True):
    # Load all .npz files referenced by the metadata rows, return stacked arrays
    all_windows, all_labels = [], []
    iterator = tqdm(df.iterrows(), total=len(df), disable=not verbose, desc="Loading windows")

    for _, row in iterator:
        try:
            w, l = load_trial_windows(row["window_file_path"])
            all_windows.append(w)
            all_labels.append(l)
        except Exception:
            continue

    X = np.concatenate(all_windows, axis=0)  # (N, 8, 50)
    y = np.concatenate(all_labels, axis=0)    # (N,)
    return X, y


# Z-score normalization 

def compute_normalization_stats(X):
    mean = np.nanmean(X, axis=(0, 2), keepdims=True)  # (1, 8, 1)
    std = np.nanstd(X, axis=(0, 2), keepdims=True)    # (1, 8, 1)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_normalization(X, mean, std):
    return np.nan_to_num((X - mean) / std, nan=0.0)


def normalize_splits(X_train, X_test):
    # Fit on train only, apply to both
    mean, std = compute_normalization_stats(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_test = apply_normalization(X_test, mean, std)
    return X_train, X_test, mean, std


# Scenario splits

def scenario_1_ideal():
    # All 36 subjects, session 0, position 0 only
    # Train: rep 0-1, Test: rep 2
    meta = load_metadata()
    subset = meta[(meta["session"] == 0) & (meta["position"] == 0)]

    train_df = subset[subset["repetition"].isin([0, 1])]
    test_df = subset[subset["repetition"] == 2]

    X_train, y_train = load_windows_from_metadata(train_df)
    X_test, y_test = load_windows_from_metadata(test_df)
    X_train, X_test, mean, std = normalize_splits(X_train, X_test)

    return X_train, y_train, X_test, y_test, {"mean": mean, "std": std}


def scenario_2_electrode_shift():
    # h0-h29 (non-fatigue), session 0 only
    # Train: position 0, all reps — Test: position 1-10, all reps
    meta = load_metadata()
    subset = meta[(meta["session"] == 0) & (meta["subject"].isin(SUBJECTS_NON_FATIGUE))]

    train_df = subset[subset["position"] == 0]
    test_df = subset[subset["position"] > 0]

    X_train, y_train = load_windows_from_metadata(train_df)
    X_test, y_test = load_windows_from_metadata(test_df)
    X_train, X_test, mean, std = normalize_splits(X_train, X_test)

    # Also return test metadata for per-position analysis
    return X_train, y_train, X_test, y_test, {"mean": mean, "std": std, "test_meta": test_df}


def scenario_2_per_position():
    # Same as scenario 2 but returns test data split by position
    # for plotting accuracy-vs-shift curves
    meta = load_metadata()
    subset = meta[(meta["session"] == 0) & (meta["subject"].isin(SUBJECTS_NON_FATIGUE))]

    train_df = subset[subset["position"] == 0]
    X_train, y_train = load_windows_from_metadata(train_df)
    mean, std = compute_normalization_stats(X_train)
    X_train = apply_normalization(X_train, mean, std)

    test_by_position = {}
    for pos in range(1, 11):
        pos_df = subset[subset["position"] == pos]
        if len(pos_df) == 0:
            continue
        X_pos, y_pos = load_windows_from_metadata(pos_df, verbose=False)
        X_pos = apply_normalization(X_pos, mean, std)
        test_by_position[pos] = (X_pos, y_pos)

    return X_train, y_train, test_by_position, {"mean": mean, "std": std}


def scenario_3_inter_subject(test_subjects=None, n_test=6):
    # h0-h29, session 0, all positions
    # Split at subject level
    meta = load_metadata()
    subset = meta[(meta["session"] == 0) & (meta["subject"].isin(SUBJECTS_NON_FATIGUE))]

    subjects = sorted(subset["subject"].unique())
    if test_subjects is None:
        rng = np.random.RandomState(RANDOM_SEED)
        test_subjects = list(rng.choice(subjects, size=n_test, replace=False))

    train_subjects = [s for s in subjects if s not in test_subjects]

    train_df = subset[subset["subject"].isin(train_subjects)]
    test_df = subset[subset["subject"].isin(test_subjects)]

    X_train, y_train = load_windows_from_metadata(train_df)
    X_test, y_test = load_windows_from_metadata(test_df)
    X_train, X_test, mean, std = normalize_splits(X_train, X_test)

    return X_train, y_train, X_test, y_test, {
        "mean": mean, "std": std,
        "train_subjects": train_subjects, "test_subjects": test_subjects,
    }


def scenario_4_interday(subjects=None, train_session=0):
    # h0-h5 by default, position 0 only
    # Train: session 0, Test: all other sessions
    if subjects is None:
        subjects = SUBJECTS_INTERDAY_LONG

    meta = load_metadata()
    subset = meta[(meta["subject"].isin(subjects)) & (meta["position"] == 0)]

    train_df = subset[subset["session"] == train_session]
    test_df = subset[subset["session"] != train_session]

    X_train, y_train = load_windows_from_metadata(train_df)
    mean, std = compute_normalization_stats(X_train)
    X_train = apply_normalization(X_train, mean, std)

    # Return test split by session for curve plotting
    test_by_session = {}
    for sess in sorted(test_df["session"].unique()):
        sess_df = test_df[test_df["session"] == sess]
        X_s, y_s = load_windows_from_metadata(sess_df, verbose=False)
        X_s = apply_normalization(X_s, mean, std)
        test_by_session[sess] = (X_s, y_s)

    return X_train, y_train, test_by_session, {"mean": mean, "std": std}


def scenario_5_fatigue():
    # h30-h35 only, 1 session
    # Train: position 0-1 (pre-fatigue), Test: position 2-10 (increasing fatigue)
    meta = load_metadata()
    subset = meta[meta["subject"].isin(SUBJECTS_FATIGUE)]

    train_df = subset[subset["position"].isin([0, 1])]
    test_df = subset[subset["position"] >= 2]

    X_train, y_train = load_windows_from_metadata(train_df)
    mean, std = compute_normalization_stats(X_train)
    X_train = apply_normalization(X_train, mean, std)

    # Return test split by position for fatigue curve
    test_by_position = {}
    for pos in sorted(test_df["position"].unique()):
        pos_df = test_df[test_df["position"] == pos]
        X_p, y_p = load_windows_from_metadata(pos_df, verbose=False)
        X_p = apply_normalization(X_p, mean, std)
        test_by_position[pos] = (X_p, y_p)

    return X_train, y_train, test_by_position, {"mean": mean, "std": std}


def scenario_6_validation_mandiri(mandiri_X, mandiri_y):
    # Train on ALL valid SeNic data, test on external recording
    meta = load_metadata()
    X_train, y_train = load_windows_from_metadata(meta)
    mean, std = compute_normalization_stats(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_test = apply_normalization(mandiri_X, mean, std)

    return X_train, y_train, X_test, mandiri_y, {"mean": mean, "std": std}