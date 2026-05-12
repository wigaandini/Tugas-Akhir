import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt

from config import (
    RANDOM_SEED, N_CLASSES, FIGURES_DIR,
    SUBJECTS_NON_FATIGUE, SUBJECTS_FATIGUE, SUBJECTS_INTERDAY_LONG,
)
from src.data_splitter import (
    load_metadata, load_windows_from_metadata,
    compute_normalization_stats, apply_normalization,
)
from src.feature_extraction import extract_features_batch, fht_envelope_batch
from src.evaluation import print_report, plot_confusion_matrix


META = load_metadata()

# ============================================================
# FIXED TRAIN/TEST SPLIT — same across all notebooks
# ============================================================
RNG = np.random.RandomState(RANDOM_SEED)
ALL_NON_FATIGUE = sorted(META[META["subject"].isin(SUBJECTS_NON_FATIGUE)]["subject"].unique())
TEST_SUBJECTS = list(RNG.choice(ALL_NON_FATIGUE, size=6, replace=False))
TRAIN_SUBJECTS = [s for s in ALL_NON_FATIGUE if s not in TEST_SUBJECTS]


def get_splits():
    """Return all scenario splits. Zero-shot: test subjects completely excluded from training."""
    import pandas as pd
    non_fat = META[META["subject"].isin(SUBJECTS_NON_FATIGUE)]

    # Interday subjects in training set — their sessions 1-9 are reserved for S4 test
    interday_subjects = [s for s in SUBJECTS_INTERDAY_LONG if s in TRAIN_SUBJECTS]
    non_interday_train = [s for s in TRAIN_SUBJECTS if s not in interday_subjects]

    # TRAINING SET: maximize diversity without leaking S4 test data
    # - Interday-long train subjects (h0,h1,h2,h4,h5): session 0 only (sessions 1-9 = S4 test)
    # - All other train subjects: ALL available sessions
    train_interday = non_fat[
        (non_fat["subject"].isin(interday_subjects)) & (non_fat["session"] == 0)
    ]
    train_others = non_fat[non_fat["subject"].isin(non_interday_train)]
    train_df = pd.concat([train_interday, train_others])

    # TEST SETS
    test_all = non_fat[(non_fat["subject"].isin(TEST_SUBJECTS)) & (non_fat["session"] == 0)]
    s1_test = test_all[test_all["position"] == 0]
    s2_test = test_all[test_all["position"] > 0]
    s3_test = test_all

    # S4: inter-day
    s4_test_df = META[
        (META["subject"].isin(interday_subjects)) & (META["position"] == 0) & (META["session"] > 0)
    ]

    # S5: fatigue
    fat_df = META[META["subject"].isin(SUBJECTS_FATIGUE)]
    s5_train = fat_df[fat_df["position"].isin([0, 1])]
    s5_test = fat_df[fat_df["position"] >= 2]

    return {
        "train_df": train_df,
        "s1_test": s1_test,
        "s2_test": s2_test,
        "s3_test": s3_test,
        "s4_test": s4_test_df,
        "s5_train": s5_train,
        "s5_test": s5_test,
        "test_subjects": TEST_SUBJECTS,
        "train_subjects": TRAIN_SUBJECTS,
        "interday_subjects": interday_subjects,
    }


def load_and_norm(df, stats=None, verbose=False):
    X, y = load_windows_from_metadata(df, verbose=verbose)
    if stats is None:
        stats = compute_normalization_stats(X)
    X = apply_normalization(X, stats[0], stats[1])
    return X, y, stats


# ============================================================
# CALIBRATION SPLITS (Option A)
# ============================================================
def split_cal_test(df, cal_reps=[0, 1]):
    cal = df[df["repetition"].isin(cal_reps)]
    test = df[~df["repetition"].isin(cal_reps)]
    return cal, test


# ============================================================
# EVALUATION RUNNER
# ============================================================
def run_zero_shot(predict_fn, splits, norm_stats, name="Model"):
    """Evaluate zero-shot (Option B) on all scenarios."""
    results = {}

    for sname, test_df in [("S1", splits["s1_test"]), ("S2", splits["s2_test"]),
                            ("S3", splits["s3_test"]), ("S4", splits["s4_test"]),
                            ("S5", splits["s5_test"])]:
        X_t, y_t, _ = load_and_norm(test_df, stats=norm_stats)
        y_pred = predict_fn(X_t)
        acc = accuracy_score(y_t, y_pred)
        results[sname] = acc
        print(f"  {sname} zero-shot: {acc:.4f}")

    return results


def run_calibration(predict_fn, finetune_fn, splits, norm_stats, name="Model"):
    """Evaluate with calibration (Option A) on all scenarios."""
    results = {}

    # S1: per-subject calibration
    s1_accs = []
    for subj in splits["test_subjects"]:
        sdf = splits["s1_test"][splits["s1_test"]["subject"] == subj]
        cal_df, test_df = split_cal_test(sdf)
        if len(cal_df) == 0 or len(test_df) == 0: continue
        X_cal, y_cal, _ = load_and_norm(cal_df, stats=norm_stats)
        X_test, y_test, _ = load_and_norm(test_df, stats=norm_stats)
        ft_predict = finetune_fn(X_cal, y_cal)
        s1_accs.append(accuracy_score(y_test, ft_predict(X_test)))
    results["S1"] = np.mean(s1_accs) if s1_accs else 0
    print(f"  S1 calibrated: {results['S1']:.4f}")

    # S2: per-position calibration
    s2_accs = []
    s2_test = splits["s2_test"]
    for pos in sorted(s2_test["position"].unique()):
        pos_df = s2_test[s2_test["position"] == pos]
        cal_df, test_df = split_cal_test(pos_df)
        if len(cal_df) == 0 or len(test_df) == 0: continue
        X_cal, y_cal, _ = load_and_norm(cal_df, stats=norm_stats)
        X_test, y_test, _ = load_and_norm(test_df, stats=norm_stats)
        ft_predict = finetune_fn(X_cal, y_cal)
        s2_accs.append(accuracy_score(y_test, ft_predict(X_test)))
    results["S2"] = np.mean(s2_accs) if s2_accs else 0
    print(f"  S2 calibrated: {results['S2']:.4f}")

    # S3: per-subject calibration (rep 0 all positions)
    s3_accs = []
    for subj in splits["test_subjects"]:
        sdf = splits["s3_test"][splits["s3_test"]["subject"] == subj]
        cal_df, test_df = split_cal_test(sdf)
        if len(cal_df) == 0 or len(test_df) == 0: continue
        X_cal, y_cal, _ = load_and_norm(cal_df, stats=norm_stats)
        X_test, y_test, _ = load_and_norm(test_df, stats=norm_stats)
        ft_predict = finetune_fn(X_cal, y_cal)
        s3_accs.append(accuracy_score(y_test, ft_predict(X_test)))
    results["S3"] = np.mean(s3_accs) if s3_accs else 0
    print(f"  S3 calibrated: {results['S3']:.4f}")

    # S4: per-session calibration
    s4_accs = []
    s4_test = splits["s4_test"]
    for sess in sorted(s4_test["session"].unique()):
        sess_df = s4_test[s4_test["session"] == sess]
        cal_df, test_df = split_cal_test(sess_df)
        if len(cal_df) == 0 or len(test_df) == 0: continue
        X_cal, y_cal, _ = load_and_norm(cal_df, stats=norm_stats)
        X_test, y_test, _ = load_and_norm(test_df, stats=norm_stats)
        ft_predict = finetune_fn(X_cal, y_cal)
        s4_accs.append(accuracy_score(y_test, ft_predict(X_test)))
    results["S4"] = np.mean(s4_accs) if s4_accs else 0
    print(f"  S4 calibrated: {results['S4']:.4f}")

    # S5: per-position calibration
    s5_accs = []
    s5_test = splits["s5_test"]
    for pos in sorted(s5_test["position"].unique()):
        pos_df = s5_test[s5_test["position"] == pos]
        cal_df, test_df = split_cal_test(pos_df)
        if len(cal_df) == 0 or len(test_df) == 0: continue
        X_cal, y_cal, _ = load_and_norm(cal_df, stats=norm_stats)
        X_test, y_test, _ = load_and_norm(test_df, stats=norm_stats)
        ft_predict = finetune_fn(X_cal, y_cal)
        s5_accs.append(accuracy_score(y_test, ft_predict(X_test)))
    results["S5"] = np.mean(s5_accs) if s5_accs else 0
    print(f"  S5 calibrated: {results['S5']:.4f}")

    return results


def print_comparison(zero_results, cal_results, name="Model"):
    print(f"\n{'='*55}")
    print(f"  {name} — RESULTS")
    print(f"{'='*55}")
    print(f"{'Scenario':<12} {'Zero-shot':>12} {'Calibrated':>12} {'Δ':>8}")
    print(f"{'-'*55}")
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        zs = zero_results.get(s, 0)
        ca = cal_results.get(s, 0)
        check_zs = "✓" if zs >= 0.85 else ""
        check_ca = "✓" if ca >= 0.85 else ""
        print(f"{s:<12} {zs*100:>10.2f}% {check_zs} {ca*100:>10.2f}% {check_ca} {(ca-zs)*100:>+7.2f}%")
    print(f"{'='*55}")
