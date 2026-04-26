"""
Mark corrupted trials in metadata.csv.

Run this ONCE after preprocessing (02_preprocess_and_window.ipynb).
Adds an `is_valid` column: True for trials with n_windows > 0, False otherwise.

Usage:
    python mark_invalid_trials.py
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METADATA_PATH


def mark_invalid_trials():
    metadata = pd.read_csv(METADATA_PATH)

    # Trials that failed preprocessing have n_windows == 0
    metadata["is_valid"] = metadata["n_windows"] > 0

    n_invalid = (~metadata["is_valid"]).sum()
    n_valid = metadata["is_valid"].sum()

    metadata.to_csv(METADATA_PATH, index=False)

    print(f"Total trials:   {len(metadata)}")
    print(f"Valid trials:   {n_valid}")
    print(f"Invalid trials: {n_invalid}")
    print()

    if n_invalid > 0:
        print("Invalid trials detail:")
        invalid = metadata[~metadata["is_valid"]]
        print(invalid[["subject", "session", "position", "repetition", "gesture"]].to_string())
        print()
        print(f"Invalid per subject:")
        print(invalid["subject"].value_counts().to_string())

    print(f"\nMetadata saved to: {METADATA_PATH}")
    print(f"Column 'is_valid' added. All downstream code should filter: "
          f"metadata[metadata['is_valid'] == True]")


if __name__ == "__main__":
    mark_invalid_trials()