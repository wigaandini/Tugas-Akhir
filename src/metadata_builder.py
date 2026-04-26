from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    DATASET_ROOT,
    SUBJECTS_INFO_PATH,
    METADATA_PATH,
    GESTURE_CLASSES,
    GESTURE_TO_LABEL,
    WINDOWS_DIR,
)


# Regex to parse the EMG filename pattern
# Example matches: "emg_p0_r2_fist.csv", "emg_p10_r0_pinch_forefinger.csv"
_FILENAME_PATTERN = re.compile(
    r"^emg_p(?P<position>\d+)_r(?P<repetition>\d+)_(?P<gesture>.+)\.csv$"
)


def _parse_emg_filename(filename: str) -> Optional[dict]:
    # Parse an EMG filename and return its components, or None if invalid
    match = _FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    return {
        "position": int(match.group("position")),
        "repetition": int(match.group("repetition")),
        "gesture": match.group("gesture"),
    }


def scan_dataset(dataset_root: Path = DATASET_ROOT) -> pd.DataFrame:
    # Walk the dataset directory and collect one row per EMG trial file.

    # Returns a DataFrame with columns:
    #     subject, session, position, repetition, gesture, raw_file_path
     
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    rows = []
    unknown_gestures = set()

    def _find_session_folders(parent: Path):
        # Handles two layouts found in SeNic:
        #   A) Flat:       hX/0/, hX/1/, ...                 (h6-h35)
        #   B) Grouped:    hX/0-4/0/, hX/0-4/1/, ...          (h0-h5, which
        #                  hX/5-9/5/, hX/5-9/6/, ...           have 10 sessions)
        
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            if child.name.isdigit():
                # Layout A: child IS the session folder
                yield int(child.name), child
            elif "-" in child.name:
                # Layout B: child is a group folder like "0-4" or "5-9".
                # Recurse one level deeper.
                for grandchild in sorted(child.iterdir()):
                    if grandchild.is_dir() and grandchild.name.isdigit():
                        yield int(grandchild.name), grandchild
            # Any other folder name is ignored silently

    # Iterate over subject folders (h0, h1, ...)
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        if not subject_dir.name.startswith("h"):
            continue

        subject_name = subject_dir.name  # e.g. "h0"

        # Iterate over session folders (handles both flat and grouped layouts)
        for session_num, session_dir in _find_session_folders(subject_dir):

            # Iterate over EMG CSV files in this session
            for csv_file in sorted(session_dir.glob("emg_p*_r*_*.csv")):
                parsed = _parse_emg_filename(csv_file.name)
                if parsed is None:
                    continue

                # Track any gestures not in our expected list (for diagnostics)
                if parsed["gesture"] not in GESTURE_CLASSES:
                    unknown_gestures.add(parsed["gesture"])
                    continue

                rows.append({
                    "subject": subject_name,
                    "session": session_num,
                    "position": parsed["position"],
                    "repetition": parsed["repetition"],
                    "gesture": parsed["gesture"],
                    "raw_file_path": str(csv_file.resolve()),
                })

    if unknown_gestures:
        print(
            f"[WARNING] Found gesture names not in GESTURE_CLASSES "
            f"(will be skipped): {sorted(unknown_gestures)}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No valid EMG files found under {dataset_root}. "
            f"Check DATASET_ROOT path and filename conventions."
        )

    # Add integer label for ML pipelines
    df["gesture_label"] = df["gesture"].map(GESTURE_TO_LABEL)
    return df


def load_subjects_info(subjects_info_path: Path = SUBJECTS_INFO_PATH) -> pd.DataFrame:
    # Load and normalize the SubjectsInfo.xlsx file
    if not subjects_info_path.exists():
        raise FileNotFoundError(f"SubjectsInfo file not found: {subjects_info_path}")

    info = pd.read_excel(subjects_info_path)

    # Standardize column names to snake_case
    info.columns = [c.strip().lower().replace(" ", "_").replace("/", "_per_")
                    for c in info.columns]

    # Rename "name" → "subject" to match scan_dataset output
    if "name" in info.columns:
        info = info.rename(columns={"name": "subject"})

    return info


def enrich_with_subject_info(
    trials_df: pd.DataFrame,
    subjects_info: pd.DataFrame,
) -> pd.DataFrame:
    # Merge trial metadata with subject demographics/description.

    # Adds a `direction` column derived from `description`:
    #     - "RI", "RO", "RM" → shift direction (in/out/small)
    #     - "FA" → fatigue group
    # Also adds `is_fatigue` boolean flag.
    
    # Only keep useful columns from subjects_info to avoid clutter
    keep_cols = ["subject", "gender", "age", "description"]
    available_cols = [c for c in keep_cols if c in subjects_info.columns]
    info_subset = subjects_info[available_cols].copy()

    merged = trials_df.merge(info_subset, on="subject", how="left")

    # Sanity check: any trial without subject info?
    missing = merged[merged["description"].isna()]["subject"].unique()
    if len(missing) > 0:
        print(f"[WARNING] Subjects in dataset but not in SubjectsInfo: {list(missing)}")

    # Derive convenience columns
    merged["direction"] = merged["description"].fillna("UNKNOWN")
    merged["is_fatigue"] = (merged["description"] == "FA")
    return merged


def _build_window_file_path(row: pd.Series) -> str:
    # Generate the output path where this trial's window array will be stored.

    # Format: WINDOWS_DIR / subject / session / emg_p{pos}_r{rep}_{gesture}.npz
    
    filename = (
        f"emg_p{row['position']}_r{row['repetition']}_{row['gesture']}.npz"
    )
    return str(WINDOWS_DIR / row["subject"] / str(row["session"]) / filename)


def attach_window_paths(df: pd.DataFrame) -> pd.DataFrame:
    # Add the `window_file_path` column (where preprocessed windows will be saved)
    df = df.copy()
    df["window_file_path"] = df.apply(_build_window_file_path, axis=1)
    # `n_windows` is initialized to 0; will be filled after preprocessing runs
    df["n_windows"] = 0
    return df


def validate_metadata(df: pd.DataFrame) -> None:
    # Print sanity-check statistics about the metadata DataFrame
    print("=" * 60)
    print("METADATA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total trials: {len(df)}")
    print(f"Unique subjects: {df['subject'].nunique()}")
    print(f"Unique gestures: {df['gesture'].nunique()} "
          f"({sorted(df['gesture'].unique())})")
    print(f"Position range: {df['position'].min()}–{df['position'].max()}")
    print(f"Repetition range: {df['repetition'].min()}–{df['repetition'].max()}")
    print()

    print("Trials per subject (first 10 rows):")
    per_subject = df.groupby("subject").size().sort_index()
    print(per_subject.head(10).to_string())
    print("...")
    print()

    print("Sessions per subject:")
    sessions_per_subject = df.groupby("subject")["session"].nunique().sort_index()
    print(sessions_per_subject.to_string())
    print()

    if "description" in df.columns:
        print("Trials per group (description):")
        print(df.groupby("description").size().to_string())
    print("=" * 60)


def build_metadata(save: bool = True) -> pd.DataFrame:
    # Full pipeline: scan dataset, merge with SubjectsInfo, attach paths.
    # Optionally save to METADATA_PATH.
    
    print(f"Scanning dataset at: {DATASET_ROOT}")
    trials = scan_dataset()
    print(f"Found {len(trials)} EMG trial files.")

    print(f"\nLoading subject info from: {SUBJECTS_INFO_PATH}")
    subjects = load_subjects_info()
    print(f"Loaded info for {len(subjects)} subjects.")

    print("\nMerging trials with subject info...")
    metadata = enrich_with_subject_info(trials, subjects)

    print("Attaching expected window output paths...")
    metadata = attach_window_paths(metadata)

    # Order columns logically
    column_order = [
        "subject", "session", "position", "repetition",
        "gesture", "gesture_label",
        "gender", "age", "description", "direction", "is_fatigue",
        "raw_file_path", "window_file_path", "n_windows",
    ]
    column_order = [c for c in column_order if c in metadata.columns]
    metadata = metadata[column_order]

    if save:
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(METADATA_PATH, index=False)
        print(f"\nMetadata saved to: {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    df = build_metadata(save=True)
    print()
    validate_metadata(df)