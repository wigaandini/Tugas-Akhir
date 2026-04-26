from pathlib import Path

# PATHS

# Path to the SeNic dataset (read-only from the project's perspective)
DATASET_ROOT = Path(
    "/Users/erdiantiwigaputriandini/Documents/Kuliah/Tugas Akhir/"
    "02. Dataset/SeNic (used)"
)

# Path to the SubjectsInfo.xlsx file inside the dataset
SUBJECTS_INFO_PATH = DATASET_ROOT / "SubjectsInfo.xlsx"

# Root of the project (where this config.py lives)
PROJECT_ROOT = Path(
    "/Users/erdiantiwigaputriandini/Documents/Kuliah/Tugas Akhir/"
    "04. TA/02. Code/Tugas-Akhir"
)

# Subdirectories for processed data and results
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.csv"
WINDOWS_DIR = PROCESSED_DATA_DIR / "windows"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure output directories exist (safe to call multiple times)
for _dir in [PROCESSED_DATA_DIR, WINDOWS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# SIGNAL ACQUISITION CONSTANTS

SAMPLING_RATE = 200          # Hz - Myo armband sampling rate
N_CHANNELS = 8               # Myo armband has 8 sEMG channels
REST_TRANSITION_SEC = 2.0    # Skip first 2 seconds of each trial (rest→gesture transition)


# PREPROCESSING HYPERPARAMETERS

# Band-pass filter range (Hz). Upper bound limited by Nyquist (100 Hz at fs=200)
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 95.0
BANDPASS_ORDER = 4

# Notch filter to remove power line interference (50 Hz for Indonesia/Europe/Asia)
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0               # Quality factor; higher = narrower notch

# Sliding window segmentation
WINDOW_MS = 250              # Window length in milliseconds
STRIDE_MS = 50               # Stride (hop) in milliseconds
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_MS / 1000)    # = 50 samples
STRIDE_SIZE = int(SAMPLING_RATE * STRIDE_MS / 1000)    # = 10 samples


# GESTURE LABELS

# The 7 active gestures in the SeNic dataset (verified from actual filenames).
# Order matters: index = integer label used by classifiers.
GESTURE_CLASSES = [
    "fist",                # 0
    "open_hand",           # 1
    "pinch_forefinger",    # 2
    "pinch_middlefinger",  # 3
    "two",                 # 4
    "eversion",            # 5
    "varus",               # 6
]

GESTURE_TO_LABEL = {g: i for i, g in enumerate(GESTURE_CLASSES)}
LABEL_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_CLASSES)}
N_CLASSES = len(GESTURE_CLASSES)


# SUBJECT GROUPING (based on SubjectsInfo.xlsx)

# Subjects by experimental role - used by data_splitter.py for scenario splits
SUBJECTS_INTERDAY_LONG = [f"h{i}" for i in range(0, 6)]      # h0-h5, 10 sessions each
SUBJECTS_INTERDAY_SHORT = [f"h{i}" for i in range(6, 14)]    # h6-h13, 3 sessions each
SUBJECTS_SHIFT_RI = [f"h{i}" for i in range(14, 18)]         # h14-h17, 1 session, Rotate Inward
SUBJECTS_SHIFT_RO = [f"h{i}" for i in range(18, 24)]         # h18-h23, 1 session, Rotate Outward
SUBJECTS_SHIFT_RM = [f"h{i}" for i in range(24, 30)]         # h24-h29, 1 session, Small angle
SUBJECTS_FATIGUE = [f"h{i}" for i in range(30, 36)]          # h30-h35, 1 session, Fatigue

SUBJECTS_NON_FATIGUE = (
    SUBJECTS_INTERDAY_LONG
    + SUBJECTS_INTERDAY_SHORT
    + SUBJECTS_SHIFT_RI
    + SUBJECTS_SHIFT_RO
    + SUBJECTS_SHIFT_RM
)

SUBJECTS_ALL = SUBJECTS_NON_FATIGUE + SUBJECTS_FATIGUE


# COMPUTE DEVICE (PyTorch)

def get_device():
    # Return the best available PyTorch device for this machine.

    # Priority:
    #     1. MPS (Apple Silicon GPU) - for M1/M2/M3/M4 Macs
    #     2. CUDA (NVIDIA GPU) - for compatibility if ever run elsewhere
    #     3. CPU - fallback

    # Usage:
    #     from config import get_device
    #     device = get_device()
    #     model = model.to(device)
    #     x = x.to(device)
    try:
        import torch
    except ImportError:
        return "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Reproducibility - set a fixed random seed across experiments
RANDOM_SEED = 42