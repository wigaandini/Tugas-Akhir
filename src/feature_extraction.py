import numpy as np
from scipy.signal import hilbert


# Statistical features for SVM (per window, per channel) 

def _rms(x):
    return np.sqrt(np.mean(x ** 2))

def _mav(x):
    return np.mean(np.abs(x))

def _waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def _zero_crossings(x):
    return np.sum(np.abs(np.diff(np.sign(x))) > 0)

def _slope_sign_changes(x):
    d = np.diff(x)
    return np.sum(np.abs(np.diff(np.sign(d))) > 0)

def _histogram(x, bins=10):
    hist, _ = np.histogram(x, bins=bins)
    return hist.astype(np.float32)


def extract_features_single_window(window):
    # window shape: (8, 50) — one window, channels-first
    features = []
    for ch in range(window.shape[0]):
        sig = window[ch]
        features.extend([
            _rms(sig),
            _mav(sig),
            _waveform_length(sig),
            _zero_crossings(sig),
            _slope_sign_changes(sig),
        ])
        features.extend(_histogram(sig, bins=10))
    return np.array(features, dtype=np.float32)


def extract_features_batch(X):
    X = np.nan_to_num(X, nan=0.0)
    return np.array([extract_features_single_window(w) for w in X], dtype=np.float32)


# Fast Hilbert Transform envelope for SCNN

def fht_envelope_single(window):
    # window shape: (8, 50) — returns analytic envelope same shape
    analytic = hilbert(window, axis=1)
    return np.abs(analytic).astype(np.float32)


def fht_envelope_batch(X):
    X = np.nan_to_num(X, nan=0.0)
    analytic = hilbert(X, axis=2)
    return np.abs(analytic).astype(np.float32)