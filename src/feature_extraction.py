import numpy as np
from scipy.signal import hilbert


# Statistical features for SVM (per window, per channel)
# Full set: RMS + MAV + WL + ZC + SSC + Histogram(10) + CWT_MAV(7) = 22 per channel

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


def _ricker_wavelet(points, a):
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec ** 2
    mod = 1.0 - tsq / wsq
    gauss = np.exp(-tsq / (2.0 * wsq))
    return A * mod * gauss


def _cwt_ricker(data, widths):
    n = len(data)
    output = np.empty((len(widths), n), dtype=np.float64)
    for i, w in enumerate(widths):
        wavelet_len = min(10 * w, n)
        wavelet_data = _ricker_wavelet(wavelet_len, w)
        output[i] = np.convolve(data, wavelet_data, mode="same")
    return output


def _cwt_mav(x, widths=np.arange(1, 8)):
    coeffs = _cwt_ricker(x, widths)
    return np.mean(np.abs(coeffs), axis=1).astype(np.float32)


def extract_features_single_window(window):
    # window shape: (8, 50) -- one window, channels-first
    # 22 features per channel: RMS(1) + MAV(1) + WL(1) + ZC(1) + SSC(1) + Hist(10) + CWT(7)
    features = []
    for ch in range(window.shape[0]):
        sig = window[ch]
        features.append(_rms(sig))
        features.append(_mav(sig))
        features.append(_waveform_length(sig))
        features.append(float(_zero_crossings(sig)))
        features.append(float(_slope_sign_changes(sig)))
        features.extend(_histogram(sig, bins=10))
        features.extend(_cwt_mav(sig))
    return np.array(features, dtype=np.float32)


def extract_features_batch(X):
    X = np.nan_to_num(X, nan=0.0)
    return np.array([extract_features_single_window(w) for w in X], dtype=np.float32)


# Fast Hilbert Transform envelope for SCNN

def fht_envelope_single(window):
    # window shape: (8, 50) -- returns analytic envelope same shape
    analytic = hilbert(window, axis=1)
    return np.abs(analytic).astype(np.float32)


def fht_envelope_batch(X):
    X = np.nan_to_num(X, nan=0.0)
    analytic = hilbert(X, axis=2)
    return np.abs(analytic).astype(np.float32)