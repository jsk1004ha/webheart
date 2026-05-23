"""rPPG signal extraction methods for raw, normalized, and reference-corrected RGB."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .signal_processing import FFTResult, bandpass_filter, estimate_bpm_fft


METHOD_LABELS = {
    "raw_green": "Raw Green (G_skin)",
    "normalized_green": "Normalized Green (g_skin)",
    "skin_rgb_combination": "Skin RGB combination (2g-r-b)",
    "ref_green": "Reference-corrected Green (A_G)",
    "ref_rgbdiff": "Reference-corrected RGB difference (2A_G-A_R-A_B)",
    # Backward-compatible labels for older saved summaries.
    "green": "Normalized Green (g_skin)",
    "rgb_combination": "Skin RGB combination (2g-r-b)",
    "green_minus_red": "Green minus Red (G-R)",
}


def extract_rppg_signals(df: pd.DataFrame, include_rg: bool = False) -> Dict[str, np.ndarray]:
    """Create rPPG candidate signals from skin and optional reference RGB channels.

    Required methods:
    - Raw Green: ``s_raw(t) = G_skin(t)``
    - Normalized Green: ``s_norm(t) = g_skin(t)``
    - Skin RGB combination: ``s_rgb(t) = 2*g(t) - r(t) - b(t)``
    - Reference Green when available: ``s_ref(t) = A_G(t)``
    - Reference RGB difference when available: ``2*A_G(t)-A_R(t)-A_B(t)``

    The reference methods analyze a skin/reference reflected-light ratio, which
    reduces illumination and auto-exposure components shared by both ROIs.
    """
    required = {"R", "G", "B", "r_norm", "g_norm", "b_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"정규화 RGB 컬럼이 없습니다: {', '.join(sorted(missing))}")

    r = df["r_norm"].to_numpy(dtype=float)
    g = df["g_norm"].to_numpy(dtype=float)
    b = df["b_norm"].to_numpy(dtype=float)

    signals: Dict[str, np.ndarray] = {
        "raw_green": df["G"].to_numpy(dtype=float),
        "normalized_green": g,
        "skin_rgb_combination": 2.0 * g - r - b,
    }
    if {"A_R", "A_G", "A_B"}.issubset(df.columns):
        a_r = df["A_R"].to_numpy(dtype=float)
        a_g = df["A_G"].to_numpy(dtype=float)
        a_b = df["A_B"].to_numpy(dtype=float)
        signals["ref_green"] = a_g
        signals["ref_rgbdiff"] = 2.0 * a_g - a_r - a_b
    if include_rg:
        signals["green_minus_red"] = g - r
    return signals


def analyze_rppg_methods(
    df: pd.DataFrame,
    fs: float,
    low_hz: float = 0.8,
    high_hz: float = 2.0,
    include_rg: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, FFTResult]]:
    """Filter each rPPG signal and estimate BPM using FFT peak search."""
    raw_signals = extract_rppg_signals(df, include_rg=include_rg)
    filtered: Dict[str, np.ndarray] = {}
    results: Dict[str, FFTResult] = {}
    for method, values in raw_signals.items():
        filtered_signal = bandpass_filter(values, fs=fs, low_hz=low_hz, high_hz=high_hz)
        filtered[method] = filtered_signal
        results[method] = estimate_bpm_fft(
            filtered_signal,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            method=method,
        )
    return filtered, results
