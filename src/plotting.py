"""Matplotlib report-quality plot generation for rPPG analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .rppg_methods import METHOD_LABELS
from .signal_processing import FFTResult

DPI = 160


def _finalize(fig: plt.Figure, output_path: str | Path) -> None:
    """Save and close a matplotlib figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_rgb_raw(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot raw ROI RGB mean values over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_sec"], df["R"], label="R mean", color="tab:red", linewidth=1.4)
    ax.plot(df["time_sec"], df["G"], label="G mean", color="tab:green", linewidth=1.4)
    ax.plot(df["time_sec"], df["B"], label="B mean", color="tab:blue", linewidth=1.4)
    if {"ref_R", "ref_G", "ref_B"}.issubset(df.columns):
        ax.plot(df["time_sec"], df["ref_R"], label="Reference R", color="tab:red", linewidth=1.0, alpha=0.45, linestyle="--")
        ax.plot(df["time_sec"], df["ref_G"], label="Reference G", color="tab:green", linewidth=1.0, alpha=0.45, linestyle="--")
        ax.plot(df["time_sec"], df["ref_B"], label="Reference B", color="tab:blue", linewidth=1.0, alpha=0.45, linestyle="--")
    ax.set_title("Raw RGB mean values: skin ROI and reference ROI")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Mean pixel value (0-255)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize(fig, output_path)


def plot_rgb_normalized(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot mean-normalized RGB traces."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_sec"], df["r_norm"], label="r_norm", color="tab:red", linewidth=1.2)
    ax.plot(df["time_sec"], df["g_norm"], label="g_norm", color="tab:green", linewidth=1.2)
    ax.plot(df["time_sec"], df["b_norm"], label="b_norm", color="tab:blue", linewidth=1.2)
    ax.set_title("Mean-normalized RGB time series")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("(channel - mean) / mean")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize(fig, output_path)


def plot_reference_absorbance(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot reference-corrected absorbance channels when available."""
    if not {"A_R", "A_G", "A_B"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_sec"], df["A_R"], label="A_R = -log(R_skin/R_ref)", color="tab:red", linewidth=1.2)
    ax.plot(df["time_sec"], df["A_G"], label="A_G = -log(G_skin/G_ref)", color="tab:green", linewidth=1.2)
    ax.plot(df["time_sec"], df["A_B"], label="A_B = -log(B_skin/B_ref)", color="tab:blue", linewidth=1.2)
    ax.set_title("Reference-corrected absorbance signals")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Relative absorbance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize(fig, output_path)


def plot_filtered_signals(time_sec: np.ndarray, filtered_signals: Dict[str, np.ndarray], output_path: str | Path) -> None:
    """Plot filtered rPPG signals for each algorithm."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "raw_green": "tab:olive",
        "normalized_green": "tab:green",
        "skin_rgb_combination": "tab:purple",
        "ref_green": "tab:cyan",
        "ref_rgbdiff": "tab:blue",
        "green": "tab:green",
        "rgb_combination": "tab:purple",
        "green_minus_red": "tab:orange",
    }
    for method, values in filtered_signals.items():
        ax.plot(time_sec, values, label=METHOD_LABELS.get(method, method), linewidth=1.2, color=colors.get(method))
    ax.set_title("Band-pass filtered rPPG candidate signals")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Filtered normalized signal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize(fig, output_path)


def plot_fft_spectrum(results: Dict[str, FFTResult], output_path: str | Path, low_hz: float, high_hz: float) -> None:
    """Plot FFT magnitude spectra and annotate each method's peak BPM."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {
        "raw_green": "tab:olive",
        "normalized_green": "tab:green",
        "skin_rgb_combination": "tab:purple",
        "ref_green": "tab:cyan",
        "ref_rgbdiff": "tab:blue",
        "green": "tab:green",
        "rgb_combination": "tab:purple",
        "green_minus_red": "tab:orange",
    }
    ymax = 0.0
    for method, result in results.items():
        freqs = result.frequencies_hz
        mask = (freqs >= max(0.0, low_hz - 0.3)) & (freqs <= high_hz + 0.3)
        label = METHOD_LABELS.get(method, method)
        ax.plot(freqs[mask], result.magnitude[mask], label=label, linewidth=1.4, color=colors.get(method))
        ax.axvline(result.f_peak_hz, linestyle="--", linewidth=1.0, color=colors.get(method), alpha=0.8)
        ymax = max(ymax, float(np.max(result.magnitude[mask])) if np.any(mask) else 0.0)
        ax.annotate(
            f"{label}\n{result.estimated_bpm:.1f} BPM",
            xy=(result.f_peak_hz, result.peak_strength),
            xytext=(result.f_peak_hz + 0.03, result.peak_strength * 1.08 + 1e-12),
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": colors.get(method, "black")},
        )
    ax.axvspan(low_hz, high_hz, color="gray", alpha=0.08, label=f"Search band {low_hz:.1f}-{high_hz:.1f} Hz")
    ax.set_title("FFT magnitude spectrum of rPPG signals")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    if ymax > 0:
        ax.set_ylim(bottom=0, top=ymax * 1.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    secax = ax.secondary_xaxis("top", functions=(lambda hz: hz * 60.0, lambda bpm: bpm / 60.0))
    secax.set_xlabel("Frequency (BPM)")
    _finalize(fig, output_path)


def plot_method_comparison(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Create bar graphs comparing estimated BPM and SNR-like metric by method."""
    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"].tolist()]
    x = np.arange(len(labels))
    palette = ["tab:olive", "tab:green", "tab:purple", "tab:cyan", "tab:blue", "tab:orange"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(x, summary_df["estimated_bpm"], color=palette[: len(labels)])
    axes[0].set_title("Estimated BPM")
    axes[0].set_ylabel("BPM")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, summary_df["snr_like"], color=palette[: len(labels)])
    axes[1].set_title("SNR-like peak clarity metric")
    axes[1].set_ylabel("SNR-like (relative)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    _finalize(fig, output_path)


def plot_commercial_error_comparison(summary_df: pd.DataFrame, output_path: str | Path, commercial_label: str) -> None:
    """Plot BPM absolute error against a simultaneous commercial monitor reading."""
    if "abs_error_vs_commercial_bpm" not in summary_df.columns:
        return

    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"].tolist()]
    colors = ["tab:cyan" if group == "corrected" else "tab:gray" for group in summary_df["correction_group"].tolist()]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, summary_df["abs_error_vs_commercial_bpm"], color=colors)
    # Keep this plot title ASCII-only so it renders cleanly on headless systems
    # that only have Matplotlib's default DejaVu font.
    ax.set_title("Absolute BPM error vs commercial webcam HR monitor")
    ax.set_ylabel("Absolute error (BPM)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    _finalize(fig, output_path)


def plot_pyvhr_error_comparison(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot BPM absolute error against pyVHR's representative BPM estimate."""
    if "abs_error_vs_pyvhr_bpm" not in summary_df.columns:
        return

    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"].tolist()]
    colors = ["tab:cyan" if group == "corrected" else "tab:gray" for group in summary_df["correction_group"].tolist()]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, summary_df["abs_error_vs_pyvhr_bpm"], color=colors)
    ax.set_title("Absolute BPM error vs pyVHR")
    ax.set_ylabel("Absolute error (BPM)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    _finalize(fig, output_path)


def plot_correction_comparison(comparison_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot corrected/uncorrected SNR and optional error improvements."""
    if comparison_df.empty:
        return

    labels = comparison_df["comparison"].tolist()
    x = np.arange(len(labels))
    has_error = "error_reduction_bpm" in comparison_df.columns
    fig, axes = plt.subplots(1, 2 if has_error else 1, figsize=(13 if has_error else 9, 4.8))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    axes[0].bar(x, comparison_df["snr_like_ratio_corrected_over_uncorrected"], color="tab:cyan")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Corrected / uncorrected SNR-like ratio")
    axes[0].set_ylabel("Ratio (>1 means corrected peak is clearer)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    if has_error:
        axes[1].bar(x, comparison_df["error_reduction_bpm"], color="tab:green")
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Error reduction vs commercial BPM")
        axes[1].set_ylabel("BPM reduction (>0 means corrected is closer)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].grid(True, axis="y", alpha=0.3)

    _finalize(fig, output_path)


def plot_baseline_vs_reference(
    time_sec: np.ndarray,
    filtered_signals: Dict[str, np.ndarray],
    results: Dict[str, FFTResult],
    output_path: str | Path,
    low_hz: float,
    high_hz: float,
) -> None:
    """Create a side-by-side baseline/reference comparison for reports."""
    pairs = [method for method in ["raw_green", "normalized_green", "ref_green", "ref_rgbdiff"] if method in filtered_signals]
    if len(pairs) < 2:
        return

    colors = {
        "raw_green": "tab:olive",
        "normalized_green": "tab:green",
        "ref_green": "tab:cyan",
        "ref_rgbdiff": "tab:blue",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for method in pairs:
        axes[0].plot(
            time_sec,
            filtered_signals[method],
            label=METHOD_LABELS.get(method, method),
            linewidth=1.1,
            color=colors.get(method),
        )
    axes[0].set_title("Filtered time-domain signals")
    axes[0].set_xlabel("Time (sec)")
    axes[0].set_ylabel("Filtered signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for method in pairs:
        result = results[method]
        freqs = result.frequencies_hz
        mask = (freqs >= max(0.0, low_hz - 0.3)) & (freqs <= high_hz + 0.3)
        axes[1].plot(
            freqs[mask],
            result.magnitude[mask],
            label=f"{METHOD_LABELS.get(method, method)} ({result.estimated_bpm:.1f} BPM)",
            linewidth=1.2,
            color=colors.get(method),
        )
        axes[1].axvline(result.f_peak_hz, linestyle="--", linewidth=0.9, color=colors.get(method), alpha=0.75)
    axes[1].axvspan(low_hz, high_hz, color="gray", alpha=0.08)
    axes[1].set_title("FFT spectra and peak BPM")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    _finalize(fig, output_path)


def create_all_plots(
    df: pd.DataFrame,
    filtered_signals: Dict[str, np.ndarray],
    results: Dict[str, FFTResult],
    summary_df: pd.DataFrame,
    correction_comparison_df: pd.DataFrame,
    output_dir: str | Path,
    low_hz: float,
    high_hz: float,
    commercial_label: str = "commercial webcam HR monitor",
) -> None:
    """Generate every required PNG graph in the session output directory."""
    out = Path(output_dir)
    plot_rgb_raw(df, out / "01_rgb_raw.png")
    plot_rgb_normalized(df, out / "02_rgb_normalized.png")
    plot_reference_absorbance(df, out / "03_reference_absorbance.png")
    plot_filtered_signals(df["time_sec"].to_numpy(dtype=float), filtered_signals, out / "04_filtered_signals.png")
    plot_fft_spectrum(results, out / "05_fft_spectrum.png", low_hz=low_hz, high_hz=high_hz)
    plot_method_comparison(summary_df, out / "06_method_comparison.png")
    plot_baseline_vs_reference(
        df["time_sec"].to_numpy(dtype=float),
        filtered_signals=filtered_signals,
        results=results,
        output_path=out / "07_baseline_vs_reference.png",
        low_hz=low_hz,
        high_hz=high_hz,
    )
    plot_commercial_error_comparison(summary_df, out / "08_commercial_error_comparison.png", commercial_label=commercial_label)
    plot_correction_comparison(correction_comparison_df, out / "09_correction_comparison.png")
    plot_pyvhr_error_comparison(summary_df, out / "10_pyvhr_error_comparison.png")
