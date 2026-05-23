"""Signal preprocessing, filtering, FFT BPM estimation, and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


@dataclass
class ProcessedRGB:
    """Container for uniformly sampled and normalized RGB time-series data."""

    dataframe: pd.DataFrame
    fs: float
    duration_sec: float
    original_duration_sec: float
    dropped_first_sec: float


@dataclass
class FFTResult:
    """FFT-based BPM estimation result for one rPPG method."""

    method: str
    estimated_bpm: float
    f_peak_hz: float
    peak_strength: float
    snr_like: float
    frequencies_hz: np.ndarray
    magnitude: np.ndarray


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a friendly error when required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV/데이터에 필수 컬럼이 없습니다: {', '.join(missing)}")


def _rgb_schema_columns(df: pd.DataFrame) -> list[str]:
    """Return supported RGB columns while preserving legacy CSV compatibility.

    The original project stored one skin ROI as ``R,G,B``.  The reference-
    corrected workflow stores explicit ``skin_R/G/B`` and ``ref_R/G/B`` columns
    while also keeping ``R,G,B`` as aliases for the skin ROI so older analysis
    code and CSV files remain readable.
    """
    base_cols = ["R", "G", "B"]
    optional_cols = ["skin_R", "skin_G", "skin_B", "ref_R", "ref_G", "ref_B"]
    return base_cols + [col for col in optional_cols if col in df.columns]


def clean_rgb_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw RGB rows by sorting time and interpolating invalid channel values.

    NaN/inf values and RGB values outside the camera-like 0..255 range are treated
    as abnormal, replaced by NaN, and linearly interpolated along time.
    """
    _require_columns(df, ("time_sec", "R", "G", "B"))
    rgb_cols = _rgb_schema_columns(df)
    cleaned = df.loc[:, ["time_sec", *rgb_cols]].copy()
    for col in ["time_sec", *rgb_cols]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=["time_sec"])
    cleaned = cleaned.sort_values("time_sec")
    cleaned = cleaned.drop_duplicates(subset=["time_sec"], keep="first")

    for col in rgb_cols:
        abnormal = (cleaned[col] < 0) | (cleaned[col] > 255)
        cleaned.loc[abnormal, col] = np.nan
        cleaned[col] = cleaned[col].interpolate(method="linear", limit_direction="both")

    cleaned = cleaned.dropna(subset=rgb_cols)
    if len(cleaned) < 3:
        raise ValueError("유효한 RGB 샘플이 너무 적습니다. 더 긴 영상 또는 CSV를 사용하세요.")
    return cleaned.reset_index(drop=True)


def estimate_sampling_frequency(time_sec: np.ndarray) -> float:
    """Estimate sampling frequency from the median positive timestamp interval."""
    diffs = np.diff(np.asarray(time_sec, dtype=float))
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        raise ValueError("timestamp 간격을 추정할 수 없습니다. time_sec 컬럼을 확인하세요.")
    median_dt = float(np.median(positive))
    if median_dt <= 0:
        raise ValueError("timestamp 간격이 0 이하입니다. time_sec 컬럼을 확인하세요.")
    return 1.0 / median_dt


def resample_uniform(cleaned: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Interpolate RGB channels onto a uniform time grid using the estimated fs."""
    time = cleaned["time_sec"].to_numpy(dtype=float)
    start, end = float(time[0]), float(time[-1])
    if end <= start:
        raise ValueError("분석 시간이 0초입니다. 더 긴 입력이 필요합니다.")
    dt = 1.0 / fs
    uniform_time = np.arange(start, end + 0.5 * dt, dt)
    if uniform_time.size < 3:
        raise ValueError("균일 보간 후 샘플 수가 너무 적습니다.")

    out = pd.DataFrame({"time_sec": uniform_time - uniform_time[0]})
    for col in _rgb_schema_columns(cleaned):
        out[col] = np.interp(uniform_time, time, cleaned[col].to_numpy(dtype=float))
    return out


def add_normalized_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean-normalized r/g/b columns: (channel - mean) / mean."""
    out = df.copy()
    for raw_col, norm_col in [("R", "r_norm"), ("G", "g_norm"), ("B", "b_norm")]:
        values = out[raw_col].to_numpy(dtype=float)
        mean_val = float(np.mean(values))
        if not np.isfinite(mean_val) or abs(mean_val) < 1e-9:
            raise ValueError(f"{raw_col} 채널 평균이 0에 가까워 정규화할 수 없습니다.")
        out[norm_col] = (values - mean_val) / mean_val
    return out


def add_reference_absorbance_channels(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Add reference-corrected absorbance channels when reference ROI exists.

    ``A_c(t) = -log((I_skin,c(t)+eps)/(I_ref,c(t)+eps))`` approximates relative
    absorbance.  Illumination or exposure changes that affect both ROIs similarly
    are reduced by the ratio before the logarithm.
    """
    out = df.copy()
    if not {"ref_R", "ref_G", "ref_B"}.issubset(out.columns):
        return out

    channel_pairs = [
        ("R", "ref_R", "A_R"),
        ("G", "ref_G", "A_G"),
        ("B", "ref_B", "A_B"),
    ]
    for skin_col, ref_col, absorbance_col in channel_pairs:
        skin = out[skin_col].to_numpy(dtype=float)
        ref = out[ref_col].to_numpy(dtype=float)
        if np.any(ref + eps <= 0) or np.any(skin + eps <= 0):
            raise ValueError("참조 보정 로그 계산을 위해 RGB 값은 양수여야 합니다.")
        out[absorbance_col] = -np.log((skin + eps) / (ref + eps))
    return out


def preprocess_rgb_timeseries(df: pd.DataFrame, drop_first_sec: float = 2.0) -> ProcessedRGB:
    """Clean, optionally drop initial exposure-settling seconds, resample, normalize.

    The first seconds are often affected by camera auto exposure/white-balance.
    ``drop_first_sec`` can be set to 0 to keep the full signal.
    """
    cleaned = clean_rgb_dataframe(df)
    original_duration = float(cleaned["time_sec"].iloc[-1] - cleaned["time_sec"].iloc[0])

    if drop_first_sec < 0:
        raise ValueError("--drop-first-sec 값은 0 이상이어야 합니다.")
    if drop_first_sec > 0:
        start_time = float(cleaned["time_sec"].iloc[0]) + drop_first_sec
        dropped = cleaned[cleaned["time_sec"] >= start_time].copy()
        if len(dropped) >= 3:
            cleaned = dropped
        else:
            raise ValueError(
                f"처음 {drop_first_sec:.1f}초를 제거하면 남는 샘플이 너무 적습니다. "
                "--drop-first-sec 값을 줄이거나 더 긴 데이터를 사용하세요."
            )

    fs = estimate_sampling_frequency(cleaned["time_sec"].to_numpy(dtype=float))
    uniform = resample_uniform(cleaned, fs=fs)
    normalized = add_reference_absorbance_channels(add_normalized_channels(uniform))
    duration = float(normalized["time_sec"].iloc[-1] - normalized["time_sec"].iloc[0])
    if duration <= 0:
        raise ValueError("분석 가능한 시간이 너무 짧습니다.")
    return ProcessedRGB(
        dataframe=normalized,
        fs=fs,
        duration_sec=duration,
        original_duration_sec=original_duration,
        dropped_first_sec=drop_first_sec,
    )


def bandpass_filter(signal_values: np.ndarray, fs: float, low_hz: float = 0.8, high_hz: float = 2.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter with zero-phase ``filtfilt``.

    The default 0.8~2.0 Hz range corresponds to 48~120 BPM.
    """
    signal_values = np.asarray(signal_values, dtype=float)
    if signal_values.ndim != 1:
        raise ValueError("필터 입력 신호는 1차원 배열이어야 합니다.")
    if not np.all(np.isfinite(signal_values)):
        raise ValueError("필터 입력 신호에 NaN/inf가 포함되어 있습니다.")
    if fs <= 0:
        raise ValueError("샘플링 주파수 fs는 0보다 커야 합니다.")
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("주파수 범위는 0 < low_hz < high_hz 이어야 합니다.")

    nyquist = fs / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"high_hz={high_hz:.2f}Hz가 Nyquist 주파수({nyquist:.2f}Hz) 이상입니다. "
            "FPS가 더 높은 영상이 필요하거나 --high-hz를 낮추세요."
        )

    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")
    padlen = 3 * max(len(a), len(b))
    if signal_values.size <= padlen:
        raise ValueError(
            f"필터링에 필요한 샘플 수가 부족합니다({signal_values.size}개). "
            f"최소 {padlen + 1}개 이상이 필요합니다. 촬영 시간을 늘리세요."
        )
    return filtfilt(b, a, signal_values)


def estimate_bpm_fft(
    filtered_signal: np.ndarray,
    fs: float,
    low_hz: float = 0.8,
    high_hz: float = 2.0,
    method: str = "method",
    peak_neighborhood_hz: float = 0.1,
) -> FFTResult:
    """Estimate BPM from the strongest FFT peak inside the heart-rate band."""
    values = np.asarray(filtered_signal, dtype=float)
    if values.size < 4:
        raise ValueError("FFT 분석에 필요한 샘플 수가 너무 적습니다.")
    values = values - float(np.mean(values))
    window = np.hanning(values.size)
    spectrum = np.fft.rfft(values * window)
    freqs = np.fft.rfftfreq(values.size, d=1.0 / fs)
    magnitude = np.abs(spectrum)

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        raise ValueError("지정한 심박 주파수 범위에 FFT bin이 없습니다. 더 긴 데이터를 사용하세요.")

    band_indices = np.where(band_mask)[0]
    peak_index = int(band_indices[np.argmax(magnitude[band_mask])])
    f_peak = float(freqs[peak_index])
    estimated_bpm = 60.0 * f_peak
    peak_strength = float(magnitude[peak_index])

    power = magnitude**2
    peak_mask = band_mask & (np.abs(freqs - f_peak) <= peak_neighborhood_hz)
    noise_mask = band_mask & ~peak_mask
    signal_power = float(np.sum(power[peak_mask]))
    if np.any(noise_mask):
        noise_power = float(np.mean(power[noise_mask]))
    else:
        noise_power = 1e-12
    snr_like = signal_power / (noise_power + 1e-12)

    return FFTResult(
        method=method,
        estimated_bpm=estimated_bpm,
        f_peak_hz=f_peak,
        peak_strength=peak_strength,
        snr_like=float(snr_like),
        frequencies_hz=freqs,
        magnitude=magnitude,
    )


def results_to_dataframe(
    results: Dict[str, FFTResult],
    commercial_bpm: float | None = None,
    pyvhr_bpm: float | None = None,
) -> pd.DataFrame:
    """Convert FFT result objects to the required summary CSV schema.

    When a commercial webcam heart-rate monitor value is supplied, each method
    also receives an absolute error column so report tables can compare this
    implementation against the simultaneously measured commercial reference.
    """
    uncorrected_methods = {"raw_green", "normalized_green", "skin_rgb_combination", "green", "rgb_combination", "green_minus_red"}
    corrected_methods = {"ref_green", "ref_rgbdiff"}
    rows = []
    for method, result in results.items():
        group = "corrected" if method in corrected_methods else "uncorrected" if method in uncorrected_methods else "other"
        row = {
            "method": method,
            "correction_group": group,
            "estimated_bpm": result.estimated_bpm,
            "f_peak_hz": result.f_peak_hz,
            "peak_strength": result.peak_strength,
            "snr_like": result.snr_like,
        }
        if commercial_bpm is not None:
            row["commercial_bpm"] = float(commercial_bpm)
            row["abs_error_vs_commercial_bpm"] = abs(result.estimated_bpm - float(commercial_bpm))
        if pyvhr_bpm is not None:
            row["pyvhr_bpm"] = float(pyvhr_bpm)
            row["abs_error_vs_pyvhr_bpm"] = abs(result.estimated_bpm - float(pyvhr_bpm))
        rows.append(row)

    columns = ["method", "correction_group", "estimated_bpm", "f_peak_hz", "peak_strength", "snr_like"]
    if commercial_bpm is not None:
        columns.extend(["commercial_bpm", "abs_error_vs_commercial_bpm"])
    if pyvhr_bpm is not None:
        columns.extend(["pyvhr_bpm", "abs_error_vs_pyvhr_bpm"])
    return pd.DataFrame(rows, columns=columns)
