"""General utilities for session folders, synthetic data, and Korean reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .rppg_methods import METHOD_LABELS
from .signal_processing import FFTResult, ProcessedRGB

DISCLAIMER = (
    "[중요] 이 프로그램은 의료 진단용이 아닙니다. 웹캠 RGB 반사광 변화를 이용한 "
    "물리·정보 융합 탐구/시뮬레이션 도구이며 건강 판단에 사용할 수 없습니다."
)


def print_disclaimer() -> None:
    """Print the non-medical-use disclaimer shown at runtime."""
    print("=" * 78)
    print(DISCLAIMER)
    print("=" * 78)


def create_session_dir(output_root: str | Path = "outputs") -> Path:
    """Create and return an ``outputs/session_YYYYMMDD_HHMMSS`` directory."""
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"session_{timestamp}"
    counter = 1
    while session_dir.exists():
        session_dir = root / f"session_{timestamp}_{counter:02d}"
        counter += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def generate_synthetic_rgb(bpm: float = 72.0, duration_sec: float = 60.0, noise: float = 0.03, fs: float = 30.0) -> pd.DataFrame:
    """Generate synthetic skin/reference RGB data with a controllable rPPG component.

    The synthetic skin ROI includes a heart-frequency sinusoid, while the
    reference ROI contains the same slow illumination drift/auto-exposure-like
    common term but no pulse.  This makes demo mode useful for validating the
    reference-corrected ratio workflow without a webcam.
    """
    if duration_sec <= 0:
        raise ValueError("demo duration은 0보다 커야 합니다.")
    if bpm <= 0:
        raise ValueError("demo BPM은 0보다 커야 합니다.")
    if noise < 0:
        raise ValueError("noise는 0 이상이어야 합니다.")

    rng = np.random.default_rng(42)
    t = np.arange(0.0, duration_sec, 1.0 / fs)
    heart_hz = bpm / 60.0
    pulse = np.sin(2.0 * np.pi * heart_hz * t)
    harmonic = 0.25 * np.sin(2.0 * np.pi * 2.0 * heart_hz * t + 0.6)
    drift = 0.018 * np.sin(2.0 * np.pi * 0.08 * t + 0.4)
    common_noise = noise * rng.normal(0.0, 1.0, size=t.size)
    common_multiplier = 1.0 + drift + common_noise * 0.22

    # Baselines are camera-like RGB means. rPPG amplitudes are deliberately small.
    r_base, g_base, b_base = 138.0, 112.0, 102.0
    r = r_base * (common_multiplier + 0.0045 * pulse + 0.0015 * harmonic + common_noise * 0.06)
    g = g_base * (common_multiplier + 0.0120 * pulse + 0.0025 * harmonic + common_noise * 0.05)
    b = b_base * (common_multiplier + 0.0030 * pulse + 0.0010 * harmonic + common_noise * 0.08)

    ref_r_base, ref_g_base, ref_b_base = 165.0, 160.0, 150.0
    ref_r = ref_r_base * (common_multiplier + common_noise * 0.025)
    ref_g = ref_g_base * (common_multiplier + common_noise * 0.025)
    ref_b = ref_b_base * (common_multiplier + common_noise * 0.025)

    channel_noise_scale = noise * 255.0 * 0.12
    r += rng.normal(0.0, channel_noise_scale, size=t.size)
    g += rng.normal(0.0, channel_noise_scale, size=t.size)
    b += rng.normal(0.0, channel_noise_scale, size=t.size)
    ref_r += rng.normal(0.0, channel_noise_scale * 0.65, size=t.size)
    ref_g += rng.normal(0.0, channel_noise_scale * 0.65, size=t.size)
    ref_b += rng.normal(0.0, channel_noise_scale * 0.65, size=t.size)

    skin_r = np.clip(r, 0, 255)
    skin_g = np.clip(g, 0, 255)
    skin_b = np.clip(b, 0, 255)

    return pd.DataFrame(
        {
            "time_sec": t,
            "R": skin_r,
            "G": skin_g,
            "B": skin_b,
            "skin_R": skin_r,
            "skin_G": skin_g,
            "skin_B": skin_b,
            "ref_R": np.clip(ref_r, 0, 255),
            "ref_G": np.clip(ref_g, 0, 255),
            "ref_B": np.clip(ref_b, 0, 255),
        }
    )


def best_peak_method(results: Dict[str, FFTResult]) -> str:
    """Return the method name with the highest SNR-like peak clarity metric."""
    if not results:
        return "N/A"
    return max(results.values(), key=lambda result: result.snr_like).method


def build_correction_comparison(
    results: Dict[str, FFTResult],
    commercial_bpm: float | None = None,
    pyvhr_bpm: float | None = None,
) -> pd.DataFrame:
    """Build a report table that explicitly compares uncorrected vs corrected methods."""
    comparison_pairs = [
        ("Raw Green vs Reference Green", "raw_green", "ref_green"),
        ("Normalized Green vs Reference Green", "normalized_green", "ref_green"),
        ("Skin RGB combination vs Reference RGB difference", "skin_rgb_combination", "ref_rgbdiff"),
    ]
    rows = []
    for label, uncorrected_method, corrected_method in comparison_pairs:
        if uncorrected_method not in results or corrected_method not in results:
            continue
        uncorrected = results[uncorrected_method]
        corrected = results[corrected_method]
        row = {
            "comparison": label,
            "uncorrected_method": uncorrected_method,
            "corrected_method": corrected_method,
            "uncorrected_bpm": uncorrected.estimated_bpm,
            "corrected_bpm": corrected.estimated_bpm,
            "bpm_difference_corrected_minus_uncorrected": corrected.estimated_bpm - uncorrected.estimated_bpm,
            "uncorrected_snr_like": uncorrected.snr_like,
            "corrected_snr_like": corrected.snr_like,
            "snr_like_ratio_corrected_over_uncorrected": corrected.snr_like / (uncorrected.snr_like + 1e-12),
        }
        if commercial_bpm is not None:
            uncorrected_error = abs(uncorrected.estimated_bpm - float(commercial_bpm))
            corrected_error = abs(corrected.estimated_bpm - float(commercial_bpm))
            row.update(
                {
                    "commercial_bpm": float(commercial_bpm),
                    "uncorrected_abs_error_bpm": uncorrected_error,
                    "corrected_abs_error_bpm": corrected_error,
                    "error_reduction_bpm": uncorrected_error - corrected_error,
                }
            )
        if pyvhr_bpm is not None:
            uncorrected_pyvhr_error = abs(uncorrected.estimated_bpm - float(pyvhr_bpm))
            corrected_pyvhr_error = abs(corrected.estimated_bpm - float(pyvhr_bpm))
            row.update(
                {
                    "pyvhr_bpm": float(pyvhr_bpm),
                    "uncorrected_abs_error_vs_pyvhr_bpm": uncorrected_pyvhr_error,
                    "corrected_abs_error_vs_pyvhr_bpm": corrected_pyvhr_error,
                    "pyvhr_error_reduction_bpm": uncorrected_pyvhr_error - corrected_pyvhr_error,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report_summary(
    output_path: str | Path,
    processed: ProcessedRGB,
    results: Dict[str, FFTResult],
    low_hz: float,
    high_hz: float,
    source_mode: str,
    correction_comparison_df: pd.DataFrame | None = None,
    commercial_bpm: float | None = None,
    commercial_label: str = "상용 웹캠 심박수 측정기",
    pyvhr_bpm: float | None = None,
    pyvhr_method: str | None = None,
    pyvhr_status: str | None = None,
) -> None:
    """Write a Korean report-ready summary paragraph and key analysis values."""
    raw_green = results.get("raw_green")
    normalized_green = results.get("normalized_green")
    ref_green = results.get("ref_green")
    ref_rgbdiff = results.get("ref_rgbdiff")
    best_method = best_peak_method(results)
    best_label = METHOD_LABELS.get(best_method, best_method)

    raw_green_bpm = f"{raw_green.estimated_bpm:.1f}" if raw_green else "N/A"
    normalized_green_bpm = f"{normalized_green.estimated_bpm:.1f}" if normalized_green else "N/A"
    ref_green_bpm = f"{ref_green.estimated_bpm:.1f}" if ref_green else "N/A"
    ref_rgbdiff_bpm = f"{ref_rgbdiff.estimated_bpm:.1f}" if ref_rgbdiff else "N/A"
    has_reference = ref_green is not None
    commercial_text = ""
    commercial_comparison_text = (
        "- 상용 웹캠 심박수 측정기와 비교하려면 같은 시간에 상용 앱/기기의 BPM을 읽고 "
        "--commercial-bpm 값으로 입력한다. 그러면 각 방식의 절대오차가 자동 계산된다.\n"
    )
    if commercial_bpm is not None:
        commercial_text = f"\n상용 기준 측정값: {commercial_label} = 약 {commercial_bpm:.1f} BPM\n"
        commercial_comparison_text = (
            f"- 상용 웹캠 심박수 측정기와 동시에 측정한 {commercial_label} 값({commercial_bpm:.1f} BPM)을 "
            "기준으로 각 방식의 절대오차를 계산하였다. commercial_comparison.csv와 "
            "08_commercial_error_comparison.png를 보고 어떤 방식이 상용 측정값에 더 가까운지 분석할 수 있다.\n"
        )
    pyvhr_text = ""
    pyvhr_comparison_text = (
        "- pyVHR와 비교하려면 저장 영상 분석에서 `--run-pyvhr`를 사용하거나 pyVHR로 따로 얻은 평균 BPM을 "
        "`--pyvhr-bpm`으로 입력한다. 그러면 각 방식의 pyVHR 대비 절대오차가 자동 계산된다.\n"
    )
    if pyvhr_bpm is not None:
        method_text = f" ({pyvhr_method})" if pyvhr_method else ""
        pyvhr_text = f"\npyVHR 비교 측정값{method_text}: 약 {pyvhr_bpm:.1f} BPM\n"
        pyvhr_comparison_text = (
            f"- pyVHR{method_text}가 같은 영상에서 추정한 대표 BPM({pyvhr_bpm:.1f} BPM)을 외부 rPPG 기준으로 두고 "
            "각 방식의 절대오차를 계산하였다. pyvhr_comparison.csv와 10_pyvhr_error_comparison.png를 보고 "
            "본 참조 보정 방식이 pyVHR 결과와 얼마나 가까운지 비교할 수 있다.\n"
        )
    elif pyvhr_status:
        pyvhr_comparison_text += f"  · 이번 실행의 pyVHR 상태: {pyvhr_status}\n"

    text = f"""비접촉 심박 신호 추출 탐구 분석 요약

{DISCLAIMER}

입력 모드: {source_mode}
촬영/입력 원본 시간: 약 {processed.original_duration_sec:.2f}초
전처리 후 분석 시간: 약 {processed.duration_sec:.2f}초
추정 샘플링 주파수(fs): 약 {processed.fs:.2f} Hz
분석한 심박 주파수 범위: {low_hz:.2f}~{high_hz:.2f} Hz ({low_hz*60:.0f}~{high_hz*60:.0f} BPM)
초기 제거 시간: {processed.dropped_first_sec:.2f}초
Raw Green 방식 추정 BPM: 약 {raw_green_bpm} BPM
Normalized Green 방식 추정 BPM: 약 {normalized_green_bpm} BPM
참조 보정 Green 방식 추정 BPM: 약 {ref_green_bpm} BPM
참조 보정 RGB difference 방식 추정 BPM: 약 {ref_rgbdiff_bpm} BPM
FFT peak가 더 뚜렷한 방식(SNR-like 기준): {best_label}
{commercial_text}
{pyvhr_text}

본 탐구의 핵심 주제는 \"피부 ROI와 기준 ROI의 반사광 비율을 이용한 참조 보정형 rPPG 심박 신호 추출\"이다. 단순 Green 방식은 피부 ROI의 초록색 평균값 G_skin(t)만 분석하므로 심박에 따른 피부 반사율 변화뿐 아니라 조명 변화, 카메라 자동 노출, 움직임 잡음도 함께 포함된다. 반면 참조 보정 방식은 같은 프레임 안에서 피부 ROI와 혈류 변화가 거의 없는 기준 ROI를 동시에 측정하고, A_c(t) = -log((I_skin,c(t)+eps)/(I_ref,c(t)+eps)) 형태의 로그 비율을 계산한다. 피부와 기준 영역에 공통으로 들어간 조명 세기 I0(t)는 비율 계산에서 약하게 상쇄되므로, 전체 밝기 변화보다 피부 자체의 상대 반사율·흡광도 변화에 더 집중할 수 있다.

본 프로그램은 웹캠 또는 영상에서 선택한 피부 ROI와 기준 ROI의 RGB 평균값을 시간에 따라 추출하고, Raw Green, Normalized Green, 참조 보정 Green(A_G), 참조 보정 RGB difference(2A_G-A_R-A_B) 신호를 같은 {low_hz:.1f}~{high_hz:.1f} Hz 범위의 Butterworth band-pass filter와 FFT로 비교하였다. Raw Green 방식에서는 약 {raw_green_bpm} BPM, Normalized Green 방식에서는 약 {normalized_green_bpm} BPM, 참조 보정 Green 방식에서는 약 {ref_green_bpm} BPM, 참조 보정 RGB difference 방식에서는 약 {ref_rgbdiff_bpm} BPM이 추정되었다. SNR-like peak 선명도 기준으로는 {best_label} 방식이 가장 뚜렷했다. 다만 조명 변화, 얼굴 움직임, 카메라 자동 노출/자동 화이트밸런스, ROI 선택 위치가 결과에 영향을 줄 수 있으므로 본 결과는 의료적 진단이 아닌 물리·정보 융합 탐구 결과로 해석해야 한다.

보정 vs 노보정 비교 방법:
- 노보정 방식은 피부 ROI만 사용하는 Raw Green, Normalized Green, Skin RGB 조합 신호이다.
- 보정 방식은 피부 ROI를 기준 ROI로 나눈 로그 비율인 A_G와 2A_G-A_R-A_B 신호이다.
- 같은 영상에서 두 계열을 동시에 계산하므로 조명 조건, 움직임, 촬영 시간 차이를 줄이고 알고리즘 차이만 비교할 수 있다.
- correction_comparison.csv에는 노보정 BPM, 보정 BPM, SNR-like 비율, 그리고 상용 기준값이 있을 때 BPM 절대오차 감소량이 저장된다.
{commercial_comparison_text}
{pyvhr_comparison_text}

실험 확장 제안:
- 같은 영상에서 기존 Raw/Normalized Green 방식과 참조 보정 방식을 동시에 계산하여 BPM 오차와 SNR-like 지표를 비교한다.
- 모니터 밝기 변화, 창문 빛 변화, 약한 실내등 흔들림처럼 공통 조명 잡음이 있는 조건에서 참조 보정 방식의 안정성을 확인한다.
- 기준 ROI를 벽, 옷, 머리카락, 책상 배경 등으로 바꾸어 어느 위치가 조명 변화를 가장 잘 대표하는지 비교한다.
{'' if has_reference else '- 현재 입력 CSV에는 ref_R/ref_G/ref_B 컬럼이 없어 참조 보정 방식은 계산되지 않았다. 웹캠/video/demo 모드 또는 참조 ROI 컬럼이 포함된 CSV를 사용하면 참조 보정 결과가 생성된다.\\n'}

방법별 세부 결과:
"""
    if correction_comparison_df is not None and not correction_comparison_df.empty:
        text += "\n보정 vs 노보정 세부 비교:\n"
        for _, row in correction_comparison_df.iterrows():
            text += (
                f"- {row['comparison']}: 노보정 {row['uncorrected_bpm']:.2f} BPM, "
                f"보정 {row['corrected_bpm']:.2f} BPM, "
                f"SNR-like 비율={row['snr_like_ratio_corrected_over_uncorrected']:.3g}"
            )
            if "error_reduction_bpm" in row and pd.notna(row["error_reduction_bpm"]):
                text += (
                    f", 상용 기준 절대오차 변화="
                    f"{row['error_reduction_bpm']:.2f} BPM 감소(양수면 보정 방식이 더 가까움)"
                )
            if "pyvhr_error_reduction_bpm" in row and pd.notna(row["pyvhr_error_reduction_bpm"]):
                text += (
                    f", pyVHR 기준 절대오차 변화="
                    f"{row['pyvhr_error_reduction_bpm']:.2f} BPM 감소(양수면 보정 방식이 더 가까움)"
                )
            text += "\n"

    for method, result in results.items():
        label = METHOD_LABELS.get(method, method)
        text += (
            f"- {label}: estimated_bpm={result.estimated_bpm:.2f}, "
            f"f_peak={result.f_peak_hz:.4f} Hz, peak_strength={result.peak_strength:.6g}, "
            f"snr_like={result.snr_like:.4g}\n"
        )
        if commercial_bpm is not None:
            text += f"  · {commercial_label} 기준 절대오차: {abs(result.estimated_bpm - commercial_bpm):.2f} BPM\n"
        if pyvhr_bpm is not None:
            text += f"  · pyVHR 기준 절대오차: {abs(result.estimated_bpm - pyvhr_bpm):.2f} BPM\n"

    Path(output_path).write_text(text, encoding="utf-8")
