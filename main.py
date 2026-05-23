"""CLI entry point for reference-corrected rPPG RGB reflection analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.capture import capture_video_rgb, capture_webcam_rgb
from src.plotting import create_all_plots
from src.pyvhr_integration import PyVHRResult, run_pyvhr_on_video, run_pyvhr_with_python, save_pyvhr_series
from src.rppg_methods import METHOD_LABELS, analyze_rppg_methods
from src.signal_processing import preprocess_rgb_timeseries, results_to_dataframe
from src.utils import build_correction_comparison, create_session_dir, generate_synthetic_rgb, print_disclaimer, write_report_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="피부 ROI와 기준 ROI의 반사광 비율로 rPPG 심박 후보 주파수를 탐구하는 비의료용 분석 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["webcam", "video", "csv", "demo"], required=True, help="실행 모드")
    parser.add_argument("--duration", type=float, default=60.0, help="webcam/demo 촬영 또는 생성 시간(초). video 모드에서는 최대 분석 시간")
    parser.add_argument("--video", type=str, help="video 모드 입력 영상 경로")
    parser.add_argument("--csv", type=str, help="csv 모드 입력 RGB CSV 경로")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam 모드 OpenCV 카메라 index")
    parser.add_argument("--roi", choices=["manual", "face"], default="manual", help="피부 ROI 선택 방식. 기준 ROI는 별도로 수동 선택")
    parser.add_argument("--low-hz", type=float, default=0.8, help="심박 탐색 최소 주파수(Hz), 0.8Hz=48BPM")
    parser.add_argument("--high-hz", type=float, default=2.0, help="심박 탐색 최대 주파수(Hz), 2.0Hz=120BPM")
    parser.add_argument("--drop-first-sec", type=float, default=2.0, help="카메라 자동노출 안정화를 위해 분석 전 제거할 첫 구간(초)")
    parser.add_argument("--bpm", type=float, default=72.0, help="demo 모드 synthetic 심박수(BPM)")
    parser.add_argument("--noise", type=float, default=0.03, help="demo 모드 synthetic 잡음 크기")
    parser.add_argument("--output-dir", type=str, default="outputs", help="session output 상위 폴더")
    parser.add_argument("--no-preview", action="store_true", help="webcam/video 미리보기 창을 표시하지 않음(자동화/헤드리스 환경용)")
    parser.add_argument(
        "--commercial-bpm",
        type=float,
        help="동시에 측정한 상용 웹캠 심박수 측정기 BPM. 입력하면 방식별 절대오차 비교를 생성",
    )
    parser.add_argument(
        "--commercial-label",
        type=str,
        default="상용 웹캠 심박수 측정기",
        help="비교 기준으로 사용할 상용 측정기 이름",
    )
    parser.add_argument("--pyvhr-bpm", type=float, help="pyVHR로 별도 계산한 대표/평균 BPM. 입력하면 pyVHR 대비 오차 비교 생성")
    parser.add_argument("--run-pyvhr", action="store_true", help="pyVHR가 설치되어 있으면 입력 영상에 pyVHR Pipeline을 함께 실행")
    parser.add_argument(
        "--pyvhr-python",
        type=str,
        default="/home/js10041530/.venvs/webheart-pyvhr/bin/python",
        help="pyVHR가 설치된 Python 실행 파일. 기본값은 이 프로젝트에서 만든 별도 pyVHR venv",
    )
    parser.add_argument("--pyvhr-video", type=str, help="pyVHR에 사용할 영상 경로. 생략하면 video 모드의 --video를 사용")
    parser.add_argument("--pyvhr-method", type=str, default="cpu_POS", help="pyVHR method 이름 예: cpu_POS, cpu_CHROM")
    parser.add_argument("--pyvhr-winsize", type=float, default=6.0, help="pyVHR window size(sec)")
    parser.add_argument("--pyvhr-roi-method", type=str, default="convexhull", help="pyVHR roi_method")
    parser.add_argument("--pyvhr-roi-approach", type=str, default="hol", help="pyVHR roi_approach 예: hol, patches")
    parser.add_argument("--pyvhr-estimate", type=str, default="median", help="pyVHR BPM estimate 방식")
    return parser


def resolve_pyvhr_comparison(
    args: argparse.Namespace,
    session_dir: Path,
    captured_video_path: Path | None = None,
) -> tuple[float | None, PyVHRResult | None, str | None]:
    """Resolve pyVHR comparison BPM from manual input or optional pyVHR execution."""
    if args.pyvhr_bpm is not None and args.pyvhr_bpm <= 0:
        raise ValueError("--pyvhr-bpm은 0보다 커야 합니다.")
    if args.pyvhr_bpm is not None:
        return float(args.pyvhr_bpm), None, "pyVHR BPM을 수동 입력값으로 사용했습니다."
    if not args.run_pyvhr:
        return None, None, None

    pyvhr_video = args.pyvhr_video or (str(captured_video_path) if captured_video_path is not None else None) or (
        args.video if args.mode == "video" else None
    )
    if pyvhr_video is None:
        raise ValueError(
            "--run-pyvhr는 영상 파일이 필요합니다. webcam 모드는 자동 녹화 영상을 사용하고, "
            "video/csv 모드는 --video 또는 --pyvhr-video를 지정하세요."
        )

    if args.pyvhr_python:
        result = run_pyvhr_with_python(
            args.pyvhr_python,
            pyvhr_video,
            output_csv=session_dir / "pyvhr_timeseries.csv",
            method=args.pyvhr_method,
            winsize=args.pyvhr_winsize,
            roi_method=args.pyvhr_roi_method,
            roi_approach=args.pyvhr_roi_approach,
            estimate=args.pyvhr_estimate,
        )
    else:
        result = run_pyvhr_on_video(
            pyvhr_video,
            method=args.pyvhr_method,
            winsize=args.pyvhr_winsize,
            roi_method=args.pyvhr_roi_method,
            roi_approach=args.pyvhr_roi_approach,
            estimate=args.pyvhr_estimate,
        )
        save_pyvhr_series(result, session_dir / "pyvhr_timeseries.csv")
    return result.bpm, result, f"pyVHR {result.method} 실행 완료"


def load_input_timeseries(args: argparse.Namespace, session_dir: Path) -> tuple[pd.DataFrame, str, Path | None]:
    """Load or capture a raw RGB time series according to the selected mode."""
    if args.mode == "webcam":
        record_video_path = session_dir / "webcam_capture_for_pyvhr.mp4" if args.run_pyvhr and not args.pyvhr_video else None
        df = capture_webcam_rgb(
            duration_sec=args.duration,
            camera_index=args.camera_index,
            roi_mode=args.roi,
            show_preview=not args.no_preview,
            record_video_path=record_video_path,
        )
        source = "webcam"
        if record_video_path is not None:
            source += f" (recorded video: {record_video_path.name})"
        return df, source, record_video_path

    if args.mode == "video":
        if not args.video:
            raise ValueError("video 모드에서는 --video 경로가 필요합니다.")
        df = capture_video_rgb(
            video_path=args.video,
            roi_mode=args.roi,
            duration_sec=args.duration,
            show_preview=not args.no_preview,
        )
        return df, f"video: {args.video}", Path(args.video)

    if args.mode == "csv":
        if not args.csv:
            raise ValueError("csv 모드에서는 --csv 경로가 필요합니다.")
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        pyvhr_video = Path(args.pyvhr_video) if args.pyvhr_video else None
        return pd.read_csv(csv_path), f"csv: {csv_path}", pyvhr_video

    if args.mode == "demo":
        df = generate_synthetic_rgb(bpm=args.bpm, duration_sec=args.duration, noise=args.noise)
        return df, f"demo synthetic: {args.bpm:.1f} BPM, noise={args.noise}", None

    raise ValueError(f"지원하지 않는 mode입니다: {args.mode}")


def run_analysis(args: argparse.Namespace) -> Path:
    """Run the full extraction, preprocessing, rPPG analysis, and output pipeline."""
    if args.low_hz <= 0 or args.high_hz <= args.low_hz:
        raise ValueError("--low-hz와 --high-hz는 0 < low < high 조건을 만족해야 합니다.")
    if args.duration <= 0:
        raise ValueError("--duration은 0보다 커야 합니다.")
    if args.commercial_bpm is not None and args.commercial_bpm <= 0:
        raise ValueError("--commercial-bpm은 0보다 커야 합니다.")

    session_dir = create_session_dir(args.output_dir)
    raw_df, source_label, captured_video_path = load_input_timeseries(args, session_dir)
    pyvhr_bpm, pyvhr_result, pyvhr_status = resolve_pyvhr_comparison(args, session_dir, captured_video_path=captured_video_path)

    processed = preprocess_rgb_timeseries(raw_df, drop_first_sec=args.drop_first_sec)
    filtered_signals, results = analyze_rppg_methods(
        processed.dataframe,
        fs=processed.fs,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        include_rg=False,
    )
    summary_df = results_to_dataframe(results, commercial_bpm=args.commercial_bpm, pyvhr_bpm=pyvhr_bpm)
    correction_comparison_df = build_correction_comparison(results, commercial_bpm=args.commercial_bpm, pyvhr_bpm=pyvhr_bpm)

    processed.dataframe.to_csv(session_dir / "rgb_timeseries.csv", index=False)
    summary_df.to_csv(session_dir / "result_summary.csv", index=False)
    correction_comparison_df.to_csv(session_dir / "correction_comparison.csv", index=False)
    if args.commercial_bpm is not None:
        summary_df.to_csv(session_dir / "commercial_comparison.csv", index=False)
    if pyvhr_bpm is not None:
        summary_df.to_csv(session_dir / "pyvhr_comparison.csv", index=False)
    write_report_summary(
        session_dir / "report_summary.txt",
        processed=processed,
        results=results,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        source_mode=source_label,
        correction_comparison_df=correction_comparison_df,
        commercial_bpm=args.commercial_bpm,
        commercial_label=args.commercial_label,
        pyvhr_bpm=pyvhr_bpm,
        pyvhr_method=pyvhr_result.method if pyvhr_result else args.pyvhr_method if pyvhr_bpm is not None else None,
        pyvhr_status=pyvhr_status,
    )
    create_all_plots(
        processed.dataframe,
        filtered_signals=filtered_signals,
        results=results,
        summary_df=summary_df,
        correction_comparison_df=correction_comparison_df,
        output_dir=session_dir,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        commercial_label=args.commercial_label,
    )

    print("\n분석 결과 요약")
    print("-" * 60)
    for _, row in summary_df.iterrows():
        label = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        print(
            f"{label}: {row['estimated_bpm']:.2f} BPM "
            f"(f_peak={row['f_peak_hz']:.4f} Hz, snr_like={row['snr_like']:.3g})"
        )
        if args.commercial_bpm is not None:
            print(f"  - {args.commercial_label} 기준 절대오차: {row['abs_error_vs_commercial_bpm']:.2f} BPM")
        if pyvhr_bpm is not None:
            print(f"  - pyVHR 기준 절대오차: {row['abs_error_vs_pyvhr_bpm']:.2f} BPM")
    if pyvhr_bpm is not None:
        print(f"pyVHR 비교 BPM: {pyvhr_bpm:.2f} BPM ({pyvhr_status})")
    print("-" * 60)
    print(f"저장된 output 폴더: {session_dir.resolve()}")
    return session_dir


def main(argv: list[str] | None = None) -> int:
    """CLI main function with friendly error handling."""
    parser = build_parser()
    args = parser.parse_args(argv)
    print_disclaimer()

    try:
        run_analysis(args)
        return 0
    except KeyboardInterrupt:
        print("\n사용자 중단으로 종료했습니다.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should show friendly messages for all expected failures.
        print(f"\n오류: {exc}", file=sys.stderr)
        if args.mode == "webcam":
            print("웹캠 없이도 검증하려면 예: python main.py --mode demo --bpm 72 --duration 60 --noise 0.03", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
