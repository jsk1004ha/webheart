"""Video/webcam capture helpers for skin and reference ROI RGB time series."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .roi import ROI, choose_reference_roi, choose_roi, clip_roi, is_valid_roi


def compute_roi_mean_rgb(frame_bgr: np.ndarray, roi: ROI) -> Tuple[float, float, float]:
    """Compute mean ``(R, G, B)`` values inside an OpenCV BGR frame ROI.

    OpenCV reads frames in BGR order. For this project the saved columns must be
    true RGB, so the ROI pixels are explicitly converted with ``COLOR_BGR2RGB``
    before averaging.
    """
    x, y, w, h = clip_roi(roi, frame_bgr.shape)
    if not is_valid_roi((x, y, w, h), min_size=5):
        raise ValueError("ROI가 프레임 밖에 있거나 너무 작아 RGB 평균을 계산할 수 없습니다.")
    roi_bgr = frame_bgr[y : y + h, x : x + w]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)
    return float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])


def _draw_capture_overlay(
    frame_bgr: np.ndarray,
    skin_roi: ROI,
    reference_roi: ROI | None,
    remaining_sec: Optional[float],
    sample_count: int,
    source_label: str,
) -> np.ndarray:
    """Draw ROI and progress text on a frame for live capture/preview."""
    display = frame_bgr.copy()
    x, y, w, h = skin_roi
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(display, "skin", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    if reference_roi is not None:
        rx, ry, rw, rh = reference_roi
        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 180, 0), 2)
        cv2.putText(
            display,
            "reference",
            (rx, max(18, ry - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
    if remaining_sec is None:
        time_text = "remaining: video"
    else:
        time_text = f"remaining: {max(0.0, remaining_sec):.1f}s"
    lines = [
        "Exploration tool only - NOT medical diagnosis",
        f"{source_label} | {time_text} | samples: {sample_count}",
        "Press q to stop early",
    ]
    y0 = 28
    for line in lines:
        cv2.putText(display, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y0 += 26
    return display


def _dataframe_from_records(records: list[dict[str, float]]) -> pd.DataFrame:
    """Build and validate an RGB time-series DataFrame from capture records."""
    if not records:
        raise RuntimeError("RGB 시계열을 추출하지 못했습니다. ROI와 입력 영상을 확인하세요.")
    columns = ["time_sec", "R", "G", "B"]
    if {"skin_R", "skin_G", "skin_B", "ref_R", "ref_G", "ref_B"}.issubset(records[0]):
        columns.extend(["skin_R", "skin_G", "skin_B", "ref_R", "ref_G", "ref_B"])
    df = pd.DataFrame.from_records(records, columns=columns)
    if len(df) < 3:
        raise RuntimeError("분석할 프레임 수가 너무 적습니다. 더 긴 영상을 사용하세요.")
    return df


def _open_video_writer(record_video_path: str | Path | None, first_frame: np.ndarray, fps: float) -> cv2.VideoWriter | None:
    """Create an MP4 writer for pyVHR/audit video recording when requested."""
    if record_video_path is None:
        return None
    path = Path(record_video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    safe_fps = float(fps) if np.isfinite(fps) and fps > 1e-6 else 30.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), safe_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"녹화 파일을 열 수 없습니다: {path}")
    return writer


def capture_webcam_rgb(
    duration_sec: float,
    camera_index: int = 0,
    roi_mode: str = "manual",
    min_roi_size: int = 20,
    show_preview: bool = True,
    record_video_path: str | Path | None = None,
) -> pd.DataFrame:
    """Capture webcam frames and return per-frame skin/reference ROI RGB means.

    The timestamp is measured with ``time.perf_counter`` so it reflects real
    capture timing instead of only frame numbers.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라(index={camera_index})를 열 수 없습니다. 웹캠 연결/권한을 확인하거나 "
            "웹캠 없이 검증하려면 `python main.py --mode demo`를 실행하세요."
        )

    try:
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise RuntimeError("카메라에서 첫 프레임을 읽지 못했습니다. demo 모드를 사용해 검증할 수 있습니다.")

        skin_roi = choose_roi(first_frame, mode=roi_mode, min_size=min_roi_size)
        reference_roi = choose_reference_roi(first_frame, min_size=min_roi_size)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        writer = _open_video_writer(record_video_path, first_frame, fps)
        records: list[dict[str, float]] = []
        start = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("프레임 읽기에 실패하여 촬영을 종료합니다.")
                    break
                elapsed = time.perf_counter() - start
                if elapsed > duration_sec:
                    break

                if writer is not None:
                    writer.write(frame)

                skin_r, skin_g, skin_b = compute_roi_mean_rgb(frame, skin_roi)
                ref_r, ref_g, ref_b = compute_roi_mean_rgb(frame, reference_roi)
                records.append(
                    {
                        "time_sec": elapsed,
                        "R": skin_r,
                        "G": skin_g,
                        "B": skin_b,
                        "skin_R": skin_r,
                        "skin_G": skin_g,
                        "skin_B": skin_b,
                        "ref_R": ref_r,
                        "ref_G": ref_g,
                        "ref_B": ref_b,
                    }
                )

                if show_preview:
                    remaining = duration_sec - elapsed
                    display = _draw_capture_overlay(frame, skin_roi, reference_roi, remaining, len(records), "webcam")
                    cv2.imshow("rPPG RGB capture", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("사용자 입력(q)으로 촬영을 조기 종료했습니다.")
                        break
        finally:
            if writer is not None:
                writer.release()
                print(f"pyVHR/검증용 녹화 영상 저장: {Path(record_video_path).resolve()}")
        return _dataframe_from_records(records)
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


def capture_video_rgb(
    video_path: str | Path,
    roi_mode: str = "manual",
    duration_sec: Optional[float] = None,
    min_roi_size: int = 20,
    show_preview: bool = True,
) -> pd.DataFrame:
    """Read a saved video file and return per-frame skin/reference ROI RGB means.

    Timestamps are based on the video FPS when available. If FPS metadata is not
    valid, OpenCV's POS_MSEC timestamp is used as a fallback.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {path}")

    try:
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise RuntimeError("영상에서 첫 프레임을 읽지 못했습니다.")

        skin_roi = choose_roi(first_frame, mode=roi_mode, min_size=min_roi_size)
        reference_roi = choose_reference_roi(first_frame, min_size=min_roi_size)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        use_fps_time = np.isfinite(fps) and fps > 1e-6

        records: list[dict[str, float]] = []
        frame_index = 0
        # Include the first frame after ROI selection.
        current_frame: Optional[np.ndarray] = first_frame

        while current_frame is not None:
            if use_fps_time:
                timestamp = frame_index / fps
            else:
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

            if duration_sec is not None and timestamp > duration_sec:
                break

            skin_r, skin_g, skin_b = compute_roi_mean_rgb(current_frame, skin_roi)
            ref_r, ref_g, ref_b = compute_roi_mean_rgb(current_frame, reference_roi)
            records.append(
                {
                    "time_sec": timestamp,
                    "R": skin_r,
                    "G": skin_g,
                    "B": skin_b,
                    "skin_R": skin_r,
                    "skin_G": skin_g,
                    "skin_B": skin_b,
                    "ref_R": ref_r,
                    "ref_G": ref_g,
                    "ref_B": ref_b,
                }
            )

            if show_preview:
                remaining = None if duration_sec is None else duration_sec - timestamp
                display = _draw_capture_overlay(current_frame, skin_roi, reference_roi, remaining, len(records), "video")
                cv2.imshow("rPPG RGB video analysis", display)
                # waitKey is also needed so the preview window remains responsive.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("사용자 입력(q)으로 영상 분석을 조기 종료했습니다.")
                    break

            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                break
            frame_index += 1
            current_frame = next_frame

        return _dataframe_from_records(records)
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
