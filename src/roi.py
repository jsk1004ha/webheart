"""ROI selection utilities for manual selection and Haar-cascade face detection."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

ROI = Tuple[int, int, int, int]


def clip_roi(roi: ROI, frame_shape: Tuple[int, int, int]) -> ROI:
    """Clip an ROI tuple ``(x, y, w, h)`` so it stays inside the frame."""
    x, y, w, h = [int(v) for v in roi]
    height, width = frame_shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return x, y, w, h


def is_valid_roi(roi: ROI, min_size: int = 20) -> bool:
    """Return True when the ROI is large enough for stable pixel averaging."""
    _, _, w, h = roi
    return w >= min_size and h >= min_size and (w * h) >= min_size * min_size


def select_manual_roi(
    frame_bgr: np.ndarray,
    min_size: int = 20,
    label: str = "skin",
    instruction: Optional[str] = None,
) -> ROI:
    """Let the user choose an ROI with OpenCV and retry when the box is too small.

    The input frame is OpenCV BGR. The user should choose a visible skin area such
    as forehead or cheek, not eyes/hair/background.
    """
    if instruction is None:
        if label == "reference":
            instruction = (
                "기준 ROI 선택: 벽, 옷, 배경처럼 혈류 변화가 없는 영역을 드래그한 뒤 "
                "Enter/Space를 누르세요. 취소하려면 c를 누릅니다."
            )
        else:
            instruction = (
                "피부 ROI 선택: 이마 또는 볼처럼 피부가 잘 보이는 영역을 드래그한 뒤 "
                "Enter/Space를 누르세요. 취소하려면 c를 누릅니다."
            )
    print(instruction)

    while True:
        display = frame_bgr.copy()
        cv2.putText(
            display,
            (
                "Select reference ROI (wall/clothes/background)"
                if label == "reference"
                else "Select skin ROI (forehead/cheek)"
            )
            + ", then Enter/Space",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        window_name = "Select reference ROI" if label == "reference" else "Select skin ROI"
        roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        clipped = clip_roi(tuple(map(int, roi)), frame_bgr.shape)
        if is_valid_roi(clipped, min_size=min_size):
            return clipped
        print(f"선택한 ROI가 너무 작습니다. 최소 {min_size}x{min_size} 픽셀 이상으로 다시 선택하세요.")


def _load_haar_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face Haar cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV Haar Cascade 파일을 불러오지 못했습니다.")
    return detector


def detect_face_roi(frame_bgr: np.ndarray, min_size: int = 20) -> Optional[ROI]:
    """Detect a face and return an approximate forehead skin ROI.

    A full face box contains non-skin regions. This function chooses a smaller
    rectangle near the upper-middle face, approximating the forehead. It is only a
    lightweight OpenCV-based helper and may need manual fallback under poor light.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = _load_haar_cascade()
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    # Use the largest detected face, usually the closest/primary subject.
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))

    # Approximate forehead ROI inside the face box.
    roi = (
        int(x + 0.25 * w),
        int(y + 0.12 * h),
        int(0.50 * w),
        int(0.18 * h),
    )
    roi = clip_roi(roi, frame_bgr.shape)
    if is_valid_roi(roi, min_size=min_size):
        return roi

    # Fallback to a cheek-like region if the forehead estimate is too small.
    cheek_roi = (
        int(x + 0.18 * w),
        int(y + 0.45 * h),
        int(0.25 * w),
        int(0.25 * h),
    )
    cheek_roi = clip_roi(cheek_roi, frame_bgr.shape)
    if is_valid_roi(cheek_roi, min_size=min_size):
        return cheek_roi
    return None


def choose_roi(frame_bgr: np.ndarray, mode: str = "manual", min_size: int = 20) -> ROI:
    """Choose an ROI using either manual selection or Haar face detection.

    If face detection fails, the function falls back to manual selection.
    """
    if mode not in {"manual", "face"}:
        raise ValueError("ROI mode는 'manual' 또는 'face'만 지원합니다.")

    if mode == "face":
        roi = detect_face_roi(frame_bgr, min_size=min_size)
        if roi is not None:
            x, y, w, h = roi
            print(f"Haar Cascade 얼굴 검출 기반 ROI 사용: x={x}, y={y}, w={w}, h={h}")
            return roi
        print("얼굴 ROI 자동 검출에 실패했습니다. manual ROI 선택으로 전환합니다.")

    return select_manual_roi(frame_bgr, min_size=min_size, label="skin")


def choose_reference_roi(frame_bgr: np.ndarray, min_size: int = 20) -> ROI:
    """Manually choose a non-blood-flow reference ROI for illumination correction."""
    return select_manual_roi(frame_bgr, min_size=min_size, label="reference")
