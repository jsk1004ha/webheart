"""Optional pyVHR comparison helpers.

pyVHR is intentionally not a hard dependency of this project because it is a
large research framework that is usually installed in its own conda environment.
These helpers make pyVHR usable as an external baseline when it is available,
while keeping the core reference-corrected rPPG workflow lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PyVHRResult:
    """Summary of one optional pyVHR run."""

    bpm: float
    method: str
    video_path: str
    times: np.ndarray
    bpm_series: np.ndarray
    status: str = "ok"
    message: str = ""


def _looks_like_posix_path(path_value: str | Path) -> bool:
    """Return True for WSL/Linux-style absolute paths such as /home/..."""
    return str(path_value).replace("\\", "/").startswith("/")


def _windows_path_to_wsl(path_value: str | Path) -> str:
    """Translate a Windows path to its WSL /mnt/<drive>/... equivalent."""
    text = str(Path(path_value).resolve())
    normalized = text.replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        drive = normalized[0].lower()
        return f"/mnt/{drive}{normalized[2:]}"
    return normalized


def _path_for_external_python(path_value: str | Path, use_wsl: bool) -> str:
    """Return a path string visible to the pyVHR Python process."""
    if use_wsl and os.name == "nt":
        return _windows_path_to_wsl(path_value)
    return str(path_value)


def _flatten_numeric(values: Any) -> np.ndarray:
    """Convert nested pyVHR outputs into a flat finite numeric array."""
    if values is None:
        return np.asarray([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        pieces = []
        for item in values:
            flat = _flatten_numeric(item)
            if flat.size:
                pieces.append(flat)
        if not pieces:
            return np.asarray([], dtype=float)
        arr = np.concatenate(pieces)
    arr = np.ravel(arr.astype(float, copy=False))
    return arr[np.isfinite(arr)]


def _parse_pipeline_output(output: Any) -> tuple[np.ndarray, np.ndarray]:
    """Handle common pyVHR Pipeline.run_on_video return shapes.

    pyVHR versions/documentation commonly show either ``(time, BPM,
    uncertainty)`` or ``(bvps, timesES, bpmES)``.  In both cases the BPM values
    are the second or third element, so this parser chooses the last numeric
    element that looks like a plausible BPM series.
    """
    if not isinstance(output, tuple):
        bpm_series = _flatten_numeric(output)
        return np.arange(bpm_series.size, dtype=float), bpm_series

    numeric_parts = [_flatten_numeric(part) for part in output]
    plausible = [part for part in numeric_parts if part.size and 30.0 <= float(np.nanmedian(part)) <= 240.0]
    bpm_series = plausible[-1] if plausible else (numeric_parts[-1] if numeric_parts else np.asarray([], dtype=float))

    times = np.asarray([], dtype=float)
    for part in numeric_parts:
        if part.size == bpm_series.size and part.size > 1 and float(np.nanmedian(part)) < 10_000:
            # Prefer a monotonic-ish time vector if one exists.
            diffs = np.diff(part)
            if np.count_nonzero(diffs >= 0) >= max(1, diffs.size // 2):
                times = part
                break
    if times.size != bpm_series.size:
        times = np.arange(bpm_series.size, dtype=float)
    return times, bpm_series


def run_pyvhr_on_video(
    video_path: str | Path,
    method: str = "cpu_POS",
    winsize: float = 6.0,
    roi_method: str = "convexhull",
    roi_approach: str = "hol",
    estimate: str = "median",
) -> PyVHRResult:
    """Run pyVHR Pipeline on a video and return the median BPM estimate.

    Raises ImportError with installation guidance when pyVHR is unavailable.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"pyVHR 입력 영상 파일을 찾을 수 없습니다: {path}")

    try:
        from pyVHR.analysis.pipeline import Pipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pyVHR가 현재 Python 환경에 설치되어 있지 않습니다. pyVHR 비교를 사용하려면 "
            "별도 conda 환경에서 `pip install pyvhr-cpu` 또는 프로젝트 안내에 맞게 pyVHR를 설치하세요. "
            "설치가 어렵다면 pyVHR로 따로 얻은 평균 BPM을 `--pyvhr-bpm`으로 입력할 수 있습니다."
        ) from exc

    pipe = Pipeline()
    requested_kwargs = {
        "winsize": winsize,
        "roi_method": roi_method,
        "roi_approach": roi_approach,
        "method": method,
        "estimate": estimate,
        "bpm_type": "welch",
        "verb": False,
    }
    signature = inspect.signature(pipe.run_on_video)
    accepted = set(signature.parameters)
    kwargs = {key: value for key, value in requested_kwargs.items() if key in accepted}
    # pyVHR 1.x hard-codes a 6 second window and accepts bpm_type; newer docs
    # show winsize/estimate.  Filtering by the live signature lets both work.
    output = pipe.run_on_video(str(path), **kwargs)

    times, bpm_series = _parse_pipeline_output(output)
    if bpm_series.size == 0:
        raise RuntimeError("pyVHR 실행은 완료됐지만 BPM 값을 추출하지 못했습니다.")

    return PyVHRResult(
        bpm=float(np.median(bpm_series)),
        method=method,
        video_path=str(path),
        times=times,
        bpm_series=bpm_series,
    )


def save_pyvhr_series(result: PyVHRResult, output_path: str | Path) -> None:
    """Save pyVHR time-varying BPM estimates for audit/reporting."""
    length = min(result.times.size, result.bpm_series.size)
    df = pd.DataFrame(
        {
            "time_sec": result.times[:length],
            "pyvhr_bpm": result.bpm_series[:length],
            "pyvhr_method": result.method,
            "pyvhr_video": result.video_path,
        }
    )
    df.to_csv(output_path, index=False)


def run_pyvhr_with_python(
    python_executable: str | Path,
    video_path: str | Path,
    output_csv: str | Path,
    method: str = "cpu_POS",
    winsize: float = 6.0,
    roi_method: str = "convexhull",
    roi_approach: str = "hol",
    estimate: str = "median",
) -> PyVHRResult:
    """Run pyVHR in another Python environment and return its summary.

    This is useful because pyVHR is often installed in a dedicated conda/venv
    while the main project runs in a lightweight environment.
    """
    use_wsl = os.name == "nt" and _looks_like_posix_path(python_executable)
    python_path = Path(python_executable)
    if not use_wsl and not python_path.exists():
        raise FileNotFoundError(f"pyVHR Python 실행 파일을 찾을 수 없습니다: {python_path}")

    repo_root = Path(__file__).resolve().parents[1]
    payload = {
        "repo_root": _path_for_external_python(repo_root, use_wsl=use_wsl),
        "video_path": _path_for_external_python(video_path, use_wsl=use_wsl),
        "output_csv": _path_for_external_python(output_csv, use_wsl=use_wsl),
        "method": method,
        "winsize": winsize,
        "roi_method": roi_method,
        "roi_approach": roi_approach,
        "estimate": estimate,
    }
    script = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
sys.path.insert(0, payload["repo_root"])
from src.pyvhr_integration import run_pyvhr_on_video, save_pyvhr_series

result = run_pyvhr_on_video(
    payload["video_path"],
    method=payload["method"],
    winsize=payload["winsize"],
    roi_method=payload["roi_method"],
    roi_approach=payload["roi_approach"],
    estimate=payload["estimate"],
)
save_pyvhr_series(result, payload["output_csv"])
print(json.dumps({
    "bpm": result.bpm,
    "method": result.method,
    "video_path": result.video_path,
    "status": result.status,
    "message": result.message,
}))
"""
    command = ["wsl.exe", str(python_executable).replace("\\", "/"), "-c", script] if use_wsl else [str(python_path), "-c", script]
    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "pyVHR 외부 Python 실행에 실패했습니다.\n"
            f"command={' '.join(command)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("pyVHR 외부 Python 실행은 성공했지만 결과 JSON을 받지 못했습니다.")
    try:
        data = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"pyVHR 결과 JSON 파싱 실패: {completed.stdout}") from exc
    return PyVHRResult(
        bpm=float(data["bpm"]),
        method=str(data.get("method", method)),
        video_path=str(data.get("video_path", video_path)),
        times=np.asarray([], dtype=float),
        bpm_series=np.asarray([], dtype=float),
        status=str(data.get("status", "ok")),
        message=str(data.get("message", "")),
    )
