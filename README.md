# 피부 ROI와 기준 ROI의 반사광 비율을 이용한 참조 보정형 rPPG 심박 신호 추출 연구

> **중요:** 이 프로젝트는 의료 진단용 프로그램이 아닙니다. 웹캠 또는 저장 영상에서 얼굴 피부 영역의 RGB 평균값 변화를 분석해 심박과 관련될 수 있는 주파수 성분을 탐구하는 **물리·정보 융합 주제연구용 시뮬레이션/분석 도구**입니다. 결과를 건강 판단, 진단, 치료 결정에 사용하면 안 됩니다.

## 1. 프로젝트 설명

이 프로그램은 컴퓨터 웹캠 또는 저장된 영상 파일에서 얼굴 **피부 ROI**와 혈류 변화가 거의 없는 **기준 ROI**(벽, 옷, 배경 등)를 같은 프레임에서 선택하고, 두 ROI 내부 픽셀의 평균 `R`, `G`, `B` 값을 시간에 따라 동시에 추출합니다. 이후 rPPG(remote photoplethysmography) 원리에 따라 RGB 시계열을 전처리하고, 피부/기준 반사광 비율을 이용한 참조 보정 신호를 기존 Green 방식과 비교해 BPM 후보를 추정합니다.

지원하는 분석 방식은 다음과 같습니다.

1. **Raw Green 방식**: 피부 ROI의 원본 초록색 평균값 `G_skin(t)`를 사용합니다.
2. **Normalized Green 방식**: 피부 ROI RGB를 평균 기준으로 정규화한 `g_skin(t)`를 사용합니다.
3. **Skin RGB 조합 방식**: 피부 ROI의 `2*g(t) - r(t) - b(t)`를 사용합니다.
4. **참조 보정 Green 방식**: `A_G(t) = -log((G_skin(t)+eps)/(G_ref(t)+eps))`를 사용합니다.
5. **참조 보정 RGB difference 방식**: `2*A_G(t) - A_R(t) - A_B(t)`를 사용합니다.

분석 후 보고서에 바로 넣을 수 있도록 CSV, PNG 그래프, 한국어 요약문을 자동 생성합니다.

## 2. 물리학적 원리

- 가시광선은 전자기파의 한 종류입니다.
- 피부와 혈액은 입사한 빛의 일부를 흡수하고 일부를 반사합니다.
- 심장 박동에 따라 말초 혈관의 혈액량이 주기적으로 변하면 피부에서 반사되는 빛의 세기도 아주 미세하게 변합니다.
- 웹캠의 이미지 센서는 이 반사광을 프레임별 RGB 값으로 기록합니다.
- 같은 피부 영역의 RGB 평균값을 시간에 따라 기록하면 미세한 반사광 변화의 시계열을 얻을 수 있습니다.
- 단순 Green 방식은 `G_skin(t)` 안에 심박 신호뿐 아니라 조명 변화, 카메라 자동 노출, 움직임 잡음이 함께 들어갑니다.
- 기준 ROI는 혈류 변화가 거의 없지만 같은 조명과 카메라 노출 변화를 받으므로, 피부 ROI와 기준 ROI의 비율을 계산하면 공통 조명 잡음을 줄일 수 있습니다.
- 참조 보정 흡광도는 다음처럼 계산합니다.

```text
A_c(t) = -log((I_skin,c(t)+eps) / (I_ref,c(t)+eps))
```

피부와 기준 영역에 들어오는 공통 조명 세기를 `I0(t)`라고 보면 `I_skin(t)=I0(t)R_skin(t)`, `I_ref(t)=I0(t)R_ref(t)`로 단순화할 수 있습니다. 두 값을 나누면 `I0(t)`가 약하게 상쇄되어 상대 반사율·흡광도 변화에 더 집중할 수 있습니다.
- 이 시계열에 band-pass filter를 적용하고 FFT로 주파수 분석을 수행하면 심박 주파수 후보를 찾을 수 있습니다.
- FFT에서 가장 큰 peak 주파수를 `f_peak`라고 하면, 심박수 후보는 다음 식으로 계산합니다.

```text
BPM = 60 × f_peak
```

기본 분석 범위는 `0.8~2.0 Hz`이며, 이는 약 `48~120 BPM`에 해당합니다.

## 3. 설치 방법

Python 3.10 이상을 권장합니다.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 실행 예시

### 4.1 웹캠 실시간 측정 모드

```bash
python main.py --mode webcam --duration 60 --roi manual
```

- 웹캠을 열고 60초 동안 촬영합니다.
- 첫 프레임에서 피부 ROI와 기준 ROI를 순서대로 직접 선택합니다.
- 피부 ROI는 이마나 볼처럼 피부가 잘 보이고 그림자/머리카락이 적은 영역을 선택하세요.
- 기준 ROI는 벽, 옷, 배경, 머리카락처럼 혈류 변화가 없고 움직임이 적은 영역을 선택하세요.
- 촬영 중 ROI 박스, 남은 시간, 샘플 수가 표시됩니다.
- `q`를 누르면 조기 종료할 수 있습니다.

Haar Cascade 얼굴 검출 기반 피부 ROI를 사용하려면 다음처럼 실행합니다. 이 경우에도 기준 ROI는 별도로 수동 선택합니다.

```bash
python main.py --mode webcam --duration 60 --roi face
```

얼굴 검출에 실패하면 manual ROI 선택으로 자동 전환됩니다.

### 4.2 저장 영상 분석 모드

```bash
python main.py --mode video --video data/sample.mp4 --roi manual
```

저장된 영상 파일을 불러와 첫 프레임에서 ROI를 선택하고 동일한 분석을 수행합니다. 기본적으로 최대 60초를 분석합니다. 더 긴 구간을 분석하려면 `--duration` 값을 늘리세요.

### 4.3 CSV 재분석 모드

```bash
python main.py --mode csv --csv outputs/session_xxx/rgb_timeseries.csv
```

이미 저장된 RGB 시계열 CSV를 다시 분석하여 그래프와 결과 파일을 새 session 폴더에 생성합니다.

### 4.4 합성 데이터 demo 모드

```bash
python main.py --mode demo --bpm 72 --duration 60 --noise 0.03
```

웹캠이 없어도 실행 가능한 검증용 모드입니다. 지정한 BPM의 심박 주파수 성분, 느린 조명 변화, 잡음을 포함한 synthetic RGB 데이터를 생성하고 같은 분석 파이프라인으로 BPM을 추정합니다.

### 4.5 보정 vs 노보정 + 상용 웹캠 심박수 측정기 비교

상용 웹캠 심박수 측정기(예: 웹캠 기반 pulse/heart-rate 앱)를 같은 시간에 켜고, 촬영이 끝날 때 표시된 BPM 또는 평균 BPM을 입력합니다.

```bash
python main.py --mode webcam --duration 60 --roi manual \
  --commercial-bpm 74 \
  --commercial-label "상용 웹캠 심박수 앱"
```

이 옵션을 넣으면 다음 비교가 자동 생성됩니다.

- **보정 vs 노보정 비교**: `correction_comparison.csv`, `09_correction_comparison.png`
- **상용 측정값 기준 오차 비교**: `commercial_comparison.csv`, `08_commercial_error_comparison.png`
- `result_summary.csv`에는 각 방식의 `correction_group`, `commercial_bpm`, `abs_error_vs_commercial_bpm` 컬럼이 추가됩니다.

보고서에서는 “상용 측정값이 절대적인 의료 기준은 아니지만, 같은 시간대에 측정한 외부 비교값으로 사용했다”고 명시하세요.

### 4.6 pyVHR와 함께 비교

`pyVHR`는 rPPG 연구용 Python 프레임워크입니다. pyVHR는 설치가 무겁고 별도 conda 환경을 권장하므로 이 프로젝트의 기본 필수 의존성에는 넣지 않았습니다. 사용할 수 있는 방식은 두 가지입니다.

#### A. pyVHR 결과 BPM을 직접 입력

pyVHR를 별도로 실행해 얻은 대표 BPM 또는 평균 BPM을 입력합니다.

```bash
python main.py --mode video --video data/sample.mp4 --roi manual \
  --pyvhr-bpm 73.5
```

#### B. pyVHR가 설치된 환경에서 같이 실행

웹캠 모드에서는 본 프로그램이 RGB/ROI 측정과 동시에 `webcam_capture_for_pyvhr.mp4`를 자동 녹화하고, 촬영이 끝나면 그 영상을 pyVHR에 넣어 비교합니다.

```bash
python main.py --mode webcam --duration 60 --roi manual \
  --run-pyvhr
```

저장 영상 파일이 이미 있으면 같은 영상 파일에 대해 pyVHR Pipeline을 함께 실행합니다.

```bash
python main.py --mode video --video data/sample.mp4 --roi manual \
  --run-pyvhr \
  --pyvhr-method cpu_POS
```

현재 기본 `--pyvhr-python` 값은 다음 별도 venv입니다.

```text
/home/js10041530/.venvs/webheart-pyvhr/bin/python
```

다른 conda/venv에 pyVHR를 설치했다면 다음처럼 바꿀 수 있습니다.

```bash
python main.py --mode video --video data/sample.mp4 --roi manual \
  --run-pyvhr \
  --pyvhr-python /path/to/pyvhr-env/bin/python
```

pyVHR를 별도 영상으로 돌리고 싶으면 다음처럼 지정합니다.

```bash
python main.py --mode csv --csv outputs/session_xxx/rgb_timeseries.csv \
  --run-pyvhr \
  --pyvhr-video data/sample.mp4
```

pyVHR 비교를 사용하면 다음 파일이 추가됩니다.

- `webcam_capture_for_pyvhr.mp4`: webcam + `--run-pyvhr`에서 자동 저장되는 pyVHR 입력 영상
- `pyvhr_comparison.csv`: 각 방식의 pyVHR 대비 절대오차
- `pyvhr_timeseries.csv`: `--run-pyvhr`로 직접 실행했을 때 pyVHR의 시간별 BPM 추정값
- `10_pyvhr_error_comparison.png`: pyVHR 대비 절대오차 그래프

설치 참고:

```bash
conda create -n pyvhr python=3.9
conda activate pyvhr
pip install pyvhr-cpu
```

pyVHR도 의료기기가 아니라 연구용 rPPG 비교 기준입니다. 따라서 보고서에서는 “상용 측정기, pyVHR, 본 알고리즘은 모두 비접촉 영상 기반 추정값이며 절대적인 의학적 정답은 아니다”라고 한계를 적는 것이 좋습니다.

## 5. 결과 파일 설명

각 실행은 `outputs/session_YYYYMMDD_HHMMSS/` 폴더를 만들고 다음 파일을 저장합니다.

| 파일 | 설명 |
| --- | --- |
| `rgb_timeseries.csv` | `time_sec, R, G, B, skin_R, skin_G, skin_B, ref_R, ref_G, ref_B, r_norm, g_norm, b_norm, A_R, A_G, A_B` 컬럼을 포함합니다. `R,G,B`는 기존 호환을 위한 피부 ROI 별칭입니다. |
| `result_summary.csv` | `method, correction_group, estimated_bpm, f_peak_hz, peak_strength, snr_like` 컬럼을 포함합니다. `--commercial-bpm` 입력 시 상용 기준 BPM과 절대오차 컬럼도 포함합니다. |
| `correction_comparison.csv` | 보정 전/후 방식 쌍의 BPM 차이, SNR-like 비율, 상용 기준 오차 감소량을 저장합니다. |
| `commercial_comparison.csv` | `--commercial-bpm` 입력 시 생성되며, 각 방식의 상용 측정값 대비 절대오차를 저장합니다. |
| `pyvhr_comparison.csv` | `--pyvhr-bpm` 또는 `--run-pyvhr` 입력 시 생성되며, 각 방식의 pyVHR 대비 절대오차를 저장합니다. |
| `pyvhr_timeseries.csv` | `--run-pyvhr` 입력 시 생성되며, pyVHR가 추정한 시간별 BPM 값을 저장합니다. |
| `report_summary.txt` | 탐구 보고서에 붙여넣을 수 있는 한국어 요약문입니다. 참조 ROI 사용 이유, 공통 조명 잡음 제거 원리, 방식별 BPM, 한계가 포함됩니다. |
| `01_rgb_raw.png` | 시간에 따른 피부 ROI와 기준 ROI의 원본 R, G, B 그래프입니다. |
| `02_rgb_normalized.png` | 평균 기준으로 정규화한 `r_norm`, `g_norm`, `b_norm` 시계열 그래프입니다. |
| `03_reference_absorbance.png` | `A_R`, `A_G`, `A_B` 참조 보정 흡광도 신호 그래프입니다. |
| `04_filtered_signals.png` | 각 방식의 band-pass filter 후 rPPG 후보 신호 그래프입니다. |
| `05_fft_spectrum.png` | FFT magnitude spectrum입니다. peak 위치와 estimated BPM이 표시됩니다. |
| `06_method_comparison.png` | 알고리즘별 estimated BPM과 SNR-like 지표를 비교하는 bar graph입니다. |
| `07_baseline_vs_reference.png` | 기존 방식과 참조 보정 방식의 시간 신호 및 FFT 스펙트럼을 나란히 비교한 그래프입니다. |
| `08_commercial_error_comparison.png` | `--commercial-bpm` 입력 시 생성되는 상용 측정값 대비 절대오차 그래프입니다. |
| `09_correction_comparison.png` | 보정 방식과 노보정 방식의 SNR-like 비율, 상용 기준 오차 감소량을 비교하는 그래프입니다. |
| `10_pyvhr_error_comparison.png` | `--pyvhr-bpm` 또는 `--run-pyvhr` 입력 시 생성되는 pyVHR 대비 절대오차 그래프입니다. |

`snr_like`는 정확한 의학적 SNR이 아니라, peak 주변 ±0.1 Hz의 파워와 나머지 탐색 범위의 평균 파워를 비교한 **상대적 peak 선명도 지표**입니다.

## 6. 보고서 작성 팁

고등학교 전자기학/물리학 주제연구 보고서에는 다음 흐름을 추천합니다.

1. **탐구 동기**
   - 비접촉 방식으로 생체 신호를 추정할 수 있다는 점에 대한 흥미
   - 스마트폰/웹캠 센서와 전자기파 반사 원리의 연결

2. **이론적 배경**
   - 가시광선이 전자기파라는 점
   - 피부와 혈액의 빛 흡수·반사
   - 심장 박동에 따른 혈액량 변화와 반사광 변화
   - RGB 이미지 센서와 디지털 영상 데이터
   - FFT와 `BPM = 60 × f_peak`

3. **프로그램 설계**
   - ROI 선택
   - RGB 평균값 추출
   - 정규화 및 band-pass filtering
   - Raw Green, Normalized Green, 참조 보정 Green, 참조 보정 RGB difference 방식 비교
   - 기준 ROI와 피부 ROI의 반사광 비율 및 로그 흡광도 계산
   - FFT peak 탐색 및 SNR-like 지표 계산

4. **실험 조건**
   - 조명 밝기, 카메라 위치, 촬영 시간, ROI 위치
   - 움직임을 줄이기 위한 조건
   - 상용 웹캠 심박수 측정기와 동시에 측정하고 `--commercial-bpm`으로 기준값 입력
   - 저장 영상이 있으면 pyVHR도 같은 영상에 적용하고 `--run-pyvhr` 또는 `--pyvhr-bpm`으로 비교
   - 가능하면 손목 맥박 등 수동 기준값도 보조 비교값으로 기록

5. **결과 분석**
   - `01_rgb_raw.png`로 원본 RGB 변화 관찰
   - `04_filtered_signals.png`로 필터링된 주기 신호 확인
   - `05_fft_spectrum.png`에서 peak 주파수와 BPM 해석
   - `06_method_comparison.png`와 `07_baseline_vs_reference.png`로 기존 방식과 참조 보정 방식 차이 비교
   - `08_commercial_error_comparison.png`로 상용 측정값 대비 각 방식의 절대오차 비교
   - `09_correction_comparison.png`로 보정 방식이 노보정 방식보다 peak 선명도 또는 기준 오차를 개선했는지 분석
   - `10_pyvhr_error_comparison.png`로 pyVHR 기준 결과와 본 알고리즘 결과의 차이 비교

6. **한계 및 개선점**
   - 자동 노출/화이트밸런스 영향
   - 얼굴 움직임 영향
   - 어두운 조명 영향
   - ROI 선택 위치 영향
   - 기준 맥박 측정 오차
   - 기준 ROI 위치별 성능 비교, 더 정교한 얼굴 추적, 조명 보정, 신호 품질 평가 방법 제안

## 7. 한계와 주의사항

- 카메라 자동 노출과 자동 화이트밸런스가 RGB 값을 변화시킬 수 있습니다.
- 얼굴 움직임, 표정 변화, ROI 이동은 신호에 큰 잡음을 만듭니다.
- 어두운 조명이나 깜빡이는 조명은 FFT peak를 왜곡할 수 있습니다.
- 이마/볼 등 ROI 선택 위치에 따라 결과가 달라질 수 있습니다.
- 수동 맥박 측정값도 기준값 오차가 있을 수 있습니다.
- 본 프로그램은 의료 목적이 아니라 물리·정보 융합 탐구 목적입니다.

## 8. 파일 구조

```text
requirements.txt
README.md
main.py
src/
  capture.py
  roi.py
  signal_processing.py
  rppg_methods.py
  plotting.py
  utils.py
outputs/
  실행 시 자동 생성
data/
  저장 영상 또는 CSV를 둘 수 있는 폴더
```
