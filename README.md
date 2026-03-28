# data-curation

로봇 trajectory segmentation과 간단한 텍스트 샘플 분할 예제를 함께 담고 있는 Python 유틸리티 패키지입니다.

이 저장소에는 크게 두 가지 축의 기능이 들어 있습니다.

1. `segmentation-cli`로 바로 실행할 수 있는 간단한 텍스트 기반 분할 파이프라인
2. 로봇 궤적 데이터를 입력받아 경계 검출, 구간 라벨링, feature 구성, 리포트 생성을 수행하는 Python API

즉, 지금 이 저장소는 "바로 돌려볼 수 있는 작은 데모 CLI"와 "실제 trajectory segmentation 작업에 붙일 수 있는 모듈형 라이브러리"를 함께 제공하는 형태입니다.

## 주요 기능

- 텍스트 파일(한 줄 = 한 샘플)을 읽어 간단한 pseudo-segmentation 수행
- `argparse` 기반 CLI 제공: `segmentation-cli`
- trajectory segmentation용 task profile/threshold 관리
- 하이브리드 방식의 temporal boundary detection
  - `ruptures` 기반 change point detection
  - 규칙 기반 이벤트 감지(speed jump, stationary zone, gripper transition)
- semantic segment labeling
  - `approach`
  - `grasp`
  - `move`
  - `insertion`
  - `place`
  - `move_to_ready`
- trajectory feature 구성
  - resampling
  - smoothing
  - joint/cartesian/gripper feature 결합
  - robust normalization
- LeRobot 스타일 데이터셋 episode 인덱싱, 샘플링, 로딩
- 샘플별 segmentation 리포트 생성
  - `segments.json`
  - `summary.csv`
  - `timeline.png`

## 폴더 구조

```text
data-curation/
├─ pyproject.toml
├─ README.md
└─ src/
   └─ segmentation/
      ├─ __init__.py
      ├─ cli.py
      ├─ config.py
      ├─ data_loader.py
      ├─ features.py
      ├─ pipeline.py
      ├─ report.py
      └─ segmenter.py
```

각 파일의 역할은 아래와 같습니다.

- `src/segmentation/cli.py`
  - 텍스트 파일 기반 데모 파이프라인용 CLI 진입점입니다.
- `src/segmentation/pipeline.py`
  - 텍스트 한 줄을 하나의 샘플로 보고 pseudo-segmentation 결과를 생성합니다.
- `src/segmentation/config.py`
  - trajectory segmentation에 필요한 task profile과 threshold를 정의합니다.
- `src/segmentation/segmenter.py`
  - boundary detection, segment 생성, semantic labeling, end-to-end segmentation 로직이 들어 있습니다.
- `src/segmentation/features.py`
  - 시계열 신호를 resample/smooth/normalize 해서 feature matrix를 만듭니다.
- `src/segmentation/data_loader.py`
  - LeRobot 스타일 데이터셋에서 episode 목록을 읽고, 샘플링하고, 실제 episode 데이터를 불러옵니다.
- `src/segmentation/report.py`
  - segmentation 결과를 JSON/CSV/PNG 리포트로 저장합니다.

## 요구 사항

- Python 3.10 이상

중요한 점:

- 이 코드베이스는 `dataclass(slots=True)`를 사용하므로 Python 3.9에서는 동작하지 않습니다.
- 시스템에 `python`이 3.9를 가리키는 경우가 있으니, 설치 전에 반드시 버전을 확인하는 것이 안전합니다.

버전 확인:

```bash
python --version
```

또는 환경에 따라:

```bash
python3 --version
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd data-curation
```

### 2. 가상환경 생성 (venv 또는 conda)

아래 두 방법 중 하나를 선택해서 사용하면 됩니다.

#### 방법 A: `venv`

macOS / Linux / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 방법 B: `conda`

```bash
conda create -n data-curation python=3.10 -y
conda activate data-curation
```

환경 제거(선택):

```bash
conda deactivate
conda env remove -n data-curation
```

### 3. 의존성 설치

현재 `pyproject.toml`에는 패키지 메타데이터만 들어 있고, 외부 런타임 의존성이 모두 선언되어 있지는 않습니다. 그래서 아래처럼 직접 설치하는 것이 가장 안전합니다.

최소 권장 설치:

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
python -m pip install -e .
```

추가 기능까지 포함한 설치:

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scipy ruptures pandas
python -m pip install -e .
```

패키지별 용도:

- `numpy`: trajectory segmentation과 feature 처리에 사실상 필수입니다.
- `matplotlib`: `report.py`의 timeline 시각화 저장에 필요합니다.
- `scipy`: Savitzky-Golay, low-pass filtering, quaternion 처리에 사용됩니다.
- `ruptures`: change point 기반 boundary detection을 사용할 때 필요합니다.
- `pandas`: episode parquet 파일을 로딩할 때 필요합니다.

선택 의존성이 없더라도 일부 기능은 fallback으로 동작합니다.

- `ruptures`가 없으면 hybrid 모드가 규칙 기반 경계 검출로 fallback 됩니다.
- `scipy`가 없으면 smoothing은 moving average 기반 안전한 fallback을 사용합니다.
- `pandas`가 없으면 parquet 로딩은 불가능하고 JSON/JSONL/NPY/NPZ 위주로 사용해야 합니다.

## 빠른 시작

### 1. CLI로 텍스트 분할 실행

이 CLI는 현재 "텍스트 파일 한 줄 = 하나의 샘플"이라는 매우 단순한 입력 형식을 사용합니다.

예시 입력 파일 `sample.txt`:

```text
첫 번째 샘플 문장입니다 예시 분할을 확인합니다
두 번째 샘플은 조금 더 길게 작성해서 분할 결과를 봅니다
```

실행:

```bash
segmentation-cli \
  --dataset-path sample.txt \
  --num-samples 2 \
  --seed 7 \
  --output results.json
```

또는 editable install 없이 바로 실행하려면:

```bash
PYTHONPATH=src python -m segmentation.cli \
  --dataset-path sample.txt \
  --num-samples 2 \
  --seed 7 \
  --output results.json
```

예시 콘솔 출력:

```text
sample=0 segments=5 label_lengths=[intro:3, body:2, conclusion:2]
sample=1 segments=5 label_lengths=[intro:6, body:2, conclusion:2]
```

출력 파일은 `.json` 또는 `.csv`만 지원합니다.

- `results.json`: 샘플별 segment 정보 전체 저장
- `results.csv`: 요약 정보 저장

### 2. Python에서 간단한 파이프라인 실행

```python
from segmentation.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    dataset_path="sample.txt",
    num_samples=3,
    seed=42,
)

results = run_pipeline(config)
print(results[0])
```

반환 예시:

```python
{
    "sample_id": 0,
    "text": "첫 번째 샘플 문장입니다 예시 분할을 확인합니다",
    "num_segments": 5,
    "label_lengths": {
        "intro": 3,
        "body": 2,
        "conclusion": 2
    },
    "segments": [
        {"label": "intro", "start": 0, "end": 3},
        {"label": "body", "start": 3, "end": 4}
    ]
}
```

주의:

- 이 파이프라인은 현재 demonstration 성격의 pseudo-segmentation입니다.
- 실제 trajectory segmentation과는 별도 모듈입니다.

## Trajectory Segmentation API

실제 로봇 trajectory segmentation은 `segmentation.segmenter` 쪽 API를 사용합니다.

### 입력 형식

`segment_trajectory()`는 아래 키를 가진 `aux_signals` 딕셔너리를 기대합니다.

- `t`: shape `[T]`, timestamp 배열
- `ee_pos_xyz`: shape `[T, 3]`, end-effector 위치
- `ee_speed`: shape `[T]`, speed
- `gripper_state`: shape `[T]`, gripper 상태
- `pose_alignment`: shape `[T]`, optional, insertion 정렬도

### 가장 간단한 사용 예시

```python
import numpy as np
from segmentation import segment_trajectory

t = np.linspace(0.0, 3.0, 31)
ee_pos_xyz = np.column_stack([
    np.linspace(0.38, 0.61, 31),
    np.linspace(0.12, -0.05, 31),
    np.linspace(0.07, 0.02, 31),
])
ee_speed = np.linspace(0.01, 0.06, 31)
gripper_state = np.concatenate([np.zeros(10), np.ones(10), np.zeros(11)])
pose_alignment = np.linspace(0.7, 0.98, 31)

segments = segment_trajectory(
    {
        "t": t,
        "ee_pos_xyz": ee_pos_xyz,
        "ee_speed": ee_speed,
        "gripper_state": gripper_state,
        "pose_alignment": pose_alignment,
    },
    profile_name="charger_insertion",
    method="hybrid",
    ruptures_method="pelt",
    penalty=5.0,
)

for seg in segments:
    print(seg)
```

반환값 형식:

```python
[
    {
        "start_t": 0.0,
        "end_t": 0.9,
        "label": "approach",
        "confidence": 0.90,
        "evidence": [
            "near pick-zone (xy)",
            "pick-zone z alignment",
            "gripper mostly open"
        ]
    }
]
```

### 내부 처리 흐름

`segment_trajectory()`는 내부적으로 아래 단계를 수행합니다.

1. `detect_boundaries()`
   - change point 또는 rule-based 방식으로 경계를 찾습니다.
2. `boundaries_to_segments()`
   - 경계 인덱스를 시간 구간으로 바꿉니다.
3. `label_segments()`
   - 각 구간에 semantic label, confidence, evidence를 부여합니다.

### 직접 단계별로 제어하고 싶을 때

```python
import numpy as np
from segmentation import detect_boundaries, boundaries_to_segments, label_segments
from segmentation.config import get_task_profile

profile = get_task_profile("charger_insertion")

boundaries = detect_boundaries(
    t=t,
    ee_pos_xyz=np.asarray(ee_pos_xyz),
    ee_speed=ee_speed,
    gripper_state=gripper_state,
    profile=profile,
    method="hybrid",
)

segments = boundaries_to_segments(
    t=t,
    boundaries=boundaries,
    min_duration_s=profile.min_segment_duration_s,
)

labeled = label_segments(
    segments=segments,
    aux_signals={
        "t": t,
        "ee_pos_xyz": ee_pos_xyz,
        "ee_speed": ee_speed,
        "gripper_state": gripper_state,
        "pose_alignment": pose_alignment,
    },
    profile=profile,
)
```

## Task Profile 설정

현재 기본 task profile은 `charger_insertion` 하나가 정의되어 있습니다.

불러오기:

```python
from segmentation.config import get_task_profile

profile = get_task_profile("charger_insertion")
print(profile.pick_zone_center)
```

profile에 들어 있는 대표 설정값:

- pick zone / insertion zone 중심 좌표
- ready pose
- zone 반경과 z tolerance
- speed threshold
- hysteresis duration
- 최소 segment 길이
- alignment tolerance

새 task를 추가하려면 `src/segmentation/config.py`의 `TASK_PROFILES`에 새 `TaskProfile`을 추가하면 됩니다.

## Feature 생성

`src/segmentation/features.py`는 trajectory segmentation용 feature matrix를 만들 때 사용합니다.

### 기대 입력

샘플 하나는 보통 아래 중 일부 키를 포함한 딕셔너리여야 합니다.

- 시간
  - `timestamps` 또는 `t` 또는 `time`
- joint state
  - `q` 또는 `joint_pos`
- joint command
  - `q_cmd`
- position
  - `position`
  - 또는 `cartesian[:, :3]`
  - 또는 `x`, `y`, `z`
- quaternion
  - `quaternion`
  - 또는 `cartesian[:, 3:7]`
- gripper
  - `gripper`
  - `gripper_opening`
  - `grip`
  - `gripper_ratio`

### 사용 예시

```python
import numpy as np
from segmentation.features import FeatureBuildConfig, build_features

sample = {
    "timestamps": np.linspace(0.0, 1.0, 101),
    "q": np.random.randn(101, 6),
    "q_cmd": np.random.randn(101, 6),
    "position": np.random.randn(101, 3),
    "quaternion": np.tile([0.0, 0.0, 0.0, 1.0], (101, 1)),
    "gripper": np.random.rand(101, 1),
}

config = FeatureBuildConfig(
    smoothing="savgol",
    normalize="mad",
)

result = build_features(sample, config=config)
print(result["feature_matrix"].shape)
print(result["timestamps"].shape)
```

반환 딕셔너리에는 아래 키가 포함됩니다.

- `feature_matrix`
- `timestamps`
- `aux_signals`

`aux_signals` 안에는 원본과 resampled/smoothed 신호가 함께 저장됩니다.

## LeRobot 스타일 데이터셋 로딩

`src/segmentation/data_loader.py`는 LeRobot 스타일 폴더 구조를 가정합니다.

### 메타데이터 파일 탐색 순서

아래 파일 중 먼저 발견되는 것을 사용합니다.

- `meta/episodes.jsonl`
- `meta/episodes.json`
- `episodes.jsonl`
- `episodes.json`

### episode 데이터 파일 탐색 순서

각 episode는 아래 경로 중 먼저 발견되는 파일로 로딩됩니다.

- `data/episode_{episode_id}.parquet`
- `data/episode_{episode_id}.json`
- `data/episode_{episode_id}.jsonl`
- `data/episode_{episode_id}.npz`
- `data/episode_{episode_id}.npy`

### 사용 예시

```python
from segmentation.data_loader import (
    list_episodes,
    uniform_sample_episodes,
    load_episode,
    save_sampling_output,
)

episodes = list_episodes("path/to/dataset")
sampled = uniform_sample_episodes(episodes, num_samples=10, seed=42)
save_sampling_output("outputs", sampled)

episode = load_episode(sampled[0])
print(episode.episode_ref)
print(episode.needs_fk)
```

`load_episode()`가 찾는 대표 키는 아래와 같습니다.

- joint states
  - `joint_states`
  - `observation.state`
  - `state`
- joint commands
  - `joint_commands`
  - `action`
  - `commands`
- ee pose
  - `ee_pose`
  - `observation.ee_pose`
  - `cartesian_pose`

만약 end-effector pose를 찾지 못하면 `needs_fk=True`로 표시됩니다. 즉, 이 모듈은 누락된 pose를 자동으로 forward kinematics로 계산하지는 않고, 상위 단계에서 추가 처리가 필요하다는 신호만 제공합니다.

## 리포트 생성

`src/segmentation/report.py`는 segment 결과를 샘플별 산출물로 정리해 줍니다.

생성되는 파일:

- `output_dir/sample_{id}/segments.json`
- `output_dir/sample_{id}/summary.csv`
- `output_dir/sample_{id}/timeline.png`

### 사용 예시

중요:

- `segment_trajectory()`의 결과는 `start_t`, `end_t` 키를 사용합니다.
- `generate_sample_report()`는 `start`, `end` 키를 기대합니다.
- 따라서 바로 넘기지 말고 한 번 매핑해서 전달하는 것이 안전합니다.

```python
from segmentation import segment_trajectory
from segmentation.report import generate_sample_report

labeled_segments = segment_trajectory(aux_signals)

report_segments = [
    {
        "start": seg["start_t"],
        "end": seg["end_t"],
        "label": seg["label"],
        "confidence": seg["confidence"],
    }
    for seg in labeled_segments
]

report = generate_sample_report(
    sample_id="demo",
    output_dir="outputs",
    segments=report_segments,
    timestamps=aux_signals["t"],
    trajectory_x=aux_signals["ee_pos_xyz"][:, 0],
    trajectory_z=aux_signals["ee_pos_xyz"][:, 2],
    gripper_state=aux_signals["gripper_state"],
    joint_speed_norm=aux_signals["ee_speed"],
    low_conf_threshold=0.5,
)

print(report.keys())
```

리포트 payload에는 아래 정보가 들어갑니다.

- `timeline`
- `label_duration_stats`
- `low_confidence_segments`
- `boundary_f1`
- `segment_iou`

GT 정보가 있다면 `gt_boundaries`, `gt_segments`를 넘겨 간단한 metric 계산도 할 수 있습니다.

## CLI 옵션 정리

`segmentation-cli`가 지원하는 옵션은 아래와 같습니다.

- `--dataset-path`
  - 입력 텍스트 파일 경로
- `--num-samples`
  - 앞에서부터 몇 개 샘플을 처리할지 지정
- `--seed`
  - pseudo-segmentation 랜덤 시드
- `--output`
  - 결과 저장 경로, `.json` 또는 `.csv`만 허용

예시:

```bash
segmentation-cli \
  --dataset-path data/sample.txt \
  --num-samples 20 \
  --seed 42 \
  --output outputs/result.csv
```

## 자주 만나는 문제

### `TypeError: dataclass() got an unexpected keyword argument 'slots'`

원인:

- Python 3.9 이하에서 실행 중일 가능성이 큽니다.

해결:

- Python 3.10 이상 인터프리터로 가상환경을 다시 만들고 설치하세요.

### `output must have .json or .csv extension`

원인:

- CLI의 `--output` 확장자가 허용 목록에 없습니다.

해결:

- `results.json` 또는 `results.csv`처럼 확장자를 명시하세요.

### `dataset path not found`

원인:

- `--dataset-path`가 잘못되었거나 현재 작업 디렉터리 기준 경로가 맞지 않습니다.

해결:

- 절대 경로를 사용하거나, 실행 위치에서 파일이 실제로 존재하는지 먼저 확인하세요.

### `ruptures`, `scipy`, `pandas`, `matplotlib` 관련 import error

원인:

- 선택 또는 외부 런타임 의존성이 설치되지 않았습니다.

해결:

```bash
python -m pip install numpy matplotlib scipy ruptures pandas
```

## 현재 상태와 한계

문서를 읽기 전에 기대치를 정확히 맞추는 것이 좋습니다.

- `pipeline.py`의 CLI 파이프라인은 데모용입니다.
- trajectory segmentation은 Python API로 사용할 수 있지만, 이를 직접 감싸는 별도 CLI는 아직 없습니다.
- `data_loader.py`는 다양한 파일 포맷을 best-effort로 읽지만, 데이터셋 스키마가 크게 다르면 추가 어댑터가 필요할 수 있습니다.
- `load_episode()`는 `ee_pose`가 없을 때 `needs_fk=True`만 반환하며, FK 자체는 구현하지 않습니다.
- `report.py`의 metric 함수는 baseline 수준의 간단한 평가 보조 유틸리티입니다.

## 추천 사용 흐름

실제 작업에서는 아래 순서로 붙이는 것이 자연스럽습니다.

1. `data_loader.py`로 episode 목록을 인덱싱하고 샘플링
2. 필요한 경우 raw sample을 `features.py`로 정규화/feature화
3. `segmenter.py`로 boundary detection과 labeling 수행
4. `report.py`로 샘플별 시각화와 요약 리포트 생성
5. threshold나 zone 기준이 안 맞으면 `config.py`의 `TaskProfile` 조정

## 진입점 요약

패키지 루트에서 바로 import 가능한 항목:

```python
from segmentation import (
    DEFAULT_TASK_PROFILE,
    TASK_PROFILES,
    TaskProfile,
    get_task_profile,
    run_pipeline,
    Segment,
    detect_boundaries,
    boundaries_to_segments,
    label_segments,
    segment_trajectory,
)
```

다만 `features`, `data_loader`, `report`는 별도 모듈에서 import해서 사용하는 구조입니다.

```python
from segmentation.features import build_features
from segmentation.data_loader import list_episodes
from segmentation.report import generate_sample_report
```

## 마무리

이 저장소는 아직 "하나의 완성된 제품형 애플리케이션"보다는, trajectory segmentation 실험과 데이터 정리를 빠르게 붙일 수 있게 만든 실용적인 모듈 모음에 가깝습니다. 그래서 README도 "무엇이 이미 구현되어 있는지", "어떻게 붙이면 되는지", "어디까지가 현재 범위인지"를 기준으로 정리했습니다.

필요하다면 다음 단계로는 아래 작업이 자연스럽습니다.

- `pyproject.toml`에 runtime dependency 및 optional extras 정리
- trajectory segmentation 전용 CLI 추가
- 실제 LeRobot sample에서 end-to-end example notebook 또는 script 추가
