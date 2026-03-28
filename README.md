# data-curation

로봇 trajectory segmentation과 Cartesian 좌표 변환 기능을 제공하는 Python 유틸리티 패키지입니다.

크게 세 가지 축의 기능이 들어 있습니다.

1. `segmentation-cli`로 바로 실행할 수 있는 간단한 텍스트 기반 분할 파이프라인
2. 로봇 trajectory 데이터를 입력받아 경계 검출, 구간 라벨링, feature 구성, 리포트 생성을 수행하는 Python API
3. RB-Y1 URDF + Pinocchio를 이용한 joint space → Cartesian pose FK 변환 및 episode 세그멘테이션 스크립트

## 주요 기능

- 텍스트 파일(한 줄 = 한 샘플)을 읽어 pseudo-segmentation 수행
- `argparse` 기반 CLI: `segmentation-cli`
- trajectory segmentation용 task profile/threshold 관리
- 하이브리드 방식의 temporal boundary detection
  - `ruptures` 기반 change point detection
  - 규칙 기반 이벤트 감지 (speed jump, stationary zone, gripper transition)
- semantic segment labeling: `approach` / `grasp` / `move` / `insertion` / `place` / `move_to_ready`
- trajectory feature 구성 (resampling, smoothing, joint/cartesian/gripper feature 결합, robust normalization)
- LeRobot 스타일 데이터셋 episode 인덱싱, 샘플링, 로딩
- **FK 지원**: Pinocchio를 통해 44-dim joint state → 6-DOF Cartesian pose (position + quaternion) 변환
- 샘플별 segmentation 리포트 생성 (`segments.json`, `summary.csv`, `timeline.png`)

## 폴더 구조

```text
data-curation/
├── environment.yml              # conda 환경 파일 (pinocchio 포함)
├── pyproject.toml
├── README.md
├── src/
│   └── segmentation/            # 메인 Python 패키지
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py       # episode 로딩 + FK 연산 연결
│       ├── features.py
│       ├── kinematics.py        # Pinocchio FK 함수들 (optional)
│       ├── pipeline.py
│       ├── report.py
│       ├── segmenter.py
│       ├── configs/
│       │   └── rby1_segment_config.yaml  # 번들 config
│       └── robots/
│           └── rby1.py          # RB-Y1 관절 레이아웃 상수
└── scripts/
    ├── joint_to_cartesian.py    # FK 계산 + 시각화 standalone 스크립트
    └── segment_episodes.py      # FK + 세그멘테이션 standalone 스크립트
```

각 파일의 역할:

- `src/segmentation/kinematics.py`: Pinocchio 기반 FK 함수. `load_robot_model()`, `compute_fk_trajectory()`, `fk_to_quaternion()` 등을 제공합니다. pinocchio가 없어도 import는 되며, 실제 호출 시에만 안내 메시지와 함께 에러가 납니다.
- `src/segmentation/robots/rby1.py`: RB-Y1 44-dim state vector 레이아웃 상수 (`DATASET_JOINT_NAMES`, `GRIPPER_MAPPING`, `EE_FRAMES`).
- `src/segmentation/configs/`: 패키지에 번들된 YAML 설정 파일. `get_config_path("rby1_segment_config.yaml")`로 접근합니다.
- `src/segmentation/data_loader.py`: LeRobot episode 로딩. `needs_fk=True` 에피소드에 대해 `compute_episode_fk()`로 Cartesian pose를 계산할 수 있습니다.
- `scripts/joint_to_cartesian.py`: FK만 수행하고 `.npz` + 그래프를 저장하는 독립 실행 스크립트.
- `scripts/segment_episodes.py`: FK + 세그멘테이션을 수행하고 결과를 JSON + 그래프로 저장하는 독립 실행 스크립트.

## 요구 사항

- Python 3.10 이상
- FK 기능(`kinematics.py`)은 **pinocchio**가 필요합니다. pinocchio는 pip으로 설치할 수 없으며 conda-forge를 통해서만 설치됩니다.

## 설치 방법

### 방법 A: conda (권장 — FK 포함 전체 환경)

```bash
git clone <repository-url>
cd data-curation
conda env create -f environment.yml
conda activate data-curation
```

`environment.yml`에는 pinocchio, numpy, scipy, pandas, pyyaml, huggingface_hub 등이 모두 포함되어 있습니다.

### 방법 B: venv (FK 기능 제외)

pinocchio를 conda 없이 설치하는 방법이 없으므로, FK 기능이 필요 없을 때만 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy matplotlib scipy ruptures pandas
pip install -e .
```

또는 optional extras를 한 번에:

```bash
pip install -e ".[full]"
```

## 빠른 시작

### 1. CLI로 텍스트 분할 실행

```bash
segmentation-cli \
  --dataset-path sample.txt \
  --num-samples 2 \
  --seed 7 \
  --output results.json
```

editable install 없이 실행:

```bash
PYTHONPATH=src python -m segmentation.cli \
  --dataset-path sample.txt \
  --num-samples 2 \
  --seed 7 \
  --output results.json
```

### 2. FK로 Cartesian pose 계산 (Python API)

```python
from segmentation import kinematics, list_episodes, load_episode, compute_episode_fk
from segmentation.robots.rby1 import EE_FRAMES

# URDF 로드
model, data = kinematics.load_robot_model("/path/to/rby1.urdf")
name_to_q_idx = kinematics.build_joint_index_map(model)

# 에피소드 로드
episodes = list_episodes("/path/to/dataset")
ep_ts = load_episode(episodes[0])

# FK로 Cartesian pose 계산
if ep_ts.needs_fk:
    ep_ts = compute_episode_fk(ep_ts, model, data, name_to_q_idx, EE_FRAMES[0])

# ep_ts.ee_pose → (N, 7) 배열: [x, y, z, qx, qy, qz, qw]
print(ep_ts.ee_pose.shape)
print(ep_ts.needs_fk)  # False
```

### 3. FK 계산 standalone 스크립트

```bash
# HuggingFace 데이터셋
python scripts/joint_to_cartesian.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/model.urdf

# 로컬 데이터셋
python scripts/joint_to_cartesian.py \
    --dataset /local/path/to/dataset \
    --urdf /path/to/model.urdf \
    --local
```

출력: `output/fk_results/episode_*.npz`, `output/plots/episode_*.png`

### 4. FK + 세그멘테이션 standalone 스크립트

```bash
python scripts/segment_episodes.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/model.urdf \
    --n-samples 5

# 로컬 + 커스텀 config
python scripts/segment_episodes.py \
    --dataset /local/path \
    --urdf /path/to/model.urdf \
    --n-samples 5 \
    --config /path/to/config.yaml \
    --local
```

출력: `output/fk_results/`, `output/segment_plots/`, `output/segmentation_results.json`

`--config`를 생략하면 패키지에 번들된 `rby1_segment_config.yaml`을 사용합니다.

## Trajectory Segmentation API

### 입력 형식

`segment_trajectory()`는 아래 키를 가진 `aux_signals` 딕셔너리를 받습니다.

- `t`: shape `[T]`, timestamp
- `ee_pos_xyz`: shape `[T, 3]`, end-effector 위치
- `ee_speed`: shape `[T]`, speed
- `gripper_state`: shape `[T]`, gripper 상태
- `pose_alignment`: shape `[T]`, optional, insertion 정렬도

### 사용 예시

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
)

for seg in segments:
    print(seg)
```

### 단계별 제어

```python
from segmentation import detect_boundaries, boundaries_to_segments, label_segments
from segmentation.config import get_task_profile

profile = get_task_profile("charger_insertion")

boundaries = detect_boundaries(
    t=t, ee_pos_xyz=ee_pos_xyz, ee_speed=ee_speed,
    gripper_state=gripper_state, profile=profile, method="hybrid",
)
segments = boundaries_to_segments(t=t, boundaries=boundaries,
                                   min_duration_s=profile.min_segment_duration_s)
labeled = label_segments(segments=segments, aux_signals={...}, profile=profile)
```

## Feature 생성

```python
import numpy as np
from segmentation.features import FeatureBuildConfig, build_features

sample = {
    "timestamps": np.linspace(0.0, 1.0, 101),
    "q": np.random.randn(101, 6),
    "cartesian": np.hstack([
        np.random.randn(101, 3),           # position
        np.tile([0., 0., 0., 1.], (101, 1)),  # quaternion (xyzw)
    ]),
    "gripper": np.random.rand(101, 1),
}

result = build_features(sample, config=FeatureBuildConfig(smoothing="savgol"))
print(result["feature_matrix"].shape)
```

FK로 계산한 `ee_pose` 배열은 `(N, 7)` 형태로 `features.build_features()`의 `"cartesian"` 키에 바로 연결됩니다.

## LeRobot 데이터셋 로딩

```python
from segmentation.data_loader import (
    list_episodes, uniform_sample_episodes, load_episode, save_sampling_output,
)

episodes = list_episodes("path/to/dataset")
sampled = uniform_sample_episodes(episodes, num_samples=10, seed=42)
save_sampling_output("outputs", sampled)

ep = load_episode(sampled[0])
print(ep.needs_fk)  # True if no cartesian pose in dataset
```

`load_episode()`가 찾는 대표 키:

| 신호 | 탐색 키 순서 |
|------|-------------|
| joint states | `joint_states`, `observation.state`, `state` |
| joint commands | `joint_commands`, `action`, `commands` |
| ee pose | `ee_pose`, `observation.ee_pose`, `cartesian_pose` |

end-effector pose를 찾지 못하면 `needs_fk=True`로 표시됩니다. `compute_episode_fk()`를 통해 FK로 채울 수 있습니다.

## 세그멘테이션 Config 조정

`scripts/segment_episodes.py`의 세그멘테이션 파라미터는 YAML 파일로 조정합니다. 패키지 기본값은 `src/segmentation/configs/rby1_segment_config.yaml`에 있습니다.

```yaml
gripper:
  thresh_closed: 0.01   # 이 값 이하 → 닫힘
  thresh_open: 0.04     # 이 값 이상 → 열림
  dwell_frames: 6       # 확인에 필요한 연속 프레임 수 (~200ms at 30Hz)
  active_hand: right

velocity:
  thresh_moving: 0.005       # m/s 이상 → 이동 중
  thresh_stationary: 0.002   # m/s 이하 → 정지
  dwell_frames: 10

preprocessing:
  gripper_median_kernel: 5
  pos_smooth_sigma: 3.0
  vel_smooth_sigma: 5.0
```

커스텀 config 사용:

```bash
python scripts/segment_episodes.py \
    --dataset /path/to/data \
    --urdf /path/to/urdf \
    --n-samples 5 \
    --config my_config.yaml \
    --local
```

## 추천 사용 흐름

1. `data_loader.list_episodes()` + `uniform_sample_episodes()`로 에피소드 샘플링
2. `data_loader.load_episode()`로 데이터 로딩
3. `needs_fk=True`이면 `kinematics.load_robot_model()` + `compute_episode_fk()`로 Cartesian pose 계산
4. `features.build_features()`로 feature matrix 구성
5. `segmenter.segment_trajectory()`로 경계 검출 및 라벨링
6. `report.generate_sample_report()`로 시각화 및 리포트 저장

또는 FK + 세그멘테이션을 한 번에 하려면:

```bash
python scripts/segment_episodes.py --dataset ... --urdf ...
```

## 진입점 요약

```python
# 패키지 루트에서 바로 import 가능
from segmentation import (
    # config
    DEFAULT_TASK_PROFILE, TASK_PROFILES, TaskProfile, get_task_profile,
    # data_loader
    EpisodeRef, EpisodeTimeseries,
    list_episodes, load_episode, compute_episode_fk,
    uniform_sample_episodes, save_sampling_output,
    # segmenter
    Segment, detect_boundaries, boundaries_to_segments,
    label_segments, segment_trajectory,
    # pipeline
    run_pipeline,
)

# 서브모듈로 import
from segmentation import kinematics
from segmentation.features import build_features, FeatureBuildConfig
from segmentation.report import generate_sample_report
from segmentation.robots.rby1 import DATASET_JOINT_NAMES, GRIPPER_MAPPING, EE_FRAMES
from segmentation.configs import get_config_path
```

## 자주 만나는 문제

### `ImportError: pinocchio is required for FK`

원인: pinocchio가 설치되지 않았습니다.

해결:
```bash
conda install -c conda-forge pinocchio
```
또는 `conda env create -f environment.yml`로 전체 환경을 새로 생성하세요.

### `TypeError: dataclass() got an unexpected keyword argument 'slots'`

원인: Python 3.9 이하에서 실행 중입니다.

해결: Python 3.10 이상의 인터프리터로 환경을 다시 만드세요.

### `ruptures`, `scipy`, `pandas` 관련 import error

```bash
pip install scipy ruptures pandas pyarrow
```
또는 `pip install -e ".[full]"`
