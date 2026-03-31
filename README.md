# data-curation

로봇 trajectory segmentation과 Cartesian 좌표 변환 기능을 제공하는 Python 유틸리티 패키지입니다.

주요 기능:

1. **데이터 기반 실험 파이프라인**: 실제 데이터셋 스키마 자동 탐지 → ruptures CPD → fast_ticc 반복 primitive 발굴 → sktime 벤치마크 → 비교 리포트 생성
2. 로봇 trajectory 데이터를 입력받아 경계 검출, 구간 라벨링, feature 구성, 리포트 생성을 수행하는 Python API
3. RB-Y1 URDF + Pinocchio를 이용한 joint space → Cartesian pose FK 변환 및 episode 세그멘테이션 스크립트
4. `segmentation-cli`로 바로 실행할 수 있는 간단한 텍스트 기반 분할 파이프라인

## 주요 기능

- 텍스트 파일(한 줄 = 한 샘플)을 읽어 pseudo-segmentation 수행
- `argparse` 기반 CLI: `segmentation-cli`
- trajectory segmentation용 task profile/threshold 관리
- 하이브리드 방식의 temporal boundary detection
  - `ruptures` 기반 change point detection
  - 규칙 기반 이벤트 감지 (speed jump, stationary zone, gripper transition)
- semantic segment labeling: `approach` / `grasp` / `move` / `insertion` / `place` / `move_to_ready`
- trajectory feature 구성 (resampling, smoothing, modality-aware normalization)
  - `joint`
  - `cartesian`
  - `joint_command`
- LeRobot 스타일 데이터셋 로딩
  - `episode_*.parquet` 레이아웃 지원
  - `file-*.parquet + episode_index` shared-parquet 레이아웃 지원
- **FK 지원**: Pinocchio를 통해 44-dim joint state → 6-DOF Cartesian pose (position + quaternion) 변환
- 샘플별 segmentation 리포트 생성 (`segments.json`, `summary.csv`, `timeline.png`)
- **데이터셋 스키마 자동 탐지**: 컬럼명 가정 없이 실제 parquet를 읽어 키 검출
- **알고리즘 기반 penalty 자동 선택**: n_bkps vs log(penalty) 곡선의 elbow로 ruptures penalty 결정
- **반복 primitive 발굴**: fast_ticc를 통한 비지도 motion primitive 클러스터링
- **sktime 벤치마크**: 설치된 sktime segmenter를 자동 탐지하여 비교 실행

## 폴더 구조

```text
data-curation/
├── environment.yml              # conda 환경 파일 (pinocchio + 실험 패키지 포함)
├── pyproject.toml
├── README.md
├── src/
│   └── segmentation/            # 메인 Python 패키지
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py       # episode 로딩 + FK 연산 연결
│       ├── evaluation.py        # 비지도 비교 지표 (boundary agreement, silhouette 등)
│       ├── features.py
│       ├── kinematics.py        # Pinocchio FK 함수들 (optional)
│       ├── lerobot_adapter.py   # 스키마 자동 탐지 + episode 인덱싱 + episode 배열 로딩
│       ├── pipeline.py
│       ├── report.py
│       ├── ruptures_segmenter.py  # penalty 자동 선택 CPD (PELT / Binseg)
│       ├── segmenter.py
│       ├── sktime_benchmark.py  # sktime segmenter 자동 탐지 + 벤치마크
│       ├── ticc_primitives.py   # fast_ticc 반복 primitive 발굴
│       ├── configs/
│       │   ├── rby1_segment_config.yaml      # 휴리스틱 세그멘테이션 config
│       │   └── lerobot_segmentation.yaml     # 실험 파이프라인 config
│       └── robots/
│           └── rby1.py          # RB-Y1 관절 레이아웃 상수
├── scripts/
│   ├── joint_to_cartesian.py    # FK 계산 + 시각화 standalone 스크립트
│   ├── run_lerobot_segmentation.py  # 실험 파이프라인 CLI (신규)
│   └── segment_episodes.py      # FK + 휴리스틱 세그멘테이션 standalone 스크립트
└── tests/
    ├── test_lerobot_adapter.py
    └── test_segmentation_pipeline.py
```

각 파일의 역할:

- `src/segmentation/lerobot_adapter.py`: 데이터셋 스키마 자동 탐지 (`inspect_dataset`) + episode 인덱싱 (`list_episode_refs`) + 표준화된 episode 배열 로딩 (`load_episode_arrays`). 컬럼명을 하드코딩하지 않고 실제 parquet를 읽어 탐지합니다.
- `src/segmentation/ruptures_segmenter.py`: `ruptures` 기반 change-point detection. penalty를 log-space elbow 방법으로 자동 선택. PELT + Binseg를 동시에 실행합니다.
- `src/segmentation/ticc_primitives.py`: `fast_ticc`를 이용한 반복 motion primitive 클러스터링. `n_clusters="auto"`로 silhouette 기반 k 자동 탐색.
- `src/segmentation/sktime_benchmark.py`: 설치된 sktime segmenter를 런타임에 탐지하여 동일 신호에 대해 실행, 결과 반환.
- `src/segmentation/evaluation.py`: boundary agreement (tolerance-based F1), segment duration 통계, cluster silhouette, 전방위 비교 테이블.
- `src/segmentation/kinematics.py`: Pinocchio 기반 FK 함수. `load_robot_model()`, `compute_fk_trajectory()`, `fk_to_quaternion()` 등. pinocchio가 없어도 import는 되며, 실제 호출 시에만 에러가 납니다.
- `src/segmentation/robots/rby1.py`: RB-Y1 44-dim state vector 레이아웃 상수 (`DATASET_JOINT_NAMES`, `GRIPPER_MAPPING`, `EE_FRAMES`).
- `src/segmentation/configs/`: 패키지에 번들된 YAML 설정 파일. `get_config_path("rby1_segment_config.yaml")`로 접근합니다.
- `src/segmentation/data_loader.py`: LeRobot episode 로딩. `needs_fk=True` 에피소드에 대해 `compute_episode_fk()`로 Cartesian pose를 계산할 수 있습니다.
- `scripts/run_lerobot_segmentation.py`: 데이터 기반 실험 파이프라인 전체를 실행하는 메인 스크립트.
- `scripts/joint_to_cartesian.py`: FK만 수행하고 `.npz` + 그래프를 저장하는 독립 실행 스크립트.
- `scripts/segment_episodes.py`: FK + 휴리스틱 세그멘테이션을 수행하고 결과를 JSON + 그래프로 저장하는 독립 실행 스크립트.

## 요구 사항

- Python 3.10 이상
- FK 기능(`kinematics.py`)은 **pinocchio**가 필요합니다. pinocchio는 pip으로 설치할 수 없으며 conda-forge를 통해서만 설치됩니다.
- 실험 파이프라인 추가 패키지 (선택): `fast-ticc`, `sktime`, `scikit-learn`

## 업데이트된 종속성 정보

- `sktime` 0.40.1 이상 버전과의 호환성 향상 (detection 모듈 지원)
- `fast-ticc` 1.0.1 이상 버전과의 호환성 향상 (ticc_labels 함수 사용)

## 설치 방법

### 방법 A: conda (권장 — FK + 실험 패키지 포함 전체 환경)

```bash
git clone <repository-url>
cd data-curation
conda env create -f environment.yml
conda activate data-curation
```

`environment.yml`에는 pinocchio, numpy, scipy, pandas, pyyaml, huggingface_hub, ruptures, fast-ticc, sktime, scikit-learn 등이 모두 포함되어 있습니다.

### 방법 B: venv (FK 기능 제외)

pinocchio를 conda 없이 설치하는 방법이 없으므로, FK 기능이 필요 없을 때만 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[full]"
```

실험 파이프라인까지 포함하려면:

```bash
pip install -e ".[experiment]"
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

### 3. 데이터 기반 실험 파이프라인 (신규)

데이터셋 스키마를 먼저 탐지하고 싶을 때 (n-episodes=0):

```bash
python scripts/run_lerobot_segmentation.py \
    --dataset /path/to/dataset \
    --local --n-episodes 0
```

Joint-space만 사용 (URDF 불필요):

```bash
python scripts/run_lerobot_segmentation.py \
    --dataset /path/to/dataset \
    --local --n-episodes 5 \
    --modalities joint \
    --output results/
```

Joint + command ablation을 실행하고 싶을 때:

```bash
python scripts/run_lerobot_segmentation.py \
    --dataset /path/to/dataset \
    --local --n-episodes 5 \
    --modalities joint_command \
    --output results/
```

FK 포함 전체 실행 (Cartesian + joint, URDF 필요):

```bash
python scripts/run_lerobot_segmentation.py \
    --dataset /path/to/dataset \
    --urdf /path/to/rby1.urdf \
    --local --n-episodes 5 \
    --modalities joint cartesian \
    --ee-frame ee_right \
    --output results/
```

출력 아티팩트:

```
results/
├── schema_report.json                  # 탐지된 데이터셋 스키마
├── episode_0/
│   ├── segmentation_joint.png          # joint 신호 + 각 알고리즘 경계 시각화
│   ├── segmentation_cartesian.png      # Cartesian 신호 + 경계 시각화
│   └── primitives_ticc_joint.png       # TICC primitive 타임라인
├── boundaries.csv                      # episode × modality × status × method 경계 인덱스
├── comparison_table.csv                # segment duration 통계 비교
├── results.json                        # 전체 결과 (경계, TICC, sktime)
└── report.md                           # 요약 리포트 (Markdown)
```

### 4. FK 계산 standalone 스크립트

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

### 5. FK + 휴리스틱 세그멘테이션 standalone 스크립트

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

## 실험 파이프라인 상세

### 데이터셋 스키마 자동 탐지

`inspect_dataset()`는 `meta/info.json`과 첫 번째 parquet 파일을 읽어 컬럼명을 자동 탐지합니다.
하드코딩된 컬럼명 없이 다양한 LeRobot 데이터셋에 적용 가능합니다.

지원 레이아웃:

- `data/chunk-*/episode_000123.parquet`
- `data/chunk-*/file-000.parquet` + `episode_index`

```python
from segmentation.lerobot_adapter import inspect_dataset, list_episode_refs, load_episode_arrays

schema = inspect_dataset("/path/to/dataset")
# DatasetSchema:
#   layout         : shared_parquet
#   fps            : 30.0
#   joint_state    : observation.state (dim=44)
#   joint_command  : action
#   ee_pose        : None
#   gripper        : None
#   timestamp      : timestamp

episodes = list_episode_refs("/path/to/dataset", schema)
arrays = load_episode_arrays(episodes[0], schema)
# arrays: {"joint_states": (T, 44), "joint_commands": (T, 44),
#          "timestamps": (T,)}
```

### Ruptures 기반 Change-Point Detection

task-specific threshold 없이 penalty를 데이터에서 자동 선택합니다.

```python
from segmentation.ruptures_segmenter import RupturesConfig, run_ruptures, run_pelt_and_binseg

# penalty 자동 선택 (elbow 방법)
cfg = RupturesConfig(penalty="auto", penalty_range=(0.5, 200.0), n_penalty_steps=40)

# PELT + Binseg 동시 실행
results = run_pelt_and_binseg(feature_matrix, cfg)
pelt_boundaries, pelt_info = results["pelt"]
binseg_boundaries, binseg_info = results["binseg"]
```

penalty 자동 선택 로직:
- log-space로 후보 penalty 40개 생성
- 각 penalty에서 PELT 실행 → `(penalty, n_bkps)` 곡선 계산
- `n_bkps` vs `log(penalty)` 곡선의 2차 도함수 최댓값 = elbow point

### fast_ticc 반복 Primitive 발굴

```python
from segmentation.ticc_primitives import TiccConfig, run_ticc

# 세그먼트 리스트 (각 원소: (T_i, D) 배열)
segments = [feature_matrix[s:e] for s, e in segment_ranges]

cfg = TiccConfig(n_clusters="auto", max_k=10, window_size=10)
result = run_ticc(segments, cfg)

if result:
    print(result.cluster_assignments)   # 각 세그먼트의 primitive ID
    print(result.transition_matrix)      # (K, K) primitive 전환 빈도
    print(result.silhouette_score)       # 클러스터 품질 지표
```

`fast_ticc`가 설치되어 있지 않으면 경고를 출력하고 `None`을 반환합니다.

### 비지도 비교 평가

```python
from segmentation.evaluation import compare_boundaries, aggregate_episode_comparisons

# episode별 비교
cmp = compare_boundaries(
    {"pelt": pelt_bounds, "binseg": binseg_bounds, "sktime_ClaSP": clasp_bounds},
    total_frames=T, fps=30.0, tolerance_frames=5,
)
# → duration_stats, pairwise_agreement, n_segments

# 여러 episode 평균
agg = aggregate_episode_comparisons([cmp_ep0, cmp_ep1, cmp_ep2])
```

## Trajectory Segmentation API (휴리스틱)

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

## Feature 생성

```python
import numpy as np
from segmentation.features import FeatureBuildConfig, build_features

joint_sample = {
    "timestamps": np.linspace(0.0, 1.0, 101),
    "q": np.random.randn(101, 6),
    "q_cmd": np.random.randn(101, 6),
}

joint_result = build_features(
    joint_sample,
    "joint",
    FeatureBuildConfig(smoothing="savgol"),
)
joint_command_result = build_features(
    joint_sample,
    "joint_command",
    FeatureBuildConfig(smoothing="savgol"),
)

cartesian_sample = {
    "timestamps": np.linspace(0.0, 1.0, 101),
    "cartesian": np.hstack([
        np.random.randn(101, 3),              # position
        np.tile([0.0, 0.0, 0.0, 1.0], (101, 1)),  # quaternion (xyzw)
    ]),
}
cartesian_result = build_features(
    cartesian_sample,
    "cartesian",
    FeatureBuildConfig(smoothing="savgol"),
)

print(joint_result["matrix"].shape)
print(joint_command_result["matrix"].shape)
print(cartesian_result["matrix"].shape)
```

`build_features()`의 canonical modality는 `joint`, `cartesian`, `joint_command`입니다.
`ablation`은 backward compatibility alias로만 허용되며 내부적으로 `joint_command`로 정규화됩니다.

FK로 계산한 `ee_pose` 배열은 `(N, 7)` 형태로 `build_features(..., "cartesian", ...)`의 `"cartesian"` 입력에 바로 연결됩니다.

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

## 실험 파이프라인 Config 조정

`scripts/run_lerobot_segmentation.py`의 파라미터는 YAML 파일로 조정합니다.
기본값은 `src/segmentation/configs/lerobot_segmentation.yaml`에 있습니다.

```yaml
dataset:
  missing_values: ffill_drop   # ffill_drop | none

features:
  modalities:
    - joint
    - joint_command

ruptures:
  penalty: auto          # "auto" → elbow 자동 선택, 또는 float 고정
  penalty_range: [0.5, 200.0]
  n_penalty_steps: 40
  model: rbf             # rbf | l2 | l1 | cosine
  min_size: 5            # 최소 세그먼트 길이 (프레임)

ticc:
  n_clusters: auto       # "auto" → silhouette 기반 k 자동 탐색
  max_k: 10
  window_size: 10

evaluation:
  boundary_tolerance_frames: 5  # 이 범위 내 경계는 동일로 간주
```

커스텀 config 사용:

```bash
python scripts/run_lerobot_segmentation.py \
    --dataset /path/to/data \
    --local \
    --config my_config.yaml
```

## 휴리스틱 세그멘테이션 Config 조정

`scripts/segment_episodes.py`의 세그멘테이션 파라미터는 `rby1_segment_config.yaml`로 조정합니다.

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

## 추천 사용 흐름

### 데이터 기반 탐색 (신규)

1. `lerobot_adapter.inspect_dataset()`로 데이터셋 스키마 확인
2. `lerobot_adapter.list_episode_refs()` + `lerobot_adapter.load_episode_arrays()`로 표준화된 배열 로딩
3. `ruptures_segmenter.run_pelt_and_binseg()`로 데이터 기반 경계 검출
4. `ticc_primitives.run_ticc()`로 반복 primitive 발굴
5. `sktime_benchmark.run_sktime_benchmark()`로 알고리즘 비교
6. `evaluation.compare_boundaries()`로 비교 테이블 생성

또는 한 번에:

```bash
python scripts/run_lerobot_segmentation.py --dataset ... --local
```

### 휴리스틱 파이프라인 (기존)

1. `data_loader.list_episodes()` + `uniform_sample_episodes()`로 에피소드 샘플링
2. `data_loader.load_episode()`로 데이터 로딩
3. `needs_fk=True`이면 `kinematics.load_robot_model()` + `compute_episode_fk()`로 Cartesian pose 계산
4. `features.build_features()`로 feature matrix 구성
5. `segmenter.segment_trajectory()`로 경계 검출 및 라벨링
6. `report.generate_sample_report()`로 시각화 및 리포트 저장

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
    # modality-aware features
    FeatureBuildConfig, build_features, normalize_modality_name,
    # LeRobot adapter
    DatasetSchema, inspect_dataset, list_episode_refs, load_episode_arrays,
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

# 실험 파이프라인 (신규)
from segmentation.lerobot_adapter import (
    inspect_dataset, list_episode_refs, load_episode_arrays, DatasetSchema
)
from segmentation.ruptures_segmenter import RupturesConfig, run_ruptures, run_pelt_and_binseg
from segmentation.ticc_primitives import TiccConfig, TiccResult, run_ticc
from segmentation.sktime_benchmark import get_available_segmenters, run_sktime_benchmark
from segmentation.evaluation import (
    boundary_agreement, segment_duration_stats, cluster_silhouette,
    compare_boundaries, aggregate_episode_comparisons,
)
```

## 테스트 실행

```bash
pytest tests/test_lerobot_adapter.py tests/test_segmentation_pipeline.py -v
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

### `fast_ticc` / `sktime` not installed

실험 파이프라인에서 TICC / sktime이 없으면 해당 모듈은 자동으로 건너뜁니다.
설치하려면:

```bash
pip install fast-ticc sktime scikit-learn
```
또는 `pip install -e ".[experiment]"`

### Version Compatibility Issues

#### `fast_ticc` 버전 호환성 문제

fast_ticc 라이브러리의 버전에 따라 API가 다를 수 있습니다. 
현재 코드는 fast-ticc 1.0.1 버전과 호환되도록 업데이트되었습니다.
이전 버전에서는 `ticc` 함수 대신 `ticc_labels` 함수를 사용하도록 수정되었습니다.

#### `sktime` 버전 호환성 문제

sktime 라이브러리의 버전에 따라 모듈 구조가 다를 수 있습니다.
현재 코드는 sktime 0.40.1 버전과 호환되도록 업데이트되었습니다.
`sktime.annotation` 모듈 대신 `sktime.detection` 모듈에서 세그멘테이션 알고리즘을 찾도록 수정되었습니다.

만약 버전 호환성 문제가 발생하면, 다음 명령어로 최신 버전을 설치하세요:

```bash
pip install --upgrade fast-ticc sktime
```

## Joint Grouping (New Feature)

The pipeline now supports processing joints in groups rather than as a single block. This can significantly reduce computational complexity and allow for more focused analysis of specific robot parts.

### Configuration

To enable joint grouping, add a `joint_groups` section to your configuration:

```yaml
features:
  joint_groups:
    # Process torso joints as a group
    torso:
      - torso_joint_1
      - torso_joint_2
      - torso_joint_3
      - torso_joint_4
      - torso_joint_5
      - torso_joint_6
    
    # Process right arm joints as a group
    right_arm:
      - right_arm_joint_1
      - right_arm_joint_2
      - right_arm_joint_3
      - right_arm_joint_4
      - right_arm_joint_5
      - right_arm_joint_6
      - right_arm_joint_7
    
    # Process left arm joints as a group
    left_arm:
      - left_arm_joint_1
      - left_arm_joint_2
      - left_arm_joint_3
      - left_arm_joint_4
      - left_arm_joint_5
      - left_arm_joint_6
      - left_arm_joint_7
    
    # Process right hand finger joints as a group
    right_hand_fingers:
      - right_hand_finger_joint_1
      - right_hand_finger_joint_2
      - right_hand_finger_joint_3
      - right_hand_finger_joint_4
      - right_hand_finger_joint_5
      - right_hand_finger_joint_6
      - right_hand_finger_joint_7
      - right_hand_finger_joint_8
      - right_hand_finger_joint_9
      - right_hand_finger_joint_10
      - right_hand_finger_joint_11
      - right_hand_finger_joint_12
    
    # Process left hand finger joints as a group
    left_hand_fingers:
      - left_hand_finger_joint_1
      - left_hand_finger_joint_2
      - left_hand_finger_joint_3
      - left_hand_finger_joint_4
      - left_hand_finger_joint_5
      - left_hand_finger_joint_6
      - left_hand_finger_joint_7
      - left_hand_finger_joint_8
      - left_hand_finger_joint_9
      - left_hand_finger_joint_10
      - left_hand_finger_joint_11
      - left_hand_finger_joint_12
```

### Benefits

When joint groups are specified:
* Each group is processed independently with its own position/velocity/acceleration features
* Dimensionality is reduced (instead of 44D joint space, you get separate groups)
* TICC and other algorithms run faster on lower-dimensional data
* Results are more interpretable as they relate to specific robot components
* Separate visualization plots are generated for each joint group

### Modular Workflow

The pipeline now supports a modular workflow where users can run rupture analysis first and then optionally run TICC and sktime analysis:

```bash
# Run rupture analysis only (no TICC/sktime)
python scripts/run_lerobot_segmentation.py --dataset /path/to/dataset --local --n-episodes 1 --modalities joint --output results --skip-ticc --skip-sktime

# Run full pipeline (rupture + TICC + sktime)
python scripts/run_lerobot_segmentation.py --dataset /path/to/dataset --local --n-episodes 1 --modalities joint --output results

# Run with interactive plotting (plots will be saved when closed)
python scripts/run_lerobot_segmentation.py --dataset /path/to/dataset --local --n-episodes 1 --modalities joint --output results --show-plots
```

### Enhanced Visualization

The pipeline now generates separate plots for each joint group:
* `segmentation_joint.png` - Combined overview plot
* `segmentation_joint_torso.png` - Torso group plot
* `segmentation_joint_right_arm.png` - Right arm group plot
* `segmentation_joint_left_arm.png` - Left arm group plot
* `segmentation_joint_right_hand_fingers.png` - Right hand fingers group plot
* `segmentation_joint_left_hand_fingers.png` - Left hand fingers group plot

Each plot shows up to 18 dimensions for better visualization of all features in each group.
