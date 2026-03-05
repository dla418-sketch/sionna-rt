# Workspace Config Guide
이 문서는 `workspace`의 노트북/유틸에서 사용하는 주요 config 변수와 함수 입력 형식을 정리한 문서입니다.

## Notebook Config
### `scene_path.ipynb`
- `XML_PATH`: 씬 XML 경로
- `MERGE_SHAPES`: `load_scene` 병합 옵션
- `ROAD_OBJECT_ID`: 정점 추출할 오브젝트 ID
- `PATH_INDICES`: 수동 경로 인덱스
- `Z_OFFSET`: 경로 z 오프셋
- `SPEED_MS`: 이동 속도 (m/s)
- `DELTA_T`: 프레임 간격 (s)

### `multipath_nxn.ipynb` / `multipath_1x1.ipynb`
- `XML_PATH`, `MERGE_SHAPES`, `DELTA_T`, `TX_POWER_DBM`
- `RX_NAMES`, `TX_NAMES`: 장치 이름 리스트
- `RX_OBJECT_IDS`, `TX_OBJECT_IDS`: 장치별 경로 소스 오브젝트 ID 리스트
- `RX_FIXED`, `TX_FIXED`: 장치별 고정 여부 리스트 (`bool`)
- `RX_FIXED_POS`, `TX_FIXED_POS`: 장치별 고정 좌표 리스트
- `RX_PATH_INDICES_LIST`, `TX_PATH_INDICES_LIST`: 장치별 경로 인덱스 리스트
- `RX_Z_OFFSET_LIST`, `TX_Z_OFFSET_LIST`: 장치별 z 오프셋 리스트
- `RX_SPEED_MS_LIST`, `TX_SPEED_MS_LIST`: 장치별 속도 리스트
- `SOLVER_KWARGS`: `solve_paths_for_frame()`로 넘기는 ray-tracing 파라미터

### `channel_estimate.ipynb`
- `XML_PATH`, `MERGE_SHAPES`
- `RX_OBJECT_ID`, `TX_OBJECT_ID`
- `RX_PATH_INDICES`, `TX_PATH_INDICES`
- `RX_Z_OFFSET`, `TX_Z_OFFSET`
- `RX_SPEED_MS`, `TX_SPEED_MS`
- `DELTA_T`
- `RX_FIXED`, `TX_FIXED`
- `RX_FIXED_POS`, `TX_FIXED_POS`
- `RX_NAMES`, `TX_NAMES`
- `TX_POWER_DBM`
- `SOLVER_KWARGS`
- `CENTER_FREQ_HZ`, `BANDWIDTH_HZ`, `NUM_FREQ`: CFR 주파수 축 설정

## `utils.py` 함수 입력 형식
### `setup_tx_rx(...)`
- `tx_positions`: `[[x, y, z], [x, y, z], ...]`
- `rx_start_positions`: `[[x, y, z], [x, y, z], ...]`
- `tx_names`: `["tx1", "tx2", ...]` (`len(tx_names) == len(tx_positions)`)
- `rx_names`: `["rx1", "rx2", ...]` (`len(rx_names) == len(rx_start_positions)`)
- `tx_velocities` (옵션): `[[vx, vy, vz], ...]` (`len(tx_velocities) == len(tx_positions)`)
- `rx_velocities` (옵션): `[[vx, vy, vz], ...]` (`len(rx_velocities) == len(rx_start_positions)`)

### `solve_paths_for_frame(...)`
- `rx_names`: `setup_tx_rx()`에서 만든 Rx 이름 리스트
  - 예: `["rx1", "rx2"]`
- `tx_names`: `setup_tx_rx()`에서 만든 Tx 이름 리스트
  - 예: `["tx1", "tx2"]`
- `rx_path_data`: Rx 경로 데이터
  - 다중 Rx: `[rx1_path_data, rx2_path_data]`
  - 단일 path 공통 적용: `rx_common_path_data`
- `tx_path_data`: Tx 경로 데이터
  - 다중 Tx: `[tx1_path_data, tx2_path_data]`
  - 단일 path 공통 적용: `tx_common_path_data`
- `frame_idx`: `int`
  - 예: `10`
- `tx_look_at_rx`: `bool`
- `tx_look_at_rx_idx`: Tx가 바라볼 Rx 인덱스
  - 예: `0` (`rx_names[0]`)

### `path_data` 내부 형식
`prepare_path_data(...)`가 반환한 dict 형식을 그대로 사용합니다.

주요 키:
- `"waypoints"`: `(N, 3)` 배열
- `"speed_ms"`: `float`
- `"frame_times"`: `(T,)` 배열
