#!/usr/bin/env bash
# v38 Jetson experiment: original CPU / adaptive v32 CPU / adaptive v32 CUDA geometry.
# Usage: bash RUN_JETSON_CUDA_V38.sh build|prepare|run|evaluate|summary|all
set -eo pipefail

ACTION=${1:-help}
WS=${COIN_WS_ROOT:-$HOME/coin_ws}
DATA=${COIN_DATA_ROOT:-$HOME/coin_data}
EXP=${JETSON_V38_EXP:-$WS/experiments/jetson_cuda_v38}
PATCH=${JETSON_V38_PATCH:-$(cd "$(dirname "$0")" && pwd)/COIN-BIEVR-jetson-cuda-v38-full.patch}
ADAPTIVE_CONF=${JETSON_ADAPTIVE_CONF:-}
ORIGINAL_CONF=${JETSON_ORIGINAL_CONF:-}
BAG=${RUNWAYD_BAG:-$DATA/enwide/RunwayD/ros2/runway_d}
GT=${RUNWAYD_GT:-$DATA/enwide/RunwayD/gt-runway_d.csv}
EVALUATOR=${COIN_EVALUATOR:-$WS/src/COIN-BIEVR-unofficial/scripts/evaluate_ate.py}
REPS=${REPS:-3}
CUDA_VALIDATE=${CUDA_VALIDATE:-true}
RESULT=${JETSON_V38_RESULT:-$DATA/results/jetson_cuda_v38/runway_d}
UNDERLAY_ROS=${UNDERLAY_ROS:-/opt/ros/humble/setup.bash}

underlay() {
  test -s "$UNDERLAY_ROS" || { echo "Missing ROS underlay: $UNDERLAY_ROS" >&2; exit 1; }
  source "$UNDERLAY_ROS"
  if test -n "${UNDERLAY_WS:-}"; then source "$UNDERLAY_WS"; fi
}

clone_at_base() {
  local dst=$1
  if ! test -d "$dst/.git"; then
    git clone https://github.com/qiyu-lu/COIN-BIEVR-unofficial.git "$dst"
    git -C "$dst" checkout --detach c560f3b
  fi
  test "$(git -C "$dst" rev-parse --short=7 HEAD)" = c560f3b
}

build() {
  test -s "$PATCH"
  mkdir -p "$EXP/src" "$EXP/configs" "$RESULT"
  local original=$EXP/src/original
  local modified=$EXP/src/cuda_v38
  clone_at_base "$original"
  clone_at_base "$modified"
  if ! git -C "$modified" apply --reverse --check --recount "$PATCH" >/dev/null 2>&1; then
    git -C "$modified" apply --check --recount "$PATCH"
    git -C "$modified" apply --recount "$PATCH"
  fi
  underlay
  colcon --log-base "$EXP/log_original" build \
    --base-paths "$original/BIEVR" "$original/interfaces/ros_common" "$original/interfaces/ros2" \
    --build-base "$EXP/build_original" --install-base "$EXP/install_original" \
    --executor sequential --packages-up-to bievr_lio_ros2 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  colcon --log-base "$EXP/log_v38" build \
    --base-paths "$modified/BIEVR" "$modified/interfaces/ros_common" "$modified/interfaces/ros2" \
    --build-base "$EXP/build_v38" --install-base "$EXP/install_v38" \
    --executor sequential --packages-up-to bievr_lio_ros2 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBIEVR_WITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87
  prepare
}

prepare() {
  test -s "$ORIGINAL_CONF" && test -s "$ADAPTIVE_CONF"
  mkdir -p "$EXP/configs" "$RESULT"
  python3 - "$ORIGINAL_CONF" "$ADAPTIVE_CONF" "$EXP/configs" "$RESULT" "$REPS" "$CUDA_VALIDATE" <<'PY'
import json, re, sys
from pathlib import Path
original, adaptive, out, result = map(Path, sys.argv[1:5])
reps, validation = int(sys.argv[5]), sys.argv[6].lower()
if reps < 1 or validation not in ('true', 'false'):
    raise SystemExit('Invalid REPS/CUDA_VALIDATE')

def set_key(text, section, key, value):
    pattern = rf'(?m)^(  {re.escape(key)}:).*$'
    text, count = re.subn(pattern, lambda m: f'{m.group(1)} {value}', text)
    if count > 1: raise SystemExit(f'duplicate {key}')
    if count == 0:
        marker = f'{section}:\n'
        if marker not in text: text += f'\n{section}:\n'
        text = text.replace(marker, marker + f'  {key}: {value}\n', 1)
    return text

for mode in ('original', 'adaptive_cpu', 'adaptive_cuda'):
    base = original.read_text() if mode == 'original' else adaptive.read_text()
    for rep in range(1, reps + 1):
        text = base
        if mode != 'original':
            for key, value in {
                'sequential_photo': 'True',
                'adaptive_photo_scale': 'True',
                'second_stage_mode': '"joint_weak_photo"',
                'photo_scale': '0.002',
                'geometry_probe_enabled': 'True',
                'geometry_probe_max_translation_m': '0.02',
                'geometry_probe_max_rotation_deg': '0.5',
                'geometry_probe_min_matches': '100',
                'reuse_direct_voxel_lookup': 'True',
                'geometry_cuda_enabled': str(mode == 'adaptive_cuda'),
                'geometry_cuda_validate': validation,
            }.items(): text = set_key(text, 'intensity', key, value)
            text = set_key(text, 'pipeline', 'overlap', 'False')
            text = set_key(text, 'pipeline', 'geo_overlap', 'False')
        trajectory = result / mode / f'rep{rep}' / 'trajectory.txt'
        text = set_key(text, 'debug', 'trajectory_path', json.dumps(str(trajectory)))
        text = set_key(text, 'debug', 'timing', 'True')
        text = set_key(text, 'debug', 'log', 'True')
        (out / f'{mode}_rep{rep}.yaml').write_text(text)
PY
}

run() {
  test -s "$BAG/metadata.yaml"
  prepare
  local mode rep install dir
  for mode in original adaptive_cpu adaptive_cuda; do
    install=$EXP/install_v38
    if test "$mode" = original; then install=$EXP/install_original; fi
    for rep in $(seq 1 "$REPS"); do
      dir=$RESULT/$mode/rep$rep
      mkdir -p "$dir"
      ( underlay; source "$install/local_setup.bash"
        ros2 launch bievr_lio_ros2 process_bag.launch.py \
          sensor_config:=enwide rosbag:="$BAG" \
          params:="$EXP/configs/${mode}_rep${rep}.yaml" rviz:=false
      ) 2>&1 | tee "$dir/run.log"
      test -s "$dir/trajectory.txt"
      if test "$mode" = adaptive_cuda; then grep -q 'cuda_geo=' "$dir/run.log"; fi
    done
  done
}

evaluate() {
  test -s "$GT" && test -s "$EVALUATOR"
  local mode rep dir
  for mode in original adaptive_cpu adaptive_cuda; do
    for rep in $(seq 1 "$REPS"); do
      dir=$RESULT/$mode/rep$rep
      python3 "$EVALUATOR" "$dir/trajectory.txt" "$GT" \
        --lever-arm -0.006253 0.011775 0.10825 --time-offset 0 --json \
        > "$dir/ate.json"
    done
  done
}

summary() {
  python3 - "$RESULT" "$REPS" <<'PY'
import json, re, statistics as st, sys
from pathlib import Path
root, reps = Path(sys.argv[1]), int(sys.argv[2])
for mode in ('original', 'adaptive_cpu', 'adaptive_cuda'):
    ate, cuda_used, cuda_valid, geo_ms, step_ms = [], 0, 0, [], []
    for rep in range(1, reps + 1):
        folder = root / mode / f'rep{rep}'
        ate.append(json.loads((folder/'ate.json').read_text())['rmse'])
        rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                for line in (folder/'run.log').read_text(errors='replace').splitlines()
                if 'CACHE_FRAME ' in line]
        cuda_used += sum(int(r.get('cuda_geo', 0)) for r in rows)
        cuda_valid += sum(int(r.get('cuda_valid', 0)) for r in rows)
        geo_ms.extend(float(r['geo_ms']) for r in rows if 'geo_ms' in r)
        step_ms.extend(float(r['step_ms']) for r in rows if 'step_ms' in r)
    print(f'{mode}: ATE mean={st.mean(ate):.6f} m, reps={reps}, '
          f'cuda_used={cuda_used}, cuda_valid={cuda_valid}, '
          f'geo_ms={st.mean(geo_ms) if geo_ms else float("nan"):.3f}, '
          f'step_ms={st.mean(step_ms) if step_ms else float("nan"):.3f}')
PY
}

case "$ACTION" in
  build) build;; prepare) prepare;; run) run;; evaluate) evaluate;; summary) summary;;
  all) build; run; evaluate; summary;;
  *) echo "Usage: bash $0 build|prepare|run|evaluate|summary|all";;
esac
