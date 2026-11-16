#!/usr/bin/env bash
# v32: per-frame geometry probe and direct voxel lookup cache ablation.
# Usage: bash RUN_GEOMETRY_PROBE_V32.sh build|run|evaluate|summary|all
set -eo pipefail

MODE=${1:-help}
WS=${COIN_WS_ROOT:-$HOME/coin_ws}
DATA=${COIN_DATA_ROOT:-$HOME/coin_data}
EXP=${GEOMETRY_V32_EXP:-$WS/experiments/geometry_probe_v32}
SRC=$EXP/src/COIN-BIEVR-unofficial
INSTALL=$EXP/install
RESULT=${GEOMETRY_V32_RESULT:-$DATA/results/geometry_probe_v32/runway_d}
CONF=$EXP/configs
PATCH=${GEOMETRY_V32_PATCH:-$(cd "$(dirname "$0")" && pwd)/COIN-BIEVR-geometry-probe-v32-full.patch}
BASE_CONF=${V9_BASE_CONF:-$WS/angle_direction_v9_configs/adaptive_joint_weak_photo_init_0p002/runway_d_base.yaml}
BAG=${RUNWAYD_BAG:-$DATA/enwide/RunwayD/ros2/runway_d}
GT=${RUNWAYD_GT:-$DATA/enwide/RunwayD/gt-runway_d.csv}
EVALUATOR=${COIN_EVALUATOR:-$WS/src/COIN-BIEVR-unofficial/scripts/evaluate_ate.py}
REPS=${REPS:-1}
PROBE_DT=${PROBE_DT:-0.05}
PROBE_DR_DEG=${PROBE_DR_DEG:-1.0}

underlay() {
  source "${UNDERLAY_ROS:-/opt/ros/jazzy/setup.bash}"
  local setup=${UNDERLAY_WS:-$HOME/colcon_ws/install/setup.bash}
  test -s "$setup" || { echo "missing ROS underlay: $setup" >&2; exit 1; }
  source "$setup"
}

build() {
  test -s "$PATCH" && test -s "$BASE_CONF"
  mkdir -p "$EXP/src" "$EXP/build" "$EXP/log" "$CONF"
  if test ! -e "$SRC"; then
    git clone --no-hardlinks --no-checkout "$WS/src/COIN-BIEVR-unofficial" "$SRC"
    git -C "$SRC" checkout --detach c560f3b
  fi
  test "$(git -C "$SRC" rev-parse --short=7 HEAD)" = c560f3b
  if ! git -C "$SRC" apply --reverse --check --recount "$PATCH" >/dev/null 2>&1; then
    git -C "$SRC" apply --check --recount "$PATCH"
    git -C "$SRC" apply --recount "$PATCH"
  fi
  # The host copy may contain a ROS conversion compatibility fix for Livox.
  local conversion=interfaces/ros_common/include/bievr_ros_common/conversions.h
  if grep -q converted_gen1 "$WS/src/COIN-BIEVR-unofficial/$conversion"; then
    cp "$WS/src/COIN-BIEVR-unofficial/$conversion" "$SRC/$conversion"
  fi
  underlay
  colcon --log-base "$EXP/log" build \
    --base-paths "$SRC/BIEVR" "$SRC/interfaces/ros_common" "$SRC/interfaces/ros2" \
    --build-base "$EXP/build" --install-base "$INSTALL" \
    --executor sequential --packages-up-to bievr_lio_ros2 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  prepare
}

prepare() {
  test -s "$BASE_CONF"
  mkdir -p "$CONF" "$RESULT"
  python3 - "$BASE_CONF" "$CONF" "$RESULT" "$REPS" "$PROBE_DT" "$PROBE_DR_DEG" <<'PY'
import re
import sys
from pathlib import Path

source, out, result, reps = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
probe_dt, probe_dr_deg = float(sys.argv[5]), float(sys.argv[6])
if not (0.0 < probe_dt <= 0.5 and 0.0 < probe_dr_deg <= 10.0):
    raise SystemExit('invalid probe thresholds')
base = source.read_text()
if not re.search(r'(?m)^map:\s*$', base):
    raise SystemExit('base config lacks map section')
for mode in ('baseline', 'lookup', 'probe', 'probe_lookup'):
    overlap = False
    anchor_period = 1
    geo_threads = 0
    joint_threads = 0
    probe_enabled = mode in ('probe', 'probe_lookup')
    lookup_enabled = mode in ('lookup', 'probe_lookup')
    for rep in range(1, reps + 1):
        text = base
        text, n = re.subn(r'(?m)^max_num_threads:.*$', 'max_num_threads: 4', text)
        if n == 0:
            text += '\nmax_num_threads: 4\n'
        text = re.sub(r'(?m)^  integrate_delay:.*\n', '', text)
        text = re.sub(r'(?m)^map:\s*$', 'map:\n  integrate_delay: 1', text, count=1)
        text, n = re.subn(r'(?m)^(  observability_state:).*$',
                          lambda m: f'{m.group(1)} "off"', text)
        if n == 0:
            text = text.replace('intensity:\n',
                                'intensity:\n  observability_state: "off"\n', 1)
        text = re.sub(r'(?ms)^pipeline:\n(?:  [^\n]*\n)*', '', text)
        text = ('pipeline:\n  overlap: false\n'
                f'  geo_overlap: {str(overlap).lower()}\n'
                '  geo_rebase_translation_m: 0.10\n'
                '  geo_rebase_rotation_deg: 1.0\n'
                f'  geo_threads: {geo_threads}\n'
                f'  joint_threads: {joint_threads}\n\n') + text
        text, n = re.subn(r'(?m)^(  geometry_anchor_period:).*$',
                          lambda m: f'{m.group(1)} {anchor_period}', text)
        if n == 0:
            text = text.replace('intensity:\n',
                                f'intensity:\n  geometry_anchor_period: {anchor_period}\n', 1)
        elif n != 1:
            raise SystemExit(f'duplicate geometry_anchor_period: {n}')
        text, n = re.subn(r'(?m)^(  reuse_geometry_linearization:).*$',
                          lambda m: f'{m.group(1)} False', text)
        if n == 0:
            text = text.replace('intensity:\n',
                                'intensity:\n  reuse_geometry_linearization: False\n', 1)
        elif n != 1:
            raise SystemExit(f'duplicate reuse_geometry_linearization: {n}')
        for key, value in (
            ('geometry_probe_enabled', str(probe_enabled)),
            ('reuse_direct_voxel_lookup', str(lookup_enabled)),
            ('geometry_probe_max_translation_m', str(probe_dt)),
            ('geometry_probe_max_rotation_deg', str(probe_dr_deg)),
            ('geometry_probe_min_matches', '100'),
        ):
            text = re.sub(rf'(?m)^  {key}:.*\n', '', text)
            text = text.replace('intensity:\n', f'intensity:\n  {key}: {value}\n', 1)
        text, n = re.subn(r'(?m)^(  sequential_photo:).*$',
                          lambda m: f'{m.group(1)} True', text)
        if n == 0:
            text = text.replace('intensity:\n', 'intensity:\n  sequential_photo: True\n', 1)
        trajectory = result / mode / f'rep{rep}' / 'trajectory.txt'
        text, n = re.subn(r'(?m)^(  trajectory_path:).*$',
                          lambda m: f'{m.group(1)} "{trajectory}"', text)
        if n != 1:
            raise SystemExit(f'expected one trajectory_path; got {n}')
        for key, value in [('log', 'True'), ('timing', 'True'), ('dashboard', 'False')]:
            text, n = re.subn(rf'(?m)^(  {key}:).*$',
                              lambda m, v=value: f'{m.group(1)} {v}', text)
            if n != 1:
                raise SystemExit(f'expected one debug.{key}; got {n}')
        path = out / f'runway_d_{mode}_rep{rep}.yaml'
        path.write_text(text)
        print(path)
PY
}

load_trial() {
  underlay
  test -s "$INSTALL/local_setup.bash" || { echo "build first" >&2; exit 1; }
  source "$INSTALL/local_setup.bash"
  local prefix
  prefix=$(ros2 pkg prefix bievr_lio_ros2)
  test "$prefix" = "$INSTALL/bievr_lio_ros2" || {
    echo "unexpected ROS package: $prefix" >&2; exit 1;
  }
}

run() {
  test -f "$BAG/metadata.yaml"
  load_trial
  prepare
  local mode rep dir log traj
  for mode in baseline lookup probe probe_lookup; do
    for rep in $(seq 1 "$REPS"); do
      dir=$RESULT/$mode/rep$rep
      log=$dir/run.log
      traj=$dir/trajectory.txt
      mkdir -p "$dir"
      echo "Running $mode rep$rep"
      ros2 launch bievr_lio_ros2 process_bag.launch.py \
        sensor_config:=enwide rosbag:="$BAG" \
        params:="$CONF/runway_d_${mode}_rep${rep}.yaml" \
        rviz:=false 2>&1 | tee "$log"
      test -s "$traj"
      grep -q 'GEOMETRY_V32' "$log"
      grep -q 'CACHE_FRAME' "$log"
    done
  done
}

evaluate() {
  test -s "$GT" && test -s "$EVALUATOR"
  local mode rep dir
  for mode in baseline lookup probe probe_lookup; do
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
import json
import re
import sys
from pathlib import Path
import numpy as np

root, reps = Path(sys.argv[1]), int(sys.argv[2])
for rep in range(1, reps + 1):
    tracks = {}
    results = {}
    for mode in ('baseline', 'lookup', 'probe', 'probe_lookup'):
        folder = root / mode / f'rep{rep}'
        log = (folder / 'run.log').read_text(errors='replace')
        summaries = [line for line in log.splitlines() if 'GEOMETRY_V32' in line]
        if not summaries:
            raise SystemExit(f'missing GEOMETRY_V32: {folder}')
        fields = dict(re.findall(r'(\w+)=([^\s]+)', summaries[-1]))
        track = np.atleast_2d(np.loadtxt(folder / 'trajectory.txt'))
        tracks[mode] = track
        ate_path = folder / 'ate.json'
        ate = json.loads(ate_path.read_text())['rmse'] if ate_path.exists() else float('nan')
        results[mode] = (ate, float(fields['mean_total_ms']))
        cache = log.count('CACHE_FRAME')
        frame_rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                      for line in log.splitlines() if 'CACHE_FRAME ' in line]
        probe_mean = (sum(float(x['probe_ms']) for x in frame_rows) / len(frame_rows)) if frame_rows else float('nan')
        lookup_hits = sum(int(x['direct_lookup_hits']) for x in frame_rows)
        print(f'rep{rep} {mode}: ATE={ate:.6f} m, trajectory={len(track)}, '
              f'CACHE_FRAME={cache}, mean_total={fields["mean_total_ms"]} ms, '
              f'probe_skipped={fields["probe_skipped"]}/{fields["probe_frames"]}, '
              f'probe_mean_ms={probe_mean:.3f}, direct_lookup_hits={lookup_hits}')
    a = tracks['baseline']
    ate0, time0 = results['baseline']
    for mode in ('lookup', 'probe', 'probe_lookup'):
        b = tracks[mode]
        if a.shape != b.shape or not np.array_equal(a[:, 0], b[:, 0]):
            print(f'FAIL: {mode} trajectory count or timestamps differ')
            continue
        pos_diff = np.linalg.norm(a[:, 1:4] - b[:, 1:4], axis=1)
        ate1, time1 = results[mode]
        print(f'rep{rep} {mode}: max position difference {pos_diff.max():.9f} m, '
              f'ATE change {(ate1 / ate0 - 1) * 100:+.2f}%, '
              f'mean time change {(time1 / time0 - 1) * 100:+.2f}%, '
              f'ATE_5pct={"PASS" if ate1 <= 1.05 * ate0 else "FAIL"}, '
              f'time_30ms={"PASS" if time1 < 30.0 else "FAIL"}')
PY
}

case "$MODE" in
  build) build ;;
  prepare) prepare ;;
  run) run ;;
  evaluate) evaluate ;;
  summary) summary ;;
  all) build; run; evaluate; summary ;;
  *) echo "Usage: bash $0 build|prepare|run|evaluate|summary|all" ;;
esac
