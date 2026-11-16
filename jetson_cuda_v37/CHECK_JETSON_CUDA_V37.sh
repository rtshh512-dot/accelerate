#!/usr/bin/env bash
# Read-only preflight for the Jetson CUDA experiment.
set -u

label() { printf '\n[%s]\n' "$1"; }
try() { if command -v "$1" >/dev/null 2>&1; then shift; "$@"; else printf 'not installed: %s\n' "$1"; fi; }

label device
uname -a
cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION_ID|UBUNTU_CODENAME)=' || true
if test -r /etc/nv_tegra_release; then cat /etc/nv_tegra_release; fi
if test -r /proc/device-tree/model; then tr -d '\0' </proc/device-tree/model; printf '\n'; fi

label cuda
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -n 4; else echo 'nvcc not found'; fi
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L; else echo 'nvidia-smi unavailable (common on Jetson; not a failure by itself)'; fi
if command -v deviceQuery >/dev/null 2>&1; then deviceQuery | tail -n 12; else echo 'deviceQuery not on PATH'; fi

label ros
if command -v ros2 >/dev/null 2>&1; then ros2 --version 2>&1 || true; else echo 'ros2 not on PATH'; fi
for distro in humble jazzy; do
  if test -r "/opt/ros/$distro/setup.bash"; then echo "ROS underlay: $distro"; fi
done

label compute
nproc
free -h
df -h .
if command -v nvpmodel >/dev/null 2>&1; then nvpmodel -q 2>&1 || true; fi
if command -v jetson_clocks >/dev/null 2>&1; then jetson_clocks --show 2>&1 | head -n 20; fi

label dependencies
for program in cmake c++ colcon python3; do
  if command -v "$program" >/dev/null 2>&1; then command -v "$program"; else echo "missing: $program"; fi
done
if command -v pkg-config >/dev/null 2>&1; then
  pkg-config --modversion tbb 2>/dev/null || true
  pkg-config --modversion eigen3 2>/dev/null || true
fi

label done
echo 'No packages, power modes, or files were changed.'
