#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${ROOT_DIR}/.cache"
TMP_DIR="${CACHE_DIR}/tmp"
UV_CACHE="${CACHE_DIR}/uv"
VENV_DIR="${CACHE_DIR}/venv"
NATIVE_BUILD_DIR="${CACHE_DIR}/native-build"
DEMO_VIDEOS_DIR="${CACHE_DIR}/demo-videos"

# Keep all temporary files, uv caches, CMake builds, and pytest artifacts on XFS under .cache/
# (NEVER under /tmp, .agents/teamwork/, or virtiofs ~/.gemini/jetski).
mkdir -p "${TMP_DIR}" "${UV_CACHE}" "${NATIVE_BUILD_DIR}" "${DEMO_VIDEOS_DIR}" "${CACHE_DIR}/video-workbench/scratch"

export TMPDIR="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TMP="${TMP_DIR}"
export UV_CACHE_DIR="${UV_CACHE}"
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="/usr/bin/python3.11"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3.11 || command -v python3)"
fi

echo "[setup_env] Provisioning virtualenv at ${VENV_DIR} using ${PYTHON_BIN}..."
uv venv \
    --allow-existing \
    --seed \
    --no-managed-python \
    --no-python-downloads \
    --python "${PYTHON_BIN}" \
    "${VENV_DIR}"

echo "[setup_env] Installing dependencies from requirements-video.txt and editable visualizeit..."
uv pip install \
    --python "${VENV_DIR}/bin/python" \
    -r "${ROOT_DIR}/requirements-video.txt"

uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --no-build-isolation \
    -e "${ROOT_DIR}"

echo "[setup_env] Building native C++17 engine in ${NATIVE_BUILD_DIR}..."
cmake \
    -S "${ROOT_DIR}/native" \
    -B "${NATIVE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${NATIVE_BUILD_DIR}"

cmake --build "${NATIVE_BUILD_DIR}" --parallel 2
ctest --test-dir "${NATIVE_BUILD_DIR}" --output-on-failure

if [[ ! -f "${NATIVE_BUILD_DIR}/libvisualizeit_pose.so" ]]; then
    echo "[setup_env] ERROR: Expected shared library missing: ${NATIVE_BUILD_DIR}/libvisualizeit_pose.so" >&2
    exit 1
fi

echo "[setup_env] Generating deterministic demo video & 3D corpus in ${DEMO_VIDEOS_DIR}..."
"${VENV_DIR}/bin/python" -B -m visualizeit.demo_assets --output-dir "${DEMO_VIDEOS_DIR}"

echo "[setup_env] Environment ready."
