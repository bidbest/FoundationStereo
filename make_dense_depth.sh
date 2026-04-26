#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  echo "Usage: $0 /path/to/source_folder [make_dense_depth.py options]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SOURCE_ARG="$1"
shift

if ! SOURCE_FOLDER_PATH="$(realpath "${SOURCE_ARG}")"; then
  echo "Failed to resolve source folder path: ${SOURCE_ARG}" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_FOLDER_PATH}" ]]; then
  echo "Source folder does not exist or is not a directory: ${SOURCE_FOLDER_PATH}" >&2
  exit 1
fi

export SOURCE_FOLDER_PATH
export TORCH_CACHE_PATH="${REPO_ROOT}/torch_cache"

mkdir -p "${TORCH_CACHE_PATH}"

echo "Using repo root: ${REPO_ROOT}"
echo "Using source folder mount: ${SOURCE_FOLDER_PATH}"
echo "Using container source path: /workspace/source_folder"
echo "Using cache path: ${TORCH_CACHE_PATH}"

cd "${SCRIPT_DIR}"
docker compose run --rm foundation-stereo \
  python3 make_dense_depth.py /workspace/source_folder "$@"
