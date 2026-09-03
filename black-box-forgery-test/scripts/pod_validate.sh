#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

: "${BBF_CONTAINER_IMAGE_DIGEST:?Set BBF_CONTAINER_IMAGE_DIGEST to the resolved sha256 image digest}"
case "${BBF_CONTAINER_IMAGE_DIGEST}" in
  sha256:[0-9a-fA-F][0-9a-fA-F]*) ;;
  *) echo "BBF_CONTAINER_IMAGE_DIGEST must be sha256:<hex>" >&2; exit 2 ;;
esac
if [ -n "${BBF_EXPECTED_IMAGE_DIGEST:-}" ] && [ "${BBF_CONTAINER_IMAGE_DIGEST}" != "${BBF_EXPECTED_IMAGE_DIGEST}" ]; then
  echo "container image digest does not match BBF_EXPECTED_IMAGE_DIGEST" >&2
  exit 2
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/bbf-uv-cache}"
export UV_CACHE_DIR
uv run --frozen bbf validate-fixtures --index data/frozen/pages.jsonl
uv run --frozen pytest -q
BBF_POD_VALIDATION_DIR="${BBF_POD_VALIDATION_DIR:-runs/pod-validation}" \
  uv run --frozen python scripts/record_pod_environment.py
echo "Pod validation passed; inspect environment.json before model download."
