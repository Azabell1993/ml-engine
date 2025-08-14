#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$ROOT/third_party/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
exec "$ROOT/build/ml_engine"
