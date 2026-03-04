#!/bin/sh
set -e
cd /workspace/otus_rl_hw_01 && uv sync
exec "$@"
