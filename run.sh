#!/usr/bin/env bash
set -euo pipefail

# 可通过同名环境变量覆盖下面所有默认值。
PROJECT_ROOT="${PROJECT_ROOT:-/home/xyj/trajectory_merge_cursor_agent}"
PYTHON_BIN="${PYTHON_BIN:-/home/xyj/.conda/envs/vlm_api/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/data/trajectory_cases}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/trajectory_merge_results}"
VOTES="${VOTES:-3}"

# 二选一：agent 或 vlm（vlm_api 也可）
BACKEND="${BACKEND:-vlm_api}"

# Cursor Agent 配置（BACKEND=agent 时生效）
CURSOR_WORKSPACE="${CURSOR_WORKSPACE:-${PROJECT_ROOT}}"
CURSOR_MODEL="${CURSOR_MODEL:-composer-2.5}"

# VLM API 配置（BACKEND=vlm_api 时生效）
# API Key 不写死在脚本中，运行前设置：export VLM_API_KEY='你的key'
export VLM_API_KEY='sk-ba184bd609e1471ba56ae29ca9506dac'
VLM_API_BASE_URL="${VLM_API_BASE_URL:-https://api.deepseek.com}"
VLM_MODEL="${VLM_MODEL:-deepseek-v4-pro}"

VLM_API_KEY_HEADER="${VLM_API_KEY_HEADER:-Authorization}"
VLM_API_KEY_PREFIX="${VLM_API_KEY_PREFIX:-Bearer}"

VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-4096}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0}"
VLM_RETRIES="${VLM_RETRIES:-2}"
VLM_IMAGE_DETAIL="${VLM_IMAGE_DETAIL:-high}"

cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --input_root "$INPUT_ROOT"
  --output_root "$OUTPUT_ROOT"
  --record_version auto
  --backend "$BACKEND"
  --votes "$VOTES"
  --frames_per_side 2
  --min_agreement 0.66
  --min_identity_confidence 0.70
  --parse_retries 1
)

case "$BACKEND" in
  agent)
    BACKEND_ARGS=(
      --cursor_workspace "$CURSOR_WORKSPACE"
      --cursor_model "$CURSOR_MODEL"
    )
    ;;
  vlm|vlm_api)
    : "${VLM_API_KEY:?BACKEND=vlm 时必须先 export VLM_API_KEY='你的API Key'}"
    : "${VLM_API_BASE_URL:?BACKEND=vlm 时必须设置 VLM_API_BASE_URL}"
    : "${VLM_MODEL:?BACKEND=vlm 时必须设置 VLM_MODEL}"
    BACKEND_ARGS=(
      --vlm_api_base_url "$VLM_API_BASE_URL"
      --vlm_model "$VLM_MODEL"
      --vlm_api_key_header "$VLM_API_KEY_HEADER"
      --vlm_api_key_prefix "$VLM_API_KEY_PREFIX"
      --vlm_max_tokens "$VLM_MAX_TOKENS"
      --vlm_temperature "$VLM_TEMPERATURE"
      --vlm_retries "$VLM_RETRIES"
      --vlm_image_detail "$VLM_IMAGE_DETAIL"
    )
    ;;
  *)
    echo "错误：BACKEND 只能是 agent、vlm 或 vlm_api，当前为：$BACKEND" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$PROJECT_ROOT/run_batch.py" \
  "${COMMON_ARGS[@]}" \
  "${BACKEND_ARGS[@]}"
