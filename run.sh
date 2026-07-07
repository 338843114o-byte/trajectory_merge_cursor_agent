#!/usr/bin/env bash
set -euo pipefail

# 基础路径
PROJECT_ROOT="${PROJECT_ROOT:-/home/xyj/trajectory_merge_cursor_agent}"
PYTHON_BIN="${PYTHON_BIN:-/home/xyj/.conda/envs/vlm_api/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/data/trajectory_cases}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/work/trajectory_merge_results}"
VOTES="${VOTES:-3}"

# 二选一：agent、vlm 或 vlm_api
BACKEND="${BACKEND:-vlm_api}"

# Cursor Agent 配置
CURSOR_WORKSPACE="${CURSOR_WORKSPACE:-${PROJECT_ROOT}}"
CURSOR_MODEL="${CURSOR_MODEL:-composer-2.5}"

# ============================================================
# YXKL OpenAI 兼容模型 (gpt-5.4) 配置
# ============================================================

# 设置 API Key 环境变量 (对应 api_key_env: YXKL_API_KEY)
export YXKL_API_KEY='sk-3gIiDHkNGEXntL5j1Z5kjPdgoAaOZYhLwN5yI9F52K7pluh7'
# 同时赋值给 VLM_API_KEY 供脚本内部校验和传参使用
export VLM_API_KEY="$YXKL_API_KEY"

# OpenAI 兼容接口地址
VLM_API_BASE_URL="${VLM_API_BASE_URL:-https://ai.yxkl.cloud/v1}"

# 模型名称
VLM_MODEL="${VLM_MODEL:-gpt-5.4}"

# Bearer 鉴权 (OpenAI 兼容标准)
VLM_API_KEY_HEADER="${VLM_API_KEY_HEADER:-Authorization}"
VLM_API_KEY_PREFIX="${VLM_API_KEY_PREFIX:-Bearer}"

# 生成参数 (对应 max_tokens: 4096, temperature: 0.1, vl_high_resolution_images: false)
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-4096}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.1}"
VLM_RETRIES="${VLM_RETRIES:-2}"
# 因为 vl_high_resolution_images 为 false，这里将图像细节设为 low
VLM_IMAGE_DETAIL="${VLM_IMAGE_DETAIL:-low}"

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
    : "${VLM_API_KEY:?BACKEND=vlm_api 时必须设置 VLM_API_KEY}"
    : "${VLM_API_BASE_URL:?BACKEND=vlm_api 时必须设置 VLM_API_BASE_URL}"
    : "${VLM_MODEL:?BACKEND=vlm_api 时必须设置 VLM_MODEL}"

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
