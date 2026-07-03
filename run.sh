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
# 阿里云百炼 Qwen3.7-Max 视觉模型配置
# ============================================================

# 不要使用已经泄露的旧 Key，替换成重新生成的 Key
export VLM_API_KEY='sk-ws-H.RXPLRRL.ZrLW.MEQCIAN42EEPeQkN61y7w-U8NuliI1P8sc0QwTlcgkJuw5vEAiAaFY8M9gs80NzEZV3n0M3XBf5dVqJ_YgjWCAxZrIx_zg'

# 阿里云百炼业务空间 OpenAI 兼容地址
VLM_API_BASE_URL="${VLM_API_BASE_URL:-https://ws-smovhssc95pp2zad.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"

# 必须使用 2026-06-08 视觉版本
VLM_MODEL="${VLM_MODEL:-qwen3.7-max-2026-06-08}"

# Bearer 鉴权
VLM_API_KEY_HEADER="${VLM_API_KEY_HEADER:-Authorization}"
VLM_API_KEY_PREFIX="${VLM_API_KEY_PREFIX:-Bearer}"

# 生成参数
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