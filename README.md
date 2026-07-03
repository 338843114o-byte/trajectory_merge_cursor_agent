# 伴随 JSON + 可切换 VLM 后端的商品轨迹合并判断

该版本在原有 Cursor 本地 Agent 后端之外，新增了 **API Key 方式的视觉模型 API 后端**。轨迹候选发现、图片构建、独立投票、格式修复和结果聚合逻辑保持一致，仅推理客户端可切换。

## 支持的后端

| `BACKEND` | 调用方式 | 是否需要 Cursor CLI | 鉴权方式 |
|---|---|---:|---|
| `agent` | Cursor 本地 Agent ACP | 是 | Cursor 登录或 `CURSOR_API_KEY` |
| `vlm` / `vlm_api` | OpenAI 兼容 `/chat/completions` 视觉 API | 否 | `VLM_API_KEY` |

## 目录结构

```text
trajectory_merge_cursor_agent/
├── data/                         # 示例或正式样本目录
├── prompts/
│   └── identity_match_v2.txt
├── cursor_acp_client.py          # Cursor Agent 后端
├── vlm_api_client.py             # 新增：API Key VLM 后端
├── trajectory_companion.py
├── vlm_identity.py
├── run_batch.py
├── run.sh
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
cd /home/xyj/trajectory_merge_cursor_agent
/home/xyj/.conda/envs/vlm_api/bin/python -m pip install -r requirements.txt
```

依赖包括 `Pillow` 和 `httpx`。

## 输入格式

程序递归查找 `trajectory.json` 或 `*.trajectory.json`：

```text
data/trajectory_cases/
└── 订单号或样本ID/
    ├── frame_001.png
    ├── frame_002.png
    ├── frame_003.png
    └── trajectory.json
```

`trajectory.json` 示例：

```json
{
  "case_id": "order_001",
  "frames": [
    {
      "frame_index": 100,
      "image_path": "frame_100.png",
      "tracks": [
        {
          "track_id": "H10",
          "bbox": [500, 150, 600, 270],
          "class_name": "product",
          "is_target": true
        }
      ]
    },
    {
      "frame_index": 101,
      "image_path": "frame_101.png",
      "tracks": [
        {
          "track_id": "H12",
          "bbox": [507, 154, 607, 274],
          "class_name": "product",
          "is_target": true
        }
      ]
    }
  ],
  "candidate_transitions": [
    {
      "old_track_id": "H10",
      "new_track_id": "H12",
      "old_frame_index": 100,
      "new_frame_index": 101
    }
  ]
}
```

`candidate_transitions` 可以省略。省略后，程序会根据相邻帧中消失和新增的轨迹自动生成候选。

## 方式一：使用 Cursor Agent

先确认 Cursor CLI 已登录：

```bash
agent status
```

运行：

```bash
BACKEND=agent \
CURSOR_MODEL=composer-2.5 \
bash /home/xyj/trajectory_merge_cursor_agent/run.sh
```

也可以直接执行 Python：

```bash
/home/xyj/.conda/envs/vlm_api/bin/python run_batch.py \
  --input_root /home/xyj/trajectory_merge_cursor_agent/data/trajectory_cases \
  --output_root /home/xyj/trajectory_merge_cursor_agent/work/trajectory_merge_results \
  --record_version auto \
  --backend agent \
  --cursor_workspace /home/xyj/trajectory_merge_cursor_agent \
  --cursor_model composer-2.5 \
  --votes 3
```

## 方式二：使用 API Key 调用 VLM

API 必须兼容 OpenAI Chat Completions 的多模态消息格式。Base URL 可以填写服务根地址、以 `/v1` 结尾的地址，或者完整的 `/chat/completions` 地址。

推荐把 Key 放在环境变量中，避免直接出现在命令历史或进程参数里：

```bash
export VLM_API_KEY='你的API-Key'

BACKEND=vlm \
VLM_API_BASE_URL='https://你的服务地址/v1' \
VLM_MODEL='你的视觉模型名称' \
bash /home/xyj/trajectory_merge_cursor_agent/run.sh
```

直接执行 Python：

```bash
export VLM_API_KEY='你的API-Key'

/home/xyj/.conda/envs/vlm_api/bin/python run_batch.py \
  --input_root /home/xyj/trajectory_merge_cursor_agent/data/trajectory_cases \
  --output_root /home/xyj/trajectory_merge_cursor_agent/work/trajectory_merge_results \
  --record_version auto \
  --backend vlm \
  --vlm_api_base_url 'https://你的服务地址/v1' \
  --vlm_model '你的视觉模型名称' \
  --vlm_max_tokens 512 \
  --vlm_temperature 0 \
  --vlm_retries 2 \
  --vlm_image_detail high \
  --votes 3
```

也支持 `--vlm_api_key`，但更建议使用 `VLM_API_KEY` 环境变量。默认使用 `Authorization: Bearer <key>`；需要 `api-key: <key>` 这类鉴权格式时，可设置：

```bash
VLM_API_KEY_HEADER='api-key' VLM_API_KEY_PREFIX=''
```

## 一行切换后端

只需要修改这一项：

```bash
BACKEND=agent bash run.sh
```

或：

```bash
export VLM_API_KEY='你的API-Key'
BACKEND=vlm VLM_API_BASE_URL='https://你的服务地址/v1' VLM_MODEL='你的模型名' bash run.sh
```

其他候选检测、投票次数和阈值配置无需改变，因此两种后端的结果可以直接对比。

## 常用环境变量

```text
PROJECT_ROOT
PYTHON_BIN
INPUT_ROOT
OUTPUT_ROOT
VOTES
BACKEND
CURSOR_WORKSPACE
CURSOR_MODEL
VLM_API_KEY
VLM_API_BASE_URL
VLM_MODEL
VLM_API_KEY_HEADER
VLM_API_KEY_PREFIX
VLM_MAX_TOKENS
VLM_TEMPERATURE
VLM_RETRIES
VLM_IMAGE_DETAIL
```

## 只验证输入，不调用模型

```bash
/home/xyj/.conda/envs/vlm_api/bin/python run_batch.py \
  --input_root /home/xyj/trajectory_merge_cursor_agent/data/trajectory_cases \
  --output_root /home/xyj/trajectory_merge_cursor_agent/work/trajectory_merge_results \
  --record_version auto \
  --dry_run
```

## 输出

每个版本目录生成一个汇总文件：

```text
work/trajectory_merge_results/1.0/结果汇总.md
```

汇总中会记录本次使用的推理后端、模型名称、每个样本的最终合并判断、候选轨迹、投票结果、一致率、身份置信度和判断原因。
