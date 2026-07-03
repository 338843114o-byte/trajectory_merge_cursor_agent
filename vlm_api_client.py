#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过 OpenAI 兼容 Chat Completions API 调用视觉语言模型。"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import httpx


class VlmApiError(RuntimeError):
    """VLM API 调用失败。"""


def _normalize_chat_completions_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("VLM API Base URL 不能为空")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _extract_text(data: Any) -> str:
    """兼容常见 OpenAI 风格返回结构，提取最终文本。"""
    if not isinstance(data, dict):
        return ""

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    chunks: list[str] = []
                    for item in content:
                        if isinstance(item, str):
                            chunks.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text") or item.get("content")
                            if isinstance(text, str):
                                chunks.append(text)
                    if chunks:
                        return "".join(chunks).strip()

                # 部分兼容服务把最终内容放在 reasoning_content 中。
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str):
                    return reasoning.strip()

            text = choice.get("text")
            if isinstance(text, str):
                return text.strip()

    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    return ""


class VlmApiClient:
    """OpenAI 兼容视觉模型客户端，接口与 CursorAcpClient 对齐。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer",
        timeout_seconds: float = 180.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
        retries: int = 2,
        image_detail: str = "high",
        debug: bool = False,
    ) -> None:
        self.base_url = base_url
        self.endpoint = _normalize_chat_completions_url(base_url)
        self.model = model.strip()
        self.api_key = api_key or os.environ.get("VLM_API_KEY")
        self.api_key_header = api_key_header.strip() or "Authorization"
        self.api_key_prefix = api_key_prefix.strip()
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = max(0, retries)
        self.image_detail = image_detail
        self.debug = debug

        self._client: httpx.Client | None = None
        self.session_create_count = 0
        self.prompt_send_count = 0

    def __enter__(self) -> "VlmApiClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._client is not None:
            return
        if not self.model:
            raise VlmApiError("VLM 模型名不能为空，请设置 --vlm_model 或 VLM_MODEL")
        if not self.api_key:
            raise VlmApiError(
                "使用 vlm_api 后端时未找到 API Key。"
                "请先设置环境变量 VLM_API_KEY，或传入 --vlm_api_key。"
            )

        auth_value = (
            f"{self.api_key_prefix} {self.api_key}".strip()
            if self.api_key_prefix
            else str(self.api_key)
        )
        headers = {
            self.api_key_header: auth_value,
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=True,
        )
        print(
            "[INFO] VLM API 客户端已启动；"
            f"endpoint={self.endpoint}；model={self.model}；"
            "每次投票使用独立请求。",
            flush=True,
        )

    def new_session(self, sample_name: str) -> str:
        if self._client is None:
            raise VlmApiError("VLM API 客户端尚未启动")
        session_id = f"api-{uuid.uuid4()}"
        self.session_create_count += 1
        if self.debug:
            print(
                f"[VLM API] 创建逻辑会话：sample={sample_name}；session={session_id}",
                flush=True,
            )
        return session_id

    def _image_content(self, image_path: Path) -> dict[str, Any]:
        path = image_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
                "detail": self.image_detail,
            },
        }

    def send_prompt(
        self,
        *,
        session_id: str,
        prompt: str,
        image_paths: Iterable[Path] = (),
    ) -> str:
        del session_id  # API 请求无服务端会话；保留参数以兼容统一客户端接口。
        client = self._client
        if client is None:
            raise VlmApiError("VLM API 客户端尚未启动")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(self._image_content(Path(path)) for path in image_paths)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self.debug:
                    print(
                        f"[VLM API ->] attempt={attempt + 1}/{self.retries + 1}；"
                        f"images={len(content) - 1}",
                        flush=True,
                    )
                response = client.post(self.endpoint, json=payload)
                self.prompt_send_count += 1
                if response.status_code >= 400:
                    body = response.text[:2000]
                    raise VlmApiError(
                        f"VLM API HTTP {response.status_code}：{body}"
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise VlmApiError(
                        f"VLM API 返回的不是 JSON：{response.text[:2000]}"
                    ) from exc
                text = _extract_text(data)
                if not text:
                    raise VlmApiError(f"VLM API 返回空文本：{str(data)[:2000]}")
                normalized = " ".join(text.casefold().split())
                missing_markers = (
                    "image_not_received",
                    "无法查看图片",
                    "未收到图片",
                    "图片不可见",
                    "no image",
                    "image was not provided",
                    "cannot view the image",
                )
                if any(marker.casefold() in normalized for marker in missing_markers):
                    raise VlmApiError(f"VLM API 模型未收到图片：{text}")
                return text
            except (httpx.HTTPError, VlmApiError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 4))

        raise VlmApiError(f"VLM API 调用失败：{last_error}")

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "vlm_api",
            "vlm_api_base_url": self.base_url,
            "vlm_api_endpoint": self.endpoint,
            "vlm_model": self.model,
            "vlm_session_create_count": self.session_create_count,
            "vlm_prompt_send_count": self.prompt_send_count,
            "vlm_api_key_configured": bool(self.api_key),
            "vlm_api_key_header": self.api_key_header,
            "vlm_api_key_prefix_configured": bool(self.api_key_prefix),
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
