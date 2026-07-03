#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cursor 本地 Agent ACP 客户端。

行为：
- 整个批次只启动一个 ``agent acp`` / ``cursor-agent acp`` 子进程；
- 每个轨迹判断样本创建独立 ACP Session，避免跨样本上下文污染；
- 图片通过 Base64 ACP image block 直接发送；
- 会话强制切换到 ask 模式，并拒绝所有文件/终端权限请求；
- 不发送 ``/max-mode off``，避免某些 CLI 版本把它当作普通问题。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable


class CursorAcpError(RuntimeError):
    """Cursor ACP 调用失败。"""


def _extract_acp_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_extract_acp_text(item) for item in content)
    if not isinstance(content, dict):
        return ""
    if content.get("type") == "text":
        return str(content.get("text") or "")
    if "content" in content:
        return _extract_acp_text(content.get("content"))
    return ""


class _AcpTransport:
    def __init__(
        self,
        command: list[str],
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float,
        debug: bool,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.debug = debug

        self.process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=100)
        self._request_id = 0

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(
            target=self._read_stdout,
            name="cursor-acp-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            name="cursor-acp-stderr",
            daemon=True,
        ).start()

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._messages.put(
                    {
                        "_transport_error": (
                            "Cursor ACP stdout 出现非 JSON 内容："
                            f"{line[:500]}；解析错误：{exc}"
                        )
                    }
                )
                continue
            if isinstance(message, dict):
                self._messages.put(message)

        self._messages.put(
            {
                "_transport_eof": True,
                "return_code": process.poll(),
            }
        )

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for raw_line in process.stderr:
            line = raw_line.rstrip("\n")
            self._stderr_lines.append(line)
            if self.debug and line:
                print(f"[Cursor ACP stderr] {line}", flush=True)

    def _write_message(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise CursorAcpError("Cursor ACP 进程尚未启动")
        if process.poll() is not None:
            raise CursorAcpError(
                "Cursor ACP 进程已经退出，"
                f"returncode={process.returncode}\n{self.stderr_tail()}"
            )
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        if self.debug:
            print(
                f"[Cursor ACP ->] id={message.get('id')} "
                f"method={message.get('method')}",
                flush=True,
            )
        process.stdin.write(payload + "\n")
        process.stdin.flush()

    def _respond_to_agent_request(self, message: dict[str, Any]) -> None:
        """拒绝 Agent 的文件、终端等权限请求。"""
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params") or {}

        if method == "session/request_permission":
            options = params.get("options") or []
            rejected_option_id: str | None = None
            for option in options:
                if not isinstance(option, dict):
                    continue
                kind = str(option.get("kind") or "").casefold()
                if kind.startswith("reject") or kind in {"deny", "denied"}:
                    rejected_option_id = str(option.get("optionId") or "") or None
                    break

            if rejected_option_id is None:
                for option in options:
                    if isinstance(option, dict):
                        rejected_option_id = str(option.get("optionId") or "") or None
                        if rejected_option_id:
                            break

            if rejected_option_id:
                result = {
                    "outcome": {
                        "outcome": "selected",
                        "optionId": rejected_option_id,
                    }
                }
            else:
                result = {"outcome": {"outcome": "cancelled"}}

            self._write_message(
                {"jsonrpc": "2.0", "id": request_id, "result": result}
            )
            return

        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"客户端未开放方法：{method}",
                },
            }
        )

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        collect_agent_text: bool = False,
        timeout_seconds: float | None = None,
    ) -> tuple[Any, str]:
        self._request_id += 1
        request_id = self._request_id
        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.timeout_seconds
        )
        deadline = time.monotonic() + timeout
        text_chunks: list[str] = []

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"等待 Cursor ACP 响应超时：method={method}, "
                    f"timeout={timeout}s\n{self.stderr_tail()}"
                )
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"等待 Cursor ACP 响应超时：method={method}\n"
                    f"{self.stderr_tail()}"
                ) from exc

            if message.get("_transport_error"):
                raise CursorAcpError(str(message["_transport_error"]))
            if message.get("_transport_eof"):
                raise CursorAcpError(
                    "Cursor ACP 进程意外退出，"
                    f"returncode={message.get('return_code')}\n"
                    f"{self.stderr_tail()}"
                )

            incoming_method = message.get("method")
            if incoming_method:
                if "id" in message:
                    self._respond_to_agent_request(message)
                    continue

                if incoming_method == "session/update":
                    update = (message.get("params") or {}).get("update") or {}
                    update_type = str(update.get("sessionUpdate") or "")
                    if collect_agent_text and update_type == "agent_message_chunk":
                        text_chunks.append(_extract_acp_text(update.get("content")))
                    if self.debug and update_type not in {
                        "agent_message_chunk",
                        "user_message_chunk",
                        "thought_message_chunk",
                        "agent_thought_chunk",
                    }:
                        print(f"[Cursor ACP update] {update_type}", flush=True)
                continue

            if message.get("id") != request_id:
                if self.debug:
                    print(
                        "[Cursor ACP] 忽略不属于当前请求的响应："
                        f"id={message.get('id')}",
                        flush=True,
                    )
                continue

            if "error" in message:
                error_info = message.get("error") or {}
                error_message = str(error_info.get("message") or error_info)
                raise CursorAcpError(
                    f"Cursor ACP 请求失败：method={method}；"
                    f"{error_message}\n{self.stderr_tail()}"
                )

            return message.get("result"), "".join(text_chunks)

    def stderr_tail(self) -> str:
        lines = [line for line in self._stderr_lines if line]
        if not lines:
            return ""
        return "Cursor ACP stderr 最近输出：\n" + "\n".join(lines[-20:])

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class CursorAcpClient:
    """一个 ACP 进程、多样本独立 Session 的 Cursor 图像客户端。"""

    def __init__(
        self,
        *,
        workspace: Path,
        model: str = "composer-2.5",
        api_key: str | None = None,
        agent_cli: str | None = None,
        timeout_seconds: float = 1200,
        debug: bool = False,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.model = model
        self.api_key = api_key
        self.agent_cli = agent_cli
        self.timeout_seconds = timeout_seconds
        self.debug = debug

        self._transport: _AcpTransport | None = None
        self._resolved_cli: str | None = None
        self._agent_capabilities: dict[str, Any] = {}
        self.session_create_count = 0
        self.prompt_send_count = 0

    def __enter__(self) -> "CursorAcpClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _resolve_cli(explicit_cli: str | None) -> str:
        candidates: list[str] = []
        if explicit_cli:
            candidates.append(explicit_cli)
        env_cli = os.environ.get("CURSOR_AGENT_CLI")
        if env_cli:
            candidates.append(env_cli)
        candidates.extend(["agent", "cursor-agent"])

        for candidate in candidates:
            found = shutil.which(candidate)
            if found:
                return found
            path = Path(candidate).expanduser()
            if path.is_file():
                return str(path.resolve())

        raise CursorAcpError(
            "未找到 Cursor Agent CLI。请先安装 Cursor CLI，"
            "并确保 `agent` 或 `cursor-agent` 位于 PATH；"
            "也可以设置 CURSOR_AGENT_CLI=/绝对路径。"
        )

    def start(self) -> None:
        if self._transport is not None:
            return
        if not self.workspace.is_dir():
            raise CursorAcpError(f"Cursor 工作目录不存在或不是目录：{self.workspace}")

        self._resolved_cli = self._resolve_cli(self.agent_cli)
        env = os.environ.copy()
        if self.api_key:
            env["CURSOR_API_KEY"] = self.api_key
        if self.debug:
            env.setdefault("CURSOR_LOG_LEVEL", "debug")

        command = [self._resolved_cli]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(["--mode", "ask", "acp"])

        self._transport = _AcpTransport(
            command=command,
            cwd=self.workspace,
            environment=env,
            timeout_seconds=self.timeout_seconds,
            debug=self.debug,
        )
        self._transport.start()

        try:
            result, _ = self._transport.request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": False,
                            "writeTextFile": False,
                        },
                        "terminal": False,
                    },
                    "clientInfo": {
                        "name": "trajectory-merge-cursor-client",
                        "title": "Trajectory Merge Cursor Client",
                        "version": "1.0.0",
                    },
                },
                timeout_seconds=60,
            )
            if not isinstance(result, dict):
                raise CursorAcpError(
                    f"Cursor ACP initialize 返回格式异常：{result!r}"
                )
            self._agent_capabilities = dict(result.get("agentCapabilities") or {})
            prompt_caps = self._agent_capabilities.get("promptCapabilities") or {}
            if not bool(prompt_caps.get("image")):
                raise CursorAcpError(
                    "当前 Cursor Agent ACP 未声明图片输入能力。"
                    "请先更新 Cursor CLI。"
                )
        except Exception:
            self.close()
            raise

        print(
            "[INFO] Cursor 本地 Agent ACP 已启动；"
            f"cli={self._resolved_cli}；model={self.model}；"
            "后续每个样本使用独立 Session。",
            flush=True,
        )

    def _switch_to_ask_mode(
        self,
        session_id: str,
        session_result: dict[str, Any],
    ) -> None:
        if self._transport is None:
            return

        for config in session_result.get("configOptions") or []:
            if not isinstance(config, dict):
                continue
            config_id = str(config.get("id") or "")
            category = str(config.get("category") or "")
            if config_id != "mode" and category != "mode":
                continue
            available_values = {
                str(option.get("value") or "")
                for option in (config.get("options") or [])
                if isinstance(option, dict)
            }
            if "ask" in available_values and config.get("currentValue") != "ask":
                self._transport.request(
                    "session/set_config_option",
                    {
                        "sessionId": session_id,
                        "configId": config_id,
                        "value": "ask",
                    },
                    timeout_seconds=60,
                )
            return

        modes = session_result.get("modes") or {}
        available_modes = {
            str(mode.get("id") or "")
            for mode in (modes.get("availableModes") or [])
            if isinstance(mode, dict)
        }
        if "ask" in available_modes and modes.get("currentModeId") != "ask":
            self._transport.request(
                "session/set_mode",
                {
                    "sessionId": session_id,
                    "modeId": "ask",
                },
                timeout_seconds=60,
            )

    def new_session(self, sample_name: str) -> str:
        if self._transport is None:
            raise CursorAcpError("Cursor ACP 客户端尚未启动")

        result, _ = self._transport.request(
            "session/new",
            {
                "cwd": str(self.workspace),
                "mcpServers": [],
            },
            timeout_seconds=60,
        )
        if not isinstance(result, dict):
            raise CursorAcpError(f"session/new 返回格式异常：{result!r}")
        session_id = result.get("sessionId")
        if not session_id:
            raise CursorAcpError(f"session/new 未返回 sessionId：{result!r}")

        session_id = str(session_id)
        self._switch_to_ask_mode(session_id, result)
        self.session_create_count += 1
        print(
            f"[INFO] 创建独立样本 Session：sample={sample_name}；"
            f"session={session_id}",
            flush=True,
        )
        return session_id

    @staticmethod
    def _image_block(image_path: Path) -> dict[str, str]:
        if not image_path.is_file():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image",
            "mimeType": mime_type,
            "data": data,
            "uri": image_path.resolve().as_uri(),
        }

    def send_prompt(
        self,
        *,
        session_id: str,
        prompt: str,
        image_paths: Iterable[Path] = (),
    ) -> str:
        if self._transport is None:
            raise CursorAcpError("Cursor ACP 客户端尚未启动")

        content: list[dict[str, str]] = [{"type": "text", "text": prompt}]
        content.extend(self._image_block(Path(path)) for path in image_paths)

        result, text = self._transport.request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": content,
            },
            collect_agent_text=True,
        )
        self.prompt_send_count += 1

        if isinstance(result, dict) and result.get("stopReason") in {
            "refusal",
            "cancelled",
        }:
            raise CursorAcpError(f"Cursor Agent 提前停止：{result}")

        text = text.strip()
        if not text:
            raise CursorAcpError(f"Cursor Agent 返回空文本：{result!r}")

        missing_markers = (
            "IMAGE_NOT_RECEIVED",
            "无法查看图片",
            "未收到图片",
            "图片不可见",
            "no image",
            "image was not provided",
            "cannot view the image",
        )
        normalized = " ".join(text.casefold().split())
        if any(marker.casefold() in normalized for marker in missing_markers):
            raise CursorAcpError(f"Cursor Agent 未收到图片：{text}")

        return text

    def metadata(self) -> dict[str, Any]:
        return {
            "cursor_model": self.model,
            "cursor_workspace": str(self.workspace),
            "cursor_cli_path": self._resolved_cli,
            "cursor_transport": "ACP_stdio_JSON-RPC",
            "cursor_process_reuse": True,
            "cursor_session_isolation": "one_session_per_case",
            "cursor_session_create_count": self.session_create_count,
            "cursor_prompt_send_count": self.prompt_send_count,
            "cursor_agent_capabilities": self._agent_capabilities,
        }

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
        self._transport = None
