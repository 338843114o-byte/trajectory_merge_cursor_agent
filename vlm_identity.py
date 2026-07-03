#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 商品身份匹配提示词与严格结果解析。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from trajectory_companion import CandidateTransition, TrackObservation, VisualItem


def load_prompt(path: Path) -> str:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"提示词为空：{path}")
    return text


def build_identity_prompt(
    template: str,
    *,
    case_id: str,
    candidate: CandidateTransition,
    old_observations: list[TrackObservation],
    new_observations: list[TrackObservation],
    visual_items: list[VisualItem],
    vote_index: int,
) -> str:
    old_frames = ", ".join(str(obs.frame_index) for obs in old_observations)
    new_frames = ", ".join(str(obs.frame_index) for obs in new_observations)
    visual_lines = "\n".join(
        f"- 图片 {index}：{item.label}"
        for index, item in enumerate(visual_items, start=1)
    )
    return f"""{template.rstrip()}

【本轮固定事实——来自伴随 JSON，不需要你 OCR】
- case_id：{case_id}
- 本次只比较旧轨迹 `{candidate.old_track_id}` 与新轨迹 `{candidate.new_track_id}`。
- 旧轨迹参考帧：{old_frames}
- 新轨迹参考帧：{new_frames}
- 旧、新轨迹 ID 不同，程序已经确认这是一个轨迹切换候选。
- 同一画面可能还有多个稳定轨迹；它们不是本轮目标，必须忽略。
- 候选来源：{candidate.source}
- JSON 配对依据：{candidate.pairing_reason or '未提供'}
- 这是第 {vote_index} 次独立判断，不得假设其他判断结果。

【附加图片顺序】
{visual_lines}

只回答：旧轨迹与新轨迹是否属于同一个真实商品。
不要重新识别轨迹 ID，不要分析其他框，不要自行改变候选轨迹对。
最终只能输出一个严格 JSON 对象，不得输出 Markdown、代码块、分析过程或额外文字。"""


def _json_candidates(text: str) -> Iterable[str]:
    stripped = text.strip()
    if stripped:
        yield stripped
    for match in re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        yield match.group(1)
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "same", "是", "同一商品"}:
            return True
        if normalized in {"false", "0", "no", "different", "否", "不同商品"}:
            return False
    raise ValueError(f"same_physical_item 不是合法布尔值：{value!r}")


def _to_confidence(value: Any) -> float:
    confidence = float(value)
    if 1 < confidence <= 100:
        confidence /= 100.0
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence 必须位于 [0,1]：{value!r}")
    return confidence


def parse_identity_response(text: str) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(data, dict):
            errors.append("结果不是 JSON 对象")
            continue
        try:
            same = _to_bool(data.get("same_physical_item"))
            confidence = _to_confidence(data.get("confidence"))
            evidence_value = data.get("evidence") or []
            if isinstance(evidence_value, list):
                evidence = [str(item).strip() for item in evidence_value if str(item).strip()]
            else:
                evidence = [str(evidence_value).strip()] if str(evidence_value).strip() else []
            reason = str(data.get("reason") or "").strip()
            return {
                "parse_ok": True,
                "same_physical_item": same,
                "confidence": confidence,
                "evidence": evidence[:4],
                "reason": reason,
                "parse_error": None,
            }
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    return {
        "parse_ok": False,
        "same_physical_item": None,
        "confidence": None,
        "evidence": [],
        "reason": "",
        "parse_error": "; ".join(errors[-5:]) or "未找到合法 JSON",
    }


def build_repair_prompt(previous_output: str, parse_error: str) -> str:
    return f"""上一条回答无法被程序解析。只修复格式，不重新分析图片。
解析错误：{parse_error}
上一条回答：
-----
{previous_output}
-----
请只输出：
{{"same_physical_item":true或false,"confidence":0到1之间的小数,"evidence":["最多4条证据"],"reason":"一句话原因"}}
不得输出代码块或其他文字。"""
