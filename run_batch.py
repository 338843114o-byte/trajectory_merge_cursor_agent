#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于伴随 JSON + 可切换 VLM 后端的稳定商品轨迹合并判断。"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cursor_acp_client import CursorAcpClient
from vlm_api_client import VlmApiClient
from trajectory_companion import (
    CandidateTransition,
    CaseData,
    build_visual_bundle,
    detect_candidate_transitions,
    discover_cases,
    load_case_json,
)
from vlm_identity import (
    build_identity_prompt,
    build_repair_prompt,
    load_prompt,
    parse_identity_response,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "identity_match_v2.txt"


@dataclass
class VoteResult:
    parse_ok: bool
    same_physical_item: bool | None
    confidence: float | None
    evidence: list[str]
    reason: str
    error: str | None = None


@dataclass
class CandidateResult:
    candidate: CandidateTransition
    votes: list[VoteResult]
    same_physical_item: bool
    identity_confidence: float
    vote_text: str
    agreement: float
    merge: int
    reason: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="伴随 JSON 负责锁定变化轨迹，VLM 只判断旧新轨迹是否为同一商品。"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input_root",
        type=Path,
        default=PROJECT_ROOT / "work" / "trajectory_cases",
        help="递归寻找 trajectory.json / *.trajectory.json",
    )
    source.add_argument("--manifest", type=Path, help="单个 JSON 或含 cases 数组的总清单")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "work" / "trajectory_merge_results",
    )
    parser.add_argument("--record_version", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("agent", "vlm", "vlm_api"),
        default=os.environ.get("TRAJECTORY_MERGE_BACKEND", "agent"),
        help="推理后端：agent=Cursor 本地 Agent；vlm/vlm_api=OpenAI 兼容视觉模型 API",
    )
    parser.add_argument("--cursor_workspace", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cursor_model", default="composer-2.5")
    parser.add_argument("--cursor_api_key", default=None)
    parser.add_argument("--agent_cli", default=None)
    parser.add_argument(
        "--vlm_api_base_url",
        default=os.environ.get("VLM_API_BASE_URL", ""),
        help="OpenAI 兼容 API Base URL，也可直接填写 /chat/completions 完整地址",
    )
    parser.add_argument(
        "--vlm_api_key",
        default=None,
        help="VLM API Key；更推荐通过环境变量 VLM_API_KEY 设置，避免出现在进程参数中",
    )
    parser.add_argument(
        "--vlm_model",
        default=os.environ.get("VLM_MODEL", ""),
        help="API 后端使用的视觉模型名",
    )
    parser.add_argument(
        "--vlm_api_key_header",
        default=os.environ.get("VLM_API_KEY_HEADER", "Authorization"),
        help="API Key 请求头名称，默认 Authorization；Azure 风格可设为 api-key",
    )
    parser.add_argument(
        "--vlm_api_key_prefix",
        default=os.environ.get("VLM_API_KEY_PREFIX", "Bearer"),
        help="API Key 前缀，默认 Bearer；不需要前缀时传空字符串",
    )
    parser.add_argument(
        "--vlm_max_tokens",
        type=int,
        default=int(os.environ.get("VLM_MAX_TOKENS", "512")),
    )
    parser.add_argument(
        "--vlm_temperature",
        type=float,
        default=float(os.environ.get("VLM_TEMPERATURE", "0")),
    )
    parser.add_argument(
        "--vlm_retries",
        type=int,
        default=int(os.environ.get("VLM_RETRIES", "2")),
    )
    parser.add_argument(
        "--vlm_image_detail",
        choices=("auto", "low", "high"),
        default=os.environ.get("VLM_IMAGE_DETAIL", "high"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("CURSOR_AGENT_TIMEOUT", "1200")),
    )
    parser.add_argument("--prompt_file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--votes", type=int, default=3, help="每个候选使用多少次独立 Session 投票，建议奇数")
    parser.add_argument("--parse_retries", type=int, default=1)
    parser.add_argument("--min_agreement", type=float, default=0.66)
    parser.add_argument("--min_identity_confidence", type=float, default=0.70)
    parser.add_argument("--frames_per_side", type=int, default=2)
    parser.add_argument("--padding_ratio", type=float, default=0.35)
    parser.add_argument("--crop_upscale", type=float, default=2.5)
    parser.add_argument("--max_gap_frames", type=int, default=2)
    parser.add_argument("--min_pairing_score", type=float, default=0.18)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--cursor_debug", action="store_true")
    parser.add_argument("--vlm_debug", action="store_true")
    return parser.parse_args()


def resolve_run_dir(output_root: Path, version: str, overwrite: bool) -> tuple[str, Path]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if version == "auto":
        found: list[tuple[int, int]] = []
        for path in output_root.iterdir():
            if path.is_dir() and (match := re.fullmatch(r"(\d+)\.(\d+)", path.name)):
                found.append((int(match.group(1)), int(match.group(2))))
        version = "1.0" if not found else f"{max(found)[0]}.{max(found)[1] + 1}"
    run_dir = output_root / version
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"结果目录已存在：{run_dir}；使用 --overwrite 或更换版本")
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return version, run_dir


def load_cases(args: argparse.Namespace) -> list[CaseData]:
    cases = load_case_json(args.manifest) if args.manifest else discover_cases(args.input_root)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError(
            "没有找到伴随 JSON。请在每个样本目录放置 trajectory.json，"
            "或通过 --manifest 指定总清单。"
        )
    return cases


def _run_one_vote(
    client: Any,
    *,
    case: CaseData,
    candidate: CandidateTransition,
    prompt_template: str,
    visual_items: list[Any],
    old_observations: list[Any],
    new_observations: list[Any],
    vote_index: int,
    parse_retries: int,
) -> VoteResult:
    session_id = client.new_session(
        f"{case.case_id}:{candidate.old_track_id}->{candidate.new_track_id}:vote{vote_index}"
    )
    prompt = build_identity_prompt(
        prompt_template,
        case_id=case.case_id,
        candidate=candidate,
        old_observations=old_observations,
        new_observations=new_observations,
        visual_items=visual_items,
        vote_index=vote_index,
    )
    raw = client.send_prompt(
        session_id=session_id,
        prompt=prompt,
        image_paths=[item.image_path for item in visual_items],
    )
    parsed = parse_identity_response(raw)
    retries = 0
    while not parsed["parse_ok"] and retries < parse_retries:
        retries += 1
        raw = client.send_prompt(
            session_id=session_id,
            prompt=build_repair_prompt(raw, str(parsed.get("parse_error") or "未知错误")),
            image_paths=(),
        )
        parsed = parse_identity_response(raw)
    if not parsed["parse_ok"]:
        return VoteResult(
            parse_ok=False,
            same_physical_item=None,
            confidence=None,
            evidence=[],
            reason="",
            error=str(parsed.get("parse_error") or "解析失败"),
        )
    return VoteResult(
        parse_ok=True,
        same_physical_item=bool(parsed["same_physical_item"]),
        confidence=float(parsed["confidence"]),
        evidence=list(parsed["evidence"]),
        reason=str(parsed["reason"]),
    )


def aggregate_votes(
    candidate: CandidateTransition,
    votes: list[VoteResult],
    *,
    requested_votes: int,
    min_agreement: float,
    min_identity_confidence: float,
) -> CandidateResult:
    valid = [vote for vote in votes if vote.parse_ok and vote.same_physical_item is not None]
    if not valid:
        return CandidateResult(
            candidate=candidate,
            votes=votes,
            same_physical_item=False,
            identity_confidence=0.0,
            vote_text="0/0",
            agreement=0.0,
            merge=0,
            reason="所有 VLM 投票均无法解析，保守不合并",
            error="no_valid_votes",
        )
    positive = [vote for vote in valid if vote.same_physical_item is True]
    negative = [vote for vote in valid if vote.same_physical_item is False]
    majority_same = len(positive) > len(negative)
    majority_votes = positive if majority_same else negative
    agreement = len(majority_votes) / len(valid)
    confidence = (
        sum(float(vote.confidence or 0.0) for vote in majority_votes) / len(majority_votes)
        if majority_votes
        else 0.0
    )
    enough_valid = len(valid) >= max(1, math.ceil(requested_votes / 2))
    accepted_same = (
        majority_same
        and enough_valid
        and agreement >= min_agreement
        and confidence >= min_identity_confidence
    )
    representative = max(
        majority_votes,
        key=lambda vote: float(vote.confidence or 0.0),
    )
    reasons: list[str] = []
    if majority_same and not enough_valid:
        reasons.append("有效投票数量不足")
    if majority_same and agreement < min_agreement:
        reasons.append(f"投票一致率 {agreement:.2f} 低于阈值 {min_agreement:.2f}")
    if majority_same and confidence < min_identity_confidence:
        reasons.append(
            f"同一商品置信度 {confidence:.2f} 低于阈值 {min_identity_confidence:.2f}"
        )
    if accepted_same:
        reason = representative.reason or "多数独立判断认为旧新轨迹属于同一商品"
    elif reasons:
        reason = "；".join(reasons) + "，保守不合并"
    else:
        reason = representative.reason or "多数独立判断认为不是同一商品"
    return CandidateResult(
        candidate=candidate,
        votes=votes,
        same_physical_item=accepted_same,
        identity_confidence=confidence,
        vote_text=f"{len(positive)}同/{len(negative)}异/{len(valid)}有效",
        agreement=agreement,
        merge=1 if accepted_same else 0,
        reason=reason,
        error=None,
    )


def escape_md(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def write_summary(
    path: Path,
    *,
    version: str,
    args: argparse.Namespace,
    case_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    errors: list[str],
    started_at: datetime,
) -> None:
    success_cases = sum(row["pipeline_ok"] for row in case_rows)
    merge_cases = sum(row["merge"] == 1 for row in case_rows)
    lines = [
        "# 商品轨迹合并判断结果",
        "",
        f"- 记录版本：{version}",
        f"- 开始时间：{started_at.astimezone().isoformat()}",
        f"- 完成时间：{datetime.now().astimezone().isoformat()}",
        f"- 样本总数：{len(case_rows)}",
        f"- 成功样本：{success_cases}",
        f"- 输出 1（至少一个候选应合并）：{merge_cases}",
        f"- 输出 0：{len(case_rows) - merge_cases}",
        f"- 推理后端：{args.backend}",
        f"- 使用模型：{args.cursor_model if args.backend == 'agent' else args.vlm_model}",
        f"- 独立投票次数：{args.votes}",
        f"- 最低投票一致率：{args.min_agreement:.2f}",
        f"- 最低商品身份置信度：{args.min_identity_confidence:.2f}",
        "",
        "## 样本级结果",
        "",
        "| case_id | JSON候选数 | 判为可合并候选数 | 最终输出 | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in case_rows:
        lines.append(
            f"| {escape_md(row['case_id'])} | {row['candidate_count']} | "
            f"{row['merge_candidate_count']} | {row['merge']} | {escape_md(row['status'])} |"
        )

    lines.extend(
        [
            "",
            "## 候选轨迹对明细",
            "",
            "| case_id | 旧轨迹→新轨迹 | JSON定位帧 | 候选来源 | 配对分数 | VLM投票 | 一致率 | 商品置信度 | 输出 | 原因 |",
            "|---|---|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    if not candidate_rows:
        lines.append("| - | - | - | - | - | - | - | - | 0 | 未检测到轨迹切换候选 |")
    else:
        for row in candidate_rows:
            score = row.get("pairing_score")
            score_text = f"{score:.3f}" if isinstance(score, (float, int)) else ""
            lines.append(
                f"| {escape_md(row['case_id'])} | "
                f"{escape_md(row['old_track_id'])}→{escape_md(row['new_track_id'])} | "
                f"{row['old_frame_index']}→{row['new_frame_index']} | "
                f"{escape_md(row['source'])} | {score_text} | "
                f"{escape_md(row['vote_text'])} | {row['agreement']:.3f} | "
                f"{row['identity_confidence']:.3f} | {row['merge']} | "
                f"{escape_md(row['reason'])} |"
            )

    if errors:
        lines.extend(["", "## 错误", ""])
        lines.extend(f"- {escape_md(error)}" for error in errors)

    lines.extend(
        [
            "",
            "## 判定规则",
            "",
            "最终输出由 Python 计算：伴随 JSON 已确认旧、新轨迹 ID 不同；"
            "只有当独立 VLM 投票认为二者属于同一真实商品，且一致率与置信度达到阈值时，才输出 1。",
            "",
            "> 该版本不会让 VLM 重新 OCR 轨迹 ID；一帧中的其他稳定轨迹由 JSON/Python 排除。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.votes <= 0 or args.votes % 2 == 0:
        raise ValueError("--votes 必须是正奇数，例如 1、3、5")
    if args.frames_per_side <= 0:
        raise ValueError("--frames_per_side 必须大于 0")
    if not 0.5 <= args.min_agreement <= 1:
        raise ValueError("--min_agreement 必须位于 [0.5,1]")
    if not 0 <= args.min_identity_confidence <= 1:
        raise ValueError("--min_identity_confidence 必须位于 [0,1]")
    if args.timeout <= 0:
        raise ValueError("--timeout 必须大于 0")
    if args.vlm_max_tokens <= 0:
        raise ValueError("--vlm_max_tokens 必须大于 0")
    if args.vlm_retries < 0:
        raise ValueError("--vlm_retries 不能小于 0")

    started_at = datetime.now()
    cases = load_cases(args)
    version, run_dir = resolve_run_dir(args.output_root, args.record_version, args.overwrite)
    summary_path = run_dir / "结果汇总.md"
    prompt_template = load_prompt(args.prompt_file)

    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    print("=" * 80)
    print("伴随 JSON 商品轨迹合并判断")
    print("=" * 80)
    print(f"样本数：{len(cases)}")
    print(f"结果：{summary_path}")
    print(f"推理后端：{args.backend}")
    print(f"使用模型：{args.cursor_model if args.backend == 'agent' else args.vlm_model}")
    print(f"每候选独立投票：{args.votes}")

    def process_without_client() -> None:
        for case in cases:
            candidates = detect_candidate_transitions(
                case,
                max_gap_frames=args.max_gap_frames,
                min_pairing_score=args.min_pairing_score,
            )
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "case_id": case.case_id,
                        "old_track_id": candidate.old_track_id,
                        "new_track_id": candidate.new_track_id,
                        "old_frame_index": candidate.old_frame_index,
                        "new_frame_index": candidate.new_frame_index,
                        "source": candidate.source,
                        "pairing_score": candidate.pairing_score,
                        "vote_text": "dry-run",
                        "agreement": 0.0,
                        "identity_confidence": 0.0,
                        "merge": 0,
                        "reason": candidate.pairing_reason,
                    }
                )
            case_rows.append(
                {
                    "case_id": case.case_id,
                    "candidate_count": len(candidates),
                    "merge_candidate_count": 0,
                    "merge": 0,
                    "pipeline_ok": True,
                    "status": "dry-run：只验证 JSON 和候选发现",
                }
            )

    if args.dry_run:
        process_without_client()
    else:
        if args.backend == "agent":
            client_context: Any = CursorAcpClient(
                workspace=args.cursor_workspace,
                model=args.cursor_model,
                api_key=args.cursor_api_key,
                agent_cli=args.agent_cli,
                timeout_seconds=args.timeout,
                debug=args.cursor_debug,
            )
        else:
            client_context = VlmApiClient(
                base_url=args.vlm_api_base_url,
                model=args.vlm_model,
                api_key=args.vlm_api_key,
                api_key_header=args.vlm_api_key_header,
                api_key_prefix=args.vlm_api_key_prefix,
                timeout_seconds=args.timeout,
                max_tokens=args.vlm_max_tokens,
                temperature=args.vlm_temperature,
                retries=args.vlm_retries,
                image_detail=args.vlm_image_detail,
                debug=args.vlm_debug,
            )

        with client_context as client:
            for case_index, case in enumerate(cases, start=1):
                print(f"\n[CASE] {case_index}/{len(cases)} {case.case_id}", flush=True)
                try:
                    candidates = detect_candidate_transitions(
                        case,
                        max_gap_frames=args.max_gap_frames,
                        min_pairing_score=args.min_pairing_score,
                    )
                    case_candidate_results: list[CandidateResult] = []
                    if not candidates:
                        case_rows.append(
                            {
                                "case_id": case.case_id,
                                "candidate_count": 0,
                                "merge_candidate_count": 0,
                                "merge": 0,
                                "pipeline_ok": True,
                                "status": "JSON 中未发现旧轨迹消失/新轨迹出现候选",
                            }
                        )
                        print("[RESULT] 无轨迹切换候选 -> 0", flush=True)
                        continue

                    with tempfile.TemporaryDirectory(prefix="trajectory_merge_") as temp_root:
                        temp_root_path = Path(temp_root)
                        for candidate_index, candidate in enumerate(candidates, start=1):
                            print(
                                f"[CANDIDATE] {candidate.old_track_id}->{candidate.new_track_id} "
                                f"({candidate_index}/{len(candidates)})",
                                flush=True,
                            )
                            candidate_dir = temp_root_path / f"candidate_{candidate_index:03d}"
                            visual_items, old_obs, new_obs = build_visual_bundle(
                                case,
                                candidate,
                                candidate_dir,
                                frames_per_side=args.frames_per_side,
                                padding_ratio=args.padding_ratio,
                                upscale=args.crop_upscale,
                            )
                            votes: list[VoteResult] = []
                            for vote_index in range(1, args.votes + 1):
                                try:
                                    vote = _run_one_vote(
                                        client,
                                        case=case,
                                        candidate=candidate,
                                        prompt_template=prompt_template,
                                        visual_items=visual_items,
                                        old_observations=old_obs,
                                        new_observations=new_obs,
                                        vote_index=vote_index,
                                        parse_retries=args.parse_retries,
                                    )
                                except Exception as exc:
                                    vote = VoteResult(
                                        parse_ok=False,
                                        same_physical_item=None,
                                        confidence=None,
                                        evidence=[],
                                        reason="",
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                votes.append(vote)
                            result = aggregate_votes(
                                candidate,
                                votes,
                                requested_votes=args.votes,
                                min_agreement=args.min_agreement,
                                min_identity_confidence=args.min_identity_confidence,
                            )
                            case_candidate_results.append(result)
                            candidate_rows.append(
                                {
                                    "case_id": case.case_id,
                                    "old_track_id": candidate.old_track_id,
                                    "new_track_id": candidate.new_track_id,
                                    "old_frame_index": candidate.old_frame_index,
                                    "new_frame_index": candidate.new_frame_index,
                                    "source": candidate.source,
                                    "pairing_score": candidate.pairing_score,
                                    "vote_text": result.vote_text,
                                    "agreement": result.agreement,
                                    "identity_confidence": result.identity_confidence,
                                    "merge": result.merge,
                                    "reason": result.reason,
                                }
                            )
                            print(
                                f"[CANDIDATE RESULT] {candidate.old_track_id}->{candidate.new_track_id} "
                                f"merge={result.merge}, votes={result.vote_text}, "
                                f"confidence={result.identity_confidence:.3f}",
                                flush=True,
                            )

                    merge_count = sum(item.merge == 1 for item in case_candidate_results)
                    case_rows.append(
                        {
                            "case_id": case.case_id,
                            "candidate_count": len(candidates),
                            "merge_candidate_count": merge_count,
                            "merge": 1 if merge_count > 0 else 0,
                            "pipeline_ok": True,
                            "status": "完成",
                        }
                    )
                except Exception as exc:
                    message = f"case={case.case_id}: {type(exc).__name__}: {exc}"
                    errors.append(message + "\n" + traceback.format_exc())
                    case_rows.append(
                        {
                            "case_id": case.case_id,
                            "candidate_count": 0,
                            "merge_candidate_count": 0,
                            "merge": 0,
                            "pipeline_ok": False,
                            "status": message,
                        }
                    )
                    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
                    if args.fail_fast:
                        break

    write_summary(
        summary_path,
        version=version,
        args=args,
        case_rows=case_rows,
        candidate_rows=candidate_rows,
        errors=errors,
        started_at=started_at,
    )
    print("\n" + "=" * 80)
    print(f"完成。唯一输出文件：{summary_path}")
    print("=" * 80)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
