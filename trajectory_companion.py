#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""伴随 JSON 读取、轨迹切换候选发现以及候选视觉材料生成。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - 在运行入口给出更明确错误
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class TrackObservation:
    frame_order: int
    frame_index: int
    image_path: Path
    track_id: str
    bbox: tuple[float, float, float, float] | None = None
    class_name: str | None = None
    confidence: float | None = None
    is_target: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class FrameRecord:
    order: int
    frame_index: int
    image_path: Path
    tracks: list[TrackObservation]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateTransition:
    old_track_id: str
    new_track_id: str
    old_frame_index: int
    new_frame_index: int
    source: str
    pairing_score: float | None = None
    pairing_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return (
            f"{self.old_track_id}->{self.new_track_id}"
            f"@{self.old_frame_index}->{self.new_frame_index}"
        )


@dataclass
class CaseData:
    case_id: str
    source_json: Path
    frames: list[FrameRecord]
    explicit_candidates: list[CandidateTransition]
    metadata: dict[str, Any] = field(default_factory=dict)

    def histories(self, target_only: bool = True) -> dict[str, list[TrackObservation]]:
        observations = [obs for frame in self.frames for obs in frame.tracks]
        if target_only and any(obs.is_target is True for obs in observations):
            observations = [obs for obs in observations if obs.is_target is True]
        histories: dict[str, list[TrackObservation]] = {}
        for obs in observations:
            histories.setdefault(obs.track_id, []).append(obs)
        for values in histories.values():
            values.sort(key=lambda item: (item.frame_order, item.frame_index))
        return histories


@dataclass(frozen=True)
class VisualItem:
    label: str
    image_path: Path


def natural_key(value: str | Path) -> list[int | str]:
    name = Path(value).name if isinstance(value, Path) else str(value)
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"不支持的图片格式：{path}")
    return path


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value in (None, "", []):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox 必须是 [x1,y1,x2,y2]，实际为：{value!r}")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox 坐标无效：{value!r}")
    return x1, y1, x2, y2


def _parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_case(item: dict[str, Any], source_json: Path, base_dir: Path) -> CaseData:
    case_id = str(item.get("case_id") or source_json.parent.name).strip()
    if not case_id:
        raise ValueError(f"case_id 为空：{source_json}")

    raw_frames = item.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"{source_json} 的 frames 必须是非空数组")

    frames: list[FrameRecord] = []
    seen_frame_indices: set[int] = set()
    for order, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            raise ValueError(f"case={case_id} 第 {order + 1} 个 frame 不是对象")
        frame_index = int(raw_frame.get("frame_index", order))
        if frame_index in seen_frame_indices:
            raise ValueError(f"case={case_id} 存在重复 frame_index={frame_index}")
        seen_frame_indices.add(frame_index)
        image_value = raw_frame.get("image_path") or raw_frame.get("image")
        if not image_value:
            raise ValueError(f"case={case_id}, frame={frame_index} 缺少 image_path")
        image_path = _resolve_path(str(image_value), base_dir)

        raw_tracks = raw_frame.get("tracks") or []
        if not isinstance(raw_tracks, list):
            raise ValueError(f"case={case_id}, frame={frame_index} 的 tracks 必须是数组")
        tracks: list[TrackObservation] = []
        for raw_track in raw_tracks:
            if not isinstance(raw_track, dict):
                raise ValueError(f"case={case_id}, frame={frame_index} 的轨迹不是对象")
            track_id = str(raw_track.get("track_id") or "").strip()
            if not track_id:
                raise ValueError(f"case={case_id}, frame={frame_index} 存在空 track_id")
            reserved = {
                "track_id", "bbox", "class_name", "class", "confidence", "score",
                "is_target", "target",
            }
            metadata = {key: value for key, value in raw_track.items() if key not in reserved}
            tracks.append(
                TrackObservation(
                    frame_order=order,
                    frame_index=frame_index,
                    image_path=image_path,
                    track_id=track_id,
                    bbox=_parse_bbox(raw_track.get("bbox")),
                    class_name=(
                        str(raw_track.get("class_name") or raw_track.get("class")).strip()
                        if raw_track.get("class_name") is not None or raw_track.get("class") is not None
                        else None
                    ),
                    confidence=_parse_optional_float(
                        raw_track.get("confidence", raw_track.get("score"))
                    ),
                    is_target=(
                        bool(raw_track.get("is_target", raw_track.get("target")))
                        if raw_track.get("is_target") is not None or raw_track.get("target") is not None
                        else None
                    ),
                    metadata=metadata,
                )
            )
        frame_reserved = {"frame_index", "image_path", "image", "tracks"}
        frame_metadata = {key: value for key, value in raw_frame.items() if key not in frame_reserved}
        frames.append(
            FrameRecord(
                order=order,
                frame_index=frame_index,
                image_path=image_path,
                tracks=tracks,
                metadata=frame_metadata,
            )
        )

    frames.sort(key=lambda frame: (frame.frame_index, natural_key(frame.image_path)))
    # 排序后重写 order，使 gap 依据真实时间顺序而不是 JSON 原始位置。
    for new_order, frame in enumerate(frames):
        frame.order = new_order
        frame.tracks = [
            TrackObservation(
                frame_order=new_order,
                frame_index=obs.frame_index,
                image_path=obs.image_path,
                track_id=obs.track_id,
                bbox=obs.bbox,
                class_name=obs.class_name,
                confidence=obs.confidence,
                is_target=obs.is_target,
                metadata=obs.metadata,
            )
            for obs in frame.tracks
        ]

    raw_candidates = item.get("candidate_transitions") or []
    if not isinstance(raw_candidates, list):
        raise ValueError(f"case={case_id} 的 candidate_transitions 必须是数组")
    explicit_candidates: list[CandidateTransition] = []
    histories_tmp: dict[str, list[TrackObservation]] = {}
    for frame in frames:
        for obs in frame.tracks:
            histories_tmp.setdefault(obs.track_id, []).append(obs)

    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"case={case_id} 的 candidate_transition 不是对象")
        old_id = str(raw_candidate.get("old_track_id") or "").strip()
        new_id = str(raw_candidate.get("new_track_id") or "").strip()
        if not old_id or not new_id:
            raise ValueError(f"case={case_id} 的候选缺少 old_track_id/new_track_id")
        if old_id == new_id:
            raise ValueError(f"case={case_id} 的候选旧新 ID 相同：{old_id}")
        old_history = histories_tmp.get(old_id, [])
        new_history = histories_tmp.get(new_id, [])
        old_frame_index = int(
            raw_candidate.get(
                "old_frame_index",
                old_history[-1].frame_index if old_history else frames[0].frame_index,
            )
        )
        new_frame_index = int(
            raw_candidate.get(
                "new_frame_index",
                new_history[0].frame_index if new_history else frames[-1].frame_index,
            )
        )
        reserved = {
            "old_track_id", "new_track_id", "old_frame_index", "new_frame_index",
            "note", "reason",
        }
        metadata = {key: value for key, value in raw_candidate.items() if key not in reserved}
        explicit_candidates.append(
            CandidateTransition(
                old_track_id=old_id,
                new_track_id=new_id,
                old_frame_index=old_frame_index,
                new_frame_index=new_frame_index,
                source="explicit_json",
                pairing_score=1.0,
                pairing_reason=str(raw_candidate.get("note") or raw_candidate.get("reason") or "伴随 JSON 显式指定"),
                metadata=metadata,
            )
        )

    reserved_case = {"case_id", "frames", "candidate_transitions"}
    metadata = {key: value for key, value in item.items() if key not in reserved_case}
    return CaseData(
        case_id=case_id,
        source_json=source_json,
        frames=frames,
        explicit_candidates=explicit_candidates,
        metadata=metadata,
    )


def load_case_json(path: Path) -> list[CaseData]:
    path = path.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [_parse_case(item, path, path.parent) for item in data["cases"]]
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return [_parse_case(data, path, path.parent)]


def discover_cases(input_root: Path) -> list[CaseData]:
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{input_root}")
    candidates = sorted(
        {
            *input_root.rglob("trajectory.json"),
            *input_root.rglob("*.trajectory.json"),
        },
        key=natural_key,
    )
    cases: list[CaseData] = []
    seen_ids: set[str] = set()
    for path in candidates:
        for case in load_case_json(path):
            if case.case_id in seen_ids:
                raise ValueError(f"发现重复 case_id：{case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _pair_score(old: TrackObservation, new: TrackObservation) -> tuple[float, str]:
    if old.class_name and new.class_name and old.class_name != new.class_name:
        return -1.0, f"类别不一致：{old.class_name}!={new.class_name}"
    if old.bbox is None or new.bbox is None:
        return 0.50, "缺少 bbox，按单一轨迹变化候选处理"

    ox1, oy1, ox2, oy2 = old.bbox
    nx1, ny1, nx2, ny2 = new.bbox
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    old_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
    new_area = max(1.0, (nx2 - nx1) * (ny2 - ny1))
    scale = max(1.0, math.sqrt((old_area + new_area) / 2))
    normalized_distance = math.hypot(ncx - ocx, ncy - ocy) / scale
    spatial = math.exp(-normalized_distance)
    area_similarity = math.exp(-abs(math.log(new_area / old_area)))
    overlap = bbox_iou(old.bbox, new.bbox)
    score = 0.55 * spatial + 0.25 * area_similarity + 0.20 * overlap
    reason = (
        f"归一化中心距离={normalized_distance:.3f}, "
        f"面积相似度={area_similarity:.3f}, IoU={overlap:.3f}"
    )
    return score, reason


def detect_candidate_transitions(
    case: CaseData,
    *,
    max_gap_frames: int = 2,
    min_pairing_score: float = 0.18,
) -> list[CandidateTransition]:
    """从伴随 JSON 自动找“旧 ID 消失 + 新 ID 出现”的候选轨迹对。

    若 JSON 已显式给出 candidate_transitions，则完全采用显式候选，不再猜测。
    自动模式会优先识别每个时间边界上唯一变化的轨迹；多对变化时使用
    bbox/类别进行互为最佳匹配，稳定存在的其他轨迹不会送给 VLM。
    """
    if case.explicit_candidates:
        return list(case.explicit_candidates)

    histories = case.histories(target_only=True)
    if not histories:
        return []

    frame_by_order = {frame.order: frame for frame in case.frames}
    pair_pool: dict[tuple[str, str], CandidateTransition] = {}

    # 1. 相邻时间边界：最符合“只有一条路径变化”的情况。
    for left_order in range(len(case.frames) - 1):
        left = frame_by_order[left_order]
        right = frame_by_order[left_order + 1]
        all_left = left.tracks
        all_right = right.tracks
        if any(obs.is_target is True for obs in all_left + all_right):
            all_left = [obs for obs in all_left if obs.is_target is True]
            all_right = [obs for obs in all_right if obs.is_target is True]
        left_map = {obs.track_id: obs for obs in all_left}
        right_map = {obs.track_id: obs for obs in all_right}
        disappeared = [left_map[key] for key in left_map.keys() - right_map.keys()]
        appeared = [right_map[key] for key in right_map.keys() - left_map.keys()]
        if not disappeared or not appeared:
            continue

        scored: list[tuple[float, TrackObservation, TrackObservation, str]] = []
        for old in disappeared:
            for new in appeared:
                score, reason = _pair_score(old, new)
                if score >= 0:
                    scored.append((score, old, new, reason))

        if len(disappeared) == 1 and len(appeared) == 1 and scored:
            score, old, new, reason = scored[0]
            candidate = CandidateTransition(
                old_track_id=old.track_id,
                new_track_id=new.track_id,
                old_frame_index=old.frame_index,
                new_frame_index=new.frame_index,
                source="auto_unique_boundary",
                pairing_score=score,
                pairing_reason="该边界只有一个旧轨迹消失和一个新轨迹出现；" + reason,
            )
            pair_pool[(old.track_id, new.track_id)] = candidate
            continue

        # 多条轨迹同时变化：仅保留互为最佳的配对。
        best_new_for_old: dict[str, tuple[float, str]] = {}
        best_old_for_new: dict[str, tuple[float, str]] = {}
        for score, old, new, _ in scored:
            if score > best_new_for_old.get(old.track_id, (-math.inf, ""))[0]:
                best_new_for_old[old.track_id] = (score, new.track_id)
            if score > best_old_for_new.get(new.track_id, (-math.inf, ""))[0]:
                best_old_for_new[new.track_id] = (score, old.track_id)
        for score, old, new, reason in scored:
            if score < min_pairing_score:
                continue
            if best_new_for_old.get(old.track_id, (None, None))[1] != new.track_id:
                continue
            if best_old_for_new.get(new.track_id, (None, None))[1] != old.track_id:
                continue
            candidate = CandidateTransition(
                old_track_id=old.track_id,
                new_track_id=new.track_id,
                old_frame_index=old.frame_index,
                new_frame_index=new.frame_index,
                source="auto_mutual_best_boundary",
                pairing_score=score,
                pairing_reason="多轨迹边界中 bbox/类别互为最佳匹配；" + reason,
            )
            existing = pair_pool.get((old.track_id, new.track_id))
            if existing is None or (existing.pairing_score or -1) < score:
                pair_pool[(old.track_id, new.track_id)] = candidate

    # 2. 允许中间存在 1~max_gap_frames 个漏检帧。
    track_ids = sorted(histories)
    for old_id in track_ids:
        old_history = histories[old_id]
        old_last = old_history[-1]
        if old_last.frame_order >= len(case.frames) - 1:
            continue
        for new_id in track_ids:
            if new_id == old_id:
                continue
            new_history = histories[new_id]
            new_first = new_history[0]
            gap = new_first.frame_order - old_last.frame_order
            if gap <= 0 or gap > max_gap_frames + 1:
                continue
            score, reason = _pair_score(old_last, new_first)
            if score < min_pairing_score:
                continue
            candidate = CandidateTransition(
                old_track_id=old_id,
                new_track_id=new_id,
                old_frame_index=old_last.frame_index,
                new_frame_index=new_first.frame_index,
                source="auto_gap_endpoint",
                pairing_score=score,
                pairing_reason=f"旧轨迹结束与新轨迹开始相隔 {gap - 1} 个中间帧；{reason}",
            )
            existing = pair_pool.get((old_id, new_id))
            if existing is None or (existing.pairing_score or -1) < score:
                pair_pool[(old_id, new_id)] = candidate

    return sorted(
        pair_pool.values(),
        key=lambda item: (
            item.old_frame_index,
            item.new_frame_index,
            -(item.pairing_score or 0.0),
            item.old_track_id,
            item.new_track_id,
        ),
    )


def _nearest_observations(
    history: list[TrackObservation],
    anchor_frame_index: int,
    *,
    side: str,
    count: int,
) -> list[TrackObservation]:
    if side == "old":
        eligible = [obs for obs in history if obs.frame_index <= anchor_frame_index]
        if not eligible:
            eligible = history
        return eligible[-count:]
    eligible = [obs for obs in history if obs.frame_index >= anchor_frame_index]
    if not eligible:
        eligible = history
    return eligible[:count]


def select_candidate_observations(
    case: CaseData,
    candidate: CandidateTransition,
    *,
    frames_per_side: int = 2,
) -> tuple[list[TrackObservation], list[TrackObservation]]:
    histories = case.histories(target_only=False)
    old_history = histories.get(candidate.old_track_id, [])
    new_history = histories.get(candidate.new_track_id, [])
    if not old_history:
        raise ValueError(f"case={case.case_id} 未找到旧轨迹 {candidate.old_track_id}")
    if not new_history:
        raise ValueError(f"case={case.case_id} 未找到新轨迹 {candidate.new_track_id}")
    return (
        _nearest_observations(
            old_history,
            candidate.old_frame_index,
            side="old",
            count=frames_per_side,
        ),
        _nearest_observations(
            new_history,
            candidate.new_frame_index,
            side="new",
            count=frames_per_side,
        ),
    )


def _safe_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding_ratio, bh * padding_ratio
    return (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(width, int(math.ceil(x2 + pad_x))),
        min(height, int(math.ceil(y2 + pad_y))),
    )


def _save_focus_images(
    obs: TrackObservation,
    destination_dir: Path,
    prefix: str,
    *,
    padding_ratio: float,
    upscale: float,
) -> list[VisualItem]:
    if Image is None:
        raise RuntimeError("缺少 Pillow。请执行：pip install -r requirements.txt")
    with Image.open(obs.image_path) as original:
        image = original.convert("RGB")
        items: list[VisualItem] = []
        focus_path = destination_dir / f"{prefix}_full.jpg"
        focus = image.copy()
        if obs.bbox is not None:
            draw = ImageDraw.Draw(focus)
            x1, y1, x2, y2 = obs.bbox
            line_width = max(4, round(min(focus.size) / 150))
            draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=line_width)
            label = f"{prefix.upper()} track={obs.track_id} frame={obs.frame_index}"
            font = ImageFont.load_default() if ImageFont is not None else None
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            banner_y = max(0, int(y1) - th - 12)
            draw.rectangle((max(0, int(x1)), banner_y, max(0, int(x1)) + tw + 12, banner_y + th + 10), fill=(0, 0, 0))
            draw.text((max(0, int(x1)) + 6, banner_y + 5), label, fill=(255, 255, 0), font=font)
        focus.save(focus_path, quality=92)
        items.append(
            VisualItem(
                label=f"{prefix}: 完整帧，黄色框为 JSON 中指定的轨迹 {obs.track_id}",
                image_path=focus_path,
            )
        )

        if obs.bbox is not None:
            box = _safe_box(obs.bbox, image.width, image.height, padding_ratio)
            crop = image.crop(box)
            if upscale > 1.0:
                crop = crop.resize(
                    (
                        max(1, round(crop.width * upscale)),
                        max(1, round(crop.height * upscale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            crop_path = destination_dir / f"{prefix}_crop.jpg"
            crop.save(crop_path, quality=95)
            items.append(
                VisualItem(
                    label=f"{prefix}: 轨迹 {obs.track_id} 的局部放大图",
                    image_path=crop_path,
                )
            )
        return items


def build_visual_bundle(
    case: CaseData,
    candidate: CandidateTransition,
    destination_dir: Path,
    *,
    frames_per_side: int = 2,
    padding_ratio: float = 0.35,
    upscale: float = 2.5,
) -> tuple[list[VisualItem], list[TrackObservation], list[TrackObservation]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    old_obs, new_obs = select_candidate_observations(
        case,
        candidate,
        frames_per_side=frames_per_side,
    )
    visual_items: list[VisualItem] = []
    for index, obs in enumerate(old_obs, start=1):
        visual_items.extend(
            _save_focus_images(
                obs,
                destination_dir,
                f"old_{index:02d}",
                padding_ratio=padding_ratio,
                upscale=upscale,
            )
        )
    for index, obs in enumerate(new_obs, start=1):
        visual_items.extend(
            _save_focus_images(
                obs,
                destination_dir,
                f"new_{index:02d}",
                padding_ratio=padding_ratio,
                upscale=upscale,
            )
        )
    return visual_items, old_obs, new_obs


def companion_json_template(case_dir: Path) -> dict[str, Any]:
    case_dir = case_dir.expanduser().resolve()
    images = sorted(
        [
            path
            for path in case_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )
    return {
        "case_id": case_dir.name,
        "frames": [
            {
                "frame_index": index,
                "image_path": image.name,
                "tracks": [],
            }
            for index, image in enumerate(images, start=1)
        ],
        "candidate_transitions": [],
        "note": "请由 YOLO/Tracker 填入每帧 tracks；可显式填写 candidate_transitions。",
    }
