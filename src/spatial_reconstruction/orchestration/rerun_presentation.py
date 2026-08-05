"""Pure presentation semantics used by the integrated S06 Rerun export."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
)

Color = tuple[int, int, int, int]


class RerunPointStyle(ContractModel):
    """Explicit visual treatment without changing spatial authority."""

    show_position: bool
    color: Color
    radius_m: NonNegativeFloat
    label_prefix: str


class RerunEventMarker(ContractModel):
    """Separate deterministic transition and semantic review identities."""

    event_kind: Literal["pickup", "carry", "place"]
    transition_frame_index: NonNegativeInt
    transition_timestamp_seconds: NonNegativeFloat
    review_frame_index: NonNegativeInt
    review_timestamp_seconds: NonNegativeFloat
    qwen_event_label: Literal["pickup", "carry", "place", "unknown"]
    qwen_summary: str
    qwen_matches_candidate: bool

    @model_validator(mode="after")
    def validate_marker(self) -> RerunEventMarker:
        if not self.qwen_summary or self.qwen_summary.strip() != self.qwen_summary:
            raise ValueError("Qwen event summary must be non-empty and trimmed")
        if self.event_kind != "carry" and (
            self.transition_frame_index != self.review_frame_index
            or self.transition_timestamp_seconds != self.review_timestamp_seconds
        ):
            raise ValueError("pickup/place review identity must equal transition identity")
        return self


def point_style(
    *,
    target: str,
    state: str,
    anchor_kind: str | None,
) -> RerunPointStyle:
    """Map authoritative presentation semantics to distinct visual styles."""

    if state in {"missing", "occluded"}:
        color: Color = (120, 120, 120, 0) if state == "missing" else (255, 70, 70, 0)
        return RerunPointStyle(
            show_position=False,
            color=color,
            radius_m=0.0,
            label_prefix=state,
        )
    if state == "stale":
        return RerunPointStyle(
            show_position=True,
            color=(255, 150, 20, 150),
            radius_m=0.045,
            label_prefix="stale display-only",
        )
    if state != "measured":
        raise ValueError(f"unsupported presentation state: {state}")
    if target == "backpack":
        return RerunPointStyle(
            show_position=True,
            color=(30, 145, 255, 255),
            radius_m=0.075,
            label_prefix="measured backpack visible cluster",
        )
    person_styles: dict[str | None, tuple[Color, str]] = {
        "person_footpoint": ((40, 235, 90, 255), "measured person footpoint"),
        "person_lower_body_surface": (
            (255, 215, 40, 255),
            "measured person lower-body surface",
        ),
        "person_upper_body_surface": (
            (200, 90, 255, 255),
            "measured person upper-body surface",
        ),
    }
    if anchor_kind not in person_styles:
        raise ValueError("measured person point requires a supported anchor kind")
    color, label = person_styles[anchor_kind]
    return RerunPointStyle(
        show_position=True,
        color=color,
        radius_m=0.065,
        label_prefix=label,
    )


def coordinate_log_text(record: Mapping[str, Any]) -> str:
    """Format timestamped localization evidence without upgrading spatial authority."""

    target = str(record["target"])
    state = str(record["state"])
    timestamp = float(record["capture_timestamp_seconds"])
    frame = int(record["source_frame_index"])
    if state == "measured":
        xyz = record["raw_world_xyz_m"]
        if xyz is None:
            raise ValueError("measured coordinate log requires raw XYZ")
        cameras = "+".join(str(value) for value in record["source_measurement_camera_ids"])
        return (
            f"{target} MEASURED t={timestamp:.3f}s frame={frame} "
            f"XYZ=({float(xyz[0]):.3f}, {float(xyz[1]):.3f}, "
            f"{float(xyz[2]):.3f})m anchor={record['anchor_kind']} "
            f"cameras={cameras}"
        )
    if state == "stale":
        xyz = record["presentation_world_xyz_m"]
        if xyz is None:
            raise ValueError("stale coordinate log requires presentation XYZ")
        return (
            f"{target} STALE DISPLAY-ONLY t={timestamp:.3f}s frame={frame} "
            f"last_XYZ=({float(xyz[0]):.3f}, {float(xyz[1]):.3f}, "
            f"{float(xyz[2]):.3f})m age={float(record['measurement_age_seconds']):.3f}s"
        )
    if state not in {"missing", "occluded"}:
        raise ValueError(f"unsupported coordinate log state: {state}")
    if record["raw_world_xyz_m"] is not None or record["presentation_world_xyz_m"] is not None:
        raise ValueError("unavailable coordinate log state must not contain XYZ")
    return f"{target} {state.upper()} t={timestamp:.3f}s frame={frame} XYZ=unavailable"


def coordinate_point_label(record: Mapping[str, Any], *, prefix: str) -> str:
    """Label a visible 3D point with timestamp and explicit coordinate provenance."""

    xyz = record["presentation_world_xyz_m"]
    if xyz is None:
        raise ValueError("visible coordinate label requires presentation XYZ")
    timestamp = float(record["capture_timestamp_seconds"])
    return (
        f"{prefix} t={timestamp:.3f}s "
        f"XYZ=({float(xyz[0]):.3f}, {float(xyz[1]):.3f}, {float(xyz[2]):.3f})m"
    )


def build_event_markers(
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[RerunEventMarker, ...]:
    """Join Qwen results to jobs without allowing review time to move events."""

    result_by_job = {str(result["job"]["job_id"]): result for result in results}
    markers: list[RerunEventMarker] = []
    for job in jobs:
        result = result_by_job.get(str(job["job_id"]))
        if result is None:
            raise ValueError("Qwen result is missing for an accepted event job")
        interpretation = result["interpretation"]
        review_frame = job.get("review_frame_index")
        review_time = job.get("review_timestamp_seconds")
        markers.append(
            RerunEventMarker(
                event_kind=job["event_kind"],
                transition_frame_index=job["source_frame_index"],
                transition_timestamp_seconds=job["capture_timestamp_seconds"],
                review_frame_index=(
                    job["source_frame_index"] if review_frame is None else review_frame
                ),
                review_timestamp_seconds=(
                    job["capture_timestamp_seconds"] if review_time is None else review_time
                ),
                qwen_event_label=interpretation["event_label"],
                qwen_summary=interpretation["summary"],
                qwen_matches_candidate=interpretation["matches_candidate"],
            )
        )
    if tuple(marker.event_kind for marker in markers) != ("pickup", "carry", "place"):
        raise ValueError("Rerun event markers must remain pickup, carry, place")
    return tuple(markers)
