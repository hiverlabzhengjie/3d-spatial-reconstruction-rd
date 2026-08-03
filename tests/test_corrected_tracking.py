from __future__ import annotations

import numpy as np
import pytest

from spatial_reconstruction.contracts import CameraIntrinsics, CameraPose, PerceptionTarget
from spatial_reconstruction.localization import (
    CorrectedAnchor,
    CorrectedAnchorKind,
    CorrectedPairState,
    CorrectedSurfaceLocalization,
    CorrectedSurfaceRole,
    CorrectedTrackingPolicy,
    PairAnchorInput,
    PersonViewValidity,
    assess_person_mask_margins,
    derive_corrected_anchor,
    localize_corrected_surface,
    resolve_corrected_pair,
)
from spatial_reconstruction.localization.mask_depth_diagnostics import MaskDepthStrategy


def _camera() -> tuple[CameraIntrinsics, CameraPose]:
    intrinsics = CameraIntrinsics(
        camera_id="camera_a",
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )
    world_from_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = CameraPose(
        camera_id="camera_a",
        T_world_from_camera=world_from_camera.tolist(),
        T_camera_from_world=np.linalg.inv(world_from_camera).tolist(),
    )
    return intrinsics, pose


def _surface(
    *, role: CorrectedSurfaceRole, points_world: np.ndarray, bottom_truncated: bool
) -> CorrectedSurfaceLocalization:
    count = len(points_world)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20 : (100 if bottom_truncated else 80), 30:70] = 1
    margins = assess_person_mask_margins(mask, margin_pixels=2)
    return CorrectedSurfaceLocalization(
        role=role,
        strategy=(
            MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER
            if role is CorrectedSurfaceRole.PERSON_UPPER_BODY
            else MaskDepthStrategy.PERSON_LOWER_BODY
        ),
        margin_assessment=margins,
        candidate_pixel_count=count,
        valid_candidate_count=count,
        retained_sample_count=count,
        confidence_threshold=1.0,
        pixels_uv=np.column_stack((np.arange(count), np.arange(count))).astype(float),
        depth_m=np.full(count, 2.0),
        confidence=np.full(count, 5.0),
        points_camera_m=np.asarray(points_world),
        points_world_m=np.asarray(points_world),
        aggregate_camera_xyz_m=(0.0, 0.0, 0.5),
        aggregate_world_xyz_m=(0.0, 0.0, 0.5),
        reprojection_max_error_px=0.0,
        round_trip_max_error_m=0.0,
    )


def _anchor(kind: CorrectedAnchorKind, xyz: tuple[float, float, float]) -> CorrectedAnchor:
    return CorrectedAnchor(
        kind=kind,
        world_xyz_m=xyz,
        support_sample_count=64,
        measured_support_world_z_m=xyz[2],
        footpoint_available=kind is CorrectedAnchorKind.PERSON_FOOTPOINT,
        selection_reason="test",
    )


def test_bottom_margin_classifies_upper_body_only() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:99, 25:75] = 1

    assessment = assess_person_mask_margins(mask, margin_pixels=2)

    assert assessment.touches_bottom_margin
    assert not assessment.footpoint_candidate_allowed
    assert assessment.validity is PersonViewValidity.UPPER_BODY_ONLY_BOTTOM_TRUNCATED


def test_top_margin_does_not_invalidate_lower_body() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:70, 25:75] = 1

    assessment = assess_person_mask_margins(mask, margin_pixels=2)

    assert assessment.touches_top_margin
    assert not assessment.touches_bottom_margin
    assert assessment.footpoint_candidate_allowed


def test_localize_bottom_truncated_person_uses_upper_body_cluster() -> None:
    intrinsics, pose = _camera()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:100, 20:80] = 1
    depth = np.full((100, 100), 1.0)
    confidence = np.full((100, 100), 5.0)

    surface = localize_corrected_surface(
        source_mask=mask,
        corrected_depth_m=depth,
        confidence=confidence,
        target=PerceptionTarget.PERSON,
        intrinsics=intrinsics,
        pose=pose,
        policy=CorrectedTrackingPolicy(),
    )

    assert surface.role is CorrectedSurfaceRole.PERSON_UPPER_BODY
    assert surface.strategy is MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER


def test_near_floor_lower_body_derives_footpoint() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 0.2, 200),
            np.linspace(1.0, 1.2, 200),
            np.linspace(0.02, 0.80, 200),
        )
    )
    surface = _surface(
        role=CorrectedSurfaceRole.PERSON_LOWER_BODY,
        points_world=points,
        bottom_truncated=False,
    )

    anchor = derive_corrected_anchor(
        surface, target=PerceptionTarget.PERSON, policy=CorrectedTrackingPolicy()
    )

    assert anchor.kind is CorrectedAnchorKind.PERSON_FOOTPOINT
    assert anchor.footpoint_available
    assert anchor.world_xyz_m[2] == 0.0


def test_elevated_lower_body_remains_surface_not_footpoint() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 0.2, 200),
            np.linspace(1.0, 1.2, 200),
            np.linspace(0.60, 1.20, 200),
        )
    )
    surface = _surface(
        role=CorrectedSurfaceRole.PERSON_LOWER_BODY,
        points_world=points,
        bottom_truncated=False,
    )

    anchor = derive_corrected_anchor(
        surface, target=PerceptionTarget.PERSON, policy=CorrectedTrackingPolicy()
    )

    assert anchor.kind is CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE
    assert not anchor.footpoint_available


def test_pair_prefers_other_camera_footpoint_over_upper_body() -> None:
    result = resolve_corrected_pair(
        target=PerceptionTarget.PERSON,
        camera_a=PairAnchorInput(
            camera_id="camera_a",
            anchor=_anchor(CorrectedAnchorKind.PERSON_FOOTPOINT, (0.0, 1.0, 0.0)),
            reliability_score=10.0,
        ),
        camera_b=PairAnchorInput(
            camera_id="camera_b",
            anchor=_anchor(
                CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE, (0.2, 1.1, 1.3)
            ),
            reliability_score=20.0,
        ),
        maximum_disagreement_m=0.35,
    )

    assert result.state is CorrectedPairState.SINGLE_CAMERA
    assert result.selected_kind is CorrectedAnchorKind.PERSON_FOOTPOINT
    assert result.selected_camera_ids == ("camera_a",)
    assert not result.fallback_surface_used


def test_pair_does_not_fuse_mixed_lower_and_upper_surface_semantics() -> None:
    result = resolve_corrected_pair(
        target=PerceptionTarget.PERSON,
        camera_a=PairAnchorInput(
            camera_id="camera_a",
            anchor=_anchor(
                CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE, (0.0, 1.0, 0.6)
            ),
            reliability_score=10.0,
        ),
        camera_b=PairAnchorInput(
            camera_id="camera_b",
            anchor=_anchor(
                CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE, (0.1, 1.1, 1.3)
            ),
            reliability_score=20.0,
        ),
        maximum_disagreement_m=0.35,
    )

    assert result.state is CorrectedPairState.SINGLE_CAMERA
    assert result.selected_kind is CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE
    assert result.selected_camera_ids == ("camera_a",)
    assert result.fallback_surface_used


def test_two_close_footpoints_fuse_on_world_floor() -> None:
    result = resolve_corrected_pair(
        target=PerceptionTarget.PERSON,
        camera_a=PairAnchorInput(
            camera_id="camera_a",
            anchor=_anchor(CorrectedAnchorKind.PERSON_FOOTPOINT, (0.0, 1.0, 0.0)),
            reliability_score=10.0,
        ),
        camera_b=PairAnchorInput(
            camera_id="camera_b",
            anchor=_anchor(CorrectedAnchorKind.PERSON_FOOTPOINT, (0.2, 1.0, 0.0)),
            reliability_score=10.0,
        ),
        maximum_disagreement_m=0.35,
    )

    assert result.state is CorrectedPairState.FUSED
    assert result.world_xyz_m == pytest.approx((0.1, 1.0, 0.0))
    assert result.contribution_weights == pytest.approx((0.5, 0.5))


def test_policy_forbids_upper_body_floor_projection() -> None:
    with pytest.raises(ValueError):
        CorrectedTrackingPolicy(upper_body_floor_projection_allowed=True)  # type: ignore[arg-type]
