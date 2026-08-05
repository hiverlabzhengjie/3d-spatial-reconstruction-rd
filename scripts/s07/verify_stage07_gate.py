"""Audit the complete S07 roadmap gate from independent verification reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_RECORDING_SHA256 = (
    "bcf84af987069151339427d57d7642cffd0e92b6c0ff05bbdbddb7c6143b64ca"
)
REQUIRED_DOCUMENTS = {
    "capture_guide": Path("docs/CAPTURE_CALIBRATION_GUIDE.md"),
    "reproduction_guide": Path("docs/REPRODUCING_FINAL_DEMONSTRATION.md"),
    "technical_report": Path("docs/FINAL_TECHNICAL_REPORT.md"),
    "stage_record": Path("docs/stages/S07_FINAL_CAPTURE_REFINEMENT_REPORTING.md"),
    "handoff": Path("docs/stages/S07_HANDOFF.md"),
    "status": Path("docs/STATUS.md"),
    "decisions": Path("docs/DECISIONS.md"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--proxies", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gate audit: {output_path}")

    verification_paths = {
        "entry": args.entry.resolve(),
        "proxies": args.proxies.resolve(),
        "rerun": args.rerun.resolve(),
    }
    reports = {
        name: _load_json(path) for name, path in verification_paths.items()
    }
    for name, report in reports.items():
        _require(report.get("stage") == "S07", f"{name} stage is not S07")
        _require(report.get("status") == "passed", f"{name} verification failed")

    entry = reports["entry"]
    proxies = reports["proxies"]
    rerun = reports["rerun"]
    documents = {
        name: PROJECT_ROOT / relative for name, relative in REQUIRED_DOCUMENTS.items()
    }
    for name, path in documents.items():
        _require(path.is_file(), f"required S07 document is missing: {name}")
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in documents.items()
    }

    _require(
        entry.get("manifest_id") == rerun.get("source_final_run_manifest_id"),
        "entry and final Rerun use different final-run identities",
    )
    _require(
        proxies.get("source_final_run_manifest_id") == entry.get("manifest_id"),
        "video proxies use a different final-run identity",
    )
    _require(
        rerun.get("recording_sha256") == FINAL_RECORDING_SHA256,
        "gate does not reference the accepted refined recording",
    )

    criteria = [
        _criterion(
            "Another task can reproduce the final run from documented commands and inputs.",
            bool(
                entry.get("manifest_regenerated")
                and entry.get("source_video_frame_counts")
                == {"camera_a": 1047, "camera_b": 1047}
                and entry.get("artifact_count") == 7
                and proxies.get("video_count") == 2
                and rerun.get("recording_parsed")
                and ".venv/bin/python scripts/s07/run_measured_final_assembly.py"
                in texts["reproduction_guide"]
                and ".venv/bin/python scripts/s07/verify_measured_final_assembly.py"
                in texts["reproduction_guide"]
            ),
            "The hash-bound entry regenerates, both 1,047-frame sources and seven "
            "accepted S06 roles verify, both proxies verify, the RRD parses, and "
            "the reproduction guide records build, playback, visual-QA, and "
            "independent-verification commands.",
        ),
        _criterion(
            "The demo shows the intended object movement and semantic event sequence.",
            bool(
                rerun.get("visual_qa_passed")
                and rerun.get("required_entity_path_count") == 20
                and rerun.get("measured_observation_counts")
                == {"backpack": 17, "person": 16}
                and rerun.get("trajectory_logging_mode")
                == "capture_time_progressive"
                and rerun.get("full_trajectory_visible_at_start") is False
                and "D044 - The refined interactive Rerun is the final demonstration"
                in texts["decisions"]
                and "pickup, carry, and place" in texts["technical_report"]
            ),
            "Six visual-QA views passed; 33 measured observations and 23 "
            "capture-time-progressive segments show pickup-to-drop-off movement; "
            "the user accepted the interactive Rerun as the final demonstration.",
        ),
        _criterion(
            "Limitations and failures are presented honestly.",
            bool(
                entry.get("unavailable_xyz_must_remain_null")
                and entry.get("measured_trajectories_must_remain_disconnected")
                and entry.get("qwen_has_spatial_authority") is False
                and rerun.get("full_trajectory_visible_at_start") is False
                and "## Principal failures and retained limitations"
                in texts["technical_report"]
                and "6.803-second carry gap" in texts["technical_report"]
                and "no surveyed dynamic ground-truth trajectory"
                in texts["technical_report"]
            ),
            "The report retains detector fragmentation, the 6.803-second null-XYZ "
            "carry gap, mixed person anchors, rejected disagreement, Qwen failures, "
            "proxy scope, RTSP limits, and the absence of dynamic ground truth.",
        ),
        _criterion(
            "The report separates demonstrated offline throughput from projected "
            "live capacity and states production measurements needed.",
            bool(
                entry.get("demonstrated_live_capacity") is False
                and rerun.get("demonstrated_live_capacity") is False
                and rerun.get("evidence_kind")
                == "measured_retained_output_assembly"
                and "## Demonstrated offline capacity versus projected live capacity"
                in texts["technical_report"]
                and "## Required production measurements and changes"
                in texts["technical_report"]
                and "p95/p99 latency" in texts["technical_report"]
                and "representative camera count" in texts["technical_report"]
            ),
            "The report keeps isolated model, virtual replay, RTSP, and retained-"
            "assembly evidence separate from live projections and lists SLO, load, "
            "tail-latency, memory, supervision, network, calibration, security, and "
            "ground-truth work required for production.",
        ),
        _criterion(
            "STATUS, DECISIONS, and every stage handoff are current.",
            bool(
                "**Stage state:** Complete" in texts["status"]
                and "## D044" in texts["decisions"]
                and "**Status:** Complete with known limitations" in texts["handoff"]
                and "Completion gate passed without weakening" in texts["stage_record"]
                and all(
                    (PROJECT_ROOT / f"docs/stages/S{stage:02d}_HANDOFF.md").is_file()
                    for stage in range(8)
                )
            ),
            "The status marks S07 complete, D042-D044 record the final selection "
            "and presentation decisions, the S07 record and handoff contain the "
            "gate result, and S00-S07 handoffs are present.",
        ),
    ]
    failed = [item["criterion"] for item in criteria if not item["passed"]]
    _require(not failed, f"S07 completion gate failed: {failed}")

    audit = {
        "schema_version": 1,
        "stage": "S07",
        "status": "passed",
        "purpose": "stage07_completion_gate_audit",
        "completion_gate_passed": True,
        "completion_gate_weakened": False,
        "all_required_outputs_present": True,
        "separate_demo_video_required": False,
        "final_demonstration_kind": "interactive_rerun_recording",
        "final_recording_sha256": FINAL_RECORDING_SHA256,
        "source_final_run_manifest_id": entry["manifest_id"],
        "criteria": criteria,
        "source_verifications": {
            name: {"ref": _relative(path), "sha256": _sha256(path)}
            for name, path in verification_paths.items()
        },
        "required_documents": {
            name: {"ref": _relative(path), "sha256": _sha256(path)}
            for name, path in documents.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def _criterion(criterion: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": passed, "evidence": evidence}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
