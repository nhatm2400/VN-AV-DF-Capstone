"""Pure temporal policy for active-speaker curation.

The inference script writes one evidence row per 200 ms bin.  This module turns
those rows into a conservative clip decision and deliberately has no CV/torch
dependency, so the policy can be calibrated and unit-tested independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


DECISIONS = {"pass", "reject", "manual"}
REASONS = {"", "static", "voiceover", "ambiguous", "inference_failure"}


@dataclass(frozen=True)
class TemporalPolicy:
    bin_ms: int = 200
    min_contiguous_bad_ms: int = 800
    min_cumulative_bad_ms: int = 500
    min_bad_voiced_ratio: float = 0.20
    light_active_threshold: float = 0.0
    light_margin: float = 0.5
    laser_active_threshold: float = 0.5
    laser_margin: float = 0.15
    mouth_freeze_threshold: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def longest_true_run_ms(flags: Iterable[bool], bin_ms: int) -> int:
    longest = current = 0
    for flag in flags:
        if flag:
            current += bin_ms
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _laser_state(score, policy: TemporalPolicy) -> str:
    if score is None:
        return "missing"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "missing"
    low = policy.laser_active_threshold - policy.laser_margin
    high = policy.laser_active_threshold + policy.laser_margin
    if score <= low:
        return "inactive"
    if score >= high:
        return "active"
    return "ambiguous"


def classify_bin(row: Mapping, policy: TemporalPolicy) -> str:
    """Return active/static/voiceover/ambiguous/silent/failure for one bin."""
    if not _as_bool(row.get("speech", False)):
        return "silent"
    if _as_bool(row.get("inference_failure", False)):
        return "failure"

    visible = _as_bool(row.get("face_visible", False))
    frozen = _as_bool(row.get("mouth_frozen", False))
    if row.get("mouth_motion") not in (None, ""):
        try:
            frozen = float(row["mouth_motion"]) <= policy.mouth_freeze_threshold
        except (TypeError, ValueError):
            return "failure"

    light = row.get("light_asd_score")
    try:
        light = None if light in (None, "") else float(light)
    except (TypeError, ValueError):
        light = None
    laser = _laser_state(row.get("laser_score"), policy)

    # Selective LASER requests are a fail-closed boundary: the preliminary
    # Light-ASD decision may be stored for diagnostics, but it cannot become an
    # automatic bin label before the requested second opinion exists.
    if _as_bool(row.get("laser_requested", False)) and laser == "missing":
        return "ambiguous"
    if _as_bool(row.get("asd_disagreement", False)):
        return "ambiguous"

    if not visible:
        # No face is a high-confidence voice-over observation; a detector/model
        # failure must be flagged separately by the inference layer.
        return "voiceover"
    if light is None:
        return "failure"

    light_active = light >= policy.light_active_threshold + policy.light_margin
    light_inactive = light <= policy.light_active_threshold - policy.light_margin
    disagreement = _as_bool(row.get("multiple_competing_faces", False))

    if light_active and not frozen and not disagreement:
        return "active"

    # Static mouth plus confident non-speaking evidence is safe enough to reject.
    if frozen and light_inactive:
        return "static"
    if frozen and laser == "inactive":
        return "static"

    # Visible faces but none speaking: LASER must confirm when Light-ASD is not
    # confidently inactive. This prevents an uncertain model score becoming an
    # automatic rejection.
    if not frozen and light_inactive:
        return "voiceover"
    if not frozen and laser == "inactive":
        return "voiceover"

    # A positive LASER result can rescue a near-threshold/frozen Light-ASD bin,
    # but contradictory models always go to people.
    if laser == "active" and not disagreement:
        return "active"
    return "ambiguous"


def _is_material(total_ms: int, longest_ms: int, voiced_ms: int,
                 policy: TemporalPolicy) -> bool:
    return (
        longest_ms >= policy.min_contiguous_bad_ms
        or (
            total_ms >= policy.min_cumulative_bad_ms
            and voiced_ms > 0
            and total_ms / voiced_ms >= policy.min_bad_voiced_ratio
        )
    )


def summarize_timeline(rows: Iterable[Mapping], policy: TemporalPolicy) -> dict:
    rows = list(rows)
    classes = [classify_bin(row, policy) for row in rows]
    voiced = [name != "silent" for name in classes]
    static = [name == "static" for name in classes]
    voiceover = [name == "voiceover" for name in classes]
    ambiguous = [name == "ambiguous" for name in classes]
    failure = [name == "failure" for name in classes]
    active = [name == "active" for name in classes]

    voiced_ms = sum(voiced) * policy.bin_ms
    static_ms = sum(static) * policy.bin_ms
    voiceover_ms = sum(voiceover) * policy.bin_ms
    unexplained = [a or b for a, b in zip(static, voiceover)]
    unexplained_ms = sum(unexplained) * policy.bin_ms
    longest_static = longest_true_run_ms(static, policy.bin_ms)
    longest_voiceover = longest_true_run_ms(voiceover, policy.bin_ms)
    longest_unexplained = longest_true_run_ms(unexplained, policy.bin_ms)
    ambiguous_ms = sum(ambiguous) * policy.bin_ms

    if any(failure):
        decision, reason = "manual", "inference_failure"
    else:
        static_material = _is_material(static_ms, longest_static, voiced_ms, policy)
        voiceover_material = _is_material(
            voiceover_ms, longest_voiceover, voiced_ms, policy
        )
        ambiguous_material = _is_material(
            ambiguous_ms,
            longest_true_run_ms(ambiguous, policy.bin_ms),
            voiced_ms,
            policy,
        )
        # Material uncertainty wins over a candidate rejection. This prevents a
        # confident sub-segment from hiding an unresolved multi-face/model case
        # elsewhere in the same clip.
        if ambiguous_material:
            decision, reason = "manual", "ambiguous"
        elif static_material or voiceover_material:
            decision = "reject"
            reason = "static" if longest_static >= longest_voiceover else "voiceover"
        else:
            decision, reason = "pass", ""

    disagreements = sum(
        _as_bool(row.get("asd_disagreement", False)) for row in rows
        if _as_bool(row.get("speech", False))
    )
    return {
        "voiced_ms": voiced_ms,
        "visible_active_speech_ratio": (sum(active) * policy.bin_ms / voiced_ms)
        if voiced_ms else 0.0,
        "unexplained_speech_ratio": unexplained_ms / voiced_ms if voiced_ms else 0.0,
        "longest_unexplained_speech_ms": longest_unexplained,
        "static_speech_ratio": static_ms / voiced_ms if voiced_ms else 0.0,
        "asd_disagreement_ratio": disagreements / sum(voiced) if any(voiced) else 0.0,
        "temporal_decision": decision,
        "temporal_reason": reason,
        "static_ms": static_ms,
        "voiceover_ms": voiceover_ms,
        "ambiguous_ms": ambiguous_ms,
        "longest_static_ms": longest_static,
        "longest_voiceover_ms": longest_voiceover,
    }
