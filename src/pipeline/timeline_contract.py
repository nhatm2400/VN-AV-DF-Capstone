"""Shared structured timeline/valid-range contract for repaired AV data."""

import hashlib
import json
import math


TIMELINE_SCHEMA_VERSION = "av_timeline_v1"
TIMELINE_MASK_POLICY = "fixed_common_window_v1"
BOUNDARY_NATIVE = "native"
BOUNDARY_CIRCULAR_WRAP = "circular_wrap"
ALLOWED_BOUNDARIES = {BOUNDARY_NATIVE, BOUNDARY_CIRCULAR_WRAP}
ALLOWED_SCOPES = {"none", "global", "local"}
TIMELINE_FIELDS = (
    "timeline_schema_version",
    "timeline_mask_policy",
    "timeline_boundary",
    "timeline_duration_s",
    "audio_valid_start_s",
    "audio_valid_end_s",
    "visual_valid_start_s",
    "visual_valid_end_s",
    "manipulation_scope",
    "manipulation_start_s",
    "manipulation_end_s",
)
_EPS = 1e-6


def _number(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Timeline field {field} không phải số") from exc
    if not math.isfinite(result):
        raise ValueError(f"Timeline field {field} không hữu hạn")
    return result


def _format(value):
    return f"{float(value):.9f}"


def build_timeline_contract(audio_duration_s, visual_duration_s, *,
                            boundary=BOUNDARY_NATIVE,
                            audio_valid=None, visual_valid=None,
                            manipulation_scope="none", manipulation=None):
    """Create canonical CSV fields; validation remains the final authority."""
    audio_duration = _number(audio_duration_s, "audio_duration_s")
    visual_duration = _number(visual_duration_s, "visual_duration_s")
    if audio_duration <= 0 or visual_duration <= 0:
        raise ValueError("Timeline duration audio/video phải > 0")
    audio_valid = audio_valid or (0.0, audio_duration)
    visual_valid = visual_valid or (0.0, visual_duration)
    common_start = max(float(audio_valid[0]), float(visual_valid[0]))
    common_end = min(float(audio_valid[1]), float(visual_valid[1]))
    if manipulation_scope == "none":
        manipulation_start = manipulation_end = ""
    else:
        manipulation = manipulation or (common_start, common_end)
        manipulation_start = _format(manipulation[0])
        manipulation_end = _format(manipulation[1])
    row = {
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline_mask_policy": TIMELINE_MASK_POLICY,
        "timeline_boundary": boundary,
        "timeline_duration_s": _format(max(audio_duration, visual_duration)),
        "audio_valid_start_s": _format(audio_valid[0]),
        "audio_valid_end_s": _format(audio_valid[1]),
        "visual_valid_start_s": _format(visual_valid[0]),
        "visual_valid_end_s": _format(visual_valid[1]),
        "manipulation_scope": manipulation_scope,
        "manipulation_start_s": manipulation_start,
        "manipulation_end_s": manipulation_end,
    }
    return validate_timeline_contract(row)


def validate_timeline_contract(row, method=None):
    """Validate and return a canonical representation suitable for CSV/meta."""
    missing = [field for field in TIMELINE_FIELDS
               if field not in row or (field not in ("manipulation_start_s",
                                                      "manipulation_end_s")
                                       and not str(row.get(field, "")).strip())]
    if missing:
        raise ValueError(f"Thiếu timeline contract fields: {missing}")
    if str(row["timeline_schema_version"]).strip() != TIMELINE_SCHEMA_VERSION:
        raise ValueError("timeline_schema_version không được hỗ trợ")
    if str(row["timeline_mask_policy"]).strip() != TIMELINE_MASK_POLICY:
        raise ValueError("timeline_mask_policy không được hỗ trợ")
    boundary = str(row["timeline_boundary"]).strip()
    scope = str(row["manipulation_scope"]).strip()
    if boundary not in ALLOWED_BOUNDARIES:
        raise ValueError(f"timeline_boundary invalid: {boundary}")
    if scope not in ALLOWED_SCOPES:
        raise ValueError(f"manipulation_scope invalid: {scope}")

    duration = _number(row["timeline_duration_s"], "timeline_duration_s")
    audio_start = _number(row["audio_valid_start_s"], "audio_valid_start_s")
    audio_end = _number(row["audio_valid_end_s"], "audio_valid_end_s")
    visual_start = _number(row["visual_valid_start_s"], "visual_valid_start_s")
    visual_end = _number(row["visual_valid_end_s"], "visual_valid_end_s")
    if duration <= 0:
        raise ValueError("timeline_duration_s phải > 0")
    for name, start, end in (
        ("audio", audio_start, audio_end),
        ("visual", visual_start, visual_end),
    ):
        if start < -_EPS or end <= start + _EPS or end > duration + 1e-3:
            raise ValueError(f"Miền {name}_valid không hợp lệ: [{start}, {end}]")
    common_start = max(audio_start, visual_start)
    common_end = min(audio_end, visual_end)
    if common_end <= common_start + _EPS:
        raise ValueError("Audio/visual không còn miền thời gian hợp lệ chung")

    raw_manip_start = str(row.get("manipulation_start_s", "") or "").strip()
    raw_manip_end = str(row.get("manipulation_end_s", "") or "").strip()
    if scope == "none":
        if raw_manip_start or raw_manip_end:
            raise ValueError("scope=none không được có manipulation interval")
        manip_start = manip_end = None
    else:
        if not raw_manip_start or not raw_manip_end:
            raise ValueError("Fake phải có manipulation interval")
        manip_start = _number(raw_manip_start, "manipulation_start_s")
        manip_end = _number(raw_manip_end, "manipulation_end_s")
        if (manip_start < common_start - 1e-3
                or manip_end > common_end + 1e-3
                or manip_end <= manip_start + _EPS):
            raise ValueError("Manipulation interval phải nằm trong common valid range")
        if scope == "global" and (abs(manip_start - common_start) > 1e-3
                                  or abs(manip_end - common_end) > 1e-3):
            raise ValueError("scope=global phải phủ toàn common valid range")

    method = str(method or row.get("method", "") or "").strip()
    expected = {
        "real": (BOUNDARY_NATIVE, "none"),
        "temporal_desync": (BOUNDARY_CIRCULAR_WRAP, "global"),
        "frame_reverse": (BOUNDARY_NATIVE, "local"),
        "pitch_flatten": (BOUNDARY_NATIVE, "global"),
        "anonymization": (BOUNDARY_NATIVE, "global"),
    }.get(method)
    if expected and (boundary, scope) != expected:
        raise ValueError(
            f"Timeline semantics sai cho {method}: {(boundary, scope)} != {expected}"
        )

    canonical = {
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline_mask_policy": TIMELINE_MASK_POLICY,
        "timeline_boundary": boundary,
        "timeline_duration_s": _format(duration),
        "audio_valid_start_s": _format(audio_start),
        "audio_valid_end_s": _format(audio_end),
        "visual_valid_start_s": _format(visual_start),
        "visual_valid_end_s": _format(visual_end),
        "manipulation_scope": scope,
        "manipulation_start_s": "" if manip_start is None else _format(manip_start),
        "manipulation_end_s": "" if manip_end is None else _format(manip_end),
    }
    return canonical


def common_valid_interval(row, method=None):
    contract = validate_timeline_contract(row, method)
    return (max(float(contract["audio_valid_start_s"]),
                float(contract["visual_valid_start_s"])),
            min(float(contract["audio_valid_end_s"]),
                float(contract["visual_valid_end_s"])))


def fixed_common_window(row, window_s, position=0.5, method=None):
    """Return a fixed-width synchronized window without exposing mask length."""
    width = _number(window_s, "window_s")
    fraction = _number(position, "position")
    if width <= 0 or not 0 <= fraction <= 1:
        raise ValueError("window_s phải > 0 và position nằm trong [0,1]")
    start, end = common_valid_interval(row, method)
    available = end - start
    if available + _EPS < width:
        raise ValueError(
            f"Common valid range {available:.6f}s ngắn hơn window {width:.6f}s"
        )
    window_start = start + max(available - width, 0.0) * fraction
    return window_start, window_start + width


def validate_timeline_against_media(row, audio_duration_s, visual_duration_s,
                                    method=None, tolerance_s=0.05):
    """Bind declared ranges to probed media durations before downstream use."""
    contract = validate_timeline_contract(row, method)
    audio_duration = _number(audio_duration_s, "audio_duration_s")
    visual_duration = _number(visual_duration_s, "visual_duration_s")
    tolerance = _number(tolerance_s, "tolerance_s")
    if audio_duration <= 0 or visual_duration <= 0 or tolerance < 0:
        raise ValueError("Media duration/tolerance không hợp lệ")
    declared_duration = float(contract["timeline_duration_s"])
    if abs(declared_duration - max(audio_duration, visual_duration)) > tolerance:
        raise ValueError("timeline_duration_s không khớp media")
    audio_start = float(contract["audio_valid_start_s"])
    audio_end = float(contract["audio_valid_end_s"])
    visual_start = float(contract["visual_valid_start_s"])
    visual_end = float(contract["visual_valid_end_s"])
    if audio_end > audio_duration + tolerance or visual_end > visual_duration + tolerance:
        raise ValueError("Valid range vượt duration media")
    if contract["timeline_boundary"] == BOUNDARY_NATIVE:
        if (abs(audio_start) > _EPS or abs(visual_start) > _EPS
                or abs(audio_end - audio_duration) > tolerance
                or abs(visual_end - visual_duration) > tolerance):
            raise ValueError("boundary=native phải dùng toàn bộ timeline từng modality")
    return contract


def timeline_contract_id(row, method=None):
    canonical = validate_timeline_contract(row, method)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
