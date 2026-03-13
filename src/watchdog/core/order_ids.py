from __future__ import annotations


def normalize_platform_order_id(*, platform: str, provider_order_id: object, fallback_order_id: str) -> str:
    """Preserve venue metadata when an exchange returns a raw order id."""
    candidate = str(provider_order_id or "").strip()
    if not candidate:
        return fallback_order_id

    lowered = candidate.lower()
    if (
        lowered.startswith("paper-")
        or lowered.startswith("live-")
        or lowered.startswith("sim-")
        or lowered.startswith("arb-")
    ):
        return candidate

    fallback_lower = fallback_order_id.lower()
    for prefix in ("paper", "live", "sim"):
        if fallback_lower.startswith(f"{prefix}-"):
            return f"{prefix}-{platform}-{candidate}"
    return f"{platform}-{candidate}"
