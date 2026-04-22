"""
Canonical RL transition contract and validation helpers.

The current walk-forward loop still records legacy outcome-only steps. Those
steps are useful for audit, but they are not a valid control dataset for PPO.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any

STATE_BLOCK_KEYS = ("macro_state", "sector_state", "portfolio_state")
TRANSITION_KEYS = ("state", "action", "reward", "next_state", "done")
TRANSITION_SCHEMA_VERSION = 1
CAUSAL_TRAINING_BACKEND = "causal_historical_env_v1"


def build_state(
    macro_state: dict[str, Any],
    sector_state: dict[str, Any],
    portfolio_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "macro_state": deepcopy(macro_state),
        "sector_state": deepcopy(sector_state),
        "portfolio_state": deepcopy(portfolio_state),
    }


def build_transition(
    state: dict[str, Any],
    action: dict[str, Any],
    reward: float,
    next_state: dict[str, Any],
    done: bool,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "state": deepcopy(state),
        "action": deepcopy(action),
        "reward": float(reward),
        "next_state": deepcopy(next_state),
        "done": bool(done),
        "info": deepcopy(info) if info is not None else {},
    }


def is_transition_step(step: Any) -> bool:
    return isinstance(step, dict) and all(key in step for key in TRANSITION_KEYS)


def is_legacy_outcome_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if is_transition_step(step):
        return False
    return all(key in step for key in (*STATE_BLOCK_KEYS, "outcome"))


def transition_errors(step: Any) -> list[str]:
    if not isinstance(step, dict):
        return ["step must be a dict"]

    errors: list[str] = []
    for key in TRANSITION_KEYS:
        if key not in step:
            errors.append(f"missing key: {key}")

    if errors:
        return errors

    for state_key in ("state", "next_state"):
        state = step.get(state_key)
        if not isinstance(state, dict):
            errors.append(f"{state_key} must be a dict")
            continue
        for block_key in STATE_BLOCK_KEYS:
            if block_key not in state:
                errors.append(f"{state_key}.{block_key} missing")
            elif not isinstance(state[block_key], dict):
                errors.append(f"{state_key}.{block_key} must be a dict")

    action = step.get("action")
    if not isinstance(action, dict):
        errors.append("action must be a dict")

    reward = step.get("reward")
    if not isinstance(reward, Real):
        errors.append("reward must be numeric")

    done = step.get("done")
    if not isinstance(done, bool):
        errors.append("done must be a bool")

    info = step.get("info", {})
    if info is not None and not isinstance(info, dict):
        errors.append("info must be a dict when present")

    return errors


def canonicalize_transition(step: dict[str, Any]) -> dict[str, Any]:
    errors = transition_errors(step)
    if errors:
        raise ValueError("; ".join(errors))

    canonical = deepcopy(step)
    canonical["schema_version"] = int(
        canonical.get("schema_version", TRANSITION_SCHEMA_VERSION)
    )
    canonical["reward"] = float(canonical["reward"])
    canonical["done"] = bool(canonical["done"])
    canonical["info"] = deepcopy(canonical.get("info", {}))
    return canonical


def summarize_buffer(buffer: list[Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "size": len(buffer),
        "valid_transition_steps": 0,
        "legacy_outcome_steps": 0,
        "invalid_steps": 0,
        "supports_transition_replay": False,
        "supports_causal_training": False,
        "disable_reason": None,
        "schema_version": TRANSITION_SCHEMA_VERSION,
    }

    if not buffer:
        summary["disable_reason"] = "RL buffer is empty."
        return summary

    first_invalid: str | None = None
    for idx, step in enumerate(buffer):
        if is_transition_step(step):
            errors = transition_errors(step)
            if errors:
                summary["invalid_steps"] += 1
                if first_invalid is None:
                    first_invalid = f"step {idx}: {'; '.join(errors)}"
            else:
                summary["valid_transition_steps"] += 1
        elif is_legacy_outcome_step(step):
            summary["legacy_outcome_steps"] += 1
        else:
            summary["invalid_steps"] += 1
            if first_invalid is None:
                first_invalid = f"step {idx}: unsupported structure"

    if summary["valid_transition_steps"] == len(buffer):
        summary["supports_transition_replay"] = True
        summary["disable_reason"] = (
            "Canonical RL transitions are present, but PPO replay training is "
            "disabled until a simulator-backed causal environment executes "
            "candidate actions through the portfolio pipeline."
        )
        return summary

    if summary["legacy_outcome_steps"]:
        summary["disable_reason"] = (
            "RL buffer contains legacy outcome-only steps without "
            "(state, action, reward, next_state, done) transitions."
        )
        return summary

    summary["disable_reason"] = (
        first_invalid or "RL buffer contains unsupported transition records."
    )
    return summary
