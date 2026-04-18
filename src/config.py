"""Central config loader — reads YAML, validates, exposes a singleton."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent  # repo root


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _ROOT / "config" / "base.yaml"
    with open(path) as f:
        cfg: dict = yaml.safe_load(f)

    # resolve relative paths relative to repo root
    paths = cfg.get("paths", {})
    for key, val in paths.items():
        paths[key] = str(_ROOT / val)
    cfg["paths"] = paths

    # resolve universe config path
    uni_path = cfg.get("universe", {}).get("config_file", "config/universe.yaml")
    cfg["universe"]["config_file"] = str(_ROOT / uni_path)

    # ensure directories exist
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    logger.info("Config loaded from %s", path)
    return cfg


@lru_cache(maxsize=1)
def load_universe_config(config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    uni_path = cfg["universe"]["config_file"]
    with open(uni_path) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s"),
    )
