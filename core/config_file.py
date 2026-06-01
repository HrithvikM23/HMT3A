from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest != "help"
    }


def load_config_defaults(parser: argparse.ArgumentParser, path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read config file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("config file must contain a JSON object")

    valid_dests = _destinations(parser)
    unknown_keys = sorted(str(key) for key in payload if key not in valid_dests)
    if unknown_keys:
        raise ValueError(f"unknown config option(s): {', '.join(unknown_keys)}")

    parser.set_defaults(**payload)
    return {str(key) for key in payload}


def config_preparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    return parser


def coerce_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return [coerce_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): coerce_config_value(item) for key, item in value.items()}
    return value
