"""Configuration loading with deterministic, explicit defaults."""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class ConfigError(ValueError):
    pass


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping from disk; no URLs or implicit remote includes."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency in pyproject
        raise ConfigError("PyYAML is required to load configuration") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return dict(data)


def load_config(path: Path, *, defaults: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load a config and recursively merge mapping defaults."""
    config = _deep_merge(dict(defaults or {}), load_yaml_config(path))
    config["_config_path"] = str(Path(path))
    return config


def load_model_configs(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_yaml_config(path)
    models = data.get("models")
    if not isinstance(models, list):
        raise ConfigError("models.yaml must contain a models list")
    result: Dict[str, Dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, Mapping) or not entry.get("id") or not entry.get("revision"):
            raise ConfigError("each model requires id and revision")
        model = dict(entry)
        result[str(model["id"])] = model
    return result


def load_dotenv_names(path: Path) -> Dict[str, str]:
    """Read a local dotenv file for opt-in API use.

    This helper is intentionally not called by the default CLI, and its return
    value is never included in a run manifest.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not name or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
            raise ConfigError(f"invalid environment variable name {name!r}")
        result[name] = value.strip().strip("\"'")
    return result


def environment_value(name: str, *, dotenv_path: Optional[Path] = None) -> Optional[str]:
    """Return an opt-in credential, preferring the process environment."""
    value = os.environ.get(name)
    if value:
        return value
    if dotenv_path:
        return load_dotenv_names(dotenv_path).get(name)
    return None


def _deep_merge(left: Dict[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(left.get(key), Mapping):
            left[key] = _deep_merge(dict(left[key]), value)
        else:
            left[key] = value
    return left

