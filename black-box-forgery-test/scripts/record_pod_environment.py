#!/usr/bin/env python3
"""Write non-secret pod/runtime metadata for a run manifest."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    output = Path(os.environ.get("BBF_POD_VALIDATION_DIR", "runs/pod-validation"))
    output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("black-box-forgery", "pydantic", "PyYAML", "uv", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "container_image_digest": os.environ.get("BBF_CONTAINER_IMAGE_DIGEST"),
        "expected_container_image_digest": os.environ.get("BBF_EXPECTED_IMAGE_DIGEST"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "cuda_visible_devices_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "credential_names_present": sorted(
            name
            for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OPENROUTER_API_KEY")
            if os.environ.get(name)
        ),
    }
    destination = output / "environment.json"
    temporary = destination.with_name("." + destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
