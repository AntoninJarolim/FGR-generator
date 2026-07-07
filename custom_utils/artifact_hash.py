"""Stable hash for artifact dicts (e.g. change detection). No wandb dependency."""

import hashlib
import json


def artifact_hash(artifacts: dict) -> str:
    """Stable hash of artifacts dict for change detection."""
    return hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
