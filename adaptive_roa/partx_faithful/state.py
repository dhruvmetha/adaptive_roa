"""Atomic pickle of the Part-X state (spec §6.6)."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_state(px, path, chash: str) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"config_hash": chash, "px": px}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_state(path, chash: str):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if blob["config_hash"] != chash:
        raise RuntimeError(f"{path}: state was saved under config {blob['config_hash']}, "
                           f"this run is {chash}; refusing to resume")
    return blob["px"]
