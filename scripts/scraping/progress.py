"""Cross-process shared character counter, so parallel per-domain scrapers for the same
language can agree on a combined total and all stop once a shared budget is reached."""

import fcntl
import json
import os


def add_chars(progress_path: str, n: int) -> int:
    """Atomically add n to the shared character total; return the new total."""
    d = os.path.dirname(progress_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd = os.open(progress_path, os.O_RDWR | os.O_CREAT, 0o644)
    f = os.fdopen(fd, "r+", encoding="utf-8")  # takes ownership of fd
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            data = json.loads(raw) if raw else {"total_chars": 0}
            data["total_chars"] = data.get("total_chars", 0) + n
            f.seek(0)
            f.truncate()
            json.dump(data, f)
            f.flush()
            os.fsync(fd)
            return data["total_chars"]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        f.close()


def read_chars(progress_path: str) -> int:
    if not os.path.exists(progress_path):
        return 0
    with open(progress_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return 0
    try:
        return json.loads(raw).get("total_chars", 0)
    except json.JSONDecodeError:
        return 0
