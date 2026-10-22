"""Run metadata and the files each evaluation leaves behind, so a result can be reproduced and
compared with a later one."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from evals.schema import REPO_ROOT


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_metadata(settings, datasets: list[Path], **extra) -> dict:
    """Everything needed to know what produced a result: models, parameters, data and code."""
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "device": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("strands-agents", "strands-agents-evals", "strands-decider")
        },
        "bedrock_model_id": settings.aws_bedrock_model_id,
        "decider_model_id": settings.decider_model_id,
        "decider_confidence_threshold": settings.decider_confidence_threshold,
        "datasets": {p.name: _sha256(p) for p in datasets},
        **extra,
    }


def new_run_dir(root: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def num(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"
