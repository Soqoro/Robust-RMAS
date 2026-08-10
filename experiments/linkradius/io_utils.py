"""Canonical hashing, atomic publication, and completion validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import (
    COMPLETION_VERSION,
    ArtifactExpectation,
    ContractError,
    validate_completion_record,
    validate_gate,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_hash(value: Any, *, domain: str = "linkradius:json:v1") -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Iterable[Any], *, domain: str) -> str:
    return content_hash(list(values), domain=domain)


def _atomic_replace_bytes(path: Path, payload: bytes, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_text(
    path: str | os.PathLike[str], text: str, *, overwrite: bool = True
) -> Path:
    target = Path(path)
    _atomic_replace_bytes(target, text.encode("utf-8"), overwrite=overwrite)
    return target


def atomic_write_json(
    path: str | os.PathLike[str], value: Any, *, overwrite: bool = True
) -> Path:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(path, payload, overwrite=overwrite)


def atomic_write_jsonl(
    path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]], *, overwrite: bool = True
) -> Path:
    lines = [canonical_json_bytes(dict(row)).decode("utf-8") for row in rows]
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""), overwrite=overwrite)


def atomic_write_csv(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    overwrite: bool = True,
) -> Path:
    row_list = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(row_list[0]) if row_list else []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
    if fieldnames:
        writer.writeheader()
        writer.writerows(row_list)
    return atomic_write_text(path, buffer.getvalue(), overwrite=overwrite)


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def json_content_hash(path: str | os.PathLike[str], *, domain: str = "linkradius:file_json:v1") -> str:
    return content_hash(load_json(path), domain=domain)


def source_hash(repo_root: str | os.PathLike[str]) -> str:
    """Hash all result-affecting LinkRadius and RecursiveMAS source files."""

    root = Path(repo_root).resolve()
    candidates: list[Path] = []
    for relative in ("RecursiveMAS", "experiments/linkradius"):
        base = root / relative
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".sh", ".json", ".txt", ".md"}:
                continue
            candidates.append(path)
    digest = hashlib.sha256(b"linkradius:source_tree:v1\0")
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def count_jsonl_rows(path: str | os.PathLike[str]) -> int:
    return len(load_jsonl(path))


def artifact_metadata(path: str | os.PathLike[str], *, row_count: int | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ContractError(f"artifact does not exist: {target}")
    result: dict[str, Any] = {
        "path": target.name,
        "sha256": file_sha256(target),
        "size_bytes": target.stat().st_size,
    }
    if row_count is not None:
        result["row_count"] = int(row_count)
    return result


def publish_completion(
    output_dir: str | os.PathLike[str],
    *,
    config_hash: str,
    source_hash_value: str,
    artifact_paths: Sequence[str | os.PathLike[str]],
    row_counts: Mapping[str, int] | None = None,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate final artifacts, then atomically publish ``.complete.json``."""

    directory = Path(output_dir)
    artifacts = []
    expectations = []
    for raw_path in artifact_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = directory / path
        if path.parent.resolve() != directory.resolve():
            raise ContractError("completion artifacts must be direct children of the output directory")
        row_count = (row_counts or {}).get(path.name)
        metadata = artifact_metadata(path, row_count=row_count)
        artifacts.append(metadata)
        expectations.append(
            ArtifactExpectation(path=path.name, sha256=metadata["sha256"], row_count=row_count)
        )
    record: dict[str, Any] = {
        "schema_version": COMPLETION_VERSION,
        "status": "complete",
        "config_hash": config_hash,
        "source_hash": source_hash_value,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    if extra:
        for key, value in extra.items():
            if key in record:
                raise ContractError(f"completion extra field would overwrite reserved field: {key}")
            record[key] = value
    validate_completion_record(
        record, expected_config_hash=config_hash, expected_artifacts=expectations
    )
    path = directory / ".complete.json"
    atomic_write_json(path, record, overwrite=overwrite)
    return path


def verify_completion(
    output_dir: str | os.PathLike[str], *, expected_config_hash: str | None = None
) -> dict[str, Any]:
    directory = Path(output_dir)
    record = load_json(directory / ".complete.json")
    validate_completion_record(record, expected_config_hash=expected_config_hash)
    for artifact in record["artifacts"]:
        path = directory / artifact["path"]
        if not path.is_file():
            raise ContractError(f"completed artifact is missing: {path}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ContractError(f"completed artifact size changed: {path}")
        if file_sha256(path) != artifact["sha256"]:
            raise ContractError(f"completed artifact hash changed: {path}")
        if "row_count" in artifact and count_jsonl_rows(path) != int(artifact["row_count"]):
            raise ContractError(f"completed artifact row count changed: {path}")
    return record


def require_passed_gate(
    path: str | os.PathLike[str],
    *,
    gate_type: str | None = None,
    required_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    gate = load_json(path)
    validate_gate(gate, gate_type=gate_type, required_hashes=required_hashes)
    return gate


def compatible_complete(
    output_dir: str | os.PathLike[str], *, expected_config_hash: str,
    expected_source_hash: str | None = None,
) -> bool:
    try:
        record = verify_completion(output_dir, expected_config_hash=expected_config_hash)
        if expected_source_hash is not None and record.get("source_hash") != expected_source_hash:
            return False
    except (OSError, ValueError, TypeError, ContractError):
        return False
    return True
