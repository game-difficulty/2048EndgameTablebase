from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "docs_and_configs" / "remote_worker.local.json"
TABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*_[1-9][0-9]*$")
WORKER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TABLE_FILE_SUFFIXES = (
    ".book",
    ".z",
    ".zbook",
    ".exzbook",
    ".exadbook",
    ".exadzbook",
    ".bccmp",
    ".bcpos",
    ".bcsuc",
)
SHARD_FILE_SUFFIXES = (".b", ".i", ".zi")
MAX_SHARD_DIRECTORIES_TO_SAMPLE = 3


class WorkerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LogConfig:
    path: Path
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3
    level: str = "INFO"


@dataclass(frozen=True)
class ResourceGroupConfig:
    name: str
    concurrency: int


@dataclass(frozen=True)
class TableConfig:
    table_id: str
    pattern: str
    target: str
    path: Path
    dtype: str
    spawn_rate: float
    resource_group: str
    concurrency: int

    @property
    def path_list(self) -> list[tuple[str, str]]:
        return [(str(self.path), self.dtype)]


@dataclass(frozen=True)
class WorkerConfig:
    server_url: str
    worker_id: str
    auth_token_env: str
    auth_token: str
    heartbeat_seconds: float
    hello_timeout_seconds: float
    connect_timeout_seconds: float
    max_batch_size: int
    reconnect_initial_seconds: float
    reconnect_max_seconds: float
    log: LogConfig
    resource_groups: dict[str, ResourceGroupConfig]
    tables: dict[str, TableConfig]


_ROOT_KEYS = {
    "server_url",
    "worker_id",
    "auth_token_env",
    "heartbeat_seconds",
    "hello_timeout_seconds",
    "connect_timeout_seconds",
    "max_batch_size",
    "reconnect_initial_seconds",
    "reconnect_max_seconds",
    "log",
    "resource_groups",
    "tables",
}
_LOG_KEYS = {"path", "max_bytes", "backup_count", "level"}
_RESOURCE_GROUP_KEYS = {"concurrency"}
_TABLE_KEYS = {
    "table_id",
    "pattern",
    "target",
    "path",
    "dtype",
    "spawn_rate",
    "resource_group",
    "concurrency",
}


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerConfigError(f"{label} must be an object")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise WorkerConfigError(f"Unknown {label} field(s): {', '.join(unknown)}")


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise WorkerConfigError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerConfigError(f"{label} must be a positive integer") from exc
    if parsed <= 0 or (maximum is not None and parsed > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise WorkerConfigError(f"{label} must be positive{suffix}")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerConfigError(f"{label} must be positive") from exc
    if parsed <= 0:
        raise WorkerConfigError(f"{label} must be positive")
    return parsed


def _resolve_local_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigError(f"{label} must be a local path")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _parse_table_id(value: Any) -> tuple[str, str, str]:
    table_id = str(value or "").strip()
    if not TABLE_ID_RE.fullmatch(table_id):
        raise WorkerConfigError(f"Invalid table_id: {table_id!r}")
    pattern, target = table_id.rsplit("_", 1)
    return table_id, pattern, target


def load_worker_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
    require_auth: bool = True,
) -> WorkerConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkerConfigError(
            f"Worker config does not exist: {config_path}. Copy the example first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise WorkerConfigError(f"Invalid JSON in worker config: {exc}") from exc

    data = _require_object(raw, "worker config")
    _reject_unknown(data, _ROOT_KEYS, "worker config")

    server_url = str(data.get("server_url") or "").strip()
    parsed_url = urlparse(server_url)
    if parsed_url.scheme != "wss" or not parsed_url.netloc:
        raise WorkerConfigError("server_url must be a valid wss:// URL")

    worker_id = str(data.get("worker_id") or "").strip()
    if not WORKER_ID_RE.fullmatch(worker_id):
        raise WorkerConfigError("worker_id contains unsupported characters")

    auth_token_env = str(data.get("auth_token_env") or "TABLEBASE_WORKER_TOKEN").strip()
    if not auth_token_env:
        raise WorkerConfigError("auth_token_env is required")
    env = os.environ if environ is None else environ
    auth_token = str(env.get(auth_token_env) or "").strip()
    if require_auth and not auth_token:
        raise WorkerConfigError(f"Environment variable {auth_token_env} is not set")

    heartbeat_seconds = _positive_float(
        data.get("heartbeat_seconds", 10), "heartbeat_seconds"
    )
    if heartbeat_seconds != 10:
        raise WorkerConfigError("heartbeat_seconds must be 10")

    log_raw = _require_object(data.get("log", {}), "log")
    _reject_unknown(log_raw, _LOG_KEYS, "log")
    log_path = _resolve_local_path(
        log_raw.get("path", "tools/tablebase_worker/logs/worker.log"), "log.path"
    )
    log_level = str(log_raw.get("level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise WorkerConfigError("log.level is invalid")
    log_config = LogConfig(
        path=log_path,
        max_bytes=_positive_int(log_raw.get("max_bytes", 5 * 1024 * 1024), "log.max_bytes"),
        backup_count=_positive_int(log_raw.get("backup_count", 3), "log.backup_count"),
        level=log_level,
    )

    resource_raw = _require_object(data.get("resource_groups"), "resource_groups")
    if not resource_raw:
        raise WorkerConfigError("At least one resource group is required")
    resource_groups: dict[str, ResourceGroupConfig] = {}
    for name, raw_group in resource_raw.items():
        if not WORKER_ID_RE.fullmatch(str(name)):
            raise WorkerConfigError(f"Invalid resource group name: {name!r}")
        group = _require_object(raw_group, f"resource_groups.{name}")
        _reject_unknown(group, _RESOURCE_GROUP_KEYS, f"resource_groups.{name}")
        resource_groups[str(name)] = ResourceGroupConfig(
            name=str(name),
            concurrency=_positive_int(
                group.get("concurrency", 1),
                f"resource_groups.{name}.concurrency",
                maximum=32,
            ),
        )

    tables_raw = data.get("tables")
    if not isinstance(tables_raw, list) or not tables_raw:
        raise WorkerConfigError("tables must be a non-empty array")
    tables: dict[str, TableConfig] = {}
    for index, raw_table in enumerate(tables_raw):
        table = _require_object(raw_table, f"tables[{index}]")
        _reject_unknown(table, _TABLE_KEYS, f"tables[{index}]")
        table_id, id_pattern, id_target = _parse_table_id(table.get("table_id"))
        if table_id in tables:
            raise WorkerConfigError(f"Duplicate table_id: {table_id}")
        pattern = str(table.get("pattern") or id_pattern).strip()
        target = str(table.get("target") or id_target).strip()
        if pattern != id_pattern or target != id_target:
            raise WorkerConfigError(f"table_id does not match pattern/target: {table_id}")
        resource_group = str(table.get("resource_group") or "").strip()
        if resource_group not in resource_groups:
            raise WorkerConfigError(
                f"Unknown resource_group {resource_group!r} for {table_id}"
            )
        dtype = str(table.get("dtype") or "uint32").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", dtype):
            raise WorkerConfigError(f"Invalid dtype for {table_id}")
        spawn_rate = float(table.get("spawn_rate", 0.1))
        if not 0 <= spawn_rate <= 1:
            raise WorkerConfigError(f"spawn_rate must be between 0 and 1 for {table_id}")
        tables[table_id] = TableConfig(
            table_id=table_id,
            pattern=pattern,
            target=target,
            path=_resolve_local_path(table.get("path"), f"tables[{index}].path"),
            dtype=dtype,
            spawn_rate=spawn_rate,
            resource_group=resource_group,
            concurrency=_positive_int(
                table.get("concurrency", 1),
                f"tables[{index}].concurrency",
                maximum=32,
            ),
        )

    reconnect_initial = _positive_float(
        data.get("reconnect_initial_seconds", 1), "reconnect_initial_seconds"
    )
    reconnect_max = _positive_float(
        data.get("reconnect_max_seconds", 60), "reconnect_max_seconds"
    )
    if reconnect_max < reconnect_initial:
        raise WorkerConfigError(
            "reconnect_max_seconds must be at least reconnect_initial_seconds"
        )

    return WorkerConfig(
        server_url=server_url,
        worker_id=worker_id,
        auth_token_env=auth_token_env,
        auth_token=auth_token,
        heartbeat_seconds=heartbeat_seconds,
        hello_timeout_seconds=_positive_float(
            data.get("hello_timeout_seconds", 15), "hello_timeout_seconds"
        ),
        connect_timeout_seconds=_positive_float(
            data.get("connect_timeout_seconds", 15), "connect_timeout_seconds"
        ),
        max_batch_size=_positive_int(
            data.get("max_batch_size", 1024), "max_batch_size", maximum=1024
        ),
        reconnect_initial_seconds=reconnect_initial,
        reconnect_max_seconds=reconnect_max,
        log=log_config,
        resource_groups=resource_groups,
        tables=tables,
    )


def table_path_status(table: TableConfig) -> tuple[bool, str | None]:
    path = table.path
    if not path.exists():
        return False, "TABLE_PATH_MISSING"
    if not path.is_dir():
        return False, "TABLE_PATH_NOT_DIRECTORY"
    prefix = f"{table.table_id}_"
    sampled_shards = 0
    try:
        for item in path.iterdir():
            name = item.name
            if not name.startswith(prefix):
                continue
            if item.is_file() and name.endswith(TABLE_FILE_SUFFIXES):
                return True, None
            if (
                item.is_dir()
                and name.endswith("b")
                and sampled_shards < MAX_SHARD_DIRECTORIES_TO_SAMPLE
            ):
                sampled_shards += 1
                try:
                    with os.scandir(item) as entries:
                        if any(
                            entry.is_file() and entry.name.endswith(SHARD_FILE_SUFFIXES)
                            for entry in entries
                        ):
                            return True, None
                except OSError:
                    continue
    except OSError:
        return False, "TABLE_PATH_UNREADABLE"
    return False, "TABLE_FILES_MISSING"
