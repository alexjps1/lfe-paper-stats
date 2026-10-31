"""Create the compact, row-preserving input snapshot used by analyses.

The module deliberately applies no quality, route, conversation, or ADAS-state
filters. Those are analysis decisions and must retain access to the original
timeline. Only columns are selected here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
import unicodedata
from typing import Any
from uuid import uuid4

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "default.toml"
CONVERSATION_COLUMNS = {
    "participant_id": pl.String,
    "conversation_id": pl.Int64,
    "trigger": pl.String,
    "started_at": pl.String,
    "ended_at": pl.String,
    "start_ms": pl.Float64,
    "end_ms": pl.Float64,
    "message_count": pl.Int64,
}


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def config_path(config_name: str) -> Path:
    candidate = Path(config_name)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / "configs" / candidate


def load_config(config_name: str = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any]]:
    path = config_path(config_name)
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {path}")
    with path.open("rb") as handle:
        return path, tomllib.load(handle)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _preparation_payload(config: dict[str, Any]) -> dict[str, Any]:
    """The configuration subset that changes prepared-data contents."""
    paths = config.get("paths", {})
    return {
        "paths": {name: paths.get(name) for name in ("silab", "webapp", "preprocessed")},
        "preprocessing": config.get("preprocessing", {}),
        "id_aliases": config.get("preprocessing", {}).get("id_aliases")
        or config.get("filter", {}).get("id_aliases", {}),
    }


def _preparation_signature(config: dict[str, Any]) -> str:
    encoded = json.dumps(_preparation_payload(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def participant_id(path: Path, aliases: dict[str, str]) -> str:
    stem = re.sub(r"^Fahrt\d+_", "", path.stem)
    stem = re.sub(r"\.\d+$", "", stem).upper()
    return aliases.get(stem, stem)


def raw_recordings(silab_dir: Path, aliases: dict[str, str]) -> list[tuple[Path, str, str]]:
    """Return one raw recording per group/participant, preferring ASC over CSV."""
    recordings: dict[tuple[str, str], Path] = {}
    for group_dir in sorted((path for path in silab_dir.iterdir() if path.is_dir()), key=lambda path: nfc(path.name)):
        group = nfc(group_dir.name)
        for path in sorted(group_dir.iterdir()):
            if path.suffix.lower() not in {".asc", ".csv"}:
                continue
            key = (group, participant_id(path, aliases))
            previous = recordings.get(key)
            if previous is None or (path.suffix.lower() == ".asc" and previous.suffix.lower() != ".asc"):
                recordings[key] = path
    return [(path, group, pid) for (group, pid), path in sorted(recordings.items())]


def _output_file(raw_file: Path, silab_dir: Path, target_silab: Path) -> Path:
    return (target_silab / raw_file.relative_to(silab_dir)).with_suffix(".parquet")


def _select_recording(raw_file: Path, columns: list[str], string_columns: set[str], group: str, pid: str) -> pl.DataFrame:
    scan = pl.scan_csv(raw_file, encoding="utf8-lossy", infer_schema_length=10_000)
    available = set(scan.collect_schema().names())
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{raw_file}: fehlende Spalte(n): {', '.join(missing)}")

    expressions = []
    for column in columns:
        dtype = pl.String if column in string_columns else pl.Float64
        expressions.append(pl.col(column).cast(dtype, strict=False).alias(column))

    # No filter is applied: collect exactly one output row per input row.
    frame = scan.select(expressions).collect()
    return frame.with_row_index("sample_index").with_columns(
        pl.lit(pid).alias("participant_id"),
        pl.lit(group).alias("source_group"),
        pl.lit(raw_file.name).alias("source_file"),
    )


def _iso_to_ms(value: str, simulation_started_at: str | None) -> float | None:
    if not value or not simulation_started_at:
        return None
    def parse(stamp: str) -> datetime:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (parse(value) - parse(simulation_started_at)).total_seconds() * 1000


def conversation_table(webapp_dir: Path, aliases: dict[str, str]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(webapp_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            session = json.load(handle)
        pid = aliases.get(str(session.get("participantId") or path.stem).upper(), str(session.get("participantId") or path.stem).upper())
        simulation_started_at = session.get("simulationStartedAt")
        for index, conversation in enumerate(session.get("voiceConversations") or [], start=1):
            started_at = conversation.get("startedAt")
            messages = conversation.get("messages") or []
            ended_at = messages[-1].get("at") if messages else started_at
            rows.append({
                "participant_id": pid,
                "conversation_id": index,
                "trigger": conversation.get("trigger"),
                "started_at": started_at,
                "ended_at": ended_at,
                "start_ms": _iso_to_ms(started_at, simulation_started_at),
                "end_ms": _iso_to_ms(ended_at, simulation_started_at),
                "message_count": len(messages),
            })
    return pl.DataFrame(rows, schema=CONVERSATION_COLUMNS, strict=False)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_directory(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    if destination.exists():
        destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        if backup.exists():
            backup.replace(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def prepared_is_current(config_name: str = DEFAULT_CONFIG) -> bool:
    """Whether a complete prepared snapshot exists for the current settings."""
    _, config = load_config(config_name)
    preprocessed = config.get("paths", {}).get("preprocessed")
    if not preprocessed:
        return False
    manifest_path = resolve_project_path(preprocessed) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("preparation_signature") == _preparation_signature(config)


def prepare(config_name: str = DEFAULT_CONFIG) -> Path:
    """Create and atomically replace the configured prepared-data directory."""
    loaded_from, config = load_config(config_name)
    paths = config.get("paths", {})
    settings = config.get("preprocessing", {})
    required_paths = ("silab", "webapp", "preprocessed")
    missing_paths = [name for name in required_paths if name not in paths]
    if missing_paths:
        raise ValueError(f"[paths] fehlt: {', '.join(missing_paths)}")

    silab_dir = resolve_project_path(paths["silab"])
    webapp_dir = resolve_project_path(paths["webapp"])
    destination = resolve_project_path(paths["preprocessed"])
    if not silab_dir.is_dir() or not webapp_dir.is_dir():
        raise FileNotFoundError("Die konfigurierten SiLAB- und Webapp-Verzeichnisse müssen existieren.")

    columns = list(settings.get("silab_columns") or [])
    string_columns = set(settings.get("string_columns") or [])
    if not columns:
        raise ValueError("[preprocessing].silab_columns darf nicht leer sein.")
    unknown_strings = string_columns - set(columns)
    if unknown_strings:
        raise ValueError(f"string_columns enthält keine ausgewählten Spalten: {', '.join(sorted(unknown_strings))}")

    aliases = config.get("preprocessing", {}).get("id_aliases") or config.get("filter", {}).get("id_aliases", {})
    recordings = raw_recordings(silab_dir, aliases)
    if not recordings:
        raise ValueError(f"Keine ASC- oder CSV-Dateien unter {silab_dir}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        target_silab = temporary / "silab"
        manifest_files = []
        for raw_file, group, pid in recordings:
            frame = _select_recording(raw_file, columns, string_columns, group, pid)
            output_file = _output_file(raw_file, silab_dir, target_silab)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(output_file, compression=settings.get("compression", "zstd"))
            manifest_files.append({
                "source": str(raw_file.relative_to(silab_dir)),
                "prepared": str(output_file.relative_to(temporary)),
                "participant_id": pid,
                "source_group": group,
                "rows": frame.height,
            })

        conversations = conversation_table(webapp_dir, aliases)
        conversations.write_parquet(temporary / "conversations.parquet", compression=settings.get("compression", "zstd"))
        _write_json(temporary / "kept_columns.json", {
            "silab_source_columns": columns,
            "silab_added_columns": ["sample_index", "participant_id", "source_group", "source_file"],
            "conversation_columns": list(CONVERSATION_COLUMNS),
        })
        _write_json(temporary / "manifest.json", {
            "config": str(loaded_from.relative_to(PROJECT_ROOT)),
            "preparation_signature": _preparation_signature(config),
            "preparation_settings": _preparation_payload(config),
            "files": manifest_files,
            "recording_count": len(manifest_files),
            "conversation_count": conversations.height,
        })
        _replace_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lossless-row SiLAB and webapp Parquets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="TOML file in configs/ (default: default.toml)")
    args = parser.parse_args()
    output = prepare(args.config)
    print(f"Prepared data written to {output}")

