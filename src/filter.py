"""
Filter script for driving data
Alexander João Peterson Santos
2026-10-16

Takes ASC files (CSV exports from the driving simulator) from a source dir and
writes filtered copies to a target dir. Two filters, usable alone or together
(a row is kept only if it passes every active filter):

  -q  Quality: drop rows whose EyesGazeQ is below the threshold.
  -c  Conversation: use the webapp session JSON of the same participant to
      keep only rows that fall inside a conversation with the AI assistant
      (default), or only rows outside all conversations (--keep outside).

A conversation spans from its `startedAt` to the `at` of its last message.
Webapp and simulator clocks are aligned via the session's `simulationStartedAt`:
a row's time is simulationStartedAt + Messzeitpunkt (simulator time in ms).
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

DEFAULT_SOURCE = "raw"
DEFAULT_TARGET = "filtered"

ASC_ENCODING = "latin-1"
QUALITY_COL = "EyesGazeQ"
TIME_COL = "Messzeitpunkt"  # ms since simulation start


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse and validate the command line.

    Usage
    -----
    filter.py [-q <quality threshold>] [-c <conversation dir>]
              [--keep {during,outside}] [<source dir> <target dir>]

    At least one of -q / -c is required. Source and target must be given
    together or both omitted (defaults are used then).

    Parameters
    ----------
    argv: list[str]
        Command line arguments without the script name.

    Returns
    -------
    argparse.Namespace
        Validated arguments with `source` and `target` as Paths.
    """
    parser = argparse.ArgumentParser(
        description="Filter ASC driving data by eye tracking quality and/or conversations.",
    )
    parser.add_argument(
        "-q", "--quality", type=float, metavar="THRESHOLD",
        help=f"keep only rows with {QUALITY_COL} >= THRESHOLD (0 to 1)",
    )
    parser.add_argument(
        "-c", "--conversations", type=Path, metavar="DIR",
        help="directory with the webapp session JSONs (<participant>.json)",
    )
    parser.add_argument(
        "--keep", choices=["during", "outside"], default="during",
        help="with -c: keep only rows during conversations (default) or only rows outside them",
    )
    parser.add_argument("dirs", nargs="*", metavar="DIR", help="<source dir> <target dir>")
    args = parser.parse_args(argv)

    if args.quality is None and args.conversations is None:
        parser.error("specify at least one filter (-q and/or -c)")
    if args.quality is not None and not 0 <= args.quality <= 1:
        parser.error("quality threshold must be between 0 and 1")
    if len(args.dirs) not in (0, 2):
        parser.error("give both <source dir> and <target dir>, or neither")

    args.source, args.target = map(Path, args.dirs or (DEFAULT_SOURCE, DEFAULT_TARGET))

    if not args.source.is_dir():
        parser.error(f"source directory {args.source} does not exist")
    if args.conversations is not None and not args.conversations.is_dir():
        parser.error(f"conversation directory {args.conversations} does not exist")
    return args


def iso_to_unix_ms(stamp: str) -> int:
    """Convert an ISO 8601 timestamp such as 2026-07-17T14:18:33.202Z to Unix ms."""
    return round(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)


def conversation_windows(json_path: Path) -> list[tuple[int, int]] | None:
    """
    Read the conversation intervals of one participant.

    Parameters
    ----------
    json_path: Path
        Webapp session JSON.

    Returns
    -------
    list[tuple[int, int]] | None
        (start, end) per conversation in ms since simulationStartedAt (negative
        if before it), end inclusive. Conversations without messages count as
        an instant at their start time. None if the session has conversations
        but no simulationStartedAt, so they can't be aligned.
    """
    with open(json_path, encoding="utf-8") as f:
        session = json.load(f)

    conversations = session.get("voiceConversations", [])
    if not conversations:
        return []
    if not session.get("simulationStartedAt"):
        return None
    origin = iso_to_unix_ms(session["simulationStartedAt"])

    windows = []
    for conv in conversations:
        start = iso_to_unix_ms(conv["startedAt"]) - origin
        ends = [iso_to_unix_ms(m["at"]) - origin for m in conv.get("messages", []) if m.get("at")]
        windows.append((start, max(ends, default=start)))
    return windows


def find_session_json(asc_path: Path, conv_dir: Path) -> Path | None:
    """Find the session JSON whose participant ID appears in the ASC file name."""
    matches = [p for p in conv_dir.glob("*.json") if p.stem.lower() in asc_path.name.lower()]
    if not matches:
        return None
    # Prefer the most specific ID, e.g. DEMO0001 over DEMO000.
    return max(matches, key=lambda p: len(p.stem))


def read_asc(path: Path) -> pl.DataFrame:
    """
    Read an ASC file with every column as string.

    Keeping the text untouched means kept rows are written back exactly as they
    were (no float reformatting). Cells are only cast for filtering.
    """
    text = path.read_bytes().decode(ASC_ENCODING)
    return pl.read_csv(io.BytesIO(text.encode("utf-8")), infer_schema=False)


def write_asc(df: pl.DataFrame, path: Path) -> None:
    """Write as the simulator does: latin-1 with CRLF line endings."""
    buf = io.BytesIO()
    df.write_csv(buf, line_terminator="\r\n")
    path.write_bytes(buf.getvalue().decode("utf-8").encode(ASC_ENCODING))


def conversation_mask(df: pl.DataFrame, windows: list[tuple[int, int]]) -> pl.Expr:
    """Expression that is True for rows whose simulator time lies in any conversation."""
    sim_ms = pl.col(TIME_COL).cast(pl.Float64, strict=False)

    in_conversation = pl.lit(False)
    for start, end in windows:
        in_conversation = in_conversation | sim_ms.is_between(start, end)
    return in_conversation


def filter_file(asc_path: Path, target_dir: Path, args: argparse.Namespace) -> bool:
    """
    Filter one ASC file and write the result to the target dir.

    Returns
    -------
    bool
        False if the file was skipped (e.g. no session JSON found), else True.
    """
    df = read_asc(asc_path)
    total = df.height

    needed = [QUALITY_COL] * (args.quality is not None) + [TIME_COL] * (args.conversations is not None)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Skipping {asc_path.name}: missing column(s) {missing}", file=sys.stderr)
        return False

    keep = pl.lit(True)

    if args.quality is not None:
        quality = pl.col(QUALITY_COL).cast(pl.Float64, strict=False)
        # Null (unparsable) quality compares as null and is dropped by filter().
        keep = keep & (quality >= args.quality)

    if args.conversations is not None:
        json_path = find_session_json(asc_path, args.conversations)
        if json_path is None:
            print(f"Skipping {asc_path.name}: no session JSON found in {args.conversations}", file=sys.stderr)
            return False
        windows = conversation_windows(json_path)
        if windows is None:
            print(f"Skipping {asc_path.name}: {json_path.name} has conversations but no simulationStartedAt", file=sys.stderr)
            return False
        in_conv = conversation_mask(df, windows)
        keep = keep & (~in_conv if args.keep == "outside" else in_conv)

    filtered = df.filter(keep)
    write_asc(filtered, target_dir / asc_path.name)
    print(f"{asc_path.name}: kept {filtered.height}/{total} rows")
    return True


def main() -> None:
    args = parse_args(sys.argv[1:])
    args.target.mkdir(parents=True, exist_ok=True)

    asc_files = sorted(args.source.glob("*.asc"))
    if not asc_files:
        sys.exit(f"No .asc files found in {args.source}")

    done = sum(filter_file(p, args.target, args) for p in asc_files)
    print(f"Converted {done}/{len(asc_files)} files into {args.target}")


if __name__ == "__main__":
    main()
