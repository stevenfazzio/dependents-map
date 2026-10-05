"""File writes that never leave a partial or unverified file at the destination.

Fetch stages append each unit of fetched work to a JSONL log as it arrives, and build
their parquet from the log once the fetch is complete. The log is both the raw record
and the resume state.
"""

import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_jsonl_log(path: Path) -> list[dict]:
    """Read a log's records, dropping a trailing line cut short by a crash."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    good_bytes = 0
    for i, line in enumerate(lines):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("line has no newline")
            records.append(json.loads(line))
        except ValueError:
            if i != len(lines) - 1:
                raise RuntimeError(f"{path} is corrupt at line {i + 1}") from None
            print(f"  Dropping a partial final line from {path.name}")
            with open(path, "rb+") as f:
                f.truncate(good_bytes)
            break
        good_bytes += len(line)
    return records


def append_jsonl(log: IO[str], record: dict) -> None:
    """Append one record and force it to disk before returning."""
    log.write(json.dumps(record) + "\n")
    log.flush()
    os.fsync(log.fileno())


def write_parquet_safely(df: pd.DataFrame, output_path: Path) -> None:
    """Write df so output_path only ever holds a complete, verified file."""
    output_path = Path(output_path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)

        # mkstemp creates 0600 and os.replace keeps the source mode, so without this
        # every rewrite would silently tighten the file's permissions.
        if output_path.exists():
            os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_path, 0o666 & ~umask)

        # Footer-only read: checks rows and schema without loading the data back.
        meta = pq.read_metadata(tmp_path)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        on_disk = set(meta.schema.to_arrow_schema().names)
        assert on_disk >= set(df.columns), f"columns missing on disk: {set(df.columns) - on_disk}"

        os.replace(tmp_path, output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Each array's shape, read from the headers alone."""
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            with archive.open(member) as f:
                version = np.lib.format.read_magic(f)
                read_header = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shapes[member.removesuffix(".npy")] = read_header(f)[0]
    return shapes


def write_npz_safely(output_path: Path, **arrays: np.ndarray) -> None:
    """Write arrays so output_path only ever holds a complete, verified file."""
    output_path = Path(output_path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".npz.tmp")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:  # a file object, so numpy can't append ".npz"
            np.savez(f, **arrays)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)

        # Headers only: reading a large matrix back would double peak memory.
        on_disk = npz_shapes(Path(tmp_path))
        expected = {name: array.shape for name, array in arrays.items()}
        assert on_disk == expected, f"shapes on disk {on_disk} != {expected}"

        os.replace(tmp_path, output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def write_bytes_safely(output_path: Path, data: bytes) -> None:
    """Write bytes so output_path only ever holds the complete file."""
    output_path = Path(output_path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)
        os.replace(tmp_path, output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
