#!/usr/bin/env python3
"""Convert ImageNet parquet shards to the H5 format used by PixelDiT-ProMoE.

The input directory is read-only.  The output directory is created separately
and has the following layout after a successful conversion:

    <output-dir>/images.h5
    <output-dir>/images_h5.json

Only ``train-*.parquet`` shards are converted.  The conversion can be resumed
with ``--resume``; progress is saved after each parquet row group.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


STATE_FILE = ".conversion_state.json"
LABELS_DB_FILE = ".labels.sqlite3"
H5_FILE = "images.h5"
H5_INDEX_FILE = "images_h5.json"
STATE_VERSION = 1


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    """Apply the ADM-style center crop used by the PixelDiT data pipeline."""
    while min(image.size) >= 2 * image_size:
        image = image.resize(
            tuple(value // 2 for value in image.size),
            resample=Image.BOX,
        )
    scale = image_size / min(image.size)
    image = image.resize(
        tuple(round(value * scale) for value in image.size),
        resample=Image.BICUBIC,
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(
        array[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    )


def image_key(source_index: int) -> str:
    index = f"{source_index:08d}"
    return f"{index[:5]}/img{index}.png"


def convert_one(task: tuple[int, bytes, int, int, int]) -> tuple[str, bytes | None, int | None, str | None]:
    """Decode one source JPEG and return a PixelDiT H5 entry."""
    source_index, jpeg_bytes, label, resolution, png_compress = task
    key = image_key(source_index)
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as source_image:
            image = source_image.convert("RGB")
            image.load()
        image = center_crop_arr(image, resolution)
        encoded = io.BytesIO()
        image.save(
            encoded,
            format="PNG",
            compress_level=png_compress,
            optimize=False,
        )
        return key, encoded.getvalue(), int(label), None
    except Exception as error:  # Keep a corrupt source image from aborting a full run.
        return key, None, None, f"{type(error).__name__}: {error}"


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def write_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def make_shard_manifest(shards: list[Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        metadata = parquet_file.metadata
        stat = shard.stat()
        manifest.append(
            {
                "name": shard.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "rows": metadata.num_rows,
                "row_group_rows": [
                    metadata.row_group(index).num_rows
                    for index in range(metadata.num_row_groups)
                ],
            }
        )
    return manifest


def source_shards(parquet_dir: Path, max_shards: int) -> list[Path]:
    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No train-*.parquet files found in {parquet_dir}. "
            "Pass the directory that directly contains the parquet shards."
        )
    return shards if max_shards == 0 else shards[:max_shards]


def validate_output_location(parquet_dir: Path, output_dir: Path) -> None:
    if output_dir == parquet_dir:
        raise ValueError("--output-dir must be different from --parquet-dir")
    try:
        output_dir.relative_to(parquet_dir)
    except ValueError:
        return
    raise ValueError("--output-dir must not be inside --parquet-dir")


def open_labels_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS labels ("
        "path TEXT PRIMARY KEY, label INTEGER NOT NULL)"
    )
    return connection


def insert_h5_bytes(h5_file: h5py.File, key: str, value: bytes) -> bool:
    """Write a byte array once, returning whether it was newly created."""
    if key in h5_file:
        return False
    group_name, dataset_name = key.rsplit("/", 1)
    group = h5_file.require_group(group_name)
    group.create_dataset(dataset_name, data=np.frombuffer(value, dtype=np.uint8))
    return True


def iter_results(
    pool: Pool | None,
    tasks: list[tuple[int, bytes, int, int, int]],
) -> Iterable[tuple[str, bytes | None, int | None, str | None]]:
    if pool is None:
        return map(convert_one, tasks)
    return pool.imap(convert_one, tasks, chunksize=min(32, len(tasks)))


def build_metadata_files(output_dir: Path, h5_path: Path, labels_db: sqlite3.Connection) -> int:
    """Create H5 labels and the image index without holding all labels in memory."""
    temp_dataset_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_dir, prefix=".dataset-json-", delete=False
        ) as dataset_file:
            temp_dataset_path = Path(dataset_file.name)
            dataset_file.write(b'{"labels":[')
            first = True
            count = 0
            for key, label in labels_db.execute(
                "SELECT path, label FROM labels ORDER BY path"
            ):
                if not first:
                    dataset_file.write(b",")
                dataset_file.write(
                    json.dumps([key, label], separators=(",", ":")).encode("utf-8")
                )
                first = False
                count += 1
            dataset_file.write(b"]}")

        with temp_dataset_path.open("rb") as dataset_file:
            dataset_json = dataset_file.read()
        with h5py.File(h5_path, "a") as h5_file:
            if "dataset.json" in h5_file:
                del h5_file["dataset.json"]
            h5_file.create_dataset(
                "dataset.json", data=np.frombuffer(dataset_json, dtype=np.uint8)
            )

        index_path = output_dir / H5_INDEX_FILE
        temporary_index_path = index_path.with_name(f".{index_path.name}.tmp")
        with temporary_index_path.open("w", encoding="utf-8") as index_file:
            index_file.write("[")
            first = True
            for (key,) in labels_db.execute("SELECT path FROM labels ORDER BY path"):
                if not first:
                    index_file.write(",")
                json.dump(key, index_file)
                first = False
            index_file.write("]\n")
        os.replace(temporary_index_path, index_path)
        return count
    finally:
        if temp_dataset_path is not None:
            temp_dataset_path.unlink(missing_ok=True)


def validate_completed_output(output_dir: Path, expected_count: int) -> None:
    h5_path = output_dir / H5_FILE
    index_path = output_dir / H5_INDEX_FILE
    if not h5_path.is_file() or not index_path.is_file():
        raise RuntimeError("Conversion completed but one or more output files are missing")

    with index_path.open("r", encoding="utf-8") as index_file:
        keys = json.load(index_file)
    if len(keys) != expected_count:
        raise RuntimeError(
            f"images_h5.json contains {len(keys)} entries, expected {expected_count}"
        )

    with h5py.File(h5_path, "r") as h5_file:
        if "dataset.json" not in h5_file:
            raise RuntimeError("images.h5 is missing dataset.json")
        labels = json.loads(np.asarray(h5_file["dataset.json"]).tobytes())["labels"]
        if len(labels) != expected_count:
            raise RuntimeError(
                f"dataset.json contains {len(labels)} labels, expected {expected_count}"
            )
        for key in keys[: min(8, len(keys))]:
            if key not in h5_file:
                raise RuntimeError(f"images.h5 is missing indexed image {key}")
            with Image.open(io.BytesIO(np.asarray(h5_file[key]).tobytes())) as image:
                if image.mode != "RGB" or image.size[0] != image.size[1]:
                    raise RuntimeError(f"Invalid converted image {key}: {image.mode} {image.size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ImageNet train parquet shards to PixelDiT-ProMoE H5 data."
    )
    parser.add_argument(
        "--parquet-dir", required=True, type=Path,
        help="Directory containing train-*.parquet source shards (read only).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New directory in which images.h5 and images_h5.json are created.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--png-compress", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0, help="0 converts every shard.")
    parser.add_argument("--max-images", type=int, default=0, help="0 converts every image.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Validate output metadata after conversion.")
    parser.add_argument(
        "--progress-seconds", type=float, default=5.0,
        help="Minimum interval between progress reports.",
    )
    arguments = parser.parse_args()
    if arguments.resolution <= 0:
        parser.error("--resolution must be positive")
    if arguments.workers <= 0:
        parser.error("--workers must be positive")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0 <= arguments.png_compress <= 9:
        parser.error("--png-compress must be between 0 and 9")
    if arguments.max_shards < 0 or arguments.max_images < 0:
        parser.error("--max-shards and --max-images must be non-negative")
    if arguments.progress_seconds < 0:
        parser.error("--progress-seconds must be non-negative")
    return arguments


def main() -> None:
    arguments = parse_args()
    Image.init()
    parquet_dir = arguments.parquet_dir.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    validate_output_location(parquet_dir, output_dir)
    if not parquet_dir.is_dir():
        raise NotADirectoryError(f"--parquet-dir does not exist: {parquet_dir}")

    shards = source_shards(parquet_dir, arguments.max_shards)
    print(f"[scan] inspecting {len(shards)} parquet shards...", flush=True)
    manifest = make_shard_manifest(shards)
    source_rows = sum(item["rows"] for item in manifest)
    total_rows = source_rows if arguments.max_images == 0 else min(source_rows, arguments.max_images)
    if total_rows == 0:
        raise ValueError("No source images selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILE
    labels_path = output_dir / LABELS_DB_FILE
    h5_path = output_dir / H5_FILE
    h5_index_path = output_dir / H5_INDEX_FILE
    requested_config = {
        "parquet_dir": str(parquet_dir),
        "resolution": arguments.resolution,
        "png_compress": arguments.png_compress,
        "max_shards": arguments.max_shards,
        "max_images": arguments.max_images,
    }

    if arguments.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"No resumable conversion state: {state_path}")
        with state_path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
        if state.get("version") != STATE_VERSION:
            raise RuntimeError("Unsupported conversion state version")
        if state.get("config") != requested_config or state.get("shards") != manifest:
            raise RuntimeError(
                "Source shards or conversion options differ from the saved state. "
                "Start in a new output directory instead."
            )
        if state.get("total_source_rows") != total_rows:
            raise RuntimeError("Saved total source rows differ from current input")
        if state.get("status") == "complete":
            expected_count = int(state["output_images"])
            if arguments.verify:
                validate_completed_output(output_dir, expected_count)
                print(f"[verify] output is valid ({expected_count:,} images)", flush=True)
            print(f"[done] conversion already complete: {output_dir}", flush=True)
            return
        if not h5_path.is_file() or not labels_path.is_file():
            raise RuntimeError("Resume state exists but images.h5 or the labels database is missing")
        print(
            f"[resume] continuing from source image {state['source_rows_completed']:,}/"
            f"{total_rows:,}",
            flush=True,
        )
    else:
        conflicting_paths = [path for path in (state_path, labels_path, h5_path, h5_index_path) if path.exists()]
        if conflicting_paths:
            names = ", ".join(path.name for path in conflicting_paths)
            raise FileExistsError(
                f"Output directory already has conversion files ({names}). "
                "Use --resume or choose a new --output-dir."
            )
        state = {
            "version": STATE_VERSION,
            "status": "in_progress",
            "config": requested_config,
            "shards": manifest,
            "total_source_rows": total_rows,
            "source_rows_completed": 0,
            "next_shard_index": 0,
            "next_row_group_index": 0,
        }
        write_state(state_path, state)

    labels_db = open_labels_database(labels_path)
    started_at = time.monotonic()
    last_report_at = started_at
    starting_source_rows = int(state["source_rows_completed"])
    current_source_rows = starting_source_rows
    newly_written = 0
    existing_images = 0
    skipped = 0

    print(
        f"[start] source={total_rows:,} images | resolution={arguments.resolution} | "
        f"workers={arguments.workers} | output={output_dir}",
        flush=True,
    )

    def report(force: bool = False) -> None:
        nonlocal last_report_at
        now = time.monotonic()
        if not force and now - last_report_at < arguments.progress_seconds:
            return
        elapsed = max(now - started_at, 1e-9)
        session_rows = current_source_rows - starting_source_rows
        rate = session_rows / elapsed
        remaining = total_rows - current_source_rows
        eta = remaining / rate if rate > 0 else None
        percent = 100 * current_source_rows / total_rows
        print(
            f"[progress] {current_source_rows:,}/{total_rows:,} ({percent:5.1f}%) | "
            f"{rate:7.1f} img/s | ETA {format_duration(eta)} | "
            f"new={newly_written:,} existing={existing_images:,} skipped={skipped:,}",
            flush=True,
        )
        last_report_at = now

    pool: Pool | None = None
    if arguments.workers > 1:
        pool = mp.Pool(processes=arguments.workers)

    try:
        with h5py.File(h5_path, "a") as h5_file:
            start_shard_index = int(state["next_shard_index"])
            start_row_group_index = int(state["next_row_group_index"])
            for shard_index in range(start_shard_index, len(shards)):
                parquet_file = pq.ParquetFile(shards[shard_index])
                row_group_start = start_row_group_index if shard_index == start_shard_index else 0
                for row_group_index in range(row_group_start, parquet_file.metadata.num_row_groups):
                    if current_source_rows >= total_rows:
                        break
                    row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
                    rows_to_process = min(row_group_rows, total_rows - current_source_rows)
                    row_group_done = 0
                    print(
                        f"[shard] {shards[shard_index].name} row-group "
                        f"{row_group_index + 1}/{parquet_file.metadata.num_row_groups} "
                        f"({rows_to_process:,} images)",
                        flush=True,
                    )

                    for batch in parquet_file.iter_batches(
                        batch_size=arguments.batch_size, row_groups=[row_group_index]
                    ):
                        if row_group_done >= rows_to_process:
                            break
                        columns = batch.to_pydict()
                        batch_rows = min(len(columns["label"]), rows_to_process - row_group_done)
                        tasks = [
                            (
                                current_source_rows + offset,
                                columns["image"][offset]["bytes"],
                                columns["label"][offset],
                                arguments.resolution,
                                arguments.png_compress,
                            )
                            for offset in range(batch_rows)
                        ]
                        labels_to_insert: list[tuple[str, int]] = []
                        for key, png_bytes, label, error in iter_results(pool, tasks):
                            if error is not None:
                                skipped += 1
                                print(f"[skip] {key}: {error}", file=sys.stderr, flush=True)
                                continue
                            assert png_bytes is not None and label is not None
                            if insert_h5_bytes(h5_file, key, png_bytes):
                                newly_written += 1
                            else:
                                existing_images += 1
                            labels_to_insert.append((key, label))
                        if labels_to_insert:
                            labels_db.executemany(
                                "INSERT OR REPLACE INTO labels(path, label) VALUES (?, ?)",
                                labels_to_insert,
                            )
                        row_group_done += batch_rows
                        current_source_rows += batch_rows
                        report()

                    if row_group_done != rows_to_process:
                        raise RuntimeError(
                            f"Processed {row_group_done} rows from a row group expected to have "
                            f"{rows_to_process}"
                        )
                    h5_file.flush()
                    labels_db.commit()
                    if current_source_rows >= total_rows:
                        next_shard_index = shard_index
                        next_row_group_index = row_group_index
                    elif row_group_index + 1 < parquet_file.metadata.num_row_groups:
                        next_shard_index = shard_index
                        next_row_group_index = row_group_index + 1
                    else:
                        next_shard_index = shard_index + 1
                        next_row_group_index = 0
                    state.update(
                        source_rows_completed=current_source_rows,
                        next_shard_index=next_shard_index,
                        next_row_group_index=next_row_group_index,
                    )
                    write_state(state_path, state)
                    report(force=True)
                if current_source_rows >= total_rows:
                    break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        labels_db.commit()

    if current_source_rows != total_rows:
        raise RuntimeError(
            f"Conversion stopped after {current_source_rows:,}/{total_rows:,} source images"
        )

    output_images = build_metadata_files(output_dir, h5_path, labels_db)
    state.update(
        status="complete",
        source_rows_completed=current_source_rows,
        output_images=output_images,
    )
    write_state(state_path, state)
    if arguments.verify:
        validate_completed_output(output_dir, output_images)
        print(f"[verify] output is valid ({output_images:,} images)", flush=True)
    labels_db.close()

    report(force=True)
    print(
        f"[done] wrote {output_images:,} images to {h5_path} "
        f"(skipped source images in this run: {skipped:,})",
        flush=True,
    )
    print(f"[done] image index: {h5_index_path}", flush=True)


if __name__ == "__main__":
    main()
