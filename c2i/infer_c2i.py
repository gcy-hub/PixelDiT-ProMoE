#!/usr/bin/env python3
"""Generate class-conditional PixelDiT-ProMoE images from a Lightning checkpoint.

Example:
    python c2i/infer_c2i.py \
      --checkpoint c2i/train_logs/exp_pixdit_promoe_xl_imagenet256/last.ckpt \
      --config c2i/configs/pix256_promoe_xl.yaml \
      --gpu-ids 0,1 \
      --class-counts '{"0": 4, "207": 2}' \
      --output-dir c2i/samples/promoe_xl
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from queue import Empty
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # Allows --help and input validation without a CUDA runtime.
    torch = None


C2I_ROOT = Path(__file__).resolve().parent
REPO_ROOT = C2I_ROOT.parent
for import_path in (str(REPO_ROOT), str(C2I_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


def import_object(class_path: str) -> Any:
    module_name, object_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), object_name)


def instantiate(specification: dict[str, Any]) -> Any:
    if not isinstance(specification, dict) or "class_path" not in specification:
        raise ValueError("Expected a Lightning-style component with class_path and init_args")
    constructor: Callable[..., Any] = import_object(specification["class_path"])
    return constructor(**specification.get("init_args", {}))


def parse_class_counts(value: str) -> dict[int, int]:
    """Parse and validate a JSON object mapping ImageNet class ids to counts."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"--class-counts must be valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict) or not parsed:
        raise argparse.ArgumentTypeError("--class-counts must be a non-empty JSON object")

    class_counts: dict[int, int] = {}
    for raw_class_id, raw_count in parsed.items():
        if isinstance(raw_class_id, bool):
            raise argparse.ArgumentTypeError("Class ids must be integers from 0 through 999")
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                f"Invalid class id {raw_class_id!r}; expected an integer from 0 through 999"
            ) from error
        if str(class_id) != str(raw_class_id).strip():
            raise argparse.ArgumentTypeError(
                f"Invalid class id {raw_class_id!r}; expected an integer from 0 through 999"
            )
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise argparse.ArgumentTypeError(
                f"Count for class {class_id} must be a positive integer"
            )
        if not 0 <= class_id < 1000:
            raise argparse.ArgumentTypeError(
                f"Class id {class_id} is outside the ImageNet-1k range [0, 999]"
            )
        class_counts[class_id] = raw_count
    return dict(sorted(class_counts.items()))


def parse_gpu_ids(value: str) -> list[int]:
    """Parse a comma-separated list of CUDA device indices."""
    if not value.strip():
        raise argparse.ArgumentTypeError("--gpu-ids must contain at least one GPU id")
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--gpu-ids must be a comma-separated list such as 0,1,2"
        ) from error
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU ids must be non-negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("GPU ids must not contain duplicates")
    return gpu_ids


def partition_tasks(
    class_counts: dict[int, int], num_workers: int
) -> list[dict[int, list[int]]]:
    """Assign every (class, image index) to exactly one inference worker."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    assignments: list[dict[int, list[int]]] = [{} for _ in range(num_workers)]
    worker_index = 0
    for class_id, count in class_counts.items():
        for image_index in range(count):
            assignments[worker_index].setdefault(class_id, []).append(image_index)
            worker_index = (worker_index + 1) % num_workers
    return assignments


def select_denoiser_state_dict(
    checkpoint: dict[str, Any],
    use_ema: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    """Extract a direct denoiser state dict from a Lightning checkpoint."""
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a state_dict")

    prefixes = ("ema_denoiser.", "denoiser.") if use_ema else ("denoiser.", "ema_denoiser.")
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if selected:
            source = "EMA" if prefix == "ema_denoiser." else "denoiser"
            return selected, source

    # Allow a checkpoint saved directly from the denoiser rather than Lightning.
    if all(isinstance(key, str) and not key.startswith(("model.", "ema_")) for key in state_dict):
        return state_dict, "denoiser"
    raise ValueError(
        "Checkpoint has neither ema_denoiser.* nor denoiser.* parameters. "
        "Use a PixelDiT-ProMoE Lightning checkpoint."
    )


def load_denoiser(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[torch.nn.Module, str]:
    model_config = config.get("model", {})
    denoiser = instantiate(model_config.get("denoiser", {}))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, source = select_denoiser_state_dict(checkpoint, use_ema=use_ema)
    missing_keys, unexpected_keys = denoiser.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing={missing_keys[:5]}")
        if unexpected_keys:
            details.append(f"unexpected={unexpected_keys[:5]}")
        raise RuntimeError(
            "Checkpoint and YAML model architecture do not match (" + "; ".join(details) + ")"
        )
    return denoiser.to(device).eval(), source


def uint8_images(samples: torch.Tensor) -> np.ndarray:
    return (
        ((samples.clamp(-1, 1) + 1) * 127.5 + 0.5)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def seeded_noise(
    class_id: int,
    output_indices: list[int],
    channels: int,
    resolution: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Give every output a stable seed regardless of class order or batch size."""
    noises = []
    for output_index in output_indices:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + class_id * 1_000_003 + output_index)
        noises.append(
            torch.randn((channels, resolution, resolution), generator=generator, device=device)
        )
    return torch.stack(noises)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate_worker(
    worker_index: int,
    gpu_id: int,
    tasks: dict[int, list[int]],
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    batch_size: int,
    seed: int,
    num_steps: int | None,
    guidance: float | None,
    use_ema: bool,
    progress_queue: Any,
) -> None:
    """Load one copy of the model and generate this worker's disjoint tasks."""
    try:
        if torch is None:
            raise RuntimeError("PyTorch is required for PixelDiT-ProMoE inference")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.set_float32_matmul_precision("medium")
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        denoiser, weights_source = load_denoiser(
            config, Path(checkpoint_path), device, use_ema=use_ema
        )
        sampler = instantiate(config["model"]["diffusion_sampler"]).to(device).eval()
        if num_steps is not None:
            sampler.num_steps = num_steps
        if guidance is not None:
            sampler.guidance = guidance

        denoiser_args = config["model"]["denoiser"]["init_args"]
        channels = int(denoiser_args["in_channels"])
        if channels != 3:
            raise ValueError(f"Expected an RGB C2I model with 3 input channels, got {channels}")
        if int(denoiser_args.get("patch_size", 16)) != 16:
            raise ValueError("This script currently supports the 256px PixelDiT C2I patch size of 16")

        progress_queue.put(("ready", worker_index, gpu_id, weights_source))
        with torch.inference_mode():
            for class_id, output_indices in tasks.items():
                class_dir = Path(output_dir) / f"class_{class_id:04d}"
                class_dir.mkdir(exist_ok=True)
                for start in range(0, len(output_indices), batch_size):
                    batch_indices = output_indices[start:start + batch_size]
                    labels = torch.full(
                        (len(batch_indices),), class_id, dtype=torch.long, device=device
                    )
                    null_labels = torch.full_like(labels, int(denoiser_args["num_classes"]))
                    noise = seeded_noise(
                        class_id, batch_indices, channels, 256, seed, device
                    )
                    samples = sampler(denoiser, noise, labels, null_labels)
                    for image_index, image in zip(batch_indices, uint8_images(samples)):
                        Image.fromarray(image).save(class_dir / f"{image_index:05d}.png")
                    progress_queue.put(("progress", worker_index, gpu_id, len(batch_indices), class_id))
        progress_queue.put(("done", worker_index, gpu_id))
    except Exception:
        progress_queue.put(("error", worker_index, gpu_id, traceback.format_exc()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate selected ImageNet classes with a PixelDiT-ProMoE checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Training checkpoint (.ckpt).")
    parser.add_argument(
        "--config", required=True, type=Path,
        help="The ProMoE YAML used to train the checkpoint; supplies model and sampler parameters.",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu-ids", type=parse_gpu_ids, default=None,
        help="Comma-separated CUDA device indices among visible GPUs, for example 0,1,2.",
    )
    gpu_group.add_argument("--gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--class-counts", required=True, type=parse_class_counts,
        help="JSON class-to-count mapping, for example '{\"0\": 4, \"207\": 2}'.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Destination directory. Defaults to <checkpoint-parent>/samples.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None, help="Override sampler num_steps.")
    parser.add_argument("--guidance", type=float, default=None, help="Override classifier-free guidance.")
    parser.add_argument(
        "--use-denoiser", action="store_true",
        help="Use raw denoiser weights instead of EMA weights when both are in the checkpoint.",
    )
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if arguments.num_steps is not None and arguments.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if arguments.gpu_ids is None:
        arguments.gpu_ids = [0 if arguments.gpu_id is None else arguments.gpu_id]
    return arguments


def main() -> None:
    arguments = parse_args()
    checkpoint_path = arguments.checkpoint.expanduser().resolve()
    config_path = arguments.config.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    if torch is None:
        raise RuntimeError("PyTorch is required for PixelDiT-ProMoE inference")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PixelDiT-ProMoE inference")
    invalid_gpu_ids = [
        gpu_id
        for gpu_id in arguments.gpu_ids
        if not 0 <= gpu_id < torch.cuda.device_count()
    ]
    if invalid_gpu_ids:
        raise ValueError(
            f"GPU ids {invalid_gpu_ids} are outside [0, {torch.cuda.device_count() - 1}] "
            "among visible CUDA devices"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    denoiser_args = config["model"]["denoiser"]["init_args"]
    sampler_args = config["model"]["diffusion_sampler"].get("init_args", {})

    output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else checkpoint_path.parent / "samples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    total = sum(arguments.class_counts.values())
    for class_id in arguments.class_counts:
        (output_dir / f"class_{class_id:04d}").mkdir(exist_ok=True)

    assignments = partition_tasks(arguments.class_counts, len(arguments.gpu_ids))
    worker_specs = [
        (gpu_id, tasks)
        for gpu_id, tasks in zip(arguments.gpu_ids, assignments)
        if tasks
    ]
    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes = []
    for worker_index, (gpu_id, tasks) in enumerate(worker_specs):
        process = context.Process(
            target=generate_worker,
            args=(
                worker_index, gpu_id, tasks, str(checkpoint_path), str(config_path),
                str(output_dir), arguments.batch_size, arguments.seed, arguments.num_steps,
                arguments.guidance, not arguments.use_denoiser, progress_queue,
            ),
        )
        process.start()
        processes.append(process)

    print(
        f"[start] checkpoint={checkpoint_path.name} samples={total} "
        f"gpus={','.join(str(gpu_id) for gpu_id, _ in worker_specs)} "
        f"sampler_steps={arguments.num_steps or sampler_args.get('num_steps')}",
        flush=True,
    )
    completed = 0
    ready_workers: dict[int, str] = {}
    finished_workers: set[int] = set()
    errors: list[str] = []
    started_at = time.monotonic()
    last_report_at = started_at

    def report_progress(force: bool = False) -> None:
        nonlocal last_report_at
        now = time.monotonic()
        if not force and now - last_report_at < 1.0:
            return
        elapsed = max(now - started_at, 1e-9)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else None
        print(
            f"[progress] {completed}/{total} ({100 * completed / total:5.1f}%) | "
            f"{rate:.2f} img/s | ETA {format_duration(eta)}",
            flush=True,
        )
        last_report_at = now

    try:
        while len(finished_workers) < len(processes) and not errors:
            try:
                event = progress_queue.get(timeout=0.25)
            except Empty:
                terminated = [
                    process for process in processes
                    if process.exitcode is not None and process.exitcode != 0
                ]
                if terminated:
                    errors.append(
                        "Inference worker exited unexpectedly: "
                        + ", ".join(str(process.exitcode) for process in terminated)
                    )
                report_progress()
                continue

            event_type, worker_index, gpu_id, *payload = event
            if event_type == "ready":
                ready_workers[worker_index] = payload[0]
                print(f"[load] gpu={gpu_id} weights={payload[0]}", flush=True)
            elif event_type == "progress":
                completed += payload[0]
                report_progress()
            elif event_type == "done":
                finished_workers.add(worker_index)
            elif event_type == "error":
                errors.append(f"GPU {gpu_id} worker failed:\n{payload[0]}")
            else:
                errors.append(f"Unknown worker event: {event_type}")
    finally:
        for process in processes:
            if errors and process.is_alive():
                process.terminate()
            process.join()
        progress_queue.close()

    if errors:
        raise RuntimeError("\n".join(errors))
    failed_processes = [process.exitcode for process in processes if process.exitcode != 0]
    if failed_processes:
        raise RuntimeError(f"Inference workers exited with codes: {failed_processes}")
    if completed != total:
        raise RuntimeError(f"Generated {completed}/{total} requested images")

    manifest = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "weights": sorted(set(ready_workers.values())),
        "gpu_ids": arguments.gpu_ids,
        "seed": arguments.seed,
        "class_counts": {str(class_id): count for class_id, count in arguments.class_counts.items()},
        "sampler": {
            "class_path": config["model"]["diffusion_sampler"]["class_path"],
            "num_steps": arguments.num_steps or sampler_args.get("num_steps"),
            "guidance": arguments.guidance if arguments.guidance is not None else sampler_args.get("guidance"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report_progress(force=True)
    print(f"[done] saved {total} images to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
