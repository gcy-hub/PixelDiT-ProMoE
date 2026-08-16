#!/usr/bin/env python3
"""Add target text labels below PixelDiT validation grid tiles."""

import argparse
import json
import os
import re
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


VISIBLE_TEXT_PATTERN = re.compile(
    r"Render\s+the\s+exact\s+visible\s+text\s+(.*?)\s+clearly\s+in\s+the\s+image",
    re.IGNORECASE | re.DOTALL,
)


def load_prompts(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompts = config["train"]["validation_prompts"]
    if not prompts:
        raise ValueError("The config has no train.validation_prompts entries.")
    return prompts


def target_text(prompt: str) -> str:
    match = VISIBLE_TEXT_PATTERN.search(prompt)
    if not match:
        return prompt
    words = re.findall(r'"([^"]*)"', match.group(1))
    return " | ".join(words) if words else match.group(1).strip()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def image_tiles_from_grid(path: Path, columns: int) -> list[Image.Image]:
    with Image.open(path) as source:
        source = source.convert("RGB")
        if source.width % columns:
            raise ValueError(
                f"Grid width {source.width} cannot be divided into {columns} columns."
            )
        tile_width = source.width // columns
        if source.height % tile_width:
            raise ValueError(
                f"Grid height {source.height} is not a multiple of inferred tile size {tile_width}."
            )
        rows = source.height // tile_width
        tile_height = source.height // rows
        return [
            source.crop(
                (
                    (index % columns) * tile_width,
                    (index // columns) * tile_height,
                    (index % columns + 1) * tile_width,
                    (index // columns + 1) * tile_height,
                )
            ).copy()
            for index in range(rows * columns)
        ]


def fit_label(label: str, font_path: str | None, max_size: int, width: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max_size
    while size > 12:
        font = ImageFont.truetype(font_path, size=size) if font_path else load_font(size)
        if ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)[2] <= width - 20:
            return font
        size -= 1
    return ImageFont.truetype(font_path, size=12) if font_path else load_font(12)


def make_grid(
    tiles: list[Image.Image],
    labels: list[str],
    output_path: Path,
    columns: int,
    tile_size: int,
    label_height: int,
    font_path: str | None = None,
) -> None:
    rows = (len(tiles) + columns - 1) // columns
    cell_height = tile_size + label_height
    canvas = Image.new("RGB", (columns * tile_size, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x = (index % columns) * tile_size
        y = (index // columns) * cell_height
        image = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        font = fit_label(label, font_path, min(30, label_height - 16), tile_size)
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_top = y + tile_size + (label_height - (label_box[3] - label_box[1])) / 2 - label_box[1]
        draw.text((x + (tile_size - label_width) / 2, label_top), label, font=font, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    save_args = {"format": "WEBP", "quality": 95, "method": 6} if output_path.suffix.lower() == ".webp" else {}
    canvas.save(temporary_path, **save_args)
    os.replace(temporary_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source_grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest_output", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--label_height", type=int, default=64)
    parser.add_argument("--font_path")
    args = parser.parse_args()

    prompts = load_prompts(args.config)
    base_labels = [target_text(prompt) for prompt in prompts]
    tiles = image_tiles_from_grid(args.source_grid, args.columns)
    if len(tiles) % len(base_labels):
        raise ValueError(
            f"Grid contains {len(tiles)} tiles, but config contains {len(base_labels)} prompts."
        )
    labels = base_labels * (len(tiles) // len(base_labels))
    make_grid(tiles, labels, args.output, args.columns, args.tile_size, args.label_height, args.font_path)

    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            f"{index:03d}": {"prompt": prompt, "target_text": label}
            for index, (prompt, label) in enumerate(zip(prompts, base_labels))
        }
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Saved {len(tiles)} labeled images to {args.output}")


if __name__ == "__main__":
    main()
