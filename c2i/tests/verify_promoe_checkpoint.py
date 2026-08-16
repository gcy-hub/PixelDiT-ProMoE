"""Validate that a ProMoE Lightning checkpoint has resumable training state."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    expert_keys = [key for key in state_dict if ".routed_experts." in key]
    prototype_keys = [key for key in state_dict if key.endswith(".prototypes")]
    if not expert_keys:
        raise SystemExit("checkpoint does not contain routed expert parameters")
    if not prototype_keys:
        raise SystemExit("checkpoint does not contain ProMoE prototype parameters")
    if "optimizer_states" not in checkpoint or not checkpoint["optimizer_states"]:
        raise SystemExit("checkpoint does not contain optimizer state required for resume")
    if "global_step" not in checkpoint:
        raise SystemExit("checkpoint does not record global_step")

    print(
        f"checkpoint ok: step={checkpoint['global_step']}, "
        f"expert_tensors={len(expert_keys)}, prototype_tensors={len(prototype_keys)}"
    )


if __name__ == "__main__":
    main()
