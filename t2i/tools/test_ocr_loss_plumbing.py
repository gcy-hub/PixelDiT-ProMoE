#!/usr/bin/env python3
"""Lightweight checks for PixelDiT whole-image OCR-loss plumbing.

This test intentionally does not load UniRec weights. It verifies the two
project-specific invariants that can be tested without an OCR checkpoint:
reading-order transcript serialization and differentiable flow x_0 recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


T2I_DIR = Path(__file__).resolve().parents[1]
if str(T2I_DIR) not in sys.path:
    sys.path.insert(0, str(T2I_DIR))

from diffusion.data.datasets.pixdit_datasets import _whole_image_ocr_transcript
from diffusion.model import gaussian_diffusion as gd


def test_transcript_reading_order() -> None:
    info = {
        "ocr_result": [
            [[[276, 254], [748, 254], [748, 283], [276, 283]], ["FOCUSING ON EGRETS", 0.9679]],
            [[[356, 197], [658, 197], [658, 226], [356, 226]], ["KEEP CALM BY", 0.9662]],
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ["LOW CONFIDENCE", 0.5]],
        ]
    }
    transcript = _whole_image_ocr_transcript(
        info, valid_only=True, min_rec_score=0.95, separator="\n"
    )
    expected = "KEEP CALM BY\nFOCUSING ON EGRETS"
    assert transcript == expected, (transcript, expected)

    # A structured OCR label that fails the configured confidence threshold
    # must not re-enter through the generic ``texts`` compatibility fallback.
    rejected = _whole_image_ocr_transcript(
        {"ocr_result": info["ocr_result"][-1:], "texts": ["LOW CONFIDENCE"]},
        valid_only=True,
        min_rec_score=0.95,
        separator="\n",
    )
    assert rejected == "", rejected


def test_flow_xstart_gradient() -> None:
    sigmas = np.array([0.1, 0.5], dtype=np.float32)
    x_t = torch.randn(2, 3, 4, 4)
    velocity_scale = torch.nn.Parameter(torch.tensor(2.0))
    velocity = velocity_scale * torch.ones_like(x_t)
    timestep = torch.tensor([0, 1], dtype=torch.long)
    sigma = gd._extract_into_tensor(sigmas, timestep, x_t.shape)
    pred_xstart = x_t - sigma * velocity
    expected = x_t - torch.tensor([0.1, 0.5]).view(2, 1, 1, 1) * velocity
    torch.testing.assert_close(pred_xstart, expected)
    pred_xstart.sum().backward()
    assert velocity_scale.grad is not None and velocity_scale.grad.abs().item() > 0


if __name__ == "__main__":
    test_transcript_reading_order()
    test_flow_xstart_gradient()
    print("OCR plumbing checks passed")
