"""Regression test for the RGB flow-matching vector-field direction."""

from unittest.mock import patch
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diffusion import LinearScheduler, ProMoEFlowTrainer


class _ReverseVelocityNet(nn.Module):
    def forward(self, x_t, t, labels, return_aux_loss=False):
        # image=2 and noise=-1 in the test below, hence noise-image=-3.
        return torch.full_like(x_t, -3.0), x_t.new_zeros(())


def test_linear_flow_target_moves_noise_to_image():
    trainer = ProMoEFlowTrainer(
        scheduler=LinearScheduler(),
        lognorm_t=False,
    )
    image = torch.full((1, 1, 1, 1), 2.0)
    noise = torch.full_like(image, -1.0)
    labels = torch.zeros(1, dtype=torch.long)

    # Fix the random path so the test isolates the sign of the target.
    with patch("torch.rand", return_value=torch.full((1,), 0.5)), patch(
        "torch.randn_like", return_value=noise
    ):
        losses = trainer._impl_trainstep(
            _ReverseVelocityNet(), None, None, image, labels
        )

    # Correct target is image-noise=3; reverse prediction=-3 gives (6)^2.
    assert torch.isclose(losses["flow_loss"], torch.tensor(36.0))
