"""Focused ProMoE tests that do not allocate the full XL expert bank on CPU."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from pixdit_core.pixeldit_c2i import PixDiTProMoE
from pixdit_core.promoe import ProMoEFeedForward


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "c2i" / "configs"

MODEL_SPECS = {
    "s": (384, 6, 12, 2, 768),
    "b": (768, 12, 12, 2, 1536),
    "l": (1024, 16, 24, 4, 2048),
    "xl": (1152, 16, 28, 4, 2304),
}


class ProMoEFeedForwardTests(unittest.TestCase):
    def _module(self, num_routed_experts=3):
        return ProMoEFeedForward(
            hidden_size=4,
            expert_intermediate_size=8,
            num_routed_experts=num_routed_experts,
            rcl_temperature=0.07,
        )

    def test_conditional_raw_cosine_routing_and_prototype_gradients(self):
        module = self._module(num_routed_experts=2)
        with torch.no_grad():
            module.prototypes.zero_()
            module.prototypes[0, 0] = 1.0
            module.prototypes[1, 1] = 1.0

        hidden = torch.tensor(
            [
                [[2.0, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0]],
                [[0.0, 3.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
            ],
            requires_grad=True,
        )
        output, rcl_loss = module(hidden, torch.tensor([4, 5]))
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(rcl_loss))
        self.assertTrue(torch.equal(module.last_routing["expert_indices"][0], torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.equal(module.last_routing["expert_indices"][1], torch.ones(2, dtype=torch.long)))
        self.assertTrue(torch.allclose(module.last_routing["weights"], torch.ones_like(module.last_routing["weights"])))

        rcl_loss.backward()
        self.assertIsNotNone(module.prototypes.grad)
        self.assertGreater(module.prototypes.grad.abs().sum().item(), 0.0)

    def test_null_tokens_use_only_the_unconditional_expert(self):
        module = self._module(num_routed_experts=2)
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        output, rcl_loss = module(hidden, torch.full((2,), 1000, dtype=torch.long))
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(rcl_loss))
        self.assertTrue(torch.all(module.last_routing["expert_indices"] == 2))
        self.assertFalse(torch.any(module.last_routing["conditional_mask"]))

        (output.mean() + rcl_loss).backward()
        self.assertIsNotNone(module.unconditional_expert.w1.weight.grad)
        self.assertIsNotNone(module.routed_experts[0].w1.weight.grad)

    def test_inactive_routed_expert_has_a_valid_zero_gradient(self):
        module = self._module(num_routed_experts=3)
        with torch.no_grad():
            module.prototypes.zero_()
            module.prototypes[0, 0] = 1.0
            module.prototypes[1:, 0] = -1.0
        hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(1, 4, 1).requires_grad_()
        output, rcl_loss = module(hidden, torch.tensor([9]))
        self.assertEqual(torch.unique(module.last_routing["expert_indices"]).tolist(), [0])
        self.assertEqual(rcl_loss.item(), 0.0)
        (output.square().mean() + rcl_loss).backward()
        inactive_gradient = module.routed_experts[1].w1.weight.grad
        self.assertIsNotNone(inactive_gradient)
        self.assertEqual(inactive_gradient.abs().sum().item(), 0.0)


class PixDiTProMoETests(unittest.TestCase):
    def test_256px_forward_and_alternating_patch_blocks(self):
        model = PixDiTProMoE(
            in_channels=3,
            patch_size=16,
            num_groups=6,
            hidden_size=24,
            patch_depth=4,
            pixel_depth=1,
            pixel_hidden_size=16,
            num_classes=1000,
            num_routed_experts=3,
            expert_intermediate_size=48,
        )
        self.assertEqual([block.use_promoe for block in model.patch_blocks], [False, True, False, True])
        self.assertTrue(all(not isinstance(block.mlp, ProMoEFeedForward) for block in model.pixel_blocks))

        prediction, rcl_loss = model(
            torch.randn(2, 3, 256, 256),
            torch.rand(2),
            torch.tensor([2, 1000]),
            return_aux_loss=True,
        )
        self.assertEqual(prediction.shape, (2, 3, 256, 256))
        self.assertTrue(torch.isfinite(rcl_loss))

    def test_all_production_configs_have_256px_meta_forward_and_expected_layout(self):
        for size, (hidden_size, num_heads, patch_depth, pixel_depth, intermediate) in MODEL_SPECS.items():
            with self.subTest(size=size):
                config = yaml.safe_load((CONFIG_DIR / f"pix256_promoe_{size}.yaml").read_text())
                init_args = config["model"]["denoiser"]["init_args"]
                self.assertEqual(init_args["hidden_size"], hidden_size)
                self.assertEqual(init_args["num_groups"], num_heads)
                self.assertEqual(init_args["patch_depth"], patch_depth)
                self.assertEqual(init_args["pixel_depth"], pixel_depth)
                self.assertEqual(init_args["expert_intermediate_size"], intermediate)
                self.assertEqual(init_args["in_channels"], 3)
                self.assertEqual(init_args["patch_size"], 16)
                self.assertEqual(init_args["pixel_hidden_size"], 16)

                with torch.device("meta"):
                    model = PixDiTProMoE(**init_args)
                    prediction = model(
                        torch.empty(1, 3, 256, 256, device="meta"),
                        torch.empty(1, device="meta"),
                        torch.zeros(1, dtype=torch.long, device="meta"),
                        s=torch.empty(1, 256, hidden_size, device="meta"),
                    )
                self.assertEqual(prediction.shape, (1, 3, 256, 256))
                self.assertEqual(model.num_promoe_patch_blocks, patch_depth // 2)
                self.assertEqual(
                    [block.use_promoe for block in model.patch_blocks],
                    [index % 2 == 1 for index in range(patch_depth)],
                )


if __name__ == "__main__":
    unittest.main()
