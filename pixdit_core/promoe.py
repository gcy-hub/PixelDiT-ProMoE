"""PixelDiT patch-level ProMoE feed-forward network.

The module keeps PixelDiT's SwiGLU ``FeedForward`` implementation and only
changes how its patch-level FFN is selected. Conditional tokens are routed by
raw cosine similarity to learned prototypes; CFG-null tokens use a dedicated
unconditional expert. A shared expert is always active.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import FeedForward


class ProMoEFeedForward(nn.Module):
    """Top-1 prototype-routed SwiGLU experts for a PixelDiT patch block.

    ``expert_intermediate_size`` is the pre-SwiGLU intermediate width passed
    to PixelDiT's ``FeedForward``. With width ``2D``, one routed expert and
    the shared expert together match the dense FFN activation cost.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_routed_experts: int = 12,
        null_label: int = 1000,
        rcl_temperature: float = 0.07,
        rcl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if num_routed_experts < 1:
            raise ValueError("num_routed_experts must be positive")
        if rcl_temperature <= 0:
            raise ValueError("rcl_temperature must be positive")

        self.hidden_size = int(hidden_size)
        self.expert_intermediate_size = int(expert_intermediate_size)
        self.num_routed_experts = int(num_routed_experts)
        self.null_label = int(null_label)
        self.rcl_temperature = float(rcl_temperature)
        self.rcl_weight = float(rcl_weight)

        self.prototypes = nn.Parameter(torch.empty(self.num_routed_experts, self.hidden_size))
        self.routed_experts = nn.ModuleList(
            [
                FeedForward(self.hidden_size, self.expert_intermediate_size)
                for _ in range(self.num_routed_experts)
            ]
        )
        self.shared_expert = FeedForward(self.hidden_size, self.expert_intermediate_size)
        self.unconditional_expert = FeedForward(self.hidden_size, self.expert_intermediate_size)

        # Lightweight diagnostics used by focused tests and startup debugging.
        self.last_routing: Dict[str, torch.Tensor] = {}
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def _route_conditional_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Top-1 expert indices and raw cosine routed weights.

        No softmax is applied before or after Top-K. The selected cosine value
        directly weights the routed expert output.
        """
        token_norm = F.normalize(tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)
        cosine = token_norm @ prototype_norm.transpose(0, 1)
        weights, indices = torch.topk(cosine, k=1, dim=-1, sorted=False)
        return indices.squeeze(-1), weights.squeeze(-1)

    def _zero_parameter_connection(self, reference: torch.Tensor) -> torch.Tensor:
        """Keep inactive experts in DDP's autograd graph without computing them.

        Top-1 routing naturally leaves many experts inactive in a batch. The
        zero-valued connection gives those parameters zero gradients so normal
        DDP can run without ``find_unused_parameters=True``.
        """
        zero = reference.new_zeros(())
        zero = zero + self.prototypes.reshape(-1)[0].to(reference.dtype) * 0
        for expert in self.routed_experts:
            for parameter in expert.parameters():
                zero = zero + parameter.reshape(-1)[0].to(reference.dtype) * 0
        for parameter in self.unconditional_expert.parameters():
            zero = zero + parameter.reshape(-1)[0].to(reference.dtype) * 0
        return zero

    def _routing_contrastive_loss(
        self,
        conditional_tokens: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Contrast active prototype rows against batch token centroids.

        Experts with no conditional tokens are excluded. A single active expert
        has no contrastive negative, so it contributes a differentiable zero.
        """
        active_experts = torch.unique(expert_indices, sorted=True)
        if active_experts.numel() < 2:
            return self.prototypes.sum() * 0

        centroids = []
        for expert_index in active_experts:
            centroids.append(conditional_tokens[expert_indices == expert_index].mean(dim=0))
        centroid_matrix = torch.stack(centroids, dim=0)
        active_prototypes = self.prototypes.index_select(0, active_experts)
        logits = F.normalize(active_prototypes, p=2, dim=-1) @ F.normalize(
            centroid_matrix, p=2, dim=-1
        ).transpose(0, 1)
        targets = torch.arange(active_experts.numel(), device=logits.device)
        return F.cross_entropy(logits / self.rcl_temperature, targets) * self.rcl_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("ProMoEFeedForward expects [batch, tokens, hidden] inputs")
        if labels.ndim != 1 or labels.shape[0] != hidden_states.shape[0]:
            raise ValueError("labels must have one class id per batch item")

        batch_size, token_count, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        flat_tokens = hidden_states.reshape(-1, hidden_size)
        flat_labels = labels.reshape(batch_size, 1).expand(-1, token_count).reshape(-1).long()
        conditional_mask = flat_labels != self.null_label
        routed_output = torch.zeros_like(flat_tokens)
        all_indices = torch.full(
            (flat_tokens.shape[0],),
            self.num_routed_experts,
            device=hidden_states.device,
            dtype=torch.long,
        )
        all_weights = torch.ones(flat_tokens.shape[0], device=hidden_states.device, dtype=hidden_states.dtype)

        if conditional_mask.any():
            conditional_tokens = flat_tokens[conditional_mask]
            conditional_indices, conditional_weights = self._route_conditional_tokens(conditional_tokens)
            all_indices[conditional_mask] = conditional_indices
            all_weights[conditional_mask] = conditional_weights.to(hidden_states.dtype)

            conditional_output = torch.zeros_like(conditional_tokens)
            for expert_index, expert in enumerate(self.routed_experts):
                token_mask = conditional_indices == expert_index
                if token_mask.any():
                    expert_output = expert(conditional_tokens[token_mask]).to(conditional_output.dtype)
                    conditional_output[token_mask] = expert_output * conditional_weights[token_mask].unsqueeze(-1)
            routed_output[conditional_mask] = conditional_output
            rcl_loss = (
                self._routing_contrastive_loss(conditional_tokens, conditional_indices)
                if return_aux_loss
                else self.prototypes.sum() * 0
            )
        else:
            rcl_loss = self.prototypes.sum() * 0

        unconditional_mask = ~conditional_mask
        if unconditional_mask.any():
            routed_output[unconditional_mask] = self.unconditional_expert(
                flat_tokens[unconditional_mask]
            ).to(routed_output.dtype)

        output = routed_output.view_as(hidden_states) + self.shared_expert(hidden_states).to(hidden_states.dtype)
        output = output + self._zero_parameter_connection(output)

        self.last_routing = {
            "expert_indices": all_indices.view(batch_size, token_count).detach(),
            "weights": all_weights.view(batch_size, token_count).detach(),
            "conditional_mask": conditional_mask.view(batch_size, token_count).detach(),
        }
        return output, rcl_loss
