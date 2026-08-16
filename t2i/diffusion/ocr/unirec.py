"""Whole-image, differentiable UniRec OCR supervision.

UniRec is frozen, but this module deliberately does not use ``torch.no_grad``
for its forward pass.  Gradients from token cross-entropy must reach the input
image so they can update the PixelDiT flow prediction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class UniRecWholeImageLoss(nn.Module):
    """Teacher-forced OCR token loss on complete predicted images.

    The module takes full images in PixelDiT's ``[-1, 1]`` convention.  It
    never crops to ground-truth polygons; dataset geometry is used only once to
    serialize annotated strings into a stable reading-order transcript.
    """

    def __init__(
        self,
        model_path: str | Path,
        max_tokens: int = 256,
        max_width: int = 960,
        max_height: int = 1408,
        size_divisor: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path).expanduser().resolve()
        self.max_tokens = int(max_tokens)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.size_divisor = int(size_divisor)
        self.dtype = dtype
        if self.max_tokens < 2:
            raise ValueError("ocr_max_tokens must be at least 2.")
        if min(self.max_width, self.max_height, self.size_divisor) <= 0:
            raise ValueError("UniRec image dimensions and divisor must be positive.")

        config_path = self.model_path / "config.json"
        weights_path = self.model_path / "model.pth"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "UniRec OCR model is incomplete. Expected official topdu/unirec-0.1b "
                f"files at {self.model_path}, including config.json and model.pth."
            )

        try:
            # Importing openocr first exposes its bundled ``openrec`` package.
            import openocr  # noqa: F401
            from openrec.modeling.encoders.focalsvtr import FocalSVTR
            from openrec.modeling.unirec_modeling.configuration_unirec import UniRecConfig
            from openrec.modeling.unirec_modeling.modeling_unirec import UniRecForConditionalGenerationNew
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Whole-image OCR loss requires openocr-python==0.1.5. "
                "Install it in the training environment before launching."
            ) from error

        # OpenOCR 0.1.5 predates Transformers' ``smart_apply`` initialization
        # API (used by the 4.52 version required by the existing Gemma stack).
        # Its visual encoder exposes ``_init_weights`` but not the new spelling.
        # This alias is needed only while constructing the model; the official
        # checkpoint below replaces every model parameter strictly afterwards.
        if not hasattr(FocalSVTR, "_initialize_weights"):
            FocalSVTR._initialize_weights = FocalSVTR._init_weights

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        config = UniRecConfig.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = UniRecForConditionalGenerationNew(config=config)

        # The checkpoint is an official, local UniRec artifact. It is not a
        # user-supplied checkpoint loaded from an untrusted source.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "UniRec checkpoint does not match its model definition. "
                f"Missing keys: {missing[:8]}; unexpected keys: {unexpected[:8]}."
            )

        self.pad_token_id = int(config.pad_token_id)
        self.bos_token_id = int(config.bos_token_id)
        self.eos_token_id = int(config.eos_token_id)
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True):
        # PixelDiT calls train() on its own model. Keep OCR deterministic and
        # frozen even when this helper is retained by the outer training loop.
        super().train(False)
        self.model.eval()
        return self

    def _resize_whole_images(self, images: torch.Tensor) -> torch.Tensor:
        """Fit every full image into UniRec's native-size envelope.

        PixelDiT's multiscale batch sampler groups identical image shapes, so a
        single differentiable resize is enough for each batch.  No crop, text
        detector, NMS, or ground-truth coordinate is used here.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] images for OCR, got {tuple(images.shape)}.")
        _, _, height, width = images.shape
        scale = min(1.0, self.max_width / width, self.max_height / height)
        target_height = max(self.size_divisor, int(math.floor(height * scale / self.size_divisor)) * self.size_divisor)
        target_width = max(self.size_divisor, int(math.floor(width * scale / self.size_divisor)) * self.size_divisor)
        if (target_height, target_width) == (height, width):
            return images
        return F.interpolate(images, size=(target_height, target_width), mode="bicubic", align_corners=False)

    def _encode_transcripts(self, transcripts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences: list[list[int]] = []
        kept_indices: list[int] = []
        for index, transcript in enumerate(transcripts):
            text = str(transcript).strip()
            if not text:
                continue
            # Match OpenOCR's UniRecLabelEncode exactly: tokenizer defaults,
            # then an explicit UniRec BOS/EOS wrapper.
            token_ids = self.tokenizer(text)["input_ids"]
            # Match OpenOCR's UniRecLabelEncode: BOS + BPE target + EOS.
            sequence = [self.bos_token_id, *token_ids, self.eos_token_id]
            if len(sequence) > self.max_tokens:
                continue
            sequences.append(sequence)
            kept_indices.append(index)

        if not sequences:
            empty_ids = torch.empty((0, 0), dtype=torch.long, device=self.device)
            empty_indices = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty_ids, empty_indices

        sequence_length = max(len(sequence) for sequence in sequences)
        labels = torch.full(
            (len(sequences), sequence_length), self.pad_token_id, dtype=torch.long, device=self.device
        )
        for row, sequence in enumerate(sequences):
            labels[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=self.device)
        return labels, torch.tensor(kept_indices, dtype=torch.long, device=self.device)

    def forward(self, images: torch.Tensor, transcripts: Sequence[str]) -> tuple[torch.Tensor, int]:
        """Return mean token CE and number of valid full-image transcripts."""
        if len(transcripts) != images.shape[0]:
            raise ValueError("Each OCR image must have exactly one target transcript.")

        labels, kept_indices = self._encode_transcripts(transcripts)
        if labels.numel() == 0:
            return images.sum() * 0.0, 0

        images = images.index_select(0, kept_indices)
        # PixelDiT is trained in [-1, 1]. The analytic x_0 reconstruction can
        # transiently exceed that range, whereas UniRec was trained on
        # normalized image pixels. This remains a whole-image operation.
        images = images.clamp(-1.0, 1.0)
        images = self._resize_whole_images(images).to(dtype=self.dtype)
        # UniRec was trained with ToTensor() followed by Normalize(0.5, 0.5),
        # exactly PixelDiT's image range.  Do not map or crop images again.
        outputs = self.model(
            pixel_values=images,
            decoder_input_ids=labels[:, :-1],
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits.float()
        targets = labels[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss, int(labels.shape[0])
