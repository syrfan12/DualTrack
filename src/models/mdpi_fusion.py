"""Fusion-only transfer of DualTrack to the nominal MDPI Y/Z trajectory task.

The original six-component checkpoint is loaded strictly before its head is
reduced to the two supervised translations. Complete local/global encoders,
including projections, positional embeddings and BatchNorm buffers, remain
frozen. No six-DoF or calibrated probe-pose claim is implied by this wrapper.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn


MODEL_SCHEMA = "dualtrack_mdpi_fusion_2d_v1"
PAIR_INDICES = torch.tensor(
    [[i, j] for j in range(1, 10) for i in range(j)], dtype=torch.long
)


def _check_coordinates(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[-1] != 2:
        raise ValueError(f"{name} must be a tensor with shape [..., time, 2]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating point dtype")


def steps_to_positions(steps: torch.Tensor) -> torch.Tensor:
    """Integrate adjacent [delta_y, delta_z] steps with an initial zero anchor."""
    _check_coordinates(steps, "steps")
    if steps.shape[-2] < 1:
        raise ValueError("At least one adjacent step is required")
    return torch.cat((torch.zeros_like(steps[..., :1, :]), steps.cumsum(dim=-2)), dim=-2)


def targets_to_pairs(positions_yz: torch.Tensor) -> torch.Tensor:
    """Return p_j - p_i in the frozen 45-pair order for ten selected frames."""
    _check_coordinates(positions_yz, "positions_yz")
    if positions_yz.shape[-2] != 10:
        raise ValueError("MDPI pair supervision requires exactly ten positions")
    pairs = PAIR_INDICES.to(positions_yz.device)
    return positions_yz[..., pairs[:, 1], :] - positions_yz[..., pairs[:, 0], :]


def pair_translations(steps: torch.Tensor) -> torch.Tensor:
    """Build all 45 pair translations from nine steps without breaking autograd."""
    return targets_to_pairs(steps_to_positions(steps))


def pair_mse_loss(steps: torch.Tensor, positions_yz: torch.Tensor) -> torch.Tensor:
    """MSE averaged over pairs and the two supervised coordinates, in nominal mm²."""
    prediction, reference = pair_translations(steps), targets_to_pairs(positions_yz)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference batch dimensions must match")
    return torch.nn.functional.mse_loss(prediction, reference)


class MDPIFusionModel(nn.Module):
    """Train the existing fusion module and a two-output head, and nothing else.

    ``dualtrack`` can be a small compatible module for scope-verification tests.
    In production, construct this wrapper using :func:`build_mdpi_fusion`.
    """

    schema = MODEL_SCHEMA

    def __init__(self, dualtrack: nn.Module):
        super().__init__()
        for name in ("local_encoder", "global_encoder", "fusion_module", "head"):
            if not isinstance(getattr(dualtrack, name, None), nn.Module):
                raise ValueError(f"DualTrack must expose a {name} module")
        if not isinstance(getattr(dualtrack.head, "fc", None), nn.Linear):
            raise ValueError("Expected a linear DualTrack head.fc")
        self.dualtrack = dualtrack
        source_head = dualtrack.head.fc
        if source_head.out_features == 6:
            head = nn.Linear(
                source_head.in_features,
                2,
                bias=source_head.bias is not None,
                device=source_head.weight.device,
                dtype=source_head.weight.dtype,
            )
            with torch.no_grad():
                head.weight.copy_(source_head.weight[[1, 2]])
                if source_head.bias is not None:
                    head.bias.copy_(source_head.bias[[1, 2]])
            dualtrack.head.fc = head
        elif source_head.out_features != 2:
            raise ValueError("Expected a six-component source head or two-component MDPI head")
        self.head_initialization = "pretrained_head_translation_rows_1_2"
        for parameter in self.dualtrack.parameters():
            parameter.requires_grad_(False)
        for module in (dualtrack.fusion_module, dualtrack.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True) -> "MDPIFusionModel":
        super().train(mode)
        # requires_grad=False alone does not freeze BatchNorm or disable dropout.
        self.dualtrack.local_encoder.eval()
        self.dualtrack.global_encoder.eval()
        return self

    def forward(
        self, global_encoder_images: torch.Tensor, local_encoder_images: torch.Tensor
    ) -> torch.Tensor:
        if (
            global_encoder_images.ndim != 5
            or local_encoder_images.ndim != 5
            or global_encoder_images.shape[:2] != local_encoder_images.shape[:2]
            or global_encoder_images.shape[1] < 2
        ):
            raise ValueError("Expected matching [batch, time, channel, height, width] inputs")
        # Enforce the boundary even if a caller toggles a child module directly.
        self.dualtrack.local_encoder.eval()
        self.dualtrack.global_encoder.eval()
        with torch.no_grad():
            local_features = self.dualtrack.local_encoder(local_encoder_images)
            indices = self.dualtrack.sampler(global_encoder_images, local_features)
            batch = torch.arange(global_encoder_images.shape[0], device=indices.device)[:, None]
            sparse_images = global_encoder_images[batch, indices].contiguous()
            global_features = self.dualtrack.global_encoder(sparse_images, indices)
        # Keep this outside no_grad: trainable fusion must receive gradients.
        fused = self.dualtrack.fusion_module(
            local_features, encoder_hidden_states=global_features
        )
        result = self.dualtrack.head(fused)
        expected = (*local_encoder_images.shape[:1], local_encoder_images.shape[1] - 1, 2)
        if result.shape != expected:
            raise ValueError(f"Unexpected MDPI head shape {tuple(result.shape)}; expected {expected}")
        return result

    def trainable_parameter_groups(
        self, fusion_lr: float = 1e-5, head_lr: float = 1e-4
    ) -> list[dict[str, Any]]:
        if fusion_lr <= 0 or head_lr <= 0:
            raise ValueError("Learning rates must be positive")
        expected = {
            id(parameter)
            for module in (self.dualtrack.fusion_module, self.dualtrack.head)
            for parameter in module.parameters()
        }
        actual = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if actual != expected:
            raise RuntimeError("Trainable parameters no longer match fusion + head scope")
        return [
            {"params": list(self.dualtrack.fusion_module.parameters()), "lr": fusion_lr, "name": "fusion"},
            {"params": list(self.dualtrack.head.parameters()), "lr": head_lr, "name": "head"},
        ]

    def scope_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "model_schema": MODEL_SCHEMA,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "module_parameters": {
                name: sum(parameter.numel() for parameter in getattr(self.dualtrack, name).parameters())
                for name in ("local_encoder", "global_encoder", "fusion_module", "head")
            },
            "trainable_names": [name for name, parameter in self.named_parameters() if parameter.requires_grad],
            "frozen_modules": ["dualtrack.local_encoder", "dualtrack.global_encoder"],
            "frozen_encoders_in_eval_mode": not (
                self.dualtrack.local_encoder.training or self.dualtrack.global_encoder.training
            ),
            "head_initialization": self.head_initialization,
        }


def build_mdpi_fusion(
    checkpoint_path: str | Path,
    grid_spacing: int = 2,
    device: str | torch.device = "cpu",
) -> MDPIFusionModel:
    """Strictly restore original weights, then adapt the head and freezing scope.

    Historical initialization paths embedded in the source checkpoint are not
    followed. This builder supports the supplied legacy CNN DualTrack checkpoint;
    it does not download foundation-model or image-backbone weights.
    """
    if isinstance(grid_spacing, bool) or not isinstance(grid_spacing, int) or grid_spacing < 1:
        raise ValueError("grid_spacing must be a positive integer")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected the original DualTrack checkpoint with a model state dictionary")
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint["config"].get("model"), dict):
        raise ValueError("Original checkpoint must contain its model configuration")
    cfg = copy.deepcopy(checkpoint["config"]["model"])
    if (
        cfg.get("name") != "dualtrack_fusion_model"
        or cfg.get("global_encoder_cfg", {}).get("name") != "global_encoder_cnn"
        or cfg.get("local_encoder_cfg", {}).get("name") != "dualtrack_loc_enc_stg3_legacy"
        or cfg.get("disable_global_encoder", False)
    ):
        raise ValueError("Expected the legacy CNN local/global DualTrack fusion architecture")
    # Reject other embedded initialization files instead of silently loading them.
    def remove_initialization_paths(node: dict[str, Any]) -> None:
        for key in ("checkpoint", "backbone_weights", "backbone_checkpoint", "load_kw", "use_cache"):
            node.pop(key, None)
        for value in node.values():
            if isinstance(value, dict):
                remove_initialization_paths(value)

    remove_initialization_paths(cfg)
    cfg["global_encoder_cfg"]["position_embedding_type"] = "absolute"
    cfg["grid_spacing"] = grid_spacing
    from src.models import get_model

    dualtrack = get_model(**cfg)
    dualtrack.load_state_dict(checkpoint["model"], strict=True)
    model = MDPIFusionModel(dualtrack)
    model.model_config = cfg
    model.pretrained_checkpoint = str(checkpoint_path)
    model.strict_pretrained_load = True
    return model.to(device)
