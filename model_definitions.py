# -*- coding: utf-8 -*-
"""Mamba model definitions matching the three training checkpoints."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Pure-PyTorch selective state-space block with [B, L, D] tensors."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, (d_model + 15) // 16)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.depthwise_conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a.unsqueeze(0).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._init_dt()

    def _init_dt(self) -> None:
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_scan = x.float()
        delta_scan = delta.float()
        b_scan = b_param.float()
        c_scan = c_param.float()
        a = -torch.exp(self.A_log.float())
        d = self.D.float()

        state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []
        for step in range(seq_len):
            dt = delta_scan[:, step, :]
            x_step = x_scan[:, step, :]
            b_step = b_scan[:, step, :]
            c_step = c_scan[:, step, :]
            discrete_a = torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0))
            discrete_bx = (
                dt.unsqueeze(-1)
                * b_step.unsqueeze(1)
                * x_step.unsqueeze(-1)
            )
            state = discrete_a * state + discrete_bx
            y_step = (state * c_step.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_step + d.unsqueeze(0) * x_step)

        return torch.stack(outputs, dim=1).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        seq_len = x.size(1)
        x = self.depthwise_conv(x.transpose(1, 2))[:, :, :seq_len]
        x = F.silu(x.transpose(1, 2))

        ssm_params = self.x_proj(x)
        dt_raw, b_param, c_param = torch.split(
            ssm_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        y = self._selective_scan(x, delta, b_param, c_param)
        y = y * F.silu(gate)
        return residual + self.dropout(self.out_proj(y))


class BidirectionalMambaLayer(nn.Module):
    """Fuse forward and backward Mamba scans."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        conv_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.forward_mamba = MambaBlock(
            d_model, d_state, expand, conv_kernel, dropout
        )
        self.backward_mamba = MambaBlock(
            d_model, d_state, expand, conv_kernel, dropout
        )
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_feat = self.forward_mamba(x)
        backward_feat = torch.flip(
            self.backward_mamba(torch.flip(x, dims=[1])),
            dims=[1],
        )
        merged = self.merge(torch.cat([forward_feat, backward_feat], dim=-1))
        return self.norm(x + self.dropout(merged))


class BidirectionalAxialMambaFeatureExtractor(nn.Module):
    """Two-dimensional row/column Mamba shared by Tasks 1 and 2."""

    def __init__(
        self,
        num_rows: int = 30,
        num_cols: int = 38,
        d_model: int = 96,
        d_state: int = 16,
        num_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.row_projection = nn.Linear(num_cols, d_model)
        self.column_projection = nn.Linear(num_rows, d_model)
        self.row_embedding = nn.Parameter(torch.zeros(1, num_rows, d_model))
        self.column_embedding = nn.Parameter(torch.zeros(1, num_cols, d_model))

        layer_args = (d_model, d_state, expand, conv_kernel, dropout)
        self.row_layers = nn.ModuleList(
            [BidirectionalMambaLayer(*layer_args) for _ in range(num_layers)]
        )
        self.column_layers = nn.ModuleList(
            [BidirectionalMambaLayer(*layer_args) for _ in range(num_layers)]
        )
        self.row_norm = nn.LayerNorm(d_model)
        self.column_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_model, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.SiLU(),
        )
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.column_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.num_rows, self.num_cols)
        if x.ndim != 4 or x.size(1) != 1 or tuple(x.shape[2:]) != expected:
            raise ValueError(
                f"Two-dimensional Mamba input must be [B, 1, "
                f"{self.num_rows}, {self.num_cols}]; received {tuple(x.shape)}."
            )

        matrix = x.squeeze(1)
        row_tokens = self.row_projection(matrix) + self.row_embedding
        column_tokens = (
            self.column_projection(matrix.transpose(1, 2))
            + self.column_embedding
        )
        for layer in self.row_layers:
            row_tokens = layer(row_tokens)
        for layer in self.column_layers:
            column_tokens = layer(column_tokens)

        row_tokens = self.row_norm(row_tokens)
        column_tokens = self.column_norm(column_tokens)
        pooled = torch.cat(
            [
                row_tokens.mean(dim=1),
                row_tokens.amax(dim=1),
                column_tokens.mean(dim=1),
                column_tokens.amax(dim=1),
            ],
            dim=-1,
        )
        return self.fusion(pooled)


class ConditionedPositionForceMamba(nn.Module):
    """Task 1: auxiliary position classification and force regression."""

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        num_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feature_dim = 128
        self.features = BidirectionalAxialMambaFeatureExtractor(
            num_rows=30,
            num_cols=38,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            dropout=dropout,
            output_dim=feature_dim,
        )
        self.position_head = nn.Linear(feature_dim, 5)
        self.position_embedding = nn.Embedding(5, 16)
        self.force_head = nn.Sequential(
            nn.Linear(feature_dim + 16, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        position_logits = self.position_head(feat)
        pos_emb = self.position_embedding(position_id)
        force_z = self.force_head(torch.cat([feat, pos_emb], dim=1))
        return force_z, position_logits

    def forward_auto(
        self,
        x: torch.Tensor,
        manual_position_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict position, then use it for force regression."""
        feat = self.features(x)
        position_logits = self.position_head(feat)
        if manual_position_id is None:
            position_id = position_logits.argmax(dim=1)
        else:
            position_id = torch.full(
                (x.size(0),),
                int(manual_position_id),
                dtype=torch.long,
                device=x.device,
            )
        pos_emb = self.position_embedding(position_id)
        force_z = self.force_head(torch.cat([feat, pos_emb], dim=1))
        return force_z, position_logits, position_id


class StaticMambaClassifier(nn.Module):
    """Task 2: static two-dimensional shape/character classification."""

    def __init__(
        self,
        num_classes: int,
        d_model: int = 96,
        d_state: int = 16,
        num_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        map_rows: int = 30,
        map_cols: int = 38,
    ) -> None:
        super().__init__()
        self.features = BidirectionalAxialMambaFeatureExtractor(
            num_rows=map_rows,
            num_cols=map_cols,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            dropout=dropout,
            output_dim=128,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class BidirectionalTriaxialMambaFeatureExtractor(nn.Module):
    """Task 3 time/height/width triaxial Mamba."""

    def __init__(
        self,
        depth: int = 50,
        height: int = 30,
        width: int = 38,
        d_model: int = 64,
        d_state: int = 12,
        num_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.15,
        output_dim: int = 192,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width

        self.depth_projection = nn.Linear(height * width, d_model)
        self.height_projection = nn.Linear(depth * width, d_model)
        self.width_projection = nn.Linear(depth * height, d_model)
        self.depth_embedding = nn.Parameter(torch.zeros(1, depth, d_model))
        self.height_embedding = nn.Parameter(torch.zeros(1, height, d_model))
        self.width_embedding = nn.Parameter(torch.zeros(1, width, d_model))

        layer_args = (d_model, d_state, expand, conv_kernel, dropout)
        self.depth_layers = nn.ModuleList(
            [BidirectionalMambaLayer(*layer_args) for _ in range(num_layers)]
        )
        self.height_layers = nn.ModuleList(
            [BidirectionalMambaLayer(*layer_args) for _ in range(num_layers)]
        )
        self.width_layers = nn.ModuleList(
            [BidirectionalMambaLayer(*layer_args) for _ in range(num_layers)]
        )
        self.depth_norm = nn.LayerNorm(d_model)
        self.height_norm = nn.LayerNorm(d_model)
        self.width_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(6 * d_model, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.SiLU(),
        )
        nn.init.normal_(self.depth_embedding, std=0.02)
        nn.init.normal_(self.height_embedding, std=0.02)
        nn.init.normal_(self.width_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.depth, self.height, self.width)
        if x.ndim != 5 or x.size(1) != 1 or tuple(x.shape[2:]) != expected:
            raise ValueError(
                f"Triaxial Mamba input must be [B, 1, {self.depth}, "
                f"{self.height}, {self.width}]; received {tuple(x.shape)}."
            )

        volume = x.squeeze(1)
        depth_tokens = (
            self.depth_projection(volume.flatten(2)) + self.depth_embedding
        )
        height_tokens = (
            self.height_projection(volume.permute(0, 2, 1, 3).flatten(2))
            + self.height_embedding
        )
        width_tokens = (
            self.width_projection(volume.permute(0, 3, 1, 2).flatten(2))
            + self.width_embedding
        )

        for layer in self.depth_layers:
            depth_tokens = layer(depth_tokens)
        for layer in self.height_layers:
            height_tokens = layer(height_tokens)
        for layer in self.width_layers:
            width_tokens = layer(width_tokens)

        depth_tokens = self.depth_norm(depth_tokens)
        height_tokens = self.height_norm(height_tokens)
        width_tokens = self.width_norm(width_tokens)
        pooled = torch.cat(
            [
                depth_tokens.mean(dim=1),
                depth_tokens.amax(dim=1),
                height_tokens.mean(dim=1),
                height_tokens.amax(dim=1),
                width_tokens.mean(dim=1),
                width_tokens.amax(dim=1),
            ],
            dim=-1,
        )
        return self.fusion(pooled)


class DynamicMambaClassifier(nn.Module):
    """Task 3: dynamic three-dimensional digit/character classification."""

    def __init__(
        self,
        num_classes: int,
        depth: int = 50,
        height: int = 30,
        width: int = 38,
        d_model: int = 64,
        d_state: int = 12,
        num_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.features = BidirectionalTriaxialMambaFeatureExtractor(
            depth=depth,
            height=height,
            width=width,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            dropout=dropout,
            output_dim=192,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
