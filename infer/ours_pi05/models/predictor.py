"""真机 pi0.5 版的 latent 预测器。

与 ours_mainline（LIBERO/SmolVLA）的差异 —— 见 spec §7.3：
  1. 维度：streams 2->3, tokens 64->256 (16x16), latent 960->2048,
          action 7->14, state 8->14。
  2. trunk 加深 rrcccc -> rrccccccc：只有 DepthwiseConvBlock 混合 token，
     4 个 conv 的 RF = 9x9，在 LIBERO 的 8x8 上是全局的，在 16x16 上只覆盖 32%；
     而 flow 范围是 ±1.0 归一化坐标 = ±7.5 token。7 个 conv -> RF 15x15，覆盖 16x16。
  3. hidden 256->384, dec_hidden 384->512：token_encoder 的压缩比从 8:1 回到 ~5:1。
  4. per-stream context pooling（三路相机统计量差异大，全局均值池化太粗）。
  5. 双臂 ego 路由：左腕的自运动只由左臂 7 维决定，右腕只由右臂决定，
     cam_high 是静态相机无自运动；但另一条臂仍会作为**画面内容**入画，
     所以 motion_full（全 14 维）依然加到每一路。

identity-start 初始化原样保留：训练从「原样拷贝陈旧 latent」起步。

维度常量在这里本地定义，不从 ours_pi05.openpi_bridge 导入 —— 那个模块拉 JAX/openpi，
会让这个纯 torch 模块依赖 GPU 栈。两边靠 test_bridge.py 之外的手工核对保持一致
（LATENT_SHAPE=(3,256,2048), TOKEN_GRID_SIDE=16）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 以下 6 个组件从 ours_mainline/models/predictor.py:1-191 逐字复制 ----
# （与维度无关，不需要任何修改）


def build_action_trajectory_features(
    motion_actions: torch.Tensor,
    delay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an explicit control-space residual trajectory from executed actions."""
    if motion_actions.ndim != 3:
        raise ValueError(f"motion_actions must have shape [B, K, A], got {tuple(motion_actions.shape)}")
    if delay.ndim != 1 or delay.shape[0] != motion_actions.shape[0]:
        raise ValueError(f"delay must have shape [B], got {tuple(delay.shape)}")
    if bool((delay <= 0).any()):
        raise ValueError("motion-prior predictor requires positive delay; d=0 must bypass it")
    if bool((delay > motion_actions.shape[1]).any()):
        raise ValueError(
            f"motion_actions length {motion_actions.shape[1]} is too short for delays "
            f"{delay.detach().cpu().tolist()}"
        )

    step_ids = torch.arange(motion_actions.shape[1], device=motion_actions.device)
    valid_mask = step_ids[None, :] < delay[:, None]
    actions = motion_actions * valid_mask[:, :, None].to(motion_actions.dtype)
    previous_actions = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)
    action_change = (actions - previous_actions) * valid_mask[:, :, None].to(actions.dtype)
    absolute_residual = torch.cumsum(actions, dim=1)
    path_magnitude = torch.cumsum(actions.abs(), dim=1)
    progress = (step_ids[None, :] + 1).to(actions.dtype) / delay[:, None].to(actions.dtype)
    progress = progress.clamp(max=1.0) * valid_mask.to(actions.dtype)
    features = torch.cat(
        [
            actions,
            absolute_residual,
            action_change,
            path_magnitude,
            progress[:, :, None],
            valid_mask[:, :, None].to(actions.dtype),
        ],
        dim=-1,
    )
    return features, valid_mask, absolute_residual


def token_grid_positions(token_count: int, *, device, dtype) -> torch.Tensor:
    """Return normalized 2D positions for square image tokens, or a 1D fallback."""
    if token_count <= 0:
        raise ValueError(f"token_count must be positive, got {token_count}")
    side = int(token_count**0.5)
    if side * side == token_count:
        axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    x = torch.linspace(-1.0, 1.0, token_count, device=device, dtype=dtype)
    return torch.stack([x, torch.zeros_like(x)], dim=-1)


class TemporalSelfAttentionBlock(nn.Module):
    """Small residual self-attention block for short action trajectories."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, steps, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        logits = torch.einsum("bthd,buhd->bhtu", q, k) / (self.head_dim**0.5)
        logits = logits.masked_fill(~valid_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhtu,buhd->bthd", weights, v).reshape(batch, steps, hidden)
        x = x + self.out(attended)
        x = x + self.ffn(self.norm2(x))
        return x * valid_mask[:, :, None].to(x.dtype)


class PerTokenResidualBlock(nn.Module):
    """Per-token pre-norm residual FFN block (no token mixing). Accepts (and ignores) a valid_mask so
    it drops into the same block loop as the attention block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class DepthwiseConvBlock(nn.Module):
    """Depthwise-separable conv block over each camera's HxW token grid (local spatial mixing within a
    camera view). Requires tokens_per_stream to be a perfect square."""

    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                            padding=kernel_size // 2, groups=hidden_dim)
        self.pw = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.act = nn.SiLU()

    def forward(self, seq: torch.Tensor, streams: int, tokens: int) -> torch.Tensor:
        b, n, h = seq.shape
        side = int(tokens ** 0.5)
        if side * side != tokens:
            raise ValueError(f"DepthwiseConvBlock needs a square token grid, got tokens={tokens}")
        x = self.norm(seq)
        x = x.reshape(b, streams, side, side, h).permute(0, 1, 4, 2, 3).reshape(b * streams, h, side, side)
        x = self.act(self.pw(self.dw(x)))
        x = x.reshape(b, streams, h, side, side).permute(0, 1, 3, 4, 2).reshape(b, streams * tokens, h)
        return seq + x


class ActionTrajectoryEncoder(nn.Module):
    """Encode explicit residual trajectories with lightweight temporal attention."""

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        max_delay: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        feature_dim = action_dim * 4 + 2
        self.delay_embedding = nn.Embedding(max_delay + 1, hidden_dim)
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [TemporalSelfAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        motion_actions: torch.Tensor,
        delay: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, valid_mask, absolute_residual = build_action_trajectory_features(motion_actions, delay)
        x = self.input_mlp(features)
        x = x * valid_mask[:, :, None].to(x.dtype)
        for block in self.blocks:
            x = block(x, valid_mask)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = (x * valid_mask[:, :, None].to(x.dtype)).sum(dim=1) / denom
        batch_indices = torch.arange(x.shape[0], device=x.device)
        endpoint = x[batch_indices, delay - 1]
        motion = self.summary_mlp(torch.cat([pooled, endpoint], dim=-1))
        motion = motion + self.delay_embedding(delay)
        return motion, {
            "action_features": features,
            "action_mask": valid_mask,
            "absolute_residual_trajectory": absolute_residual,
        }


# ---- 结束逐字复制段 ----

TRUNK_SPEC = "rrccccccc"  # LIBERO: "rrcccc" —— 16x16 上需要 7 个 conv 才近似全局 (RF 15x15)
DECODER_SPEC = "rrc"

# 与 ours_pi05.openpi_bridge.LATENT_SHAPE=(3,256,2048) 保持一致；不在这里导入 openpi_bridge
# （会拉 JAX），靠 tests/test_predictor.py::test_predictor_dims_match_bridge 手工核对。
NUM_STREAMS = 3
TOKENS = 256
LATENT_DIM = 2048

LEFT_ARM_DIMS = tuple(range(0, 7))
RIGHT_ARM_DIMS = tuple(range(7, 14))
# stream 顺序与 openpi_bridge.STREAM_KEYS 一致：
#   0 = base_0_rgb        (cam_high，静态外视角 -> 无自运动)
#   1 = left_wrist_0_rgb  (由左臂驱动)
#   2 = right_wrist_0_rgb (由右臂驱动)
STREAM_EGO_MASK: tuple[tuple[int, ...], ...] = ((), LEFT_ARM_DIMS, RIGHT_ARM_DIMS)


class MotionPriorLatentPredictor(nn.Module):
    """真机 pi0.5 版：action-conditioned warp token transport + residual innovation,
    with init-frame context, 14-dim proprio-state conditioning, per-stream latent
    context pooling, and (optional) bimanual ego-motion routing."""

    def __init__(self, ego_routing: bool = True):
        super().__init__()
        latent_dim, action_dim, num_streams, max_delay = LATENT_DIM, 14, NUM_STREAMS, 20
        hidden_dim = action_hidden_dim = 384      # LIBERO: 256
        dec_hidden = 512                          # LIBERO: 384
        action_num_heads, action_num_layers = 4, 2
        max_transport_offset = 1.0
        max_gain_delta = 1.0
        interaction_gate_bias = -4.0

        self.ego_routing = ego_routing
        self.num_streams = num_streams
        self.max_gain_delta = max_gain_delta
        self.max_transport_offset = max_transport_offset
        self.state_dim = 14
        self.action_dim = action_dim
        self.max_delay = max_delay

        self.stream_embedding = nn.Embedding(num_streams, hidden_dim)
        self.frame_role_embedding = nn.Embedding(2, hidden_dim)  # 0=stale, 1=init
        self.action_trajectory_encoder = ActionTrajectoryEncoder(
            action_dim=action_dim,
            hidden_dim=action_hidden_dim,
            max_delay=max_delay,
            num_heads=action_num_heads,
            num_layers=action_num_layers,
        )
        # per-stream context pooling（LIBERO 是 z.mean(dim=(1,2)) 的单一全局向量）
        self.latent_context_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.SiLU()
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim), nn.Linear(self.state_dim, hidden_dim), nn.SiLU()
        )
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.SiLU()
        )

        trunk = []
        for ch in TRUNK_SPEC:
            trunk.append(PerTokenResidualBlock(hidden_dim) if ch == "r" else DepthwiseConvBlock(hidden_dim))
        self.token_attention = nn.ModuleList(trunk)

        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.transport_flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.transport_strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.transport_gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.transport_delta_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.SiLU()
        )

        self.residual_in_proj = nn.Linear(hidden_dim * 3, dec_hidden)
        dec = []
        for ch in DECODER_SPEC:
            dec.append(PerTokenResidualBlock(dec_hidden) if ch == "r" else DepthwiseConvBlock(dec_hidden))
        self.residual_decoder = nn.ModuleList(dec)
        self.residual_out_head = nn.Linear(dec_hidden, latent_dim + 1)

        # --- identity-start init（与 ours_mainline 一致，必须保留）---
        nn.init.zeros_(self.transport_flow_head[-1].weight)
        nn.init.zeros_(self.transport_flow_head[-1].bias)
        nn.init.zeros_(self.transport_strength_head[-1].weight)
        nn.init.constant_(self.transport_strength_head[-1].bias, -6.0)
        nn.init.zeros_(self.transport_gain_head[-1].weight)
        nn.init.zeros_(self.transport_gain_head[-1].bias)   # tanh(0)=0 -> gain 1
        nn.init.zeros_(self.residual_out_head.weight)
        nn.init.zeros_(self.residual_out_head.bias)
        nn.init.constant_(self.residual_out_head.bias[-1:], interaction_gate_bias)

    def _ego_motion(self, motion_actions: torch.Tensor, delay: torch.Tensor, stream: int) -> torch.Tensor:
        """把不相关那条臂的维度置零后，再过一遍 action encoder。"""
        dims = STREAM_EGO_MASK[stream]
        masked = torch.zeros_like(motion_actions)
        if dims:
            idx = list(dims)
            masked[..., idx] = motion_actions[..., idx]
        motion, _ = self.action_trajectory_encoder(masked, delay)
        return motion

    def forward(
        self,
        z: torch.Tensor,
        motion_actions: torch.Tensor,
        delay: torch.Tensor,
        *,
        z_init: torch.Tensor,
        state: torch.Tensor,
        return_aux: bool = False,
    ):
        if z.ndim != 4:
            raise ValueError(f"z must be [B, C, T, D], got {tuple(z.shape)}")
        if z.shape[0] != motion_actions.shape[0] or z.shape[0] != delay.shape[0]:
            raise ValueError("z, motion_actions, and delay must use the same batch size")
        if z.shape[1] != self.num_streams:
            raise ValueError(f"expected {self.num_streams} streams, got {z.shape[1]}")
        if z_init.shape != z.shape:
            raise ValueError(f"z_init must match z shape {tuple(z.shape)}, got {tuple(z_init.shape)}")
        if int(delay.min()) < 1:
            raise ValueError("delay must be >= 1; d=0 必须由调用方绕过预测器（直接用真实 obs）")
        if int(delay.max()) > motion_actions.shape[1]:
            raise ValueError(
                f"delay {int(delay.max())} 超过了 motion_actions 的步数 {motion_actions.shape[1]}"
            )
        if int(delay.max()) > self.max_delay:
            raise ValueError(f"delay {int(delay.max())} 超过 max_delay={self.max_delay}")

        batch, streams, tokens, dim = z.shape

        motion_full, _ = self.action_trajectory_encoder(motion_actions, delay)  # [B, hidden]
        stream_ids = torch.arange(streams, device=z.device)
        stream_condition = self.stream_embedding(stream_ids)[None, :, :]        # [1, C, hidden]

        # per-stream latent context（LIBERO 是单一全局池化）
        stream_context = self.latent_context_encoder(z.mean(dim=2))             # [B, C, hidden]

        condition = motion_full[:, None, :] + stream_condition + stream_context
        condition = condition + self.state_encoder(state.to(z.dtype))[:, None, :]

        if self.ego_routing:
            ego = torch.stack(
                [self._ego_motion(motion_actions, delay, s) for s in range(streams)], dim=1
            )  # [B, C, hidden]
            condition = condition + ego

        positions = token_grid_positions(tokens, device=z.device, dtype=z.dtype)
        pos_embed = self.position_encoder(positions)[None, None]
        role = self.frame_role_embedding.weight

        token_hidden = self.token_encoder(z) + condition[:, :, None, :] + pos_embed + role[0]
        init_token = self.token_encoder(z_init)
        init_hidden = init_token + stream_condition[:, :, None, :] + pos_embed + role[1]
        token_hidden = torch.cat([token_hidden, init_hidden], dim=1)  # [B, 2C, T, hidden]

        b, s2, t, h = token_hidden.shape
        seq = token_hidden.reshape(b, s2 * t, h)
        valid = seq.new_ones(seq.shape[:2], dtype=torch.bool)
        for block in self.token_attention:
            seq = block(seq, s2, t) if isinstance(block, DepthwiseConvBlock) else block(seq, valid)
        token_hidden = seq.reshape(b, s2, t, h)[:, :streams]  # 丢掉 init 流

        # --- warp transport ---
        flow = torch.tanh(self.transport_flow_head(token_hidden)) * self.max_transport_offset
        transport_strength = torch.sigmoid(self.transport_strength_head(token_hidden))
        gain = 1.0 + self.max_gain_delta * torch.tanh(self.transport_gain_head(token_hidden))

        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError(f"warp needs a square token grid, got tokens={tokens}")
        base_grid = positions.reshape(side, side, 2)
        sample_grid = base_grid[None] - flow.reshape(batch * streams, side, side, 2)
        z_map = z.reshape(batch * streams, side, side, dim).permute(0, 3, 1, 2)
        z_warp = F.grid_sample(
            z_map, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        z_warp = z_warp.permute(0, 2, 3, 1).reshape(batch, streams, tokens, dim)
        z_transport = (1.0 - transport_strength) * z + transport_strength * (gain * z_warp)

        # --- innovation residual ---
        transport_delta_hidden = self.transport_delta_encoder(z_transport - z)
        dec = self.residual_in_proj(
            torch.cat([token_hidden, transport_delta_hidden, init_token], dim=-1)
        )
        db, ds, dt_, dh = dec.shape
        seq = dec.reshape(db, ds * dt_, dh)
        valid = seq.new_ones(seq.shape[:2], dtype=torch.bool)
        for block in self.residual_decoder:
            seq = block(seq, ds, dt_) if isinstance(block, DepthwiseConvBlock) else block(seq, valid)
        out = self.residual_out_head(seq.reshape(db, ds, dt_, dh))
        residual = out[..., :-1]
        gate = torch.sigmoid(out[..., -1:])
        z_hat = z_transport + gate * residual

        if not return_aux:
            return z_hat
        return z_hat, {
            "flow": flow,
            "transport_strength": transport_strength,
            "gain": gain,
            "interaction_gate": gate,
            "z_transport": z_transport,
        }
