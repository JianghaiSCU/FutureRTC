"""Action-conditioned visual-latent forecaster: continuous token transport plus a gated innovation.

Given the stale visual latent ``z`` observed ``d`` steps before the handoff, the ``d`` actions
committed since, an estimate of the proprioceptive state at the handoff, and the episode's
first-frame latent, predict the visual latent the policy would have seen at the handoff.

Forward pass:

1. encode the committed action trajectory (temporal attention over an explicit control-space
   residual trajectory) into a motion summary;
2. build a per-stream condition = motion summary + stream embedding + latent context + state;
3. encode tokens, add positional and condition terms, and append the init-frame latent as extra
   streams so the stale tokens can consult the un-occluded initial layout;
4. run the ``rrcccc`` block trunk (per-token residual FFNs and depthwise convs over each camera's
   8x8 patch grid);
5. predict a per-token flow field, a transport strength, and an intensity gain, then advect the
   stale feature map by bilinear sampling:
   ``z_transport = (1 - s) * z + s * gain * warp(z, flow)``;
6. decode a gated residual from [token hidden, transport delta, init token] through an ``rrc``
   decoder: ``z_hat = z_transport + sigmoid(gate) * residual``.

Zero-initialized flow / gain / residual output heads make ``z_hat == z`` at initialization, so
training starts from the identity.

Only ``latent_dim`` varies between backbones (pi0.5 2048, SmolVLA 960); every other hyperparameter
is a module constant, fixed to the configuration the released checkpoints were trained with.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PREDICTOR_ARCHITECTURE = "motion_prior_transport_interaction_v1"
PREDICTOR_FORMAT_VERSION = 5

# --- fixed architecture -----------------------------------------------------------------
HIDDEN_DIM = 256
ACTION_HIDDEN_DIM = 256
ACTION_NUM_HEADS = 4
ACTION_NUM_LAYERS = 2
NUM_STREAMS = 2
ACTION_DIM = 7
MAX_DELAY = 20
STATE_DIM = 8
TOKEN_BLOCK_SPEC = "rrcccc"
RESIDUAL_DECODER_SPEC = "rrc"
RESIDUAL_DECODER_HIDDEN = 384
MAX_TRANSPORT_OFFSET = 1.0
MAX_GAIN_DELTA = 1.0
INTERACTION_GATE_BIAS = -4.0


def build_action_trajectory_features(
    motion_actions: torch.Tensor,
    delay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an explicit control-space residual trajectory from the executed actions.

    Each action is treated as a control-space increment; the features are the action itself, its
    running sum, its step-to-step change, its running absolute path length, a normalized progress
    scalar, and the validity flag -- ``action_dim * 4 + 2`` channels.
    """

    if motion_actions.ndim != 3:
        raise ValueError(
            f"motion_actions must have shape [B, K, A], got {tuple(motion_actions.shape)}")
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
    """Normalized 2D positions for square image tokens, or a 1D fallback."""

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
    """Small masked residual self-attention block for short action trajectories."""

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
    """Per-token pre-norm residual FFN ('r'). Adds depth to the visual path with no token mixing,
    and is cheaper than an attention block (no QKV/output projections)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class DepthwiseConvBlock(nn.Module):
    """Depthwise-separable conv ('c') over each camera's HxW token grid: local spatial mixing at a
    fraction of an attention block's parameters. Mixes tokens WITHIN a camera view, not across
    cameras. Requires a square token grid."""

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
        x = x.reshape(b, streams, side, side, h).permute(0, 1, 4, 2, 3).reshape(
            b * streams, h, side, side)
        x = self.act(self.pw(self.dw(x)))
        x = x.reshape(b, streams, h, side, side).permute(0, 1, 3, 4, 2).reshape(
            b, streams * tokens, h)
        return seq + x


class ActionTrajectoryEncoder(nn.Module):
    """Encode the committed-action residual trajectory with lightweight temporal attention."""

    def __init__(self, *, action_dim: int, hidden_dim: int, max_delay: int,
                 num_heads: int = 4, num_layers: int = 2):
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

    def forward(self, motion_actions: torch.Tensor,
                delay: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, valid_mask, absolute_residual = build_action_trajectory_features(
            motion_actions, delay)
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


def _build_blocks(spec: str, hidden_dim: int) -> nn.ModuleList:
    blocks = []
    for ch in spec:
        if ch == "r":
            blocks.append(PerTokenResidualBlock(hidden_dim))
        elif ch == "c":
            blocks.append(DepthwiseConvBlock(hidden_dim))
        else:
            raise ValueError(f"block spec chars must be 'r' or 'c', got {ch!r}")
    return nn.ModuleList(blocks)


def _run_blocks(blocks: nn.ModuleList, token_hidden: torch.Tensor) -> torch.Tensor:
    """Run a block stack over the flattened cross-camera token set [B, S, T, H]."""
    batch, streams, tokens, hidden = token_hidden.shape
    seq = token_hidden.reshape(batch, streams * tokens, hidden)
    for block in blocks:
        if isinstance(block, DepthwiseConvBlock):
            seq = block(seq, streams, tokens)
        else:
            seq = block(seq)
    return seq.reshape(batch, streams, tokens, hidden)


class MotionPriorLatentPredictor(nn.Module):
    """One-shot action-conditioned token transport plus interaction innovation."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_init_frame = True
        self.state_dim = STATE_DIM

        self.stream_embedding = nn.Embedding(NUM_STREAMS, HIDDEN_DIM)
        # 2 roles: 0 = stale (query frame), 1 = init (episode frame 0).
        self.frame_role_embedding = nn.Embedding(2, HIDDEN_DIM)
        self.action_trajectory_encoder = ActionTrajectoryEncoder(
            action_dim=ACTION_DIM,
            hidden_dim=ACTION_HIDDEN_DIM,
            max_delay=MAX_DELAY,
            num_heads=ACTION_NUM_HEADS,
            num_layers=ACTION_NUM_LAYERS,
        )
        # the action encoder already runs at the visual width, so this is a no-parameter passthrough
        self.action_proj = nn.Identity()
        self.latent_context_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        # shared by the stale latent and the init-frame latent, so the init stream costs no encoder
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.token_attention = _build_blocks(TOKEN_BLOCK_SPEC, HIDDEN_DIM)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.transport_flow_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 2),
        )
        self.transport_strength_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        # per-token intensity gain, so the warp can change magnitude instead of being a pure
        # (convex) rearrangement. Zero-init -> tanh(0) = 0 -> gain 1, i.e. a no-op at init.
        self.transport_gain_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.transport_delta_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.SiLU(),
        )
        # innovation decoder over [token hidden, transport delta, init token]
        self.residual_in_proj = nn.Linear(HIDDEN_DIM * 3, RESIDUAL_DECODER_HIDDEN)
        self.residual_decoder = _build_blocks(RESIDUAL_DECODER_SPEC, RESIDUAL_DECODER_HIDDEN)
        self.residual_out_head = nn.Linear(RESIDUAL_DECODER_HIDDEN, latent_dim + 1)

        nn.init.zeros_(self.transport_flow_head[-1].weight)
        nn.init.zeros_(self.transport_flow_head[-1].bias)
        nn.init.zeros_(self.transport_strength_head[-1].weight)
        nn.init.constant_(self.transport_strength_head[-1].bias, -6.0)
        nn.init.zeros_(self.transport_gain_head[-1].weight)
        nn.init.zeros_(self.transport_gain_head[-1].bias)
        # zero-init the innovation output so z_hat == z_transport at init; the last bias entry is
        # the gate, started near-closed (sigmoid(-4) ~= 0.018).
        nn.init.zeros_(self.residual_out_head.weight)
        nn.init.zeros_(self.residual_out_head.bias)
        nn.init.constant_(self.residual_out_head.bias[-1:], INTERACTION_GATE_BIAS)

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
            raise ValueError(f"z must have shape [B, C, T, D], got {tuple(z.shape)}")
        if motion_actions.ndim != 3:
            raise ValueError(
                f"motion_actions must have shape [B, K, A], got {tuple(motion_actions.shape)}")
        if delay.ndim != 1:
            raise ValueError(f"delay must have shape [B], got {tuple(delay.shape)}")
        if z.shape[0] != motion_actions.shape[0] or z.shape[0] != delay.shape[0]:
            raise ValueError("z, motion_actions, and delay must use the same batch size")
        if z.shape[1] > NUM_STREAMS:
            raise ValueError(
                f"z contains {z.shape[1]} streams but the predictor supports {NUM_STREAMS}")
        if z_init is None:
            raise ValueError("z_init (the episode's first-frame latent) is required")
        if z_init.shape != z.shape:
            raise ValueError(f"z_init must match z shape {tuple(z.shape)}, got {tuple(z_init.shape)}")
        if state is None:
            raise ValueError(f"state [B, {STATE_DIM}] is required")

        motion, action_aux = self.action_trajectory_encoder(
            motion_actions, delay.to(motion_actions.device))
        motion = self.action_proj(motion)
        stream_ids = torch.arange(z.shape[1], device=z.device)
        stream_condition = self.stream_embedding(stream_ids)[None, :, :]
        joint_latent_context = self.latent_context_encoder(z.mean(dim=(1, 2)))[:, None, :]
        condition = motion[:, None, :] + stream_condition + joint_latent_context
        condition = condition + self.state_encoder(state.to(z.dtype))[:, None, :]

        positions = token_grid_positions(z.shape[2], device=z.device, dtype=z.dtype)
        pos_embed = self.position_encoder(positions)[None, None]
        token_hidden = self.token_encoder(z)
        token_hidden = token_hidden + condition[:, :, None, :] + pos_embed
        n_stale_streams = z.shape[1]

        role = self.frame_role_embedding.weight  # [2, hidden]
        token_hidden = token_hidden + role[0]
        # the init stream is a static reference layout, so it gets camera identity + position +
        # its role tag, but NOT the motion-conditioned term.
        init_token = self.token_encoder(z_init)
        init_hidden = init_token + stream_condition[:, :, None, :] + pos_embed + role[1]
        token_hidden = torch.cat([token_hidden, init_hidden], dim=1)  # [B, 2C, T, hidden]

        token_hidden = _run_blocks(self.token_attention, token_hidden)
        token_hidden = token_hidden[:, :n_stale_streams]  # drop init streams before transport

        flow = torch.tanh(self.transport_flow_head(token_hidden)) * MAX_TRANSPORT_OFFSET
        transport_strength = torch.sigmoid(self.transport_strength_head(token_hidden))

        # Continuous motion-conditioned feature advection: warp each camera's sqrt(T) x sqrt(T)
        # feature map by the flow field via bilinear sampling, then apply a per-token intensity
        # gain. Zero flow + gain 1 at init -> z_transport == z.
        batch, streams, tokens, dim = z.shape
        side = int(tokens ** 0.5)
        if side * side != tokens:
            raise ValueError(f"warp transport needs a square token grid, got tokens={tokens}")
        gain = 1.0 + MAX_GAIN_DELTA * torch.tanh(self.transport_gain_head(token_hidden))
        base_grid = positions.reshape(side, side, 2)
        sample_grid = base_grid[None] - flow.reshape(batch * streams, side, side, 2)
        z_map = z.reshape(batch * streams, side, side, dim).permute(0, 3, 1, 2)
        z_warp = F.grid_sample(z_map, sample_grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        z_warp = z_warp.permute(0, 2, 3, 1).reshape(batch, streams, tokens, dim)
        z_transport = (1.0 - transport_strength) * z + transport_strength * (gain * z_warp)

        transport_delta_hidden = self.transport_delta_encoder(z_transport - z)
        dec = self.residual_in_proj(
            torch.cat([token_hidden, transport_delta_hidden, init_token], dim=-1))
        interaction_output = self.residual_out_head(_run_blocks(self.residual_decoder, dec))
        interaction_residual = interaction_output[..., :-1]
        interaction_gate = torch.sigmoid(interaction_output[..., -1:])
        z_hat = z_transport + interaction_gate * interaction_residual

        if not return_aux:
            return z_hat
        return z_hat, {
            **action_aux,
            "token_positions": positions,
            "transport_flow": flow,
            "transport_strength": transport_strength,
            "transport_gain": gain,
            "z_transport": z_transport,
            "interaction_gate": interaction_gate,
            "interaction_residual": interaction_residual,
        }


def load_predictor(ckpt_path, device: str | torch.device = "cpu"):
    """Load a released predictor checkpoint. The architecture is fixed; only latent_dim varies."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("format_version") != PREDICTOR_FORMAT_VERSION:
        raise RuntimeError(
            f"{ckpt_path}: expected format_version {PREDICTOR_FORMAT_VERSION}, "
            f"got {ckpt.get('format_version')!r}")
    if ckpt.get("predictor_architecture") != PREDICTOR_ARCHITECTURE:
        raise RuntimeError(
            f"{ckpt_path}: expected architecture {PREDICTOR_ARCHITECTURE!r}, "
            f"got {ckpt.get('predictor_architecture')!r}")
    model = MotionPriorLatentPredictor(int(ckpt["args"]["latent_dim"])).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    norm = ckpt.get("latent_norm")
    if norm is not None:
        norm = {"mean": norm["mean"].to(device), "std": norm["std"].to(device)}
    return model, norm


@torch.no_grad()
def forecast(model, latent_norm, z_stale, motion_actions_qnorm, delay, *, z_init, state):
    """Forecast the handoff-time latent.

    ``motion_actions_qnorm`` must be in BANK (quantile-normalized) space -- the space every
    predictor was trained on. Returns ``z_hat`` in ``z_stale``'s dtype.
    """
    def _norm(x):
        if latent_norm is None:
            return x
        return (x - latent_norm["mean"][None, :, None, :]) / latent_norm["std"][None, :, None, :]

    device = z_stale.device
    delay_t = (delay if isinstance(delay, torch.Tensor)
               else torch.full((z_stale.shape[0],), int(delay), dtype=torch.long))
    z_hat = model(
        _norm(z_stale.float()),
        torch.as_tensor(motion_actions_qnorm, dtype=torch.float32, device=device),
        delay_t.to(device=device, dtype=torch.long),
        z_init=_norm(z_init.float()),
        state=torch.as_tensor(state, dtype=torch.float32, device=device),
    )
    if latent_norm is not None:
        z_hat = z_hat * latent_norm["std"][None, :, None, :] + latent_norm["mean"][None, :, None, :]
    return z_hat.to(z_stale.dtype)
