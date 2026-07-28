"""On-the-fly (t, t+d) pair sampling over a per-frame visual-latent bank.

A visual latent is a per-frame quantity -- ``z_s`` depends only on frame ``t`` and ``z_target`` only
on frame ``t+d`` -- so the bank caches each selected frame's latent ONCE and the delay is resampled
on every access. That gives full (frame x delay) coverage with no per-pair duplication and no VLM in
the training loop.

Latents stay memory-mapped on disk: ``__getitem__`` faults in only the frames it addresses, and
``release_resident`` periodically drops resident pages so a long run over a multi-hundred-GB bank
does not accrete the whole thing into RSS.
"""
from __future__ import annotations

import bisect
import ctypes
import ctypes.util
import pathlib
import random

import numpy as np
import torch

from common.run_index import build_index, contiguous_run_lengths

PERFRAME_FORMAT_VERSION = "perframe_latents_v1"

_MADV_RANDOM = 1    # asm-generic/mman-common.h: stop read-ahead
_MADV_DONTNEED = 4  # drop resident pages now (re-faulted from page cache/disk on next access)


def _madvise(tensor, advice: int) -> None:
    """Best-effort madvise on a mmap-backed tensor's storage; silently skips if unavailable.

    MADV_RANDOM stops read-ahead so RSS tracks only the indexed frames; MADV_DONTNEED releases
    resident pages so a long full-coverage run stays bounded on a memory-contended host.
    """
    try:
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        if nbytes == 0:
            return
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        pagesize = 4096
        addr = storage.data_ptr()
        start = addr - (addr % pagesize)
        length = nbytes + (addr - start)
        libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), advice)
    except Exception:  # noqa: BLE001 - madvise is an optimization, never a requirement
        pass


class ConcatShardView:
    """Read-only concatenated view over per-shard latent tensors, indexed by a global frame index.

    Avoids materializing the whole [N, C, T, D] bank: integer indexing touches only the addressed
    shard/frame, which for a mmap-backed tensor is a page-cache read.
    """

    def __init__(self, shards):
        self.shards = list(shards)
        if not self.shards:
            raise ValueError("ConcatShardView needs at least one shard")
        self.offsets = [0]
        for shard in self.shards:
            self.offsets.append(self.offsets[-1] + int(shard.shape[0]))
        self._len = self.offsets[-1]
        self.shape = (self._len,) + tuple(int(x) for x in self.shards[0].shape[1:])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, g):
        if not isinstance(g, (int, np.integer)):
            raise TypeError("ConcatShardView supports integer frame indexing only")
        if g < 0:
            g += self._len
        si = bisect.bisect_right(self.offsets, g) - 1
        return self.shards[si][g - self.offsets[si]]


def compute_latent_stats(latents, eps: float = 1e-6, chunk: int = 1024):
    """Per-(camera, hidden_dim) mean/std over frames and tokens.

    Accumulated in float64 over small frame chunks so the bank is never fully upcast at once;
    streams shard-by-shard for a ConcatShardView. Returns (mean, std), each [C, D] float32, with
    std floored at ``eps``.
    """
    def frame_chunks():
        if hasattr(latents, "shards"):
            for shard in latents.shards:
                for i in range(0, shard.shape[0], chunk):
                    yield shard[i:i + chunk]
        else:
            for i in range(0, latents.shape[0], chunk):
                yield latents[i:i + chunk]

    total = 0
    s = ss = None
    for x in frame_chunks():
        xf = x.to(torch.float64)  # [b, C, T, D]
        if s is None:
            C, D = xf.shape[1], xf.shape[3]
            s = torch.zeros(C, D, dtype=torch.float64)
            ss = torch.zeros(C, D, dtype=torch.float64)
        s += xf.sum(dim=(0, 2))
        ss += (xf * xf).sum(dim=(0, 2))
        total += xf.shape[0] * xf.shape[2]
    mean = s / total
    var = (ss / total - mean * mean).clamp_min(0.0)
    std = var.sqrt().clamp_min(eps)
    return mean.to(torch.float32), std.to(torch.float32)


def select_episode_row_mask(episode_index, max_episodes):
    """Boolean row mask keeping the first ``max_episodes`` distinct episodes (by first appearance).

    ``None`` or <= 0 keeps every row.
    """
    ei = np.asarray(episode_index)
    if max_episodes is None or max_episodes <= 0:
        return np.ones(len(ei), dtype=bool)
    seen: list = []
    keep: set = set()
    for value in ei:
        key = int(value)
        if key not in keep and len(seen) < max_episodes:
            seen.append(key)
            keep.add(key)
    return np.isin(ei, list(keep))


def bank_shards(directory) -> list[pathlib.Path]:
    """``shard_*.pt`` only -- a bank directory also holds latent_stats.pt and the state sidecar.

    Sorted LEXICOGRAPHICALLY by filename, which is the bank's global frame order and what the
    sidecar's alignment check assumes. Do not sort by parsed integer.
    """
    paths = sorted(pathlib.Path(directory).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.pt found in {directory}")
    return paths


class PerFrameLatentDataset(torch.utils.data.Dataset):
    """Sample (z_s, z_target, motion_actions, delay) triples on the fly from a per-frame bank.

    ``latents`` [N, C, T, D] and ``actions`` [N, A] are frame-contiguous within each run, runs
    concatenated; ``run_lengths`` gives the run structure so pairs stay in-run. Delay is drawn
    fresh per access from ``[1, min(d_max, room)]``.

    Actions are in BANK (quantile-normalized) space, the space the predictor's motion prior
    consumes. ``state_fn`` (if given) receives them in that space.
    """

    def __init__(self, latents, actions, run_lengths, camera_keys, *, d_max: int = 20,
                 seed: int = 0, norm_mean=None, norm_std=None, state_fn=None):
        self.latents = latents
        self.actions = actions
        self.run_lengths = list(run_lengths)
        self.camera_keys = tuple(camera_keys)
        self.d_max = int(d_max)
        # Optional callable (g0, d, committed_actions[d,7]) -> Tensor[8] filling an item["state"]
        # proprio channel, computed in the worker. None => no "state" key.
        self.state_fn = state_fn
        self._rng = random.Random(seed)
        self.index = build_index(self.run_lengths, self.d_max)
        self.set_normalization(norm_mean, norm_std)

    def set_normalization(self, norm_mean, norm_std):
        """Per-(camera, hidden_dim) normalization applied to z_s, z_target and z_init.
        mean/std are [C, D] and broadcast over the token axis. None disables it."""
        self.norm_mean = None if norm_mean is None else torch.as_tensor(norm_mean,
                                                                        dtype=torch.float32)
        self.norm_std = None if norm_std is None else torch.as_tensor(norm_std,
                                                                      dtype=torch.float32)

    def _normalize(self, z):
        if self.norm_mean is None:
            return z
        return (z - self.norm_mean[:, None, :]) / self.norm_std[:, None, :]

    def release_resident(self) -> None:
        """Drop resident mmap pages of the latent shards so RSS stays near the working set."""
        if hasattr(self.latents, "shards"):
            for shard in self.latents.shards:
                _madvise(shard, _MADV_DONTNEED)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        run_id, local_t, run_start = self.index[i]
        room = self.run_lengths[run_id] - 1 - local_t
        d = self._rng.randint(1, min(self.d_max, room))
        g0 = run_start + local_t
        item = {
            "z_s": self._normalize(self.latents[g0].float()),
            "z_target": self._normalize(self.latents[g0 + d].float()),
            # episode/run frame 0: the un-occluded static layout reference
            "z_init": self._normalize(self.latents[run_start].float()),
            "motion_actions": self.actions[g0:g0 + d].float(),
            "delay": torch.tensor(d, dtype=torch.long),
            "g_target": torch.tensor(g0 + d, dtype=torch.long),
            "camera_keys": self.camera_keys,
        }
        if self.state_fn is not None:
            item["state"] = self.state_fn(g0, d, self.actions[g0:g0 + d])
        return item

    @classmethod
    def from_shards(cls, paths, *, d_max: int = 20, seed: int = 0, mmap: bool = True,
                    state_fn=None):
        """Memory-map per-frame shards as one ConcatShardView, recomputing the run structure over
        the concatenated frames (robust to runs split across shard boundaries)."""
        paths = [pathlib.Path(p) for p in paths]
        shards, actions_parts, ei_parts, fi_parts = [], [], [], []
        camera_keys = None
        for path in paths:
            payload = torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)
            if payload.get("format_version") != PERFRAME_FORMAT_VERSION:
                raise RuntimeError(
                    f"{path} must use per-frame latent format '{PERFRAME_FORMAT_VERSION}'")
            if camera_keys is None:
                camera_keys = tuple(payload["camera_keys"])
            elif tuple(payload["camera_keys"]) != camera_keys:
                raise ValueError(f"{path} camera_keys differ from earlier shards")
            latents_shard = payload["latents"]
            if mmap:
                _madvise(latents_shard, _MADV_RANDOM)
            shards.append(latents_shard)
            actions_parts.append(torch.as_tensor(payload["actions"]).clone())  # small -> RAM
            ei_parts.append(np.asarray(payload["episode_index"]))
            fi_parts.append(np.asarray(payload["frame_index"]))
        if camera_keys is None:
            raise FileNotFoundError("no per-frame shards provided")
        latents = ConcatShardView(shards)
        actions = torch.cat(actions_parts, dim=0)
        run_lengths = contiguous_run_lengths(np.concatenate(ei_parts), np.concatenate(fi_parts))
        return cls(latents, actions, run_lengths, camera_keys, d_max=d_max, seed=seed,
                   state_fn=state_fn)


def collate(batch):
    """Stack a batch, right-padding ``motion_actions`` to the batch's maximum delay."""
    camera_keys = batch[0]["camera_keys"]
    if any(row["camera_keys"] != camera_keys for row in batch[1:]):
        raise ValueError("all samples in a batch must use the same ordered camera_keys")
    max_motion_steps = max(row["motion_actions"].shape[0] for row in batch)
    action_dim = batch[0]["motion_actions"].shape[-1]
    motion_actions = batch[0]["motion_actions"].new_zeros(
        (len(batch), max_motion_steps, action_dim))
    for index, row in enumerate(batch):
        if row["motion_actions"].shape[-1] != action_dim:
            raise ValueError("all samples in a batch must use the same action dimension")
        motion_actions[index, : row["motion_actions"].shape[0]] = row["motion_actions"]
    out = {
        "z_s": torch.stack([x["z_s"] for x in batch]),
        "z_target": torch.stack([x["z_target"] for x in batch]),
        "z_init": torch.stack([x["z_init"] for x in batch]),
        "motion_actions": motion_actions,
        "delay": torch.stack([x["delay"] for x in batch]),
        "g_target": torch.stack([x["g_target"] for x in batch]),
        "camera_keys": camera_keys,
    }
    if "state" in batch[0]:
        out["state"] = torch.stack([x["state"] for x in batch])
    return out
