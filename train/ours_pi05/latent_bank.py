"""Latent bank 的落盘格式。

latent 用 bf16 存（SigLIP 本来就跑 bf16，不损失精度），通过 torch bf16 -> uint16
view 走 numpy memmap（无损、零拷贝）。plates_stacking 的 bank ≈ 89 GB
（28,360 frames × 3 × 256 × 2048 × 2 bytes），磁盘只剩 ~206 GB，所以一次只留
一个任务的 bank，且 latents 必须走 memmap —— 训练时随机索引，绝不能整个读进内存。

LATENT_SHAPE 在这里本地定义（而不是从 openpi_bridge 导入），因为 openpi_bridge
会拉起 JAX + 加载 openpi，是重量级 GPU 依赖；latent_bank 是纯存储层，测试要快、
要能在无 GPU 环境跑。两边的 shape 是否一致由
tests/test_latent_bank.py::test_latent_shape_matches_bridge 钉死。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

LATENT_SHAPE = (3, 256, 2048)
ACTION_DIM = 14
STATE_DIM = 14


def _latent_path(root: pathlib.Path) -> pathlib.Path:
    return root / "latents.npy"


class BankWriter:
    def __init__(self, root: str | pathlib.Path, num_frames: int):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.num_frames = int(num_frames)

        self._latents = np.lib.format.open_memmap(
            _latent_path(self.root),
            mode="w+",
            dtype=np.uint16,
            shape=(self.num_frames, *LATENT_SHAPE),
        )
        self._actions = np.zeros((self.num_frames, ACTION_DIM), dtype=np.float32)
        self._states = np.zeros((self.num_frames, STATE_DIM), dtype=np.float32)
        self._episode_index = np.zeros((self.num_frames,), dtype=np.int32)
        self._frame_index = np.zeros((self.num_frames,), dtype=np.int32)

    def write(
        self,
        idx: int,
        *,
        latent: np.ndarray,
        action: np.ndarray,
        state: np.ndarray,
        episode_index: int,
        frame_index: int,
    ) -> None:
        z = torch.as_tensor(np.asarray(latent, dtype=np.float32)).to(torch.bfloat16)
        self._latents[idx] = z.view(torch.uint16).numpy()
        self._actions[idx] = np.asarray(action, dtype=np.float32)
        self._states[idx] = np.asarray(state, dtype=np.float32)
        self._episode_index[idx] = int(episode_index)
        self._frame_index[idx] = int(frame_index)

    def finalize(self, *, latent_norm: dict, action_quantiles: dict, meta: dict) -> None:
        self._latents.flush()
        np.save(self.root / "actions.npy", self._actions)
        np.save(self.root / "states.npy", self._states)
        np.save(self.root / "episode_index.npy", self._episode_index)
        np.save(self.root / "frame_index.npy", self._frame_index)
        np.savez(
            self.root / "latent_norm.npz",
            mean=np.asarray(latent_norm["mean"], dtype=np.float32),
            std=np.asarray(latent_norm["std"], dtype=np.float32),
        )
        np.savez(
            self.root / "action_quantiles.npz",
            q01=np.asarray(action_quantiles["q01"], dtype=np.float32),
            q99=np.asarray(action_quantiles["q99"], dtype=np.float32),
        )
        (self.root / "meta.json").write_text(
            json.dumps({**meta, "num_frames": self.num_frames, "latent_shape": list(LATENT_SHAPE)}, indent=2)
        )


class Bank:
    """只读 bank。latents 是 memmap，不会一次性读进内存。"""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.latents = np.load(_latent_path(self.root), mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy")
        self.states = np.load(self.root / "states.npy")
        self.episode_index = np.load(self.root / "episode_index.npy")
        self.frame_index = np.load(self.root / "frame_index.npy")

        norm = np.load(self.root / "latent_norm.npz")
        self.latent_norm = {"mean": norm["mean"], "std": norm["std"]}
        q = np.load(self.root / "action_quantiles.npz")
        self.action_quantiles = {"q01": q["q01"], "q99": q["q99"]}

        # episode -> 该 episode 首帧在 bank 里的行号（predictor 用来取 z_init）。
        # 依赖 episode_index 内每个 episode 的行是按 frame_index 升序连续写入的
        # （采集时天然满足：按 episode 顺序逐帧写）；用 setdefault 只记第一次出现，
        # 即最小行号 = 该 episode 的第一帧。
        self.episode_starts: dict[int, int] = {}
        for row, ep in enumerate(self.episode_index):
            self.episode_starts.setdefault(int(ep), row)

    def __len__(self) -> int:
        return len(self.actions)

    def read_latent(self, idx: int) -> np.ndarray:
        """[3, 256, 2048] float32。"""
        # .copy(): self.latents is opened read-only (mmap_mode="r"), so the slice is a
        # non-writable view; torch.from_numpy requires a writable buffer.
        u16 = np.array(self.latents[idx], copy=True)
        return torch.from_numpy(u16).view(torch.bfloat16).to(torch.float32).numpy()
