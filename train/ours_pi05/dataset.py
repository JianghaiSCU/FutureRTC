"""预测器的训练数据集。

bank 是**逐帧**的（每帧的 latent 只 embed 一次）。样本在 __getitem__ 时才组装：
采一个 d，切出 (z[t], a[t:t+d], state[t], z_init[episode], z[t+d])。
这样「帧 × delay」全覆盖、无逐对重复，训练循环里也不用跑 VLM。
（做法沿用 ours_mainline/data_collection/collect_latents.py:3-6 的注释。）

样本绝不跨 episode 边界。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ours_pi05.action_space import qnorm, to_delta
from ours_pi05.latent_bank import Bank, LATENT_SHAPE
from ours_pi05.models.corrector import Corrector

from ours_pi05.deploy_protocol import MAX_DELAY  # 单一来源在 deploy_protocol；此处 re-export 供 eval/fast_loader 沿用


class PerFrameLatentDataset(Dataset):
    """预测器的训练数据集。

    ``raw_mode`` —— 训练用的性能通路（默认关，保持语义最直白的行为）
    ------------------------------------------------------------------
    默认（``raw_mode=False``）每个样本返回 3 个**归一化后的 float32** latent
    （z / z_target / z_init），每个 3×256×2048×4 B = 6.3 MB，一个样本 18.9 MB。
    batch 256 时 DataLoader 每步要通过共享内存从 worker 搬 **4.8 GB** 到主进程 ——
    实测这一项就占了每步 2860 ms 里的 1838 ms（64%），远超 GPU 计算本身。

    ``raw_mode=True`` 改成：
      * latent 按**磁盘上的原始 bf16** 原样返回（不转 float32、不归一化）——
        IPC 从 18.9 MB/样本降到 6.3 MB，且**零精度损失**（归一化改在 GPU 上用
        float32 做）；
      * ``z_init`` **不再逐样本返回**（100 个 episode 只有 100 个唯一值，却被复制了
        256 份），改为只返回 ``episode``，由训练脚本在 GPU 上建一张查找表
        （100 × 3.15 MB = 315 MB）。

    ``raw_mode=True`` 时调用方必须自己做：
        z = (z_raw.float() - mean[:, None, :]) / std[:, None, :]
        z_init = z_init_table[episode]
    用 ``latent_norm_tensors()`` 拿 mean/std，用 ``z_init_table()`` 拿查找表。
    """

    def __init__(
        self,
        bank: Bank,
        delay_set: Sequence[int],
        seed: int = 0,
        raw_mode: bool = False,
    ):
        self.bank = bank
        self.delay_set = [int(d) for d in delay_set]
        if any(d < 1 or d > MAX_DELAY for d in self.delay_set):
            raise ValueError(f"delay_set 必须在 [1, {MAX_DELAY}]，得到 {self.delay_set}；d=0 走旁路")
        self.seed = int(seed)
        self.raw_mode = bool(raw_mode)
        self.corrector = Corrector()

        # 合法锚点：t 和 t+max(delay_set) 必须在同一个 episode 内
        dmax = max(self.delay_set)
        ep = bank.episode_index
        n = len(bank)
        self.anchors = np.array(
            [t for t in range(n - dmax) if ep[t] == ep[t + dmax]], dtype=np.int64
        )

        m = bank.latent_norm["mean"][:, None, :]  # [3, 1, 2048]
        s = bank.latent_norm["std"][:, None, :]
        if m.shape != (LATENT_SHAPE[0], 1, LATENT_SHAPE[2]):
            raise ValueError(f"latent_norm shape mismatch with bank LATENT_SHAPE {LATENT_SHAPE}: {m.shape}")
        self._mean, self._std = m, s

        # z_init 是 episode 首帧的 latent —— 每个 episode 只有一个唯一值（这里 100 个）。
        # raw_mode 下**不逐样本返回**它（否则 100 个唯一值会被复制 256 份塞进共享内存），
        # 由训练脚本用 z_init_table() 在 GPU 上建查找表。非 raw_mode 下缓存在内存里，
        # 避免每个样本都从 84 GB 的 memmap 再随机读 3.15 MB。
        self._z_init_cache: dict[int, np.ndarray] | None = None
        if not self.raw_mode:
            self._z_init_cache = {
                ep_id: self._norm(bank.read_latent(row)).astype(np.float32)
                for ep_id, row in bank.episode_starts.items()
            }

    def __len__(self) -> int:
        return len(self.anchors)

    def _norm(self, z: np.ndarray) -> np.ndarray:
        return (z - self._mean) / self._std

    def _read_bf16(self, row: int) -> torch.Tensor:
        """磁盘上的 bf16 原样读出，不转 float32、不归一化。零拷贝、零精度损失。"""
        u16 = np.array(self.bank.latents[row], copy=True)  # uint16 view of bf16
        return torch.from_numpy(u16).view(torch.bfloat16)

    # ---- raw_mode 的配套：给训练脚本在 GPU 上重建归一化与 z_init ----

    def latent_norm_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (mean, std)，形状 [3, 1, 2048]，可直接广播到 [B, 3, 256, 2048]。"""
        return (
            torch.from_numpy(self._mean.astype(np.float32)),
            torch.from_numpy(self._std.astype(np.float32)),
        )

    def z_init_table(self) -> torch.Tensor:
        """episode -> 归一化后的首帧 latent，形状 [max_ep+1, 3, 256, 2048] float32。

        用 `table[episode]` 索引。100 个 episode 是 315 MB，直接放 GPU。
        """
        eps = self.bank.episode_starts
        table = torch.zeros(
            (max(eps) + 1, *LATENT_SHAPE), dtype=torch.float32
        )
        for ep_id, row in eps.items():
            table[ep_id] = torch.from_numpy(
                self._norm(self.bank.read_latent(row)).astype(np.float32)
            )
        return table

    def __getitem__(self, i: int) -> dict:
        row = int(self.anchors[i])
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        d = int(rng.choice(self.delay_set))

        bank = self.bank
        ep = int(bank.episode_index[row])

        # s_anchor：陈旧 obs 时刻（t）的 state —— to_delta 的锚点，语义是
        # 「相对陈旧位姿的位移」，这个锚点本身就该是陈旧的，不能换成交接态。
        s_anchor = bank.states[row]
        committed = bank.actions[row : row + d]                 # [d, 14] 绝对关节角
        delta = to_delta(committed, s_anchor)                   # [d, 14]
        motion = qnorm(delta, bank.action_quantiles)            # [d, 14]

        # state：喂给预测器的 state 输入，是交接时刻（t+d）的**校正后** state，
        # 不是陈旧的 s_anchor —— 训练和部署都必须过 Corrector，否则两边条件的
        # 分布不一致（部署时 server 传入的就是 corr(state_t, committed, d)）。
        # 见 ours_mainline/README.md:80 与该 checkpoint 的 state_source=corrector。
        s_hat = self.corrector(bank.states[row], bank.actions[row : row + d], d)

        # **左对齐**补零到 MAX_DELAY —— 必须与 predictor 的 valid_mask 一致：
        #   valid_mask = step_ids < delay   (ours_mainline/models/predictor.py:34)
        #   endpoint   = x[:, delay - 1]    (:184)
        # 搞成右对齐会静默毁掉训练：模型读到全零的「有效」步，cumsum 特征全废。
        padded = np.zeros((MAX_DELAY, 14), dtype=np.float32)
        padded[:d] = motion

        common = {
            "motion_actions": torch.from_numpy(padded),
            "delay": torch.tensor(d, dtype=torch.long),
            "state": torch.from_numpy(s_hat.astype(np.float32)),
        }

        if self.raw_mode:
            # bf16 原样，不归一化、不带 z_init —— 见类 docstring。
            # 6.3 MB/样本，而非 18.9 MB。
            return {
                "z_raw": self._read_bf16(row),
                "z_target_raw": self._read_bf16(row + d),
                "episode": torch.tensor(ep, dtype=torch.long),
                **common,
            }

        z = self._norm(bank.read_latent(row))
        z_target = self._norm(bank.read_latent(row + d))
        z_init = self._z_init_cache[ep]  # 已归一化，见 __init__
        return {
            "z": torch.from_numpy(z.astype(np.float32)),
            "z_target": torch.from_numpy(z_target.astype(np.float32)),
            "z_init": torch.from_numpy(z_init.astype(np.float32)),
            **common,
        }
