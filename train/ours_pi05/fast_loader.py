"""直读 memmap 的批加载器 —— 替代 torch DataLoader。

为什么不用 DataLoader
---------------------
预测器每个样本要两个 latent（z 和 z_target），各 3×256×2048 bf16 = 3.15 MB。
有效 batch 256 时每步就是 **1.6 GB**（accum 2 则是 3.2 GB）。

走 DataLoader 的话这些字节要经过：
    page cache -> worker 进程 -> /dev/shm 序列化 -> 主进程反序列化 -> pin -> H2D
worker 那两跳是纯浪费 —— bank（84 GB）本来就已经整个躺在 page cache 里了
（机器 629 GB 内存），从 memmap 读就是一次 memcpy。

实测 DataLoader 通路：32 个 worker、prefetch 6，GPU 利用率仍在 100/0/0 之间跳，
主进程卡在 futex_wait 等数据，稳态只有 0.6~0.7 it/s（而纯计算是 2.35 it/s）。

这里的做法
----------
  * 主进程直接从 memmap 读进**预分配的 pinned buffer**（pinned 才能异步 H2D）；
  * 用线程池并行填充（numpy 的 memmap 拷贝会释放 GIL，所以线程真能并行）；
  * 一个后台线程预取下一批，与当前批的计算重叠（双缓冲）。

latent 全程保持磁盘上的 bf16（uint16 view），不在 CPU 上转 float32 —— 传输量
减半，且零精度损失。归一化在 GPU 上用 float32 做。
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ours_pi05.action_space import qnorm, to_delta
from ours_pi05.dataset import MAX_DELAY
from ours_pi05.latent_bank import LATENT_SHAPE, Bank
from ours_pi05.models.corrector import Corrector


class FastBatchLoader:
    """无限迭代器，产出已在 GPU 上的 batch。

    每次 next() 返回 dict：
        z, z_target : [B, 3, 256, 2048] float32，已归一化（GPU）
        z_init      : [B, 3, 256, 2048] float32，已归一化（GPU，来自查找表）
        motion_actions : [B, 20, 14] float32（GPU），qnorm 的 delta，**左对齐**
        delay       : [B] int64（GPU）
        state       : [B, 14] float32（GPU），交接时刻的**校正后** state
    """

    def __init__(
        self,
        bank: Bank,
        delay_set,
        *,
        batch_size: int,
        device: str = "cuda",
        seed: int = 0,
        read_threads: int = 16,
        prefetch: int = 2,
    ):
        self.bank = bank
        self.delay_set = [int(d) for d in delay_set]
        if any(d < 1 or d > MAX_DELAY for d in self.delay_set):
            raise ValueError(f"delay_set 必须在 [1, {MAX_DELAY}]，得到 {self.delay_set}")
        self.bs = int(batch_size)
        self.device = device
        self.corrector = Corrector()

        dmax = max(self.delay_set)
        ep = bank.episode_index
        self.anchors = np.array(
            [t for t in range(len(bank) - dmax) if ep[t] == ep[t + dmax]], dtype=np.int64
        )

        # 归一化常量（GPU）
        m = bank.latent_norm["mean"][:, None, :].astype(np.float32)  # [3, 1, 2048]
        s = bank.latent_norm["std"][:, None, :].astype(np.float32)
        self.z_mean = torch.from_numpy(m).to(device)
        self.z_std = torch.from_numpy(s).to(device)

        # z_init 查找表（GPU）：每个 episode 只有一个唯一值，不该逐样本搬运
        eps = bank.episode_starts
        table = torch.zeros((max(eps) + 1, *LATENT_SHAPE), dtype=torch.float32)
        for ep_id, row in eps.items():
            z0 = torch.from_numpy(bank.read_latent(row))
            table[ep_id] = (z0 - torch.from_numpy(m)) / torch.from_numpy(s)
        self.z_init_table = table.to(device)

        # pinned buffer 环（异步 H2D 要求源是 pinned 内存）
        self._pool = ThreadPoolExecutor(max_workers=read_threads)
        self._bufs = [
            {
                "z": torch.empty((self.bs, *LATENT_SHAPE), dtype=torch.uint16).pin_memory(),
                "zt": torch.empty((self.bs, *LATENT_SHAPE), dtype=torch.uint16).pin_memory(),
                "ma": torch.empty((self.bs, MAX_DELAY, 14), dtype=torch.float32).pin_memory(),
                "dl": torch.empty((self.bs,), dtype=torch.long).pin_memory(),
                "st": torch.empty((self.bs, 14), dtype=torch.float32).pin_memory(),
                "ep": torch.empty((self.bs,), dtype=torch.long).pin_memory(),
            }
            for _ in range(prefetch + 1)
        ]

        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._rng = np.random.default_rng(seed)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    # ---- 生产者：填 pinned buffer ----

    def _fill_one(self, buf, k: int, row: int, d: int) -> None:
        bank = self.bank
        # memmap -> pinned buffer，一次 memcpy（bank 在 page cache 里，不碰盘）
        buf["z"][k] = torch.from_numpy(np.asarray(bank.latents[row]))
        buf["zt"][k] = torch.from_numpy(np.asarray(bank.latents[row + d]))

        s_anchor = bank.states[row]
        committed = bank.actions[row : row + d]
        motion = qnorm(to_delta(committed, s_anchor), bank.action_quantiles)
        buf["ma"][k].zero_()
        buf["ma"][k, :d] = torch.from_numpy(motion)  # 左对齐，见 dataset.py
        buf["dl"][k] = d
        # state 是交接时刻的校正后 state，不是 s_anchor（见 dataset.py 的注释）
        buf["st"][k] = torch.from_numpy(self.corrector(s_anchor, committed, d))
        buf["ep"][k] = int(bank.episode_index[row])

    def _producer(self) -> None:
        i = 0
        while not self._stop.is_set():
            buf = self._bufs[i % len(self._bufs)]
            i += 1
            rows = self._rng.choice(self.anchors, size=self.bs, replace=False)
            ds = self._rng.choice(self.delay_set, size=self.bs)
            futs = [
                self._pool.submit(self._fill_one, buf, k, int(r), int(d))
                for k, (r, d) in enumerate(zip(rows, ds, strict=True))
            ]
            for f in futs:
                f.result()
            try:
                self._q.put(buf, timeout=5)
            except queue.Full:
                if self._stop.is_set():
                    return

    # ---- 消费者 ----

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        buf = self._q.get()
        dev = self.device
        # uint16 -> bfloat16（零拷贝 view）-> float32 -> 归一化，全在 GPU 上
        z_raw = buf["z"].to(dev, non_blocking=True).view(torch.bfloat16)
        zt_raw = buf["zt"].to(dev, non_blocking=True).view(torch.bfloat16)
        ep = buf["ep"].to(dev, non_blocking=True)
        return {
            "z": (z_raw.float() - self.z_mean) / self.z_std,
            "z_target": (zt_raw.float() - self.z_mean) / self.z_std,
            "z_init": self.z_init_table[ep],
            "motion_actions": buf["ma"].to(dev, non_blocking=True),
            "delay": buf["dl"].to(dev, non_blocking=True),
            "state": buf["st"].to(dev, non_blocking=True),
        }

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False)
