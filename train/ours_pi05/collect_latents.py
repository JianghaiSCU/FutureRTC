#!/usr/bin/env python3
"""为一个任务采 per-frame latent bank。

图像走 policy 自己的完整 transform 栈（repack -> data_transforms -> Normalize ->
model_transforms），保证喂进 SigLIP 的像素与推理时逐位一致。

state / action 则从 LeRobotDataset 直接另读**原始绝对关节角** —— 因为 openpi 的
AlohaInputs 带 adapt_to_pi=True，会翻转部分关节符号并重映射夹爪
（openpi/src/openpi/policies/aloha_policy.py），那是模型内部空间，不是我们要的。
预测器全程在原始关节空间工作，部署时客户端发来的也是原始关节角。action 按绝对
关节角存（不在采集时转 delta —— delta 的锚点依赖查询时刻，只能在 __getitem__
里现算，见 action_space.py 的模块 docstring）。

**norm_stats 的一个不明显的坑**：`openpi_bridge.load_policy` 内部
（`policy_config.create_trained_policy`）加载的 Normalize 统计量来自
**checkpoint 自带的** `<ckpt>/assets/<asset_id>/norm_stats.json`，而不是
`TrainConfig.assets_dirs`（cwd 相对的 `./assets/...`，训练时的产物，两者内容
通常一致，但路径解析方式不同 —— 从仓库根目录跑本脚本时 `./assets/...` 根本不
存在）。如果这里图省事直接用 `cfg.data.create(cfg.assets_dirs, cfg.model)` 产出
的 data_config 喂给 `transform_dataset`，要么在 `./assets` 不存在时直接
ValueError 崩溃，要么 —— 万一 cwd 凑巧能解析到别的 `./assets` 目录 —— 静默用
一份跟 policy 实际用的不一定是同一份的 norm_stats，Normalize 出来的 state/action
就会跟 policy 推理时看到的不一致（虽然 encode_images 只吃图像、不吃 state，这个
坑不会污染 latent 本身，但会污染 bank 里 Normalize 过的中间产物如果之后有人复用
这条 pipeline 去存 state）。所以这里显式从 `<ckpt>/assets` 加载 norm_stats 并
`dataclasses.replace` 进 data_config，跟 `create_trained_policy` 完全同源。

用法：
    export HF_LEROBOT_HOME=/root/.cache OPENPI_DATA_HOME=/root/.cache/openpi
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    export CUDA_VISIBLE_DEVICES=2
    openpi/.venv/bin/python -m ours_pi05.collect_latents \
        --train-config pi05_cobot_plates_stacking \
        --ckpt /root/zyx/ckpt/cvpr2026_RTC/openpi/pi05_cobot_plates_stacking/pi05_cobot_plates_stacking/30000 \
        --repo-id plates_stacking \
        --out banks/plates_stacking
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.data

from ours_pi05 import openpi_bridge
from ours_pi05.action_space import compute_delta_quantiles, to_delta
from ours_pi05.latent_bank import LATENT_SHAPE, BankWriter

from openpi.models import model as _model
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DELAY = 10  # 一期固定；只影响 action_quantiles 的统计口径
NORM_CHUNK = 1000  # latent_norm 流式累加的分块大小（帧数），控制常驻内存


def _bytes_per_frame() -> int:
    """一帧 latent 的落盘字节数：LATENT_SHAPE 全维乘积 * 2 bytes（bf16 view 成 uint16）。

    不要硬编码 3.15MB —— 若 LATENT_SHAPE 再变（例如 token 数或 hidden dim 改了），
    这里必须跟着变，否则预检查会静默用错误的估计值。
    """
    return int(np.prod(LATENT_SHAPE)) * 2


def _free_gb(path: pathlib.Path) -> float:
    """磁盘可用空间（GB）。path 若不存在则往上找最近的已存在祖先目录。"""
    p = path.resolve()
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return shutil.disk_usage(str(p)).free / 1e9


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _raw_obs_dict(raw_item: dict) -> dict:
    """从 create_torch_dataset() 的原始（未 repack）item 里手搭 obs_from_dict() 要的格式。

    只用于部署路径的 parity 检查（openpi_bridge.obs_from_dict 走 policy._input_transform，
    这条 transform 链*不*含 RepackTransform —— 见 openpi_bridge.py 里
    create_trained_policy 的调用点，repack_transforms 用的是默认空 Group()），
    调用方必须自己把 LeRobotDataset 的列名 repack 成 AlohaInputs 期望的
    {"images": {...}, "state": [...], "prompt": str} 形状。
    """
    return {
        "images": {
            "cam_high": _to_numpy(raw_item["observation.images.cam_high"]),
            "cam_left_wrist": _to_numpy(raw_item["observation.images.cam_left_wrist"]),
            "cam_right_wrist": _to_numpy(raw_item["observation.images.cam_right_wrist"]),
        },
        "state": _to_numpy(raw_item["observation.state"]),
        "prompt": raw_item["prompt"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="只采前 N 帧（冒烟用，0=全采）")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="SigLIP 前向的 batch。jit 后 bs=32 是 5.7 ms/帧（bs=8 是 7.4）。"
        "改这个值会触发 jit 重编译，但每次运行只编译一次。",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader 进程数，用来把 CPU 侧（transform ~55ms + 读盘 ~24ms 每帧）"
        "藏到 GPU 后面。0 = 主进程串行。",
    )
    args = p.parse_args()

    out = pathlib.Path(args.out)
    ckpt_path = pathlib.Path(args.ckpt)

    # --- 图像侧：policy 的完整 transform 栈 ---
    cfg = _config.get_config(args.train_config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    # 关键修正：Normalize 用的 norm_stats 必须和 policy 实际用的同源 —— 从 checkpoint
    # 自带的 assets 加载，而不是 cwd 相对的 cfg.assets_dirs（见模块 docstring）。
    norm_stats = _checkpoints.load_norm_stats(ckpt_path / "assets", data_config.asset_id)
    if norm_stats is None:
        raise SystemExit(f"norm_stats not found under {ckpt_path / 'assets' / (data_config.asset_id or '?')}")
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    raw_ds = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tf_ds = _data_loader.transform_dataset(raw_ds, data_config)

    # --- state/action 侧：原始绝对关节角，直接另读一份 LeRobotDataset（不带
    # delta_timestamps，不经 AlohaInputs/adapt_to_pi） ---
    lerobot_ds = LeRobotDataset(args.repo_id)

    n_total = len(tf_ds)
    assert len(lerobot_ds) == n_total, (
        f"tf_ds 和 lerobot_ds 帧数不一致：{n_total} vs {len(lerobot_ds)}；两者必须逐行对齐"
    )
    n = n_total if args.limit <= 0 else min(args.limit, n_total)

    bpf = _bytes_per_frame()
    est_gb = n * bpf / 1e9
    free_gb = _free_gb(out)
    print(
        f"[bank] frames={n}/{n_total}  bytes/frame={bpf}  est_size={est_gb:.1f} GB  "
        f"free={free_gb:.1f} GB (at {out.resolve() if out.exists() else out})"
    )
    if est_gb > free_gb * 0.9:
        raise SystemExit(
            f"磁盘不够：需要 ~{est_gb:.1f} GB，可用 {free_gb:.1f} GB（含 10% 安全余量）。"
            "先删掉上一个任务的 bank，或减小 --limit。"
        )

    policy = openpi_bridge.load_policy(args.train_config, str(ckpt_path))
    # **必须用 jit 版编码器。** 急切执行是 ~6000 ms/帧（flax nnx 的 Python dispatch
    # 开销），28k 帧要 47 小时；jit + bs=32 是 5.7 ms/帧 -> 40 分钟。
    encode = openpi_bridge.make_jitted_encoder(policy)

    writer = BankWriter(out, num_frames=n)
    t0 = time.time()

    def _collate(items):
        """把 batch 个 tf_ds item 堆成一个 batched Observation。

        tf_ds[i] 已经是 transform 后的 dict（numpy/torch 混合）。这条路径与
        obs_from_dict() 内部的 policy._input_transform -> add batch dim ->
        Observation.from_dict 是同一条（唯一差异是 tf_ds 多走了一步 RepackTransform，
        把 LeRobotDataset 的列名 repack 成 AlohaInputs 期望的格式）。
        """
        return _model.Observation.from_dict(
            jax.tree.map(
                lambda *xs: jnp.stack([jnp.asarray(_to_numpy(x)) for x in xs]), *items
            )
        )

    # CPU 侧（tf_ds 的 transform ~55 ms/帧 + LeRobotDataset 读盘 ~24 ms/帧）串行执行的话
    # 会盖过 GPU 的 5.7 ms/帧。用多进程 DataLoader 把它并行掉并预取，藏到 GPU 后面。
    class _FrameSource(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            item = tf_ds[i]
            raw = lerobot_ds[i]
            return {
                "i": i,
                # 必须用 tree.map 递归转换：item["image"] 是个嵌套 dict，
                # 用 {k: _to_numpy(v)} 平铺会把它 np.asarray 成 object 数组，
                # 后面 jnp.asarray 就炸「Dtype object is not a valid JAX array type」。
                "item": jax.tree.map(_to_numpy, item),
                "state": _to_numpy(raw["observation.state"]).astype(np.float32).reshape(-1)[:14],
                "action": _to_numpy(raw["action"]).astype(np.float32).reshape(-1, 14)[0],
                "episode_index": int(raw["episode_index"]),
                "frame_index": int(raw["frame_index"]),
            }

    bs = args.batch_size
    loader = torch.utils.data.DataLoader(
        _FrameSource(),
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,  # 保持 list[dict]，自己 stack
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    done = 0
    for batch in loader:
        items = [b["item"] for b in batch]
        # 最后一个不满的 batch：pad 到 bs 再截回来，避免 jit 为新的 batch 形状重编译
        pad = bs - len(items)
        if pad:
            items = items + [items[-1]] * pad
        z_batch = np.asarray(encode(_collate(items)))[: len(batch)]  # [b, 3, 256, 2048]

        for k, b in enumerate(batch):
            writer.write(
                b["i"],
                latent=z_batch[k],
                action=b["action"],
                state=b["state"],
                episode_index=b["episode_index"],
                frame_index=b["frame_index"],
            )

        done += len(batch)
        if done % (bs * 20) < bs or done == n:
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            print(
                f"[bank] {done}/{n}  {rate:.1f} fps  "
                f"eta {(n - done) / max(rate, 1e-6) / 60:.1f} min",
                flush=True,
            )

    # --- latent norm：按 (stream, dim) z-score，over frames+tokens，流式累加 ---
    print("[bank] computing latent norm ...")
    mean = np.zeros(LATENT_SHAPE[:1] + LATENT_SHAPE[2:], dtype=np.float64)  # [3, 2048]
    sq = np.zeros_like(mean)
    count = 0
    for start in range(0, n, NORM_CHUNK):
        end = min(start + NORM_CHUNK, n)
        chunk = np.array(writer._latents[start:end], copy=True)  # [b, 3, 256, 2048] uint16
        zf = torch.from_numpy(chunk).view(torch.bfloat16).to(torch.float64).numpy()
        # 归约 frames(axis0) + tokens(axis2)，留 stream(axis1) + dim(axis3)
        mean += zf.sum(axis=(0, 2))
        sq += (zf**2).sum(axis=(0, 2))
        count += zf.shape[0] * zf.shape[2]
    mean /= count
    var = sq / count - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))

    # --- action quantiles：在 delta 表示上算，delay=DELAY，跳过跨 episode 的窗口 ---
    print("[bank] computing action quantiles ...")
    states = writer._states
    actions = writer._actions
    eps = writer._episode_index
    deltas = []
    for i in range(max(n - DELAY + 1, 0)):
        j = i + DELAY - 1  # 窗口 actions[i:i+DELAY] 覆盖的最后一帧
        if eps[i] != eps[j]:
            continue
        deltas.append(to_delta(actions[i : i + DELAY], states[i]))
    deltas = np.concatenate(deltas, axis=0) if deltas else np.zeros((1, 14), np.float32)
    quantiles = compute_delta_quantiles(deltas)

    writer.finalize(
        latent_norm={"mean": mean.astype(np.float32), "std": std.astype(np.float32)},
        action_quantiles=quantiles,
        meta={
            "task": args.repo_id,
            "train_config": args.train_config,
            "ckpt": str(ckpt_path),
            "delay": DELAY,
            "stream_keys": list(openpi_bridge.STREAM_KEYS),
        },
    )
    print(f"[bank] done -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
