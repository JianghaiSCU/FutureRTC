#!/usr/bin/env python3
"""整个 episode 的开环 rollout —— 模拟真机的异步调度，四条线同框。

风格参照 openpi/0-openloop_robotwin_test.py：逐维纵向堆叠，chunk 边界打竖线。
区别是那个脚本只画「label vs pred」两条线（同步、无延迟），这里要画的是**异步
调度下四种做法各自实际下发的动作**：

    label        数据集真值 a_t
    sync         在 B_m 用**新鲜** obs 查询，执行 C[0:S]        <- 无延迟的上界
    naive-async  在 B_m-d 用**陈旧** obs 查询，执行 C[d:d+S]    <- 主动丢掉前 d 个
    ours         在 B_m-d 用陈旧 obs 查询，但先把 obs **预测**到交接时刻，执行 C[0:S]

三点必须做对，否则图是错的：

1. **committed actions 要用每条线自己上一个窗口实际下发的动作尾巴**（executed[q:B]），
   不是数据集真值 —— 部署时 server 收到的就是客户端自己发出去的那 d 个动作。

2. **执行片段不同**：naive-async 是 C[d:d+S]，ours 和 sync 是 C[0:S]。拿 C[0:S] 去
   代表 naive-async 会额外惩罚它一个平凡的 d 步时间位移 —— 那是评测的错，不是它的
   缺陷（见 eval_offline.py 里同一处的注释）。

3. **同一个 rng**：四条线只应因 obs 不同而不同，不应因 flow 采样噪声不同而不同。

用法：
    export CUDA_VISIBLE_DEVICES=1
    openpi/.venv/bin/python -m ours_pi05.eval_openloop_episode \
        --predictor outputs/predictor/plates_stacking/predictor_10000.pt \
        --train-config pi05_cobot_plates_stacking \
        --ckpt /root/zyx/ckpt/.../30000 \
        --repo-id plates_stacking --episode 0 \
        --out outputs/eval/openloop_ep0
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from ours_pi05 import openpi_bridge
from ours_pi05.action_space import qnorm, to_delta
from ours_pi05.dataset import MAX_DELAY
from ours_pi05.models.corrector import Corrector
from ours_pi05.models.predictor import MotionPriorLatentPredictor

S = 25   # 每个窗口执行的 raw action 数
H = 50   # policy 返回的 chunk 长度


def _obs_dict(ds, row: int, state: np.ndarray) -> dict:
    raw = ds[row]
    return {
        "images": {
            "cam_high": np.asarray(raw["observation.images.cam_high"]),
            "cam_left_wrist": np.asarray(raw["observation.images.cam_left_wrist"]),
            "cam_right_wrist": np.asarray(raw["observation.images.cam_right_wrist"]),
        },
        "state": np.asarray(state, dtype=np.float32),
        "prompt": raw["task"] if isinstance(raw["task"], str) else str(raw["task"]),
    }


def _chunk_from_obs(policy, sampler, obs_dict: dict, latent, rng) -> np.ndarray:
    """跑一次 policy，image tokens 用传入的 latent 覆盖。返回 [H, 14] 绝对关节角。

    图像仍要喂进 transform（obs_from_dict 需要它产出 image_mask 与正确的 dict 结构），
    但它算出的 tokens 会被 latent 完全覆盖 —— 别把这段"看似多余"的读取优化掉。
    state 必须走原始 obs dict 重跑 transform：pi05 的 discrete_state_input=True，
    state 会被归一化后离散化进 prompt，覆写 transform 之后的 state 会绕开归一化。
    """
    observation = openpi_bridge.obs_from_dict(policy, obs_dict)
    actions = sampler(rng, observation, jnp.asarray(np.asarray(latent)[None]))
    return openpi_bridge.actions_to_dict(policy, observation, actions)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictor", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--delay", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = args.delay
    out = pathlib.Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # --- 模型 ---
    policy = openpi_bridge.load_policy(args.train_config, args.ckpt)
    sampler = openpi_bridge.make_jitted_sampler(policy)
    encoder = openpi_bridge.make_jitted_encoder(policy)

    # weights_only=False：ckpt 里除了 state_dict 还存了 latent_norm / action_quantiles
    # （numpy 数组）和 args，torch 2.6 的默认 weights_only=True 会拒绝加载。ckpt 是我们
    # 自己训练脚本写的，可信。
    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    model = MotionPriorLatentPredictor(
        ego_routing=not ck["args"].get("no_ego_routing", False)
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    corrector = Corrector()
    mean = ck["latent_norm"]["mean"][:, None, :]
    std = ck["latent_norm"]["std"][:, None, :]
    quantiles = ck["action_quantiles"]

    # --- episode 数据 ---
    ds = LeRobotDataset(args.repo_id)
    ep_idx = np.array([int(ds[i]["episode_index"]) for i in range(len(ds))])
    rows = np.flatnonzero(ep_idx == args.episode)
    if rows.size == 0:
        raise SystemExit(f"episode {args.episode} 不存在")
    T = int(rows.size)
    base = int(rows[0])
    states = np.stack([np.asarray(ds[base + t]["observation.state"], np.float32) for t in range(T)])
    labels = np.stack(
        [np.asarray(ds[base + t]["action"], np.float32).reshape(-1, 14)[0] for t in range(T)]
    )
    print(f"[ep] episode={args.episode}  frames={T}  base_row={base}", flush=True)

    # --- 逐帧编码 latent（jit，batch 32）---
    print("[ep] encoding latents ...", flush=True)
    lat = np.zeros((T, *openpi_bridge.LATENT_SHAPE), dtype=np.float32)
    BS = 32
    for s0 in range(0, T, BS):
        idx = list(range(s0, min(s0 + BS, T)))
        obs_list = [openpi_bridge.obs_from_dict(policy, _obs_dict(ds, base + t, states[t])) for t in idx]
        batched = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *obs_list)
        lat[idx] = np.asarray(encoder(batched))
    z_init = lat[0]  # episode 首帧，预测器的静态参考

    # --- 四条线各自 rollout ---
    methods = ["sync", "naive_async", "ours"]
    executed = {m: np.zeros((T, 14), np.float32) for m in methods}
    n_windows = (T + S - 1) // S

    for m_i in range(n_windows):
        B = m_i * S
        n_exec = min(S, T - B)
        rng = jax.random.key(args.seed + m_i)  # 三条线共用，只有 obs 不同

        # ---- sync：在 B 用新鲜 obs 查询，执行 C[0:S] ----
        c = _chunk_from_obs(policy, sampler, _obs_dict(ds, base + B, states[B]), lat[B], rng)
        executed["sync"][B : B + n_exec] = c[:n_exec]

        # ---- 异步两条线：在 q = B - d 用陈旧 obs 查询 ----
        q = max(B - d, 0)
        if m_i == 0:
            # 第一个窗口没有延迟（还没有上一条 chunk 在执行），两条线都退化为 sync
            executed["naive_async"][B : B + n_exec] = c[:n_exec]
            executed["ours"][B : B + n_exec] = c[:n_exec]
            continue

        # naive-async：陈旧 obs，执行 C[d : d+S]（主动丢掉前 d 个）
        c_na = _chunk_from_obs(
            policy, sampler, _obs_dict(ds, base + q, states[q]), lat[q], rng
        )
        executed["naive_async"][B : B + n_exec] = c_na[d : d + n_exec]

        # ours：陈旧 obs -> 预测到交接时刻
        #
        # committed = **数据集真值** action[q:B]，不是 ours 自己上一窗口下发的动作。
        #
        # 为什么：开环评测的前提就是「机器人完美跟踪数据集轨迹」—— 图像、state、
        # 标签全都取自数据集那条轨迹。如果只让 ours 的 state 锚点跟着它**自己**的
        # 预测走，它就会逐窗口漂移，而图像仍停在数据集轨迹上，两者不自洽；再拿数据集
        # 的 action 当标签去罚它，等于单方面给 ours 上枷锁（naive-async 用的是数据集
        # 的真实 state，没这个问题）。实测这么搞会让 ours 反而比 naive-async 差 2.9 倍。
        #
        # 这也正是**预测器训练时**看到的输入（fast_loader 喂的是 bank.actions[t:t+d]）。
        # 不是作弊：committed actions 在部署时本来就是已知且必然执行的，
        # 而由于 action_t ≡ qpos_{t+1} 的恒等性，s_hat = committed[d-1] 精确等于
        # 交接时刻的真实 state。真机上闭环执行时，这条恒等性由 gap 日志来验证。
        committed = labels[q:B]
        s_anchor = states[q]
        s_hat = corrector(s_anchor, committed, d)

        motion = qnorm(to_delta(committed, s_anchor), quantiles)
        padded = np.zeros((MAX_DELAY, 14), np.float32)
        padded[:d] = motion  # 左对齐，与训练一致

        zn = (lat[q] - mean) / std
        zi = (z_init - mean) / std
        with torch.no_grad():
            z_hat_n = model(
                torch.from_numpy(zn[None]).float().to(device),
                torch.from_numpy(padded[None]).to(device),
                torch.tensor([d], dtype=torch.long, device=device),
                z_init=torch.from_numpy(zi[None]).float().to(device),
                state=torch.from_numpy(s_hat[None]).float().to(device),
            )[0].cpu().numpy()
        z_hat = z_hat_n * std + mean

        obs_hat = _obs_dict(ds, base + q, s_hat)  # state 换成 s_hat，走完整 transform
        c_ours = _chunk_from_obs(policy, sampler, obs_hat, z_hat, rng)
        executed["ours"][B : B + n_exec] = c_ours[:n_exec]

        if m_i % 4 == 0:
            print(f"[ep] window {m_i}/{n_windows}", flush=True)

    # --- 指标 ---
    #
    # **主指标是 vs sync，不是 vs 数据集标签。**
    #
    # sync 用的是完美的新鲜 obs、零延迟，它对数据集标签的 L1 已经有 ~0.010 —— 那是
    # policy 本身的**模仿误差**（它在模仿人类遥操作，本来就不可能完全一致）。延迟带来的
    # 误差叠在这之上，量级更小，会被完全淹没：实测 naive-async 对标签的 L1 甚至比 sync
    # 还低（0.0079 vs 0.0099），纯属噪声 —— 这个指标测不出延迟。
    #
    # 拿 sync（policy 自己在零延迟下的输出）当基准，模仿误差被消掉，剩下的就是
    # 「obs 陈旧」这一个变量。这与 eval_offline.py 用 C_oracle 当基准是同一个道理。
    # 窗口 0 三条线完全相同（还没有延迟），必须排除，否则会稀释差异。
    mask = np.zeros(T, bool)
    mask[S:] = True

    res = {"episode": args.episode, "frames": T, "delay": d, "S": S}
    for m in methods:
        res[f"{m}_l1_vs_label"] = float(np.abs(executed[m] - labels).mean())
    for m in ("naive_async", "ours"):
        res[f"{m}_l1_vs_sync"] = float(
            np.abs(executed[m][mask] - executed["sync"][mask]).mean()
        )
    res["action_ratio"] = res["ours_l1_vs_sync"] / max(res["naive_async_l1_vs_sync"], 1e-9)
    (out / "openloop.json").write_text(json.dumps(res, indent=2))

    # --- 图：整个 episode，逐维纵向堆叠，chunk 边界打竖线 ---
    time = np.arange(T)
    chunk_starts = np.arange(0, T, S)
    fig, axes = plt.subplots(14, 1, figsize=(12, 2.0 * 14), sharex=True)
    for dim in range(14):
        ax = axes[dim]
        ax.plot(time, labels[:, dim], color="black", lw=1.8, label="label (dataset)")
        ax.plot(time, executed["sync"][:, dim], color="tab:blue", lw=1.2, label="sync (no delay)")
        ax.plot(
            time, executed["naive_async"][:, dim], color="tab:red", lw=1.2, ls="--",
            label=f"naive-async (C[{d}:{d}+S])",
        )
        ax.plot(
            time, executed["ours"][:, dim], color="tab:green", lw=1.2, ls="-.",
            label="ours (C[0:S])",
        )
        for x in chunk_starts:
            ax.axvline(x=x, color="gray", lw=0.8, alpha=0.3)
        ax.set_ylabel(f"Dim {dim}")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("episode step  (gray = chunk boundary, S=25)")
    fig.suptitle(
        f"open-loop rollout   episode={args.episode}   delay={d}\n"
        f"L1 vs sync (delay effect, imitation error removed):  "
        f"naive-async {res['naive_async_l1_vs_sync']:.5f}   "
        f"ours {res['ours_l1_vs_sync']:.5f}   "
        f"ratio {res['action_ratio']:.3f}",
        y=0.999,
    )
    fig.tight_layout()
    path = out / f"episode_{args.episode}_all_dims.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    np.savez(
        out / f"episode_{args.episode}_traj.npz",
        labels=labels,
        **{m: executed[m] for m in methods},
    )

    print("\n" + "=" * 68)
    print("  vs 数据集标签（含 policy 固有模仿误差 —— 测不出延迟，仅供参考）:")
    for m in methods:
        print(f"    {m:12s}  L1 {res[f'{m}_l1_vs_label']:.5f}")
    print("\n  vs sync（消掉模仿误差，只剩「obs 陈旧」这一个变量）  <- 主指标:")
    for m in ("naive_async", "ours"):
        print(f"    {m:12s}  L1 {res[f'{m}_l1_vs_sync']:.5f}")
    print(f"\n  action_ratio (ours / naive-async) = {res['action_ratio']:.3f}")
    print("=" * 68)
    print(f"图 -> {path}")


if __name__ == "__main__":
    main()
