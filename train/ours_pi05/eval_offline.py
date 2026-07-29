#!/usr/bin/env python3
"""M3 门禁：离线验收预测器。

第一层 —— latent MSE：
    z_hat vs 真实 z_{t+d}，对比两个 baseline：
      stale : 直接用 z_t（naive-async 实际在做的事）
      zero  : 预测 = 输入（等价于 stale，此处作为数值 sanity）

第二层 —— 开环 action 一致性（**更重要**）：
    latent 准不准不重要，policy 行为一致才重要。
    三个 chunk 用同一个 rng 采样，只有 obs 不同：
      C_oracle = policy(真实未来 latent, 真实未来 state)
      C_stale  = policy(陈旧 latent,     陈旧 state)
      C_ours   = policy(z_hat,          s_hat)
    要求 ||C_ours - C_oracle|| 显著小于 ||C_stale - C_oracle||。

用法：
    openpi/.venv/bin/python -m ours_pi05.eval_offline \
        --bank banks/plates_stacking \
        --predictor outputs/predictor/plates_stacking/predictor_150000.pt \
        --train-config pi05_cobot_plates_stacking \
        --ckpt /root/zyx/ckpt/.../30000 \
        --repo-id plates_stacking \
        --delay 10 --num-samples 200 \
        --out outputs/eval/plates_stacking
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

from ours_pi05 import openpi_bridge
from ours_pi05.action_space import qnorm, to_delta
from ours_pi05.dataset import MAX_DELAY
from ours_pi05.latent_bank import Bank
from ours_pi05.models.corrector import Corrector
from ours_pi05.models.predictor import MotionPriorLatentPredictor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _chunk(
    policy, lerobot_ds, frame_row: int, latent: np.ndarray, state: np.ndarray, rng, sampler
) -> np.ndarray:
    """跑一次 policy，image tokens 用传入的 latent 覆盖、state 用传入的 state。

    **必须从原始 obs dict 出发重跑 policy._input_transform**，不能直接覆写
    transform 之后的 state —— pi05 的 discrete_state_input=True，state 会被
    Normalize 之后再由 TokenizePrompt 离散化进 prompt
    （openpi/src/openpi/transforms.py:255-266）。覆写已 transform 的 state 会
    绕开归一化，模型读到的是垃圾。

    图像照常喂进 transform 栈（因为 obs_from_dict 需要它产出 image_mask 与
    正确的 dict 结构），但它算出的 tokens **会被 latent 完全覆盖**，所以图像
    内容不影响结果 —— 真正起作用的只有 prompt、state、image_mask。别把这段
    "看似多余"的图像读取优化掉。
    """
    raw = lerobot_ds[frame_row]
    obs_dict = {
        "images": {
            "cam_high": np.asarray(raw["observation.images.cam_high"]),
            "cam_left_wrist": np.asarray(raw["observation.images.cam_left_wrist"]),
            "cam_right_wrist": np.asarray(raw["observation.images.cam_right_wrist"]),
        },
        "state": np.asarray(state, dtype=np.float32),
        "prompt": raw["task"] if isinstance(raw["task"], str) else str(raw["task"]),
    }
    observation = openpi_bridge.obs_from_dict(policy, obs_dict)
    # **必须用 jit 版采样器。** 急切执行下 pi0.5 的 flow 采样每次要好几秒（flax nnx 的
    # Python 层 dispatch 开销，不是算力），而每个样本要采 3 次 chunk（oracle/stale/ours），
    # 200 个样本就是 600 次 —— 实测 20 个样本要 10 分钟。sampler 由调用方建一次、反复用。
    actions = sampler(rng, observation, jnp.asarray(np.asarray(latent)[None]))
    # 用 bridge 的输出侧接口，别碰 policy._output_transform 私有属性。
    # 原始 sampler 返回的是 [B, 50, 32]，**模型归一化空间**，不是机器人的 14 维。
    # actions_to_dict 走 AbsoluteActions（会把我们传进去的 state 加回来，
    # openpi/src/openpi/transforms.py:241）+ AlohaOutputs（切 [:, :14] + 反归一化），
    # 所以 c_ours 天然锚定在 s_hat 上。
    return openpi_bridge.actions_to_dict(policy, observation, actions)  # [50, 14]


def _plot_sample(row: int, d: int, c_oracle: np.ndarray, c_stale: np.ndarray, c_ours: np.ndarray, out_path: pathlib.Path) -> None:
    """每维一个子图，三条线画在同一坐标系里，跟
    openpi/0-openloop_lerobot_test_dos.py 的纵向堆叠风格一致。

    画的是**时间对齐后、真机上实际下发的那 S=25 个动作**：
        oracle      = C_oracle[0:S]
        naive-async = C_stale[d:d+S]   （它主动丢掉前 d 个陈旧动作）
        ours        = C_ours[0:S]
    三者都对应窗口 [B_m, B_m+S)，所以可以逐点比较。

    别画整条 50 步的 chunk 去比 —— C_stale 是从 t 起算的、C_oracle 是从 t+d 起算的，
    按 index 0 对齐会看到一个平凡的 d 步位移，那是评测的错觉，不是 naive-async 的缺陷。
    """
    T, D = c_oracle.shape
    time = np.arange(T)
    fig, axes = plt.subplots(D, 1, figsize=(8, 1.8 * D), sharex=True)
    for dim in range(D):
        ax = axes[dim]
        ax.plot(time, c_oracle[:, dim], color="tab:blue", lw=1.6, label="oracle = C_oracle[0:S]")
        ax.plot(
            time, c_stale[:, dim], color="tab:red", lw=1.2, ls="--",
            label=f"naive-async = C_stale[{d}:{d}+S]",
        )
        ax.plot(time, c_ours[:, dim], color="tab:green", lw=1.2, ls="-.", label="ours = C_ours[0:S]")
        ax.set_ylabel(f"dim {dim}")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=7)
        if dim == D - 1:
            ax.set_xlabel("chunk step")
    fig.suptitle(f"open-loop action consistency  row={row}  delay={d}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True)
    p.add_argument("--predictor", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--delay", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-plot-samples", type=int, default=4, help="画图的样本数（图较大，别设太多）")
    p.add_argument("--device", default="cuda", help="predictor(torch) 的 device；GPU 显存不足时可临时切 cpu 验证流程")
    args = p.parse_args()

    d = args.delay
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    bank = Bank(args.bank)
    # weights_only=False：checkpoint 里带 numpy 数组（latent_norm/action_quantiles）和
    # 一个 argparse Namespace 转的 dict（args），PyTorch>=2.6 的默认 weights_only=True
    # 不认 numpy 的 _reconstruct global，会直接炸。这是我们自己训练脚本产出的可信文件。
    ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
    model = MotionPriorLatentPredictor(ego_routing=not ck["args"].get("no_ego_routing", False))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    corrector = Corrector()

    mean = bank.latent_norm["mean"][:, None, :]
    std = bank.latent_norm["std"][:, None, :]

    policy = openpi_bridge.load_policy(args.train_config, args.ckpt)
    # 建一次，600 次采样共用（见 _chunk 的注释）。第一次调用会付一次 jit 编译。
    sampler = openpi_bridge.make_jitted_sampler(policy)
    # 原始 LeRobot 帧（给 _chunk 取图像与 task prompt）；bank 的行号与它一一对应
    # （collect_latents.py 逐行写入，无 shuffle，bank row i == lerobot_ds[i]）。
    lerobot_ds = LeRobotDataset(args.repo_id)

    ep = bank.episode_index
    valid = [t for t in range(len(bank) - d) if ep[t] == ep[t + d]]
    rng_pick = np.random.default_rng(args.seed)
    rows = rng_pick.choice(valid, size=min(args.num_samples, len(valid)), replace=False)

    lat_ours, lat_stale = [], []
    act_ours, act_stale = [], []
    act_stale_naive_unaligned = []  # 诊断：不做时间对齐的 stale（会虚高，见下）
    plotted = 0

    for k, row in enumerate(rows):
        row = int(row)
        ep_id = int(bank.episode_index[row])
        init_row = bank.episode_starts[ep_id]

        z_raw = bank.read_latent(row)              # [3,256,2048] 陈旧
        z_true = bank.read_latent(row + d)         # 真实未来
        z_init_raw = bank.read_latent(init_row)

        s_anchor = bank.states[row]
        committed = bank.actions[row : row + d]
        s_hat = corrector(s_anchor, committed, d)          # == committed[-1]
        s_true = bank.states[row + d]

        # --- 第一层：latent ---
        zn = (z_raw - mean) / std
        zt = (z_true - mean) / std
        zi = (z_init_raw - mean) / std

        motion = qnorm(to_delta(committed, s_anchor), bank.action_quantiles)
        padded = np.zeros((MAX_DELAY, 14), dtype=np.float32)
        padded[:d] = motion  # 左对齐；与 dataset.py 保持一致

        with torch.no_grad():
            z_hat_n = model(
                torch.from_numpy(zn[None]).float().to(device),
                torch.from_numpy(padded[None]).to(device),
                torch.tensor([d], dtype=torch.long, device=device),
                z_init=torch.from_numpy(zi[None]).float().to(device),
                # 预测器的 state 输入是**交接时刻的校正后 state**（s_hat），不是陈旧的
                # s_anchor —— 必须与训练时一致（fast_loader.py:126 喂的就是
                # corrector(s_anchor, committed, d)）。喂错不会报错，只会让 z_hat 在一个
                # 训练时从未见过的条件下生成。
                #
                # 注意 s_anchor 仍然是 to_delta 的锚点（上面第 179 行）——那是"相对陈旧
                # 位姿的位移"，锚点本来就该是陈旧的。两者是紧挨着的两个不同的量。
                state=torch.from_numpy(s_hat[None]).float().to(device),
            )[0].cpu().numpy()

        lat_ours.append(float(np.mean((z_hat_n - zt) ** 2)))
        lat_stale.append(float(np.mean((zn - zt) ** 2)))

        z_hat = z_hat_n * std + mean  # 反归一化回 SigLIP 空间

        # --- 第二层：开环 action 一致性 ---
        # 三个 chunk 共用同一个 rng —— 否则比的是 flow 采样噪声，不是 obs 差异。
        rng = jax.random.key(args.seed + k)

        c_oracle_full = _chunk(policy, lerobot_ds, row + d, z_true, s_true, rng, sampler)
        c_stale_full = _chunk(policy, lerobot_ds, row, z_raw, s_anchor, rng, sampler)
        c_ours_full = _chunk(policy, lerobot_ds, row, z_hat, s_hat, rng, sampler)

        # --- 时间对齐：必须比「真机上实际执行的那一段」，不是整条 chunk ---
        #
        # C_stale 是用 t 时刻的 obs 采的，它输出的是**从 t 开始**的 chunk：a_t, a_{t+1}, ...
        # C_oracle / C_ours 对应的是**从 t+d 开始**的 chunk：a_{t+d}, a_{t+d+1}, ...
        # 直接按 index 0 对齐相减，量到的主要是这个平凡的 d 步时间位移，跟 obs 陈不陈旧
        # 无关 —— 那会把 naive-async 的表现严重高估其误差（结果虚高）。
        #
        # 真机上 naive-async 并不会犯这个错：它在交接时刻执行 Cm[d : d+S]，**主动丢掉
        # 前 d 个陈旧动作**（见 deploy_policy_local_rtc_naive_async_local.py）。而 ours
        # 因为 obs 已经是交接时刻的，整条 chunk 都可用，执行 Cm[0 : S]。
        #
        # 三者对应的都是窗口 [B_m, B_m + S) 内实际下发的 S 个动作，这才可比。
        S = 25
        c_oracle = c_oracle_full[0:S]
        c_stale = c_stale_full[d : d + S]   # naive-async 实际执行的片段
        c_ours = c_ours_full[0:S]

        act_ours.append(float(np.mean(np.abs(c_ours - c_oracle))))
        act_stale.append(float(np.mean(np.abs(c_stale - c_oracle))))

        # 诊断用：不做时间对齐的话会是什么数（用来量化「对齐前 vs 对齐后」的差距）
        act_stale_naive_unaligned.append(
            float(np.mean(np.abs(c_stale_full[0:S] - c_oracle)))
        )

        if plotted < args.num_plot_samples:
            _plot_sample(row, d, c_oracle, c_stale, c_ours, plot_dir / f"sample_row{row}.png")
            plotted += 1

        if k % 20 == 0:
            print(f"[eval] {k}/{len(rows)}")

    res = {
        "delay": d,
        "num_samples": len(rows),
        "latent_mse_ours": float(np.mean(lat_ours)),
        "latent_mse_stale": float(np.mean(lat_stale)),
        "latent_ratio": float(np.mean(lat_ours) / max(np.mean(lat_stale), 1e-9)),
        # 都是窗口 [B_m, B_m+S) 内实际下发的 S=25 个动作：
        #   oracle      = C_oracle[0:25]
        #   naive-async = C_stale[d:d+25]   <- 它主动丢掉前 d 个陈旧动作
        #   ours        = C_ours[0:25]      <- 整条 chunk 可用，这是 ours 的意义
        "action_l1_ours_vs_oracle": float(np.mean(act_ours)),
        "action_l1_stale_vs_oracle": float(np.mean(act_stale)),
        "action_ratio": float(np.mean(act_ours) / max(np.mean(act_stale), 1e-9)),
        # 诊断：若不做时间对齐（拿 C_stale[0:25] 去比），naive-async 会被额外惩罚一个
        # 平凡的 d 步时间位移 —— 那是评测的错，不是它的错。两者差多少即为高估的幅度。
        "_diag_action_l1_stale_unaligned": float(np.mean(act_stale_naive_unaligned)),
        "_diag_action_ratio_unaligned": float(
            np.mean(act_ours) / max(np.mean(act_stale_naive_unaligned), 1e-9)
        ),
    }
    (out / "offline.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 60)
    print(f"latent MSE   ours {res['latent_mse_ours']:.5f}  "
          f"stale {res['latent_mse_stale']:.5f}  ratio {res['latent_ratio']:.3f}")
    print(f"action L1    ours {res['action_l1_ours_vs_oracle']:.5f}  "
          f"stale {res['action_l1_stale_vs_oracle']:.5f}  ratio {res['action_ratio']:.3f}")
    print("=" * 60)
    print("M3 GATE:", "PASS" if res["action_ratio"] < 0.9 else "FAIL")
    print(f"plots saved under {plot_dir}")


if __name__ == "__main__":
    main()
