#!/usr/bin/env python3
"""训练 latent 预测器。

Loss：**纯 MSE on normalized latents，没有 policy loss**。这是最终结论（zou
2026-07-17 拍板保持），不是"暂定"。

为什么是纯 MSE（三条，按重要性）：

1. **它够用，门禁没响。** M3（开环 action 一致性）就是为"纯 latent MSE 不保证
   policy 行为一致"这个风险设的门禁——latent 各维度对下游动作的重要性天差地别，
   MSE 一视同仁。结果 action_ratio 降到 0.09~0.14，三个真机任务全 work，
   实测甚至优于 sync。既然门禁没响，加 policy loss 就没有依据。
2. **真机加 policy loss 非常贵——贵在注入点深度，不在框架。** 真机 latent 截在
   `PaliGemma.img()` 输出 = SigLIP 之后、**gemma_2b LLM 之前**，梯度要穿
   `z_hat -> gemma_2b prefix(2B) -> action expert -> flow 多步去噪 -> actions`；
   而 mainline 的 latent 是 SmolVLA **VLM 的输出**，蒸馏只穿 flow head
   （ours_mainline/data_collection/collect_state_sidecar.py:6）。每步开销从 12M
   前反传变 4.1B 前反传，1.05 it/s -> 0.0x 量级。
   **注意**：这跟 policy 是 JAX 无关（policy 全程冻结、只有 predictor 吃梯度；
   jax.vjp + torch.autograd.Function 跨框架接梯度是可行的）。早期文档里
   "跨框架反传代价过大"的说法**归因错了**，别照抄。
3. LIBERO 版据说配了 policy_target='distill' 但 policy_weight=0.0（想过、没走）。
   **此说法未经证实**——trainer 不在本机，全仓的命中都是本仓文档在自我引用。

若将来真要加：便宜的做法是拿冻结 policy 在少量样本上算 dA/dz 敏感度、固化成
latent 各维的静态权重给 MSE 加权，而不是端到端蒸馏。见 spec §11 的风险行。

要盯的核心数字是 `ratio = loss / stale_baseline`：stale_baseline 是
「原样拷贝陈旧 latent、什么都不预测」的 MSE —— 也正是 naive-async 部署今天
在做的事。predictor 用 identity-start 初始化（零 flow head、transport/
innovation gate 近似关闭），所以 step 0 时它就是原样拷贝陈旧 latent，
ratio 从 ~1.00 起步；如果 ratio 不能显著降到 1 以下，这个方法就没有存在意义。

用法：
    openpi/.venv/bin/python -m ours_pi05.train_predictor \
        --bank banks/plates_stacking \
        --out outputs/predictor/plates_stacking \
        --delay-set 10 --steps 150000
"""

from __future__ import annotations

import argparse
import functools
import math
import pathlib
import time

import torch
from torch.utils.checkpoint import checkpoint


from ours_pi05.fast_loader import FastBatchLoader
from ours_pi05.latent_bank import Bank
from ours_pi05.models.predictor import MotionPriorLatentPredictor


def cosine_lr(step: int, *, base: float, warmup: int, total: int, lr_min: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (base - lr_min) * (1 + math.cos(math.pi * min(p, 1.0)))


def _forward(model, z, motion_actions, delay, z_init, state, *, grad_checkpointing: bool):
    if not grad_checkpointing:
        return model(z, motion_actions, delay, z_init=z_init, state=state)

    # 整体 checkpoint 前向：把整棵 trunk/decoder 的中间激活换成重算，换取显存。
    # use_reentrant=False 是关键——非重入实现不要求传入的张量本身 requires_grad
    # （z/motion_actions/... 都是 dataloader 出来的叶子张量，天然不需要梯度），
    # 只要模型参数 requires_grad=True，反向时仍能正确重算出参数梯度。
    fn = functools.partial(model, z_init=z_init, state=state)
    return checkpoint(fn, z, motion_actions, delay, use_reentrant=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delay-set", type=int, nargs="+", default=[10])
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--batch-size", type=int, default=128, help="实际 batch；见 --accum")
    p.add_argument("--accum", type=int, default=2, help="梯度累积；有效 batch = batch_size * accum")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="机器有 80 核。8 个 worker 喂不饱 GPU：每步要经 /dev/shm 搬 3.2 GB "
        "latent，实测主进程卡在 futex_wait 等数据、GPU 利用率掉到 0%。",
    )
    p.add_argument("--ckpt-every", type=int, default=30_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="bf16 autocast。实测 bs=128 时 fwd+bwd 从 511ms -> 213ms（2.4x），"
        "峰值显存 31 -> 22 GB。loss 仍在 float32 里算。",
    )
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-ego-routing", action="store_true", help="ablation：关掉双臂 ego 路由")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    bank = Bank(args.bank)

    # 直读 memmap 的加载器。torch DataLoader 在这里喂不饱 GPU：每步要经 /dev/shm
    # 搬 3.2 GB latent，实测 GPU 利用率在 100/0/0 之间跳、主进程卡在 futex_wait，
    # 稳态只有 0.6 it/s（纯计算是 2.35）。见 fast_loader.py 的 docstring。
    loader = FastBatchLoader(
        bank,
        args.delay_set,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        read_threads=args.num_workers,
    )
    print(
        f"[train] bank={args.bank}  anchors={len(loader.anchors)}  delay_set={args.delay_set}",
        flush=True,
    )
    it = iter(loader)

    model = MotionPriorLatentPredictor(ego_routing=not args.no_ego_routing).to(device)
    n_param = sum(p_.numel() for p_ in model.parameters())
    print(f"[train] predictor params: {n_param / 1e6:.2f} M  ego_routing={not args.no_ego_routing}"
          f"  amp_bf16={args.amp}")

    if args.grad_checkpointing:
        print("[train] gradient checkpointing ON")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # baseline：z_hat = z（原样拷贝陈旧 latent）。这是必须打败的对手。
    step = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    while step < args.steps:
        losses, stale_losses = [], []
        for _ in range(args.accum):
            # FastBatchLoader 是无限迭代器，产出的张量已在 GPU 上、已归一化。
            batch = next(it)
            z, z_target, z_init = batch["z"], batch["z_target"], batch["z_init"]
            ma, delay, state = batch["motion_actions"], batch["delay"], batch["state"]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                z_hat = _forward(
                    model, z, ma, delay, z_init, state,
                    grad_checkpointing=args.grad_checkpointing,
                )
            loss = torch.nn.functional.mse_loss(z_hat.float(), z_target)
            (loss / args.accum).backward()

            losses.append(loss.detach())
            with torch.no_grad():
                stale_losses.append(torch.nn.functional.mse_loss(z, z_target).detach())

        lr = cosine_lr(step, base=args.lr, warmup=args.warmup, total=args.steps, lr_min=args.lr_min)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            l = torch.stack(losses).mean().item()
            sl = torch.stack(stale_losses).mean().item()
            el = time.time() - t0
            ratio = l / max(sl, 1e-9)
            print(
                f"[train] step {step}/{args.steps}  loss {l:.5f}  "
                f"stale_baseline {sl:.5f}  RATIO {ratio:.3f}  "
                f"lr {lr:.2e}  {step / max(el, 1e-6):.2f} it/s",
                flush=True,
            )

        if step % args.ckpt_every == 0 or step == args.steps:
            path = out / f"predictor_{step}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "latent_norm": bank.latent_norm,
                    "action_quantiles": bank.action_quantiles,
                    "args": vars(args),
                    "step": step,
                },
                path,
            )
            print(f"[train] saved {path}", flush=True)

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[train] peak GPU memory allocated: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
