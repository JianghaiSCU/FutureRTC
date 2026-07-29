"""唯一与 openpi 耦合的模块：从 pi0.5 里 capture image tokens，以及把预测出的
tokens 注入回去。

采集/注入点是同一个：openpi/src/openpi/models/pi0.py:114
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
它是 SigLIP 视觉塔 + PaliGemma 投影的输出，已经在 LLM 的 token embedding 空间里，
位于 LLM transformer 之前 —— 正是 embed_prefix 拼进前缀序列的那一段。

sample_actions_with_tokens 是 pi0.Pi0.sample_actions（pi0.py:216-275）的镜像，
唯一改动就是 embed_prefix 里的 img() 调用换成传入的 image_tokens。

等价性（tests/test_bridge.py 强制）
----------------------------------
把 encode_images 采出的真实 tokens 原样注入，本模块与 openpi 自己的
`model.sample_actions` **逐位相等**（实测 max|Δ| = 0.0）。前缀层面更强：
_embed_prefix_with_tokens 产出的 tokens / input_mask / ar_mask 与
`model.embed_prefix` 逐位相等。

**不要拿它去比 `policy._sample_actions`。** 后者被 nnx_utils.module_jit 包过
（openpi/src/openpi/policies/policy.py:63），而模型 dtype 是 bfloat16
（openpi/src/openpi/models/pi0_config.py:20）。openpi **自己的** sample_actions
急切执行 vs jit 执行就差 max|Δ| ≈ 0.045（action 量级 |a|.mean ≈ 0.097），纯粹是
XLA 算子融合在 bf16 下的舍入差异，与本模块无关。比错对象会得出"注入坏了"的
错误结论 —— 这个坑踩过一次了。

部署要速度就用 make_jitted_sampler()。
"""

from __future__ import annotations

import pathlib
import sys

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

_OPENPI_SRC = pathlib.Path(__file__).resolve().parents[1] / "openpi" / "src"
if str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

# Observation.images 的迭代顺序 == embed_prefix 的拼接顺序 == 我们的 stream 顺序。
# 真正钉死这个顺序的不是 AlohaInputs（它只是恰好按这个顺序建字典，不构成保证）
# —— 是 openpi/src/openpi/models/model.py：IMAGE_KEYS（~39 行）+
# preprocess_observation（~162 行）。preprocess_observation 用
# `for key in image_keys` 按 IMAGE_KEYS 的顺序重新构建 out_images，
# 所以不管输入 transform 产出的字典原始顺序是什么，到达 embed_prefix 的顺序
# 永远固定为 IMAGE_KEYS。下面的断言把这个保证钉死成一条可执行检查。
STREAM_KEYS: tuple[str, str, str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
assert STREAM_KEYS == _model.IMAGE_KEYS, (
    f"STREAM_KEYS {STREAM_KEYS} != openpi IMAGE_KEYS {_model.IMAGE_KEYS}；"
    "两者必须一致，否则 encode_images/_embed_prefix_with_tokens 的 stream 顺序假设不成立"
)

# 256 tokens = 16x16 网格。注意 **不是** 196：SigLIP 的 patch 大小是从 variant 字符串
# 解析出来的（openpi/src/openpi/models/siglip.py:306），PaliGemma 用 So400m/**14**，
# 224/14 = 16 -> 16x16 = 256。siglip.py:192 那个 patch_size=(16,16) 只是类的默认值，
# 不是实际生效的值。16x16 仍是完美正方形，所以预测器的 grid_sample warp 成立。
LATENT_SHAPE: tuple[int, int, int] = (3, 256, 2048)
TOKEN_GRID_SIDE: int = 16


def load_policy(train_config_name: str, ckpt_path: str):
    """加载训好的 pi0.5 policy（JAX）。"""
    config = _config.get_config(train_config_name)
    return _policy_config.create_trained_policy(config, ckpt_path)


def obs_from_dict(policy, obs_dict: dict) -> _model.Observation:
    """跑完 policy 自己的 input transform 并加 batch 维，与 Policy.infer 逐步一致
    （openpi/src/openpi/policies/policy.py:70-89）。"""
    inputs = jax.tree.map(lambda x: x, obs_dict)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    return _model.Observation.from_dict(inputs)


def _encode_images_core(model, observation: _model.Observation):
    """SigLIP 前向，取出 image tokens。返回 jax array [B, 3, 256, 2048]。"""
    obs = _model.preprocess_observation(None, observation, train=False)
    tokens = [model.PaliGemma.img(obs.images[name], train=False)[0] for name in obs.images]
    return jnp.stack(tokens, axis=1)  # [B, C, T, D]


def encode_images(policy, observation: _model.Observation) -> np.ndarray:
    """跑 SigLIP，取出 image tokens。返回 [B, 3, 256, 2048]（stream 顺序 = STREAM_KEYS）。

    **急切执行，慢。** 实测 ~6000 ms/帧（bs=1）—— flax nnx 的 Python 层 dispatch 开销
    完全压过了实际计算。单次调用（部署时每个 window 一次）无所谓，但要采 28k 帧的
    latent bank 就是 47 小时。批量采集**必须**用 make_jitted_encoder()（bs=32 时
    5.7 ms/帧，快 1000 倍）。
    """
    z = np.asarray(_encode_images_core(policy._model, observation))
    assert z.shape[1:] == LATENT_SHAPE, (
        f"encode_images 输出形状与 LATENT_SHAPE 不符：expected {LATENT_SHAPE}, got {z.shape[1:]}"
        "（配置或分辨率变了会导致这里静默写出维度错误的 latent bank）"
    )
    return z


def make_jitted_encoder(policy):
    """给批量采集用的 jit 版 encode_images。返回 fn(observation) -> [B, 3, 256, 2048]。

    与 make_jitted_sampler 同一个模式（nnx.split 冻结状态 + jax.jit + nnx.merge，
    照搬 openpi/src/openpi/shared/nnx_utils.py:15-43）。

    实测（plates_stacking，1×A100）：
        急切 bs=1   : 5969 ms/frame
        jit   bs=8  :    7.4 ms/frame
        jit   bs=32 :    5.7 ms/frame
    采 28,360 帧：47.6 小时 -> 40 分钟（此时瓶颈转到 CPU 侧的 transform + 读盘）。

    注意 batch size 变化会触发重新编译，所以采集时要固定 batch，最后一个不满的
    batch 要 pad 或单独处理。
    """
    graphdef, state = nnx.split(policy._model)

    @jax.jit
    def fn(state, observation):
        model = nnx.merge(graphdef, state)
        return _encode_images_core(model, observation)

    def wrapper(observation: _model.Observation):
        return fn(state, observation)

    return wrapper


def _embed_prefix_with_tokens(model, obs: _model.Observation, image_tokens: jnp.ndarray):
    """pi0.embed_prefix（pi0.py:106-137）的镜像，img() 换成传入的 image_tokens。"""
    input_mask = []
    ar_mask: list[bool] = []
    tokens = []

    for i, name in enumerate(obs.images):
        img_tok = image_tokens[:, i]  # [B, T, D]
        tokens.append(img_tok)
        input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=img_tok.shape[1]))
        ar_mask += [False] * img_tok.shape[1]

    if obs.tokenized_prompt is not None:
        tokenized_inputs = model.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

    return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.array(ar_mask)


def _sample_actions_with_tokens(
    model,
    rng,
    observation: _model.Observation,
    image_tokens,
    *,
    num_steps: int = 10,
    noise: jnp.ndarray | None = None,
):
    """核心逻辑，接收 model（而不是 policy），这样 make_jitted_sampler 可以直接把
    nnx.merge 出来的 model 传进来，不用再造一个假 policy 壳子。"""
    image_tokens = jnp.asarray(image_tokens)
    obs = _model.preprocess_observation(None, observation, train=False)

    dt = -1.0 / num_steps
    batch_size = obs.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = _embed_prefix_with_tokens(model, obs, image_tokens)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            obs, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def sample_actions_with_tokens(
    policy,
    rng,
    observation: _model.Observation,
    image_tokens,
    *,
    num_steps: int = 10,
    noise: jnp.ndarray | None = None,
):
    """pi0.sample_actions（pi0.py:216-275）的镜像，prefix 用传入的 image_tokens。

    约定与 openpi 一致：t=1 是噪声、t=0 是目标，dt = -1/num_steps。noise=None 时
    与原版一样内部用 jax.random.normal(rng, ...) 采样；显式传 noise 可以让"同一份
    噪声、不同观测（stale/predicted/oracle）"这种离线对比实验更稳，不必依赖"同一个
    rng 在等价的调用路径下会产生等价噪声"这种隐式假设。

    返回 raw actions [B, action_horizon=50, action_dim=32]（pi0_config.py 的
    Pi0Config 默认值：action_dim=32, action_horizon=50）。**这是模型的归一化
    action 空间，没有 un-normalize，也没有裁到机器人的 14 维** —— 千万不要把这个
    直接喂给机器人。想要机器人能执行的 14 维绝对关节角，要再过一遍
    policy._output_transform（AlohaOutputs，aloha_policy.py:99-101：先切片
    `actions[:, :14]` 再反归一化）。用本模块的 actions_to_dict() 做这一步，
    不要绕开它直接摸 policy._output_transform —— 那样就把 openpi 耦合面泄漏到
    了这个模块之外。
    """
    return _sample_actions_with_tokens(
        policy._model, rng, observation, image_tokens, num_steps=num_steps, noise=noise
    )


def actions_to_dict(policy, observation: _model.Observation, actions) -> np.ndarray:
    """model 空间的 raw actions -> 机器人空间的 14 维绝对关节角。

    是 obs_from_dict 的对称镜像：obs_from_dict 包 policy._input_transform，
    这个函数包 policy._output_transform。复现 Policy.infer 里 sample_actions
    之后到 output_transform 之前的那一步（policy.py:90-102）：

        outputs = {"state": inputs["state"], "actions": self._sample_actions(...)}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)  # 去掉 batch 维
        outputs = self._output_transform(outputs)

    只支持 batch=1（与 Policy.infer 一致，它也是靠 `x[0, ...]` 硬去掉 batch 维）。

    返回 actions [action_horizon=50, 14]，已反归一化、可以直接下发给机器人。
    """
    outputs = {
        "state": np.asarray(observation.state),
        "actions": np.asarray(actions),
    }
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
    outputs = policy._output_transform(outputs)
    return outputs["actions"]


def make_jitted_sampler(policy, *, num_steps: int = 10):
    """给部署用的 jit 版 sample_actions_with_tokens。

    照搬 openpi 自己的做法（openpi/src/openpi/shared/nnx_utils.py:15-43）：nnx.split
    冻结模型状态，jax.jit 编译，调用时 nnx.merge 还原。急切执行每次查询要几秒，
    真机 d=10 步的预算根本不够。

    返回 fn(rng, observation, image_tokens) -> raw actions [B, 50, 32]，与
    sample_actions_with_tokens 一样是模型的归一化 action 空间，同样需要过
    actions_to_dict()/policy._output_transform 才能拿到机器人能执行的 14 维。

    注意：jit 后的结果与急切版会有 bf16 融合级别的差异（openpi 自己也一样，
    见模块 docstring）。等价性测试测的是急切版；部署用 jit 版。
    """
    graphdef, state = nnx.split(policy._model)

    def fn(state, rng, observation, image_tokens):
        model = nnx.merge(graphdef, state)
        return _sample_actions_with_tokens(model, rng, observation, image_tokens, num_steps=num_steps, noise=None)

    jitted = jax.jit(fn)

    def wrapper(rng, observation, image_tokens):
        return jitted(state, rng, observation, jnp.asarray(image_tokens))

    return wrapper
