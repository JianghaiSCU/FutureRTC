"""Bridge 的等价性测试。需要真实 ckpt + GPU，跑得慢（分钟级），但这是整条链路的地基。"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CKPT = pathlib.Path(
    "/root/zyx/ckpt/cvpr2026_RTC/openpi/pi05_cobot_plates_stacking/pi05_cobot_plates_stacking/30000"
)
TRAIN_CONFIG = "pi05_cobot_plates_stacking"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason=f"ckpt not found: {CKPT}")


@pytest.fixture(scope="module")
def policy():
    from ours_pi05.openpi_bridge import load_policy

    return load_policy(TRAIN_CONFIG, str(CKPT))


def _fake_obs():
    """AlohaInputs 要求图像是 CHW —— 它内部走 einops.rearrange(img, "c h w -> h w c")
    （openpi/src/openpi/policies/aloha_policy.py:194）。喂 HWC 会被当成 c=480,h=640,w=3，
    静默变成垃圾并在 PIL resize 时炸掉。现有部署 server 也是先 transpose(2,0,1) 再喂的
    （infer/openpi/deploy_policy_server_local.py:73-74）。
    """
    rng = np.random.default_rng(0)
    return {
        "images": {
            "cam_high": rng.integers(0, 256, (3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": rng.integers(0, 256, (3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": rng.integers(0, 256, (3, 480, 640), dtype=np.uint8),
        },
        "state": rng.normal(size=14).astype(np.float32),
        "prompt": "stack the plates",
    }


def test_encode_images_shape_and_stream_order(policy):
    from openpi.models import model as _model

    from ours_pi05.openpi_bridge import LATENT_SHAPE, STREAM_KEYS, encode_images, obs_from_dict

    obs = obs_from_dict(policy, _fake_obs())
    z = encode_images(policy, obs)
    # 256 tokens = 16x16（SigLIP So400m/**14**，224/14=16）。不是 196 —— siglip.py:192
    # 那个 patch_size=(16,16) 只是类默认值，实际值从 variant 字符串解析（siglip.py:306）。
    assert z.shape == (1, *LATENT_SHAPE), z.shape
    # stream 顺序必须与 embed_prefix 实际迭代的顺序一致 —— 那是 preprocess_observation
    # 之后的 obs.images（model.py 用 IMAGE_KEYS 重建过），不是 pre-preprocess 的
    # obs.images。之前直接断言 pre-preprocess 的顺序只是"碰巧"和 STREAM_KEYS 一致
    # （因为 obs_from_dict 走的 input transform 恰好也按这个顺序建字典），并不是
    # embed_prefix 实际吃到的顺序。
    pre = _model.preprocess_observation(None, obs, train=False)
    assert tuple(pre.images.keys()) == STREAM_KEYS


def test_rebuilt_prefix_is_bitwise_identical_to_embed_prefix(policy):
    """最强的一条：我们重建的前缀必须与 pi0.embed_prefix 逐位相同。

    tokens / input_mask / ar_mask 三者全等 => capture 点确实等于 inject 点，
    且前缀组装（拼接顺序、mask、ar_mask）没有任何偏差。这条不依赖下游数值，
    是纯结构性的证据。
    """
    from openpi.models import model as _model

    from ours_pi05.openpi_bridge import _embed_prefix_with_tokens, obs_from_dict

    model = policy._model
    obs = obs_from_dict(policy, _fake_obs())
    pre = _model.preprocess_observation(None, obs, train=False)

    z = jnp.stack(
        [model.PaliGemma.img(pre.images[n], train=False)[0] for n in pre.images], axis=1
    )
    t_stock, mask_stock, ar_stock = model.embed_prefix(pre)
    t_ours, mask_ours, ar_ours = _embed_prefix_with_tokens(model, pre, z)

    assert t_ours.dtype == t_stock.dtype  # bf16，不能被悄悄升成 f32
    assert bool(jnp.array_equal(t_ours, t_stock))
    assert bool(jnp.array_equal(mask_ours, mask_stock))
    assert bool(jnp.array_equal(ar_ours, ar_stock))


def test_inject_reproduces_stock_sampler_bitwise(policy):
    """把 encode_images 的输出原样注入，必须与 openpi 自己的 sample_actions 逐位相等。

    **比对对象是 model.sample_actions（急切），不是 policy._sample_actions。**
    后者被 nnx_utils.module_jit 包过（policy.py:63），而模型 dtype 是 bfloat16
    （pi0_config.py:20）。openpi 自己的 sample_actions 急切 vs jit 就差
    max|Δ| ≈ 0.045（action 量级 |a|.mean ≈ 0.097）—— 纯 XLA 融合舍入差异。
    拿急切的镜像去比 jit 过的 stock，会得出「注入坏了」的错误结论。

    同 execution mode 下比，实测 max|Δ| = 0.0，逐位相等。
    """
    from ours_pi05.openpi_bridge import encode_images, obs_from_dict, sample_actions_with_tokens

    obs = obs_from_dict(policy, _fake_obs())
    rng = jax.random.key(42)

    z = encode_images(policy, obs)
    ours = np.asarray(sample_actions_with_tokens(policy, rng, obs, z))
    stock_eager = np.asarray(policy._model.sample_actions(rng, obs))

    assert ours.shape == stock_eager.shape
    np.testing.assert_allclose(ours, stock_eager, atol=0, rtol=0)


def test_jit_eager_gap_matches_openpis_own(policy):
    """把上面那条 docstring 里的说法钉死成断言，免得后人再踩一次。

    我们的 jit 版 vs openpi 的 jit 版，差距不应超过 openpi 自身 jit-vs-eager 的差距
    的量级。这条同时保证 make_jitted_sampler 没写错（它若真错了，差距会是数量级的）。
    """
    from ours_pi05.openpi_bridge import (
        encode_images,
        make_jitted_sampler,
        obs_from_dict,
    )

    obs = obs_from_dict(policy, _fake_obs())
    rng = jax.random.key(42)
    z = encode_images(policy, obs)

    stock_jit = np.asarray(policy._sample_actions(rng, obs))
    stock_eager = np.asarray(policy._model.sample_actions(rng, obs))
    openpi_own_gap = np.abs(stock_jit - stock_eager).max()

    ours_jit = np.asarray(make_jitted_sampler(policy)(rng, obs, z))
    ours_gap = np.abs(ours_jit - stock_jit).max()

    # openpi 自己就有这个 gap；我们的不应显著更差
    assert ours_gap <= openpi_own_gap * 2 + 1e-6, (
        f"our jitted sampler diverges from stock by {ours_gap}, "
        f"but openpi's own jit-vs-eager gap is only {openpi_own_gap}"
    )


def test_inject_different_tokens_changes_output(policy):
    """反向哨兵：换掉 tokens 必须改变输出。否则说明注入根本没生效（测试 2 会假通过）。"""
    from ours_pi05.openpi_bridge import encode_images, obs_from_dict, sample_actions_with_tokens

    obs = obs_from_dict(policy, _fake_obs())
    rng = jax.random.key(42)

    z = encode_images(policy, obs)
    base = np.asarray(sample_actions_with_tokens(policy, rng, obs, z))
    perturbed = np.asarray(sample_actions_with_tokens(policy, rng, obs, z + 1.0))

    assert not np.allclose(base, perturbed, atol=1e-3)


def test_inject_per_stream_changes_output(policy):
    """针对性哨兵：test_inject_different_tokens_changes_output 一次性扰动全部 3 个
    stream，如果实现里只有 stream 0 真正被注入、stream 1/2 悄悄 fallback 回真实
    img() 输出，那条测试仍然会通过（因为 stream 0 的扰动足够让输出变化）。
    这里逐个 stream 单独扰动，任何一个 stream 没被真正注入都会让对应的这条断言
    失败。
    """
    from ours_pi05.openpi_bridge import encode_images, obs_from_dict, sample_actions_with_tokens

    obs = obs_from_dict(policy, _fake_obs())
    rng = jax.random.key(42)

    z = encode_images(policy, obs)
    base = np.asarray(sample_actions_with_tokens(policy, rng, obs, z))

    for i in range(z.shape[1]):
        z_i = z.copy()
        z_i[:, i] += 1.0
        perturbed = np.asarray(sample_actions_with_tokens(policy, rng, obs, z_i))
        assert not np.allclose(base, perturbed, atol=1e-3), f"stream {i} 的注入似乎没生效"


def test_actions_to_dict_shape_and_values(policy):
    """actions_to_dict 必须把 [B, 50, 32] 的 model 空间 raw actions 变成机器人
    能直接执行的 [50, 14]，且数值上要真的经过了裁剪 + 反归一化（不是简单切片）。
    """
    from ours_pi05.openpi_bridge import (
        actions_to_dict,
        encode_images,
        obs_from_dict,
        sample_actions_with_tokens,
    )

    obs = obs_from_dict(policy, _fake_obs())
    rng = jax.random.key(42)

    z = encode_images(policy, obs)
    raw = sample_actions_with_tokens(policy, rng, obs, z)
    actions = actions_to_dict(policy, obs, raw)

    assert actions.shape == (50, 14), actions.shape
    raw_np = np.asarray(raw)[0]
    assert raw_np.shape == (50, 32)
    assert not np.allclose(actions, raw_np[:, :14], atol=1e-6)
