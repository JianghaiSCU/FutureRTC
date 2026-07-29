import pytest
import torch

from ours_pi05.models.predictor import (
    LATENT_DIM,
    NUM_STREAMS,
    STREAM_EGO_MASK,
    TOKENS,
    MotionPriorLatentPredictor,
)

B, C, T, D = 2, 3, 256, 2048
K, A, S = 10, 14, 14


def _inputs(device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(0)
    return dict(
        z=torch.randn(B, C, T, D, generator=g),
        motion_actions=torch.randn(B, K, A, generator=g),
        delay=torch.full((B,), 10, dtype=torch.long),
        z_init=torch.randn(B, C, T, D, generator=g),
        state=torch.randn(B, S, generator=g),
    )


def test_forward_shape():
    m = MotionPriorLatentPredictor().eval()
    with torch.no_grad():
        out = m(**_inputs())
    assert out.shape == (B, C, T, D)
    assert torch.isfinite(out).all()


def test_identity_start_output_is_close_to_input():
    """identity-start 初始化：未训练时 z_hat 必须 ≈ z（原样拷贝陈旧 latent）。

    这是模型稳定的根基。flow=0、strength=sigmoid(-6)≈0.0025、
    gate=sigmoid(-4)≈0.018 且 residual 输出权重全零 -> residual 项恒为 0。
    """
    m = MotionPriorLatentPredictor().eval()
    inp = _inputs()
    with torch.no_grad():
        out = m(**inp)
    # strength≈0.0025 且 warp 在 flow=0 时是恒等 -> z_transport == z（到数值精度）
    torch.testing.assert_close(out, inp["z"], atol=1e-4, rtol=1e-3)


def test_identity_start_init_values():
    """直接钉住 identity-start 初始化的具体数值，而不只是靠行为测试间接推断。

    行为测试 test_identity_start_output_is_close_to_input 无法区分
    「strength bias = -6」和「strength bias = 0」（因为 flow=0 时 warp 是恒等，
    (1-s)*z + s*1*z == z 对任意 s 都成立），也无法区分 gate bias -4 和 0
    （因为 residual_out_head.weight=0 时 residual 恒为 0，gate*0 == 0 对任意 gate 都成立）。
    这里直接断言初始化数值本身。
    """
    m = MotionPriorLatentPredictor()
    assert m.transport_flow_head[-1].weight.abs().max().item() == 0.0
    assert m.transport_flow_head[-1].bias.abs().max().item() == 0.0
    assert m.transport_gain_head[-1].weight.abs().max().item() == 0.0
    assert m.transport_gain_head[-1].bias.abs().max().item() == 0.0
    assert m.transport_strength_head[-1].bias.min().item() == -6.0
    assert m.residual_out_head.weight.abs().max().item() == 0.0
    assert m.residual_out_head.bias[:-1].abs().max().item() == 0.0
    assert m.residual_out_head.bias[-1].item() == -4.0


def test_stream_ego_mask_is_bimanual():
    """cam_high 是静态相机（无自运动）；左腕只由左臂驱动、右腕只由右臂驱动。"""
    assert STREAM_EGO_MASK[0] == ()                    # base_0_rgb  (cam_high)
    assert STREAM_EGO_MASK[1] == tuple(range(0, 7))    # left_wrist_0_rgb
    assert STREAM_EGO_MASK[2] == tuple(range(7, 14))   # right_wrist_0_rgb


def test_ego_routing_can_be_disabled():
    m = MotionPriorLatentPredictor(ego_routing=False).eval()
    with torch.no_grad():
        out = m(**_inputs())
    assert out.shape == (B, C, T, D)


def test_trunk_receptive_field_covers_grid():
    """16x16 网格上 trunk 必须仍是（近似）全局的。

    只有 DepthwiseConvBlock 做 token 混合，kernel=3 -> 每块 RF 半径 +1。
    7 个 conv -> RF = 1 + 7*2 = 15，覆盖 16x16。
    """
    from ours_pi05.models.predictor import DepthwiseConvBlock, TRUNK_SPEC

    n_conv = TRUNK_SPEC.count("c")
    rf = 1 + 2 * n_conv
    side = int(T**0.5)
    assert side == 16
    assert rf >= side - 1, f"trunk RF {rf} 太小，覆盖不了 {side}x{side} 网格"

    # 光核对 TRUNK_SPEC 字符串还不够——如果 __init__ 实际构建的 trunk 和 TRUNK_SPEC 对不上
    # （例如硬编码了别的 spec 字符串），上面的断言也会照样通过。这里核对构建出来的模块本身。
    m = MotionPriorLatentPredictor().eval()
    n_conv_blocks = sum(1 for block in m.token_attention if isinstance(block, DepthwiseConvBlock))
    assert n_conv_blocks == n_conv


def test_rejects_delay_zero():
    """d=0 必须由调用方绕过预测器，不能进来。"""
    m = MotionPriorLatentPredictor().eval()
    inp = _inputs()
    inp["delay"] = torch.zeros(B, dtype=torch.long)
    with pytest.raises(ValueError):
        m(**inp)


def test_rejects_delay_exceeding_action_window():
    m = MotionPriorLatentPredictor().eval()
    inp = _inputs()
    inp["delay"] = torch.full((B,), K + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        m(**inp)


def test_param_count_is_small():
    m = MotionPriorLatentPredictor()
    n = sum(p.numel() for p in m.parameters())
    assert 5e6 < n < 40e6, f"param count {n} 超出预期（spec 估 ~18M）"


def test_rejects_mismatched_batch_size():
    """z 的 batch 维必须和 motion_actions/delay 的 batch 维一致。

    移植时这道检查被漏掉了：z [2,...] 配 motion_actions [1,K,A]、delay [1] 时，
    `motion_full[:, None, :] + stream_condition + stream_context` 会把 [1,1,h] 广播到
    [2,3,h] —— 不报错、输出形状也对，但两个样本都被静默地条件化到了样本 0 的 motion 上。
    """
    m = MotionPriorLatentPredictor().eval()
    inp = _inputs()
    inp["motion_actions"] = inp["motion_actions"][:1]
    inp["delay"] = inp["delay"][:1]
    with pytest.raises(ValueError):
        m(**inp)


def test_padding_rows_are_ignored():
    """build_action_trajectory_features 是左对齐的：只有前 delay 行有效，其余是 padding。

    如果调用方误把有效动作右对齐传入，网络会把 padding（本该是零）当成有效轨迹，
    而这不会崩溃——cumsum 出来的 motion 特征只是静默地错了。这里确认末尾的 padding 行
    （garbage）不会改变输出。
    """
    m = MotionPriorLatentPredictor().eval()
    inp = _inputs()
    inp["delay"] = torch.full((B,), 3, dtype=torch.long)
    with torch.no_grad():
        a = m(**inp)
        inp["motion_actions"][:, 3:] = 999.0  # garbage in the padding rows
        b = m(**inp)
    torch.testing.assert_close(a, b)


def test_predictor_dims_match_bridge():
    """预测器里本地定义的维度常量必须和 openpi_bridge.LATENT_SHAPE 手工保持一致。

    predictor.py 故意不 import openpi_bridge（会拉 JAX 进纯 torch 模块），
    所以两边的 (num_streams, tokens, latent_dim) 只能靠这条测试交叉核对，
    而不是靠共享一份真源头。import 放在测试里，和 test_latent_bank.py 的做法一致。
    """
    from ours_pi05.openpi_bridge import LATENT_SHAPE

    assert (NUM_STREAMS, TOKENS, LATENT_DIM) == LATENT_SHAPE
