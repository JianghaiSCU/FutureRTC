"""SmolVLA flow-head hooks for the policy-distillation loss.

SmolVLA takes the proprioceptive state as a differentiable projected token passed straight to
``embed_prefix(state=...)``, so the student's corrected state and the teacher's GT state enter
without any prompt re-tokenization.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint

from common.latent_io import inject_latent
from common.runtime import build_runtime
from predictor.policy_loss.base import OfflinePolicyLoss


def flow_velocity(model, images, img_masks, lang_tokens, lang_masks, state, x_t, time,
                  grad_checkpoint=False):
    """SmolVLA ``VLAFlowMatching.forward`` up to the velocity ``v_t`` (no ground-truth MSE)."""
    from lerobot.policies.smolvla import modeling_smolvla as M

    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state)
    suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, time)

    pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att, suffix_att], dim=1)
    att_2d_masks = M.make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    def run_expert(prefix_embs, suffix_embs):
        (_, suffix_out), _ = model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        return suffix_out

    if grad_checkpoint:
        suffix_out = torch.utils.checkpoint.checkpoint(
            run_expert, prefix_embs, suffix_embs, use_reentrant=False)
    else:
        suffix_out = run_expert(prefix_embs, suffix_embs)
    suffix_out = suffix_out[:, -model.config.chunk_size:].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


class OfflineSmolvlaPolicyLoss(OfflinePolicyLoss):
    def __init__(self, *, policy_path, sidecar_paths, dataset_actions, corrector, device,
                 lerobot_root, action_stats=None, tokenizer_path=None, seed=0, max_delay=20):
        from common.action_space import load_action_stats
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS,
        )

        runtime = build_runtime("smolvla", policy_path=policy_path, lerobot_root=lerobot_root,
                                device=device, tokenizer_path=tokenizer_path)
        self.policy = runtime.policy
        self.pre = runtime.preprocessor
        self.obs_state_key = runtime.obs_state_key
        self.OBS_LANGUAGE_TOKENS = OBS_LANGUAGE_TOKENS
        self.OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE_ATTENTION_MASK
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.vlm_dtype = next(self.policy.model.vlm_with_expert.parameters()).dtype

        self._init_shared(
            sidecar_paths=sidecar_paths, dataset_actions=dataset_actions,
            action_stats=action_stats if action_stats is not None else load_action_stats(),
            corrector=corrector, device=device,
            chunk_size=self.policy.model.config.chunk_size,
            action_dim=dataset_actions.shape[1],
            max_action_dim=self.policy.model.config.max_action_dim,
            seed=seed, max_delay=max_delay,
        )

    def _prepare(self, g_target, state8):
        n = g_target.shape[0]
        tasks = [self.task_strings[int(self.task_index[int(g)])] for g in g_target.cpu()]
        dummy = torch.zeros((n, *self.dummy_hw), dtype=torch.float32)
        obs = {
            "observation.images.image": dummy.clone().to(self.device),
            "observation.images.image2": dummy.clone().to(self.device),
            self.obs_state_key: state8.to(self.device),
            "task": tasks,
        }
        batch = self.pre(obs)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        return (images, img_masks, batch[self.OBS_LANGUAGE_TOKENS],
                batch[self.OBS_LANGUAGE_ATTENTION_MASK], state)

    def _single_step_action(self, z, prepared, x_t, time, grad_checkpoint=False):
        images, img_masks, lang_tokens, lang_masks, state = prepared
        with inject_latent("smolvla", self.policy, z.to(self.vlm_dtype)):
            v_t = flow_velocity(self.policy.model, images, img_masks, lang_tokens, lang_masks,
                                state, x_t, time, grad_checkpoint=grad_checkpoint)
        return x_t - time[:, None, None] * v_t
