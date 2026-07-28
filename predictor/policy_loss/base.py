"""Backbone-independent half of the action-space policy-distillation loss.

Teacher: the frozen flow head on the TRUE handoff latent with the GT handoff state.
Student: the same head on the PREDICTED latent with the corrector's state estimate.
Both see the same task, the same flow noise, and the same ``x_t``, so the loss isolates the
(latent, state) pair. Gradients reach only the predicted latent.

Subclasses supply two hooks:

* ``_prepare(g_target, state8)``            -> backbone-specific policy inputs
* ``_single_step_action(z, prepared, x_t, time, grad_checkpoint=False)`` -> ``a_hat``

Images fed to the flow head are dummy zeros: the injected latent overrides them, which is what
lets this run without reading a single image or re-embedding the latent bank.
"""
from __future__ import annotations

import numpy as np
import torch

from common.action_space import load_action_stats, qnorm_to_policy
from corrector.state_fn import right_align_env_actions


class OfflinePolicyLoss:
    """Shared state, action-space handling, and the distillation loss itself."""

    def _init_shared(self, *, sidecar_paths, dataset_actions, action_stats, corrector, device,
                     chunk_size, action_dim, max_action_dim, seed, max_delay=20):
        self.device = device
        self.actions = dataset_actions          # [N, A] BANK space, global-frame indexed
        self.stats = action_stats
        self.corrector = corrector
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.max_action_dim = int(max_action_dim)
        self.max_delay = int(max_delay)
        self.rng = torch.Generator(device="cpu").manual_seed(int(seed) + 777)
        self.dummy_hw = (3, 256, 256)

        # Sidecars concatenate bank-major, matching the latent bank's global frame order; the
        # per-bank task_index values are remapped into one merged task_strings list.
        states, task_indices, strings = [], [], []
        for path in sidecar_paths:
            sidecar = torch.load(path, map_location="cpu", weights_only=False)
            if sidecar.get("format_version") != "state_task_sidecar_v1":
                raise RuntimeError(f"{path}: unexpected sidecar format")
            remap = []
            for s in list(sidecar["task_strings"]):
                if s not in strings:
                    strings.append(s)
                remap.append(strings.index(s))
            states.append(sidecar["state"].float())
            task_indices.append(
                torch.tensor(remap, dtype=torch.long)[sidecar["task_index"].long()])
        self.state = torch.cat(states, 0)        # [N, 8] raw observation.state
        self.task_index = torch.cat(task_indices, 0)
        self.task_strings = strings
        if self.state.shape[0] != self.actions.shape[0]:
            raise RuntimeError(
                f"sidecar frames {self.state.shape[0]} != bank frames {self.actions.shape[0]} "
                "(bank/sidecar order mismatch?)")

    # --- action spaces ------------------------------------------------------------------

    def _gt_action_chunk(self, g_target: torch.Tensor) -> torch.Tensor:
        """actions[g : g+chunk] converted BANK -> POLICY and padded to max_action_dim.

        The bank already stores quantile-normalized actions, and both backbones normalize ACTION
        with MEAN_STD, so the conversion is qnorm -> env -> (env - mean)/std. A chunk running off
        the end of the bank repeats its last available row.
        """
        n = g_target.shape[0]
        chunk = torch.zeros(n, self.chunk_size, self.action_dim, dtype=torch.float32)
        total = self.actions.shape[0]
        for j, g in enumerate(g_target.tolist()):
            rows = self.actions[g:min(g + self.chunk_size, total)].float()
            chunk[j, : rows.shape[0]] = rows
            if 0 < rows.shape[0] < self.chunk_size:
                chunk[j, rows.shape[0]:] = rows[-1]
        chunk = torch.as_tensor(qnorm_to_policy(chunk, self.stats), dtype=torch.float32)
        padded = torch.zeros(n, self.chunk_size, self.max_action_dim, dtype=torch.float32)
        padded[..., : self.action_dim] = chunk
        return padded.to(self.device)

    def _student_state8(self, g_target, committed_actions, delay) -> torch.Tensor:
        """The deployable state estimate the student sees: the corrector's output from the state
        at ``g_target - delay`` plus the committed actions. Falls back to the GT state when no
        corrector is configured, which isolates the visual channel."""
        if self.corrector is None or committed_actions is None or delay is None:
            return self.state[g_target.cpu()]
        base = self.state[(g_target - delay).cpu()].numpy()
        out = np.empty_like(base)
        committed = committed_actions.detach().cpu()
        for j, d in enumerate(delay.cpu().tolist()):
            buf = right_align_env_actions(committed[j], int(d), self.max_delay,
                                          action_space="qnorm", stats=self.stats)
            out[j] = self.corrector.correct_batch(base[j][None], buf, int(d))[0]
        return torch.from_numpy(out).float()

    # --- hooks --------------------------------------------------------------------------

    def _prepare(self, g_target, state8):
        raise NotImplementedError

    def _single_step_action(self, z, prepared, x_t, time, grad_checkpoint=False):
        raise NotImplementedError

    # --- loss ---------------------------------------------------------------------------

    def loss(self, z_hat, z_target, g_target, committed_actions=None, delay=None, reduce=True):
        """Action-space distillation MSE.

        ``reduce=True`` returns the scalar batch mean; ``reduce=False`` returns the per-sample
        loss ``[B]`` so the caller can apply per-sample weights before averaging.
        """
        gt_state8 = self.state[g_target.cpu()]
        teacher_prepared = self._prepare(g_target, gt_state8)          # true latent + GT state
        student_state8 = self._student_state8(g_target, committed_actions, delay)
        student_prepared = self._prepare(g_target, student_state8)     # z_hat + corrected state

        actions = self._gt_action_chunk(g_target)
        noise = torch.randn(actions.shape, generator=self.rng).to(self.device)
        time = (torch.rand(actions.shape[0], generator=self.rng) * 0.8 + 0.1).to(self.device)
        x_t = build_x_t(actions, noise, time)

        a_pred = self._single_step_action(z_hat, student_prepared, x_t, time,
                                          grad_checkpoint=True)
        with torch.no_grad():
            target = self._single_step_action(z_target, teacher_prepared, x_t, time).float()
        squared = (a_pred.float() - target).pow(2)
        if reduce:
            return squared.mean()
        return squared.reshape(squared.shape[0], -1).mean(dim=1)


def build_x_t(actions, noise, time):
    """``x_t = t*noise + (1-t)*actions``, matching both backbones' flow interpolation."""
    te = time[:, None, None]
    return te * noise + (1.0 - te) * actions
