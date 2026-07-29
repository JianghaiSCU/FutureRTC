""""Ours" (motion-prior latent prediction) policy server for the real robot
(openpi / JAX pi05).

Independent counterpart of ``deploy_policy_server_local.py`` /
``deploy_policy_server_rtc_realtime_chunking_local.py``. Same HTTP interface
(base64+pickle obs, port 8001, ``flag='reset'``), but instead of RTC's
inpainting-guided flow sampling, this server predicts the observation the
policy WOULD have seen at handoff time and runs a single ordinary flow sample
against that predicted observation. Nothing in ``src/openpi`` is modified.

Why the server (not the client) owns this
------------------------------------------
Latent injection has to happen between the vision tower and the language
model -- the exact seam ``ours_pi05.openpi_bridge`` exposes
(``encode_images`` / ``sample_actions_with_tokens``). That is JAX-side model
internals, not something reachable from a black-box ``policy.infer()`` call,
so it has to live in the process that holds the JAX model, i.e. this server.

Per-request pipeline (see task-10 brief, ``.superpowers/sdd/task-10-brief.md``):
    1. decode obs (stale images + stale state) + committed_actions [d,14]
       (absolute joint angles) + inference_delay
    2. z_stale = jitted SigLIP encode of the stale obs
    3. z_init: episode's first-frame latent, cached on the first query after
       a reset (``flag='reset'`` clears it)
    4. s_hat = Corrector()(stale_state, committed_actions, delay) -- on this
       robot an exact identity: returns committed_actions[delay-1]
    5. z_hat = predictor(normalized z_stale, motion_actions, delay,
       z_init=normalized z_init, state=s_hat), de-normalized back to SigLIP
       space
    6. rebuild the observation FROM THE RAW OBS DICT with s_hat substituted
       for state (see trap 1 below), and sample with z_hat injected in place
       of the real image tokens
    7. return the [50, 14] robot-space chunk

Five traps this file exists to not fall into
---------------------------------------------
1. Cannot overwrite ``.state`` on an already-transformed ``Observation``.
   pi0.5 has ``discrete_state_input=True``: state is normalized and then
   tokenized INTO THE LANGUAGE PROMPT. To inject s_hat we put it in the RAW
   obs dict and re-run ``openpi_bridge.obs_from_dict`` from scratch. Upside,
   for free: because this config uses delta joint actions, the output
   transform ADDS THE SUPPLIED STATE BACK, so the returned absolute joint
   targets are automatically anchored at the pose the arm will actually be in
   at handoff.
2. The raw sampler output is NOT robot actions -- it is [B, 50, 32] in the
   model's normalized action space. Always go through
   ``openpi_bridge.actions_to_dict``; never touch ``policy._output_transform``
   directly.
3. The predictor's inputs must be assembled EXACTLY as in training (see
   ``ours_pi05.dataset.PerFrameLatentDataset.__getitem__`` and
   ``ours_pi05.fast_loader.FastBatchLoader._fill_one``, mirrored here via
   ``ours_pi05.deploy_protocol.build_padded_motion``): motion_actions is
   qnorm(to_delta(committed, s_anchor)) LEFT-aligned into [MAX_DELAY, 14];
   s_anchor for to_delta is the STALE state; the ``state`` fed to the
   predictor is the CORRECTED HANDOFF state s_hat -- a different quantity.
   latent_norm / action_quantiles come from the predictor checkpoint, not
   re-derived.
4. Use the jitted paths (``make_jitted_encoder`` / ``make_jitted_sampler``),
   built once at startup and warmed up with a dummy observation. Eager
   ``encode_images`` is ~6000 ms/frame and eager sampling takes seconds --
   async only works if inference fits inside d=10 raw-action steps of arm
   motion.
5. ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is mandatory: a torch predictor and
   the JAX policy share one GPU in this process. Set (via ``setdefault``,
   still overridable) before jax is imported by anything, including
   transitively via ``ours_pi05.openpi_bridge``.
"""

from __future__ import annotations

import os

# Trap 5: must happen before jax is imported anywhere in this process (this
# module or anything it imports, e.g. ours_pi05.openpi_bridge). setdefault so
# an operator's explicit `export XLA_PYTHON_CLIENT_PREALLOCATE=...` still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import base64
import gzip
import pathlib
import pickle
import sys
import time

import cv2
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
import numpy as np
import torch
import uvicorn

# ---------------------------------------------------------------------------
# Path setup: make `ours_pi05` importable. Importing ours_pi05.openpi_bridge
# inserts <repo_root>/openpi/src at the FRONT of sys.path as a side effect
# (see that module's top) -- do that via the import below rather than
# hand-rolling a second path insertion here, so this process only ever binds
# one `openpi` module: the dev tree ours_pi05 is pinned to, not the
# infer/openpi/src copy the other (unmodified) deploy_policy_server_*.py
# scripts in this directory use directly. Mixing the two would silently give
# two different `openpi.models.model.Observation` classes in the same
# process.
# ---------------------------------------------------------------------------
# 往上逐级找包含 `ours_pi05/` 的目录，把它加进 sys.path。这样无论 deploy 脚本嵌套
# 多深都能定位到 ours_pi05：开发机是 realworld_cobot/infer/openpi/（往上两级），
# 机器人主机是 <root>/openpi/（往上一级，openpi 与 ours_pi05 同级）。
def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    for cand in [start, *start.parents]:
        if (cand / "ours_pi05" / "__init__.py").is_file():
            return cand
    raise RuntimeError(
        f"往上找不到 ours_pi05/ 包（从 {start} 起）。请把 ours_pi05/ 放在 openpi/ 的同级目录。"
    )


_REPO_ROOT = _find_repo_root(pathlib.Path(__file__).resolve().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from ours_pi05 import openpi_bridge  # noqa: E402
from ours_pi05.deploy_protocol import CHUNK_LEN  # noqa: E402
from ours_pi05.deploy_protocol import METHOD  # noqa: E402
from ours_pi05.deploy_protocol import build_padded_motion  # noqa: E402
from ours_pi05.models.corrector import Corrector  # noqa: E402
from ours_pi05.models.predictor import MotionPriorLatentPredictor  # noqa: E402

CAM_NAMES: tuple[str, str, str] = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def decode_client_obs(input_rgb_arr, input_state, prompt: str) -> dict:
    """Client wire format -> the RAW obs dict ``openpi_bridge.obs_from_dict``
    expects: ``{"images": {cam_high, cam_left_wrist, cam_right_wrist}, "state":
    [14] float32, "prompt": str}``.

    ``input_rgb_arr`` is ``[head, left_wrist, right_wrist]`` (see the client's
    ``encode_obs``); mirrors the image squeeze/CHW handling already proven in
    ``deploy_policy_server_rtc_realtime_chunking_local.py::
    PI0.update_observation_window`` -- AlohaInputs requires CHW
    (openpi/src/openpi/policies/aloha_policy.py), HWC silently becomes garbage.
    """
    if len(input_rgb_arr) != len(CAM_NAMES):
        raise ValueError(f"expected {len(CAM_NAMES)} camera frames, got {len(input_rgb_arr)}")
    images = {}
    for name, raw in zip(CAM_NAMES, input_rgb_arr, strict=True):
        img = np.asarray(raw, dtype=np.uint8)
        img = np.squeeze(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)
        images[name] = np.ascontiguousarray(img)

    state = np.asarray(input_state, dtype=np.float32).reshape(-1)
    return {"images": images, "state": state, "prompt": prompt}


class PI0Ours:
    """Wraps a trained pi0.5 policy + motion-prior predictor + identity
    corrector behind the ``infer_ours`` entry point. See module docstring for
    the full per-request pipeline."""

    def __init__(
        self,
        train_config_name: str,
        checkpoint_path: str,
        predictor_ckpt: str,
        *,
        prompt: str,
        pi0_step: int = CHUNK_LEN,
        device: str = "cuda",
    ):
        self.policy = openpi_bridge.load_policy(train_config_name, checkpoint_path)
        print(f"[ours-server] loaded pi0.5 policy: config={train_config_name} ckpt={checkpoint_path}")

        # weights_only=False: this checkpoint carries numpy arrays
        # (latent_norm / action_quantiles) and a vars(argparse.Namespace) dict
        # (args); torch>=2.6's default weights_only=True rejects numpy's
        # _reconstruct global. This is our own training script's output, so
        # it's trusted (same rationale as ours_pi05/eval_offline.py:134-137).
        ck = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)
        ego_routing = not ck["args"].get("no_ego_routing", False)
        self.predictor = MotionPriorLatentPredictor(ego_routing=ego_routing)
        self.predictor.load_state_dict(ck["state_dict"])
        self.device = device
        self.predictor.to(device).eval()
        print(
            f"[ours-server] loaded predictor: {predictor_ckpt} "
            f"(step={ck.get('step')}, ego_routing={ego_routing})"
        )

        self.action_quantiles = ck["action_quantiles"]  # {"q01": [14], "q99": [14]}
        latent_norm = ck["latent_norm"]                 # {"mean": [3,2048], "std": [3,2048]}
        self._mean = np.asarray(latent_norm["mean"], dtype=np.float32)[:, None, :]  # [3,1,2048]
        self._std = np.asarray(latent_norm["std"], dtype=np.float32)[:, None, :]

        self.corrector = Corrector()
        self.prompt = prompt
        self.pi0_step = int(pi0_step)

        # Trap 4: jitted paths, built once.
        self._jit_encode = openpi_bridge.make_jitted_encoder(self.policy)
        self._jit_sample = openpi_bridge.make_jitted_sampler(self.policy)
        self._rng = jax.random.key(0)

        self.z_init: np.ndarray | None = None  # episode's first-frame latent, float32 [3,256,2048]

        self._warmup()

    def _warmup(self) -> None:
        """Compile both jitted paths at startup so the first real query does
        not eat the JIT compile (trap 4)."""
        t0 = time.time()
        rng = np.random.default_rng(0)
        dummy = {
            "images": {
                name: rng.integers(0, 256, (3, 480, 640), dtype=np.uint8) for name in CAM_NAMES
            },
            "state": np.zeros(14, dtype=np.float32),
            "prompt": self.prompt,
        }
        obs = openpi_bridge.obs_from_dict(self.policy, dummy)
        z = np.asarray(jnp.asarray(self._jit_encode(obs), dtype=jnp.float32))
        self._rng, rng_key = jax.random.split(self._rng)
        _ = self._jit_sample(rng_key, obs, jnp.asarray(z, dtype=jnp.bfloat16))
        print(f"[ours-server] warmup done in {time.time() - t0:.1f}s (encoder+sampler jit-compiled)")

    def reset(self) -> None:
        self.z_init = None
        print("[ours-server] episode reset: cleared z_init cache")

    def _encode(self, obs_dict: dict) -> np.ndarray:
        """Stale obs -> SigLIP encode (jitted). Returns float32 [3,256,2048]
        (batch dim squeezed)."""
        observation = openpi_bridge.obs_from_dict(self.policy, obs_dict)
        z = self._jit_encode(observation)  # jax array [1,3,256,2048], native (bf16) dtype
        return np.asarray(jnp.asarray(z, dtype=jnp.float32))[0]

    def infer_ours(self, obs_dict: dict, committed_actions: np.ndarray, delay: int) -> np.ndarray:
        delay = int(delay)
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        committed_actions = np.asarray(committed_actions, dtype=np.float32).reshape(-1, 14)
        if committed_actions.shape[0] != delay:
            raise ValueError(
                f"committed_actions has {committed_actions.shape[0]} rows, expected exactly "
                f"delay={delay} (protocol requires the caller to slice to exactly d rows)"
            )

        # s_anchor: the STALE state (query-time obs), the to_delta anchor.
        # NOT the same quantity as s_hat below -- see trap 3.
        s_anchor = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)

        # 2) stale obs -> SigLIP encode
        z_stale = self._encode(obs_dict)

        # 3) episode's first-frame latent, cached on first query after reset.
        # The window-0 bootstrap query (delay=0, see below) is exactly the
        # episode's true first frame, so this also doubles as z_init capture.
        if self.z_init is None:
            self.z_init = z_stale.copy()
            print("[ours-server] cached z_init for this episode")

        # 4) identity-proxy corrected handoff state (delay=0 -> base_state,
        # unchanged; see Corrector.__call__).
        s_hat = self.corrector(s_anchor, committed_actions, delay)

        if delay == 0:
            # Nothing to compensate for -- the window-0 bootstrap query, where
            # the obs used for inference is consumed immediately (arm still
            # stationary, no queued committed actions exist yet).
            # predictor.forward explicitly forbids delay=0 ("d=0 必须由调用方
            # 绕过预测器"), so skip it and sample directly against the real
            # latent/state. This reduces exactly to the ordinary policy.infer()
            # path (bitwise, per openpi_bridge's own equivalence tests).
            z_hat = z_stale
        else:
            # 5) predict handoff latent
            motion_padded = build_padded_motion(committed_actions, s_anchor, self.action_quantiles, delay)
            z_n = (z_stale - self._mean) / self._std
            z_init_n = (self.z_init - self._mean) / self._std
            with torch.no_grad():
                z_hat_n = self.predictor(
                    torch.from_numpy(z_n[None]).float().to(self.device),
                    torch.from_numpy(motion_padded[None]).float().to(self.device),
                    torch.tensor([delay], dtype=torch.long, device=self.device),
                    z_init=torch.from_numpy(z_init_n[None]).float().to(self.device),
                    state=torch.from_numpy(s_hat[None]).float().to(self.device),
                )[0].cpu().numpy()
            z_hat = z_hat_n * self._std + self._mean  # de-normalize back to SigLIP space

        # 6) rebuild the observation from the RAW obs dict with s_hat
        #    substituted for state -- trap 1. discrete_state_input=True
        #    tokenizes state into the prompt; you cannot overwrite `.state` on
        #    an already-transformed Observation and expect that to matter.
        obs_hat_dict = dict(obs_dict)
        obs_hat_dict["state"] = s_hat
        observation_hat = openpi_bridge.obs_from_dict(self.policy, obs_hat_dict)

        # 7) sample with z_hat injected in place of the real image tokens.
        # Cast to bf16 -- the model's native token dtype (pi0_config bf16);
        # leaving z_hat as float32 would silently upcast the whole prefix via
        # jnp.concatenate's dtype promotion instead of erroring, which is both
        # slower and diverges from the bf16 pipeline the predictor was
        # trained against.
        self._rng, rng_key = jax.random.split(self._rng)
        actions = self._jit_sample(rng_key, observation_hat, jnp.asarray(z_hat[None], dtype=jnp.bfloat16))
        # Trap 2: raw sampler output is [B,50,32] in the model's normalized
        # action space, not robot actions -- always go through actions_to_dict.
        chunk = openpi_bridge.actions_to_dict(self.policy, observation_hat, actions)
        return chunk[: self.pi0_step]


def decode_pickle_base64(encoded_str):
    decoded = base64.b64decode(encoded_str)
    return pickle.loads(decoded)


def decode_pickle_base64_zip(b64_str):
    zip_data = base64.b64decode(b64_str.encode())
    bin_data = gzip.decompress(zip_data)
    return pickle.loads(bin_data)


app = FastAPI()


@app.post("/send")
async def handle_obs(request: Request):
    json_data = await request.json()

    try:
        obs = decode_pickle_base64(json_data["data"])
    except Exception:
        obs = decode_pickle_base64_zip(json_data["data"])

    flag = json_data.get("flag", "return")

    if flag == "reset":
        model.reset()
        return {"result": "reset done"}

    method = json_data.get("method", "return")
    if method != METHOD:
        raise HTTPException(
            status_code=400,
            detail=f"deploy_policy_server_ours_local only serves method={METHOD!r}, got {method!r}",
        )

    committed_actions = json_data.get("committed_actions")
    inference_delay = json_data.get("inference_delay")
    if committed_actions is None or inference_delay is None:
        raise HTTPException(
            status_code=400,
            detail="method='ours' requires both 'committed_actions' and 'inference_delay' in the payload",
        )

    for key in obs:
        print(f"obs-{key}:{np.array(obs[key]).shape}")

    obs_dict = decode_client_obs(obs["input_rgb_arr"], obs["input_state"], model.prompt)
    committed_actions = np.asarray(committed_actions, dtype=np.float32)
    delay = int(inference_delay)

    if flag == "return":
        chunk = model.infer_ours(obs_dict, committed_actions, delay)
        print(f"return actions {chunk.shape}")
        return {"result": chunk.tolist()}
    # Parity with the other deploy servers' non-'return' branch (validate
    # + decode only, no inference). ours has no persistent
    # observation_window to pre-warm separately, so this is a no-op
    # beyond confirming the payload decodes.
    return {"result": "no"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ours (motion-prior latent prediction) deploy server")
    p.add_argument("--train-config", required=True, help="openpi train config name, e.g. pi05_cobot_plates_stacking")
    p.add_argument("--ckpt", required=True, help="pi0.5 policy checkpoint dir (JAX / openpi)")
    p.add_argument("--predictor", required=True, help="predictor_<step>.pt checkpoint path (ours_pi05.train_predictor output)")
    p.add_argument("--task", "--prompt", dest="prompt", required=True, help="language instruction sent to the policy")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--pi0-step", type=int, default=CHUNK_LEN, help="number of chunk rows returned per query")
    p.add_argument("--device", default="cuda", help="device for the torch predictor")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = PI0Ours(
        args.train_config,
        args.ckpt,
        args.predictor,
        prompt=args.prompt,
        pi0_step=args.pi0_step,
        device=args.device,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
