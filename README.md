# FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking
<h4 align="center">Hai Jiang<sup>1</sup>, Yixian Zou<sup>2</sup>, Binbin Liang<sup>1</sup>, Boqian Liu<sup>3</sup>, Fanman Meng<sup>2</sup>, Shuaicheng Liu<sup>2</sup></center>
<h4 align="center">1.Sichuan University,
<h4 align="center">2.University of Electronic Science and Technology of China,</center></center>
<h4 align="center">3.University of Alberta</center></center>

<h4 align="center"> <div>
  <p>
    <a href="https://arxiv.org/abs/2607.24008"><img src="https://img.shields.io/badge/Paper-FutureRTC-b31b1b.svg" alt="Paper" /></a>
    <a href="https://jianghaiscu.github.io/FutureRTC_proj/"><img src="https://img.shields.io/badge/Project-Page-35b8a9.svg" alt="Project page" /></a>
  </p>

</div>

---

## Overview

Under asynchronous control, inference is launched at `t = n·K − d` but the handoff only happens at
`t = n·K`, so the policy is queried on a **stale observation**. FutureRTC uses the `d` actions that
are already committed — and will certainly be executed — to predict the observation **at the handoff
moment** (image latent + proprioceptive state), and feeds that into a **frozen** π₀.₅. The policy
then behaves as if it had seen a fresh frame, which is what *anticipatory-conditioned* means:
conditioning the policy on a predicted future observation.

### Hardware and tasks

- **Robot** — AgileX / Piper dual-arm cobot, 14 DoF (left arm 6 + left gripper 1 + right arm 6 +
  right gripper 1).
- **Cameras** — 3× RealSense D435i (head, left wrist, right wrist).
- **Base policy** — π₀.₅ from [openpi](https://github.com/Physical-Intelligence/openpi) (JAX).
- **Tasks** — `plates_stacking`, `towel_folding`, `cup_hanging`.


---

## 1. Layout

```
realworld_cobot_release/
├── train/                      # training side (a dev machine with GPUs)
│   ├── openpi/                 # openpi (JAX) -- trains the pi0.5 policy, plus open-loop eval
│   └── ours_pi05/              # FutureRTC side model: latent bank, predictor training, eval
└── infer/                      # deployment side (the robot host)
    ├── openpi/                 # openpi deployment snapshot -- the FutureRTC policy server
    ├── ours_pi05/              # minimal deployment closure (bridge / protocol / model defs)
    └── cobot-magic-real/       # robot execution: Piper SDK / CAN / RealSense + deploy client
```

- `train/ours_pi05` is the **full** package (collect bank, train predictor, evaluate, deploy contract).
- `infer/ours_pi05` keeps only the **deployment runtime**: `__init__ / action_space /
  deploy_protocol / openpi_bridge` plus `models/{corrector, predictor}`.
- The **shared contract files** (`deploy_protocol`, `action_space`, `models`, `openpi_bridge`) must
  stay in sync. If you change them — or the policy config — on the training side, mirror the change
  to the deployment side by hand.

### Implementation frameworks

- The π₀.₅ policy is **JAX/Flax** (openpi).
- The FutureRTC predictor is a standalone **PyTorch** model, attached to the
  **frozen** JAX policy through a capture/inject bridge (`ours_pi05/openpi_bridge.py`).
- On the robot, policy inference (JAX) and the predictor (PyTorch) **share one GPU**. Always set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` so XLA does not claim all the memory and the two coexist.

---

## 2. Environment

### 2.1 openpi (training + deployment, JAX)

`train/openpi` and `infer/openpi` are openpi checkouts managed with **uv** (each carries its own
`uv.lock`):

```bash
cd train/openpi        # or infer/openpi
uv sync
```

### 2.2 ours_pi05 (the PyTorch side model)

Run its scripts inside openpi's uv environment — they import openpi to capture the policy's image
latents:

```bash
cd train
openpi/.venv/bin/python -m ours_pi05.train_predictor --help
```

### 2.3 Robot host (cobot-magic-real)

Piper SDK + CAN + RealSense. See `infer/cobot-magic-real/README.md`.

---

## 3. Training

### 3.1 The π₀.₅ policy

Trained once per task with openpi; FutureRTC then leaves it frozen.

```bash
cd train/openpi
# 1) compute norm stats
uv run scripts/compute_norm_stats.py --config-name pi05_cobot_plates_stacking
# 2) train (50k steps, global batch 8, EMA off, full fine-tune)
uv run scripts/train.py pi05_cobot_plates_stacking --exp-name <run>   # add --resume to continue
```

Configs for the three tasks: `pi05_cobot_plates_stacking`, `pi05_cobot_towel_folding`,
`pi05_cobot_cup_hanging`.

### 3.2 The FutureRTC predictor

Three steps: collect the latent bank → train the predictor → evaluate. All run in openpi's JAX
environment, because they need to capture the policy's image latents.

```bash
cd train

# (a) collect the latent bank -- run the frozen policy frame by frame and capture image latents
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  openpi/.venv/bin/python -m ours_pi05.collect_latents \
  --train-config pi05_cobot_plates_stacking --ckpt <policy_30000> \
  --repo-id plates_stacking --out banks/plates_stacking

# (b) train the predictor -- multi-delay (d = 1..10), pure latent MSE
CUDA_VISIBLE_DEVICES=0 openpi/.venv/bin/python -m ours_pi05.train_predictor \
  --bank banks/plates_stacking --out outputs/predictor/plates_stacking_d1to10 \
  --delay-set 1 2 3 4 5 6 7 8 9 10 --steps 150000 --ckpt-every 10000 \
  --batch-size 128 --accum 2

# (c) offline acceptance test -- chunk level, a fixed 200-sample set, a few minutes
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  openpi/.venv/bin/python -m ours_pi05.eval_offline \
  --bank banks/plates_stacking --predictor <predictor.pt> \
  --train-config pi05_cobot_plates_stacking --ckpt <policy_30000> \
  --repo-id plates_stacking --delay 10 --num-samples 200 --out outputs/eval/plates_d10

# (d) open-loop episode rollout -- trajectory plot with chunk boundaries, qualitative check
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  openpi/.venv/bin/python -m ours_pi05.eval_openloop_episode \
  --predictor <predictor.pt> --train-config pi05_cobot_plates_stacking \
  --ckpt <policy_30000> --repo-id plates_stacking --episode 0 --out outputs/eval/openloop_plates
```

---

## 4. Deployment

Three processes, all on the robot host:

```
control_arm_server.py x2   <-TCP(9990/9991)-  deploy client (cobot-magic-real)  --HTTP POST /send-->  policy server (infer/openpi, :8001)
  (CAN, left/right arm)                       reads cameras + joints,                                 openpi pi05 inference
                                              interpolates control points                             + FutureRTC prediction
```

| Component | File |
|---|---|
| Policy server | `infer/openpi/deploy_policy_server_ours_local.py` |
| Deploy client | `infer/cobot-magic-real/deploy_policy_local_ours_batch_test.py` |

### 4.1 Startup

```bash
# --- robot host, terminal 1: CAN + arm servers ---
cd infer/cobot-magic-real
source set_player.sh 1        # source it in every terminal; it exports CUDA_VISIBLE_DEVICES
bash can_config.sh            # configure CAN (adjust USB ports / number of CAN modules)
bash run_server.sh            # two control_arm_server processes (left 9990 / right 9991)
python reset.py               # move the arms to the zero pose

# --- terminal 2: policy server ---
cd infer/openpi
XLA_PYTHON_CLIENT_PREALLOCATE=false python deploy_policy_server_ours_local.py \
  --train-config pi05_cobot_plates_stacking --ckpt <policy_30000> \
  --predictor <predictor_60000.pt> --prompt "<task instruction>"

# --- terminal 3: deploy client ---
cd infer/cobot-magic-real
source set_player.sh 1
python deploy_policy_local_ours_batch_test.py --task plates_stacking \
  --episodes 20 --max-step 300 --save-video
```

> Changing hardware means updating the RealSense serial numbers (grep `RealSenseCam`) and the CAN
> ports.

### 4.2 Timing contract

```
S = 25                  raw actions executed per window
d = 10                  inference delay
H = 50                  chunk length returned by the server
FIRE_STEP = S - d = 15  at the 15th raw action, a background thread launches the next query
```

### 4.3 Batch testing

The client runs a fixed protocol, 20 episodes per task by default: wait for Enter → run until Enter
or `max_step` → ask keep/discard (a discarded episode is not counted and is re-run in place) → ask
success/failure for kept episodes → `fsync` to disk immediately → return to the zero pose (gripper
opens to release, then closes) → next episode. If it crashes, re-running the same `--task` resumes
where it stopped.

Outputs land in `outputs/batch_test/<method>/<task>/`: `results.jsonl`, `results.txt`,
`summary.json`, `video/*.mp4` (with `--save-video`), `actions/*.npz` (the executed raw actions, not
the interpolated control points).

---

## 5. Data contract

- Actions and states are **14-dimensional absolute joint angles** (not deltas). Grippers sit at
  0-based indices `[6, 13]`, physical range 0–0.1 (0–1 on the model side).
- Joint units are radians; `control_arm_server.py` multiplies by a factor to reach Piper SDK integer
  units, and maps the grippers separately.
- In this dataset `action_t ≡ qpos_{t+1}` holds exactly (measured error 0.0). The FutureRTC state
  correction module therefore degenerates to the identity here — unlike LIBERO, where it has to
  learn a real residual.
- The client interpolates each chunk into dense control points and sends them at
  `CONTROL_FREQUENCY`.

### HTTP interface

`POST http://<host>:8001/send` with body
`{'data': base64(pickle(obs)), 'flag': 'return'|'reset', ...}`, where
`obs = {'input_rgb_arr': [head, left, right], 'input_state': joints14}`. The response is
`{"result": [[...] x H]}`.

The prompt is decided by the **server** (the observation carries no instruction); the
`instruction.txt` the client reads is only for printing and cross-checking.

---

## Tests

```bash
cd train
pytest ours_pi05/tests -q
```

55 of the 64 tests run without the JAX stack. The 7 in `test_bridge.py` and two bridge-dependent
cases in `test_latent_bank.py` / `test_predictor.py` need `jax` and `flax` — that is, openpi's
environment.

## Citation

If FutureRTC helps your research, please cite our paper:

```bibtex
@article{jiang2027futurertc,
  title={FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking},
  author={Jiang, Hai and Zou, Yixian and Liang, Binbin and Liu, Boqian and Meng, Fanman and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2607.24008},
  year={2026}
}
```

## License

Released under the MIT License (see `LICENSE`). The RTC and Kinetix dependencies carry their own
licenses.

## Acknowledgments

The real-world deployment builds on [openpi](https://github.com/Physical-Intelligence/openpi) and
the π model family from [Physical Intelligence](https://www.physicalintelligence.company/), and runs
on AgileX [Cobot Magic](https://global.agilex.ai/products/cobot-magic) hardware. `infer/openpi` and
`train/openpi` retain openpi's own license (see `LICENSE` / `LICENSE_GEMMA.txt` inside them).
