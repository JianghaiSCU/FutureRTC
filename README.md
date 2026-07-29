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

FutureRTC supplies the missing execution-time observation instead of smoothing over the seam. On
Kinetix the observation splits cleanly into *robot* dimensions and *environment* dimensions, and the
paper's two modules land on the two halves:

| Paper | On Kinetix |
|---|---|
| State correction module | **Exact forward simulation.** The `d` already-executed actions are replayed from the current state. Proprioception is fully known here, so this half needs no learning — where LIBERO must *learn* a residual correction, Kinetix gets it analytically. |
| Observation prediction module | **Learned per-level latent predictor.** Given the current latent, the executed-action prefix and the delay, it predicts the environment's future observation latent. |


## Results

FutureRTC on the bc31 base policy, averaged over the 12 levels and the execute-horizon sweep
(2048 trials per level):

| inference delay | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| avg. sucess rate | 89.0 | 89.0 | 88.0 | 86.4 | 86.0 |
| avg. execution steps | 92.3 | 92.5 | 94.8 | 96.6 | 98.1 |

Solve rate stays essentially flat as the delay grows — a **3.0%** drop from `d = 0` to `d = 4` —
because the learned environment prediction absorbs the delay that a naive handoff would otherwise
pay in lost control. See the paper for the comparison against Naive Async., TE, BID, RTC, T-RTC,
VLASH and REMAC.

---

## Requirements

| Item | Recommendation |
|------|----------------|
| Python | **3.12** |
| Stack | JAX 0.4.35, flax, optax, numpy |
| External | The Real-Time Chunking Kinetix repo |

## Setup

This code builds on the Real-Time Chunking Kinetix codebase and does not vendor it:

- **RTC repo** — <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>. Clone it and
  initialize its `third_party/kinetix` submodule (<https://github.com/FlairOx/Kinetix>). It provides
  the Kinetix environment, the `FlowPolicy` (`src/model.py`), and `src/train_expert.py`.
- **Base policy (bc31)** — a per-level BC flow-matching policy. Download one from
  `gs://rtc-assets/bc/`, or train it with the RTC repo's `src/train_flow.py`. We use the epoch-31
  checkpoint ("bc31"). The loaders expect the layout `<run-path>/<step>/policies/<level_name>.pkl`
  and select the highest-numbered `<step>`.

```bash
pip install -r requirements.txt
```

Stage 2 (training) needs only those packages. **Stages 1 and 3 additionally import the RTC repo**
(its Kinetix env + `FlowPolicy`), so run them where that repo is importable — the simplest option is
the RTC repo's own environment with `optax` added.

### Paths

Nothing is hardcoded; both variables are also settable per-command via `--rtc-root` / `--run-path`.

| Variable | Description |
|----------|-------------|
| `RTC_ROOT` | Your Real-Time Chunking Kinetix checkout |
| `RTC_BC_RUN_PATH` | Base-policy run path, laid out as `<run-path>/<step>/policies/<level>.pkl` |

```bash
export RTC_ROOT=/path/to/real-time-chunking-kinetix
export RTC_BC_RUN_PATH=$RTC_ROOT/pretrained_bc31
```

---

## Pipeline

The method is trained and evaluated **per Kinetix level** (12 levels by default). To evaluate the
shipped weights, skip to Stage 3.

### Stage 1 — collect on-policy environment latents

Rolls out the base policy and records `(z_s, motion_actions, delay) -> z_env` shards.

> The predictor **must** be trained on-policy; off-policy or expert data collapses it.

```bash
python scripts/collect_latents.py \
  --rtc-root "$RTC_ROOT" --run-path "$RTC_BC_RUN_PATH" \
  --output-dir outputs/latents
```

### Stage 2 — train the observation prediction module

```bash
python scripts/train_predictor.py \
  --data-dir outputs/latents \
  --output-dir outputs/predictors
```

### Stage 3 — evaluate

Sweeps inference delays `0..4` over the execute-horizon grid, 2048 trials per level by default, and
writes `results.jsonl` + `summary.csv`:

```bash
python scripts/eval_handoff.py \
  --rtc-root "$RTC_ROOT" --run-path "$RTC_BC_RUN_PATH" \
  --predictor-dir weights/predictors \
  --delays 0,1,2,3,4 --output-dir outputs/eval
```

### Pretrained weights

`weights/predictors/` ships the 12 per-level predictors (one `<level>.pkl` each, ~15 MB) that
produce the [results](#results) above on the bc31 base policy. With them you can run Stage 3
directly and skip Stages 1–2 — you still need the RTC repo and the bc31 base policy to evaluate.
They are the same architecture this release trains; `train_predictor.py` reproduces them from
scratch with the MSE-only recipe.

---

## Tests

```bash
python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

`test_predictor.py` needs JAX; the delay-grid and results tests run without it.

---

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

The Kinetix experiments build on
[Real-Time Chunking Kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
from [Physical Intelligence](https://www.physicalintelligence.company/) and on
[Kinetix](https://github.com/FlairOx/Kinetix).
