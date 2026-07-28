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

Real-time deployment of Vision-Language-Action (VLA) policies requires **asynchronous execution**:
the next action chunk is computed while the current one is still running. That creates a
**prediction–execution misalignment** — by the time a chunk takes over, the observation it was
computed from is already stale — which shows up as inter-chunk discontinuity. Existing methods
either smooth the chunk boundary superficially, pay for costly policy optimization, or roll the
proprioceptive state forward while ignoring the visual observation entirely.

**FutureRTC** supplies the missing observation instead. It has two modules and one loss:

- a **state correction module** that compensates for the gap between the rolled-forward and the
  actual execution-time proprioceptive state;
- an **observation prediction module** that forecasts the execution-time visual representation,
  using robot motion as an explicit physical prior through motion-aware feature transport and
  reconstruction;
- a **policy consistency loss** that aligns the action chunks generated from predicted contexts with
  those the policy would have produced from its true execution-time inputs.

The base VLA policy stays untouched throughout — FutureRTC is a plug-and-play module, not a
fine-tune.

![](Figures/execution_compare.png)

---

## Repository structure

The code for each experimental setting lives on its own branch, so that each one can carry the
environment, dependencies and launch scripts it actually needs. Pick the branch matching what you
want to reproduce:

| Branch | Setting | Backbones | Contents |
|---|---|---|---|
| [`sim/libero`](../../tree/sim/libero) | LIBERO simulation | π₀.₅, SmolVLA-450M | Latent-bank collection, state-correction and observation-prediction training, delay-swept evaluation, trained checkpoints |
| [`sim/Kinetix`](../../tree/sim/Kinetix) | Kinetix simulation | Action-chunking flow policies | 12 dynamic environments, delays `d ∈ [0, 4]` |
| [`dev/realworld`](../../tree/dev/realworld) | Real-world bimanual | — | AgileX Cobot Magic deployment: *Stack Plates*, *Fold Towel*, *Hang Cups* |
