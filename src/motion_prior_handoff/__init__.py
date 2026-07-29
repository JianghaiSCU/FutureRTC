"""Motion-prior split-handoff for real-time action chunking on Kinetix.

Under an inference delay ``d``, at chunk handoff the robot observation dims are taken from
forward-simulating the ``d`` already-executed actions (proprioception is known) and the
environment observation dims are supplied by a learned per-level latent predictor. The two
are combined in FlowPolicy's observation-latent space and decoded into the next action
chunk. Eval method name: ``futurertc``.

Submodules:
  predictor   - the single-token motion-prior latent predictor (the learned component)
  flow_policy - FlowPolicy observation-latent interface + level/delay utilities
  rtc_env     - RTC/Kinetix bootstrap + env / policy / checkpoint loaders
  robot_mask  - per-level robot vs. environment observation split
  delay_grid  - (delay, execute_horizon) sweep enumeration
  results     - record schema, aggregation, and output writers
"""

__version__ = "0.1.0"
