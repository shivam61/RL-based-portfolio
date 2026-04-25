# Current System State

This document is the concise source of truth for what the repo is doing now, why those decisions were taken, and what remains research-only.

The operating rules for how future decisions should be made are defined in:

- [docs/DECISION_PROTOCOL.md](docs/DECISION_PROTOCOL.md)

## Production Track

The current production policy is:

- `tilt_only_rl`
- posture forced to `neutral`
- sector-first breadth enabled
- learned sector tilts enabled
- full optimizer / risk / execution stack enabled

This was an explicit decision, not a temporary hack.

Reason:

- learned sector tilts continued to add value on holdout
- learned posture control did not
- the posture-learning objective was not aligned with realized posture economics
- keeping both problems coupled was slowing iteration and obscuring what was actually working

Current interpretation:

- RL is currently useful as a tilt engine
- RL is not yet trusted as a posture controller

## Research Track

Posture research is now separate from production.

We no longer treat PPO posture behavior as the main research path. Instead, posture is being studied from realized forward outcomes:

- fixed `risk_on`
- fixed `neutral`
- fixed `risk_off`

over short forward horizons in rebalance units.

Reason:

- realized fixed-posture portfolios diverged materially in the execution path
- the previous training-time regret proxy said those postures were nearly identical
- that mismatch explained the single-posture collapse seen in PPO posture experiments

The next posture work therefore focuses on:

- label quality
- utility definition
- learnability

before any new posture policy training.

## Key Decisions And Why

### 1. Freeze production posture to `neutral`

Decision:

- force `neutral` posture in the mainline RL path

Reason:

- on the 2016 holdout, fixed `neutral` was the strongest static posture baseline
- the trained policy’s uplift survived even when posture was frozen
- that proved the current live edge is in sector tilts, not posture switching

### 2. Keep sector-first breadth

Decision:

- keep sector-first breadth in the current structural stack

Reason:

- stock-breadth-only gating created brittle execution and did not improve separability
- sector-first breadth improved the base portfolio shape and holdout economics while keeping execution stable

### 3. Stop tuning PPO posture reward for now

Decision:

- stop iterating on the old posture regret / PPO path as the mainline posture solution

Reason:

- realized fixed-posture outcomes were different
- training-time posture utilities were not
- that meant the learning signal itself was misaligned with realized PnL

### 4. Move posture research to realized labels

Decision:

- build a realized forward-outcome dataset for posture choice

Reason:

- posture should first be tested as a prediction problem
- if realized posture winners are not learnable from state, PPO will not solve that cleanly either

## Current Baselines

These baselines should remain stable while posture research is ongoing:

- `neutral_full_stack`
- `tilt_only_rl`
- `optimizer_only`

All new posture research should compare back to these references, not quietly redefine them.

## What Is Not Yet Solved

- dynamic posture choice
- a trusted posture label utility
- a posture model that beats static `neutral` out of sample

Until those are solved, the correct interpretation of the repo is:

- production-ready tilt RL
- research-stage posture learning
