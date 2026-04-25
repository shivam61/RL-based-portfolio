# Decision Protocol

This document defines the operating rules for future work on this repo.

The goal is to stop the system from drifting into:

- implementation before diagnosis
- reward tuning before structural diagnosis
- multiple simultaneous changes with unclear attribution
- promotion based on a single attractive metric

These rules are meant to be followed by both humans and future agents working in this repository.

## Core Principle

Do not move to the next phase until the current phase has a clear diagnosis, a bounded plan, and a measurable success condition.

## Rules

### 1. Root Cause Before Fix

Before changing code, first answer:

- what failed?
- where is it failing?
- what evidence supports that?
- what are the top plausible causes?
- which cause is most likely?

Do not implement speculative fixes without first writing down the current diagnosis.

### 2. Plan Before Implementation

Before substantial implementation work:

- write the proposed next step
- explain why this step is next
- list alternatives considered
- list expected outcomes
- list side effects / failure modes
- define what would count as success or rejection

If the step is contentious or subtle, discuss and debate it before editing code.

### 3. One Major Change Per Measured Run

Each measured experiment should change one primary thing only:

- objective
- execution path
- action mapping
- breadth / sector structure
- label definition

Do not combine multiple major changes in one measured run unless the work is strictly coupled and impossible to separate.

Reason:

- attribution becomes unreliable
- regressions become harder to localize

### 4. Preserve Stable Baselines

Never redefine baselines mid-stream.

Current stable references:

- `neutral_full_stack`
- `tilt_only_rl`
- `optimizer_only`

All research changes should be compared back to those references unless there is an explicit decision to replace a baseline.

### 5. Diagnose Structure Before Reward

If behavior is weak, first check:

- execution feasibility
- realized control realization
- separability of realized outcomes
- optimizer compression
- label quality

Do not default to reward tuning unless structure and measurement are already credible.

This branch already showed why:

- several “reward problems” were actually execution or separability problems first

### 6. Separate Production From Research

If a component is working in production but another coupled component is not:

- freeze the working component
- isolate the failing component into research

Do not keep both coupled in one loop if one is already demonstrably useful and the other is not.

Current example:

- production: tilt-only RL
- research: posture learning

### 7. Prefer Simpler Learning Problems First

When a learning problem is not yet well-posed:

- start with supervised prediction or regression
- use realized outcomes where possible
- use PPO / RL only after simpler formulations prove signal exists

Current application:

- posture should be treated as utility prediction first, not PPO control first

### 8. Make Execution Noise Explicit

If execution artifacts can affect labels or evaluation:

- log them directly
- separate clean vs degraded samples
- do not let hidden execution asymmetry masquerade as alpha

At minimum, track:

- fallback count
- solver / relaxation status
- cash gap
- turnover mismatch
- cap-binding pressure

### 9. Promotion Requires Economics And Mechanism

Do not promote a change only because one outcome metric improved.

A change is only promotable if:

- economics improved or remained acceptably stable
- the intended mechanism actually happened
- diagnostics support the claimed reason for improvement

Example:

- if a controller is supposed to switch posture, it must show credible posture-related evidence
- if a label redesign is supposed to remove defensive skew, the label distribution must actually rebalance

### 10. Rejected Runs Must Still Be Logged

Every rejected or non-promoted run should record:

- what changed
- what was measured
- why it was rejected
- what was learned

Reason:

- rejected runs are part of the research memory
- future agents should not repeat the same dead ends

### 11. New-Machine Reproducibility Is Mandatory

Any meaningful workflow change should leave the repo in a state where a new machine can:

- install dependencies
- download or rebuild required data
- rebuild features
- run the relevant workflow

If data is intentionally not committed, the rebuild path must be documented clearly.

### 12. Docs Are Part Of The Change

If a decision changes:

- production behavior
- baseline interpretation
- research direction
- setup / reproducibility

then the decision and its reasoning must be reflected in docs during the same change, not later.

## Required Decision Template

For any meaningful next step, capture the following before implementation:

1. Current diagnosis
2. Why this is the next step
3. Alternatives considered
4. Expected upside
5. Expected side effects / risks
6. Exact success criteria
7. Exact rejection criteria
8. What baseline it will be compared against

## Phase Gates

Use this order unless there is strong evidence to skip a phase:

1. Diagnose
2. Plan
3. Debate / compare options
4. Implement one bounded change
5. Validate
6. Measure
7. Log
8. Commit / push
9. Decide keep / reject

Do not collapse these phases just to move faster. That tends to slow the program down later.

## Current Implications

Based on the work so far, future posture research should obey these specific rules:

- do not return to PPO posture learning yet
- do not tune reward before label quality and execution noise are understood
- do not treat hard classification as the first posture model if margins remain small
- do not change production posture away from `neutral` without evidence that the replacement beats `tilt_only_rl`

## Status

This protocol is active now and should be treated as part of the repo contract for future work.
