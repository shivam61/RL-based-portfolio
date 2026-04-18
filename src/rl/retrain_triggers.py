"""
Adaptive RL retrain trigger detection.

Three tiers:
  Tier 1 — Portfolio/PnL events   (hard, immediate retrain)
  Tier 2 — Macro regime events    (require persistence across periods)
  Tier 3 — Model drift events     (subtle, need baseline comparison)

Usage in walk_forward.py:
    detector = EventDetector(cfg)
    ...
    events = detector.update(period_outcome)
    if events:
        _retrain_rl(reason=events[0].reason)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrainEvent:
    tier: int           # 1, 2, or 3
    name: str           # machine-readable identifier
    reason: str         # human-readable description
    severity: float     # 0.0–1.0  (1.0 = most urgent)


class EventDetector:
    """
    Monitors every rebalance period's outcome and fires retraining
    events when conditions exceed configured thresholds.

    Call update() after each period. It returns a (possibly empty)
    list of RetrainEvent — the caller decides whether to retrain.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg.get("rl_triggers", {})
        self._enabled = self._cfg.get("enabled", False)

        t1 = self._cfg.get("tier1_portfolio", {})
        t2 = self._cfg.get("tier2_macro", {})
        t3 = self._cfg.get("tier3_drift", {})

        # Tier 1 thresholds
        self._dd_threshold       = t1.get("drawdown_threshold", 0.10)
        self._shock_threshold    = t1.get("period_shock", -0.04)
        self._consec_loss_n      = t1.get("consecutive_losses", 3)
        self._hit_window         = t1.get("hit_rate_window", 10)
        self._hit_min            = t1.get("hit_rate_min", 0.45)

        # Tier 2 thresholds
        self._vix_jump_pct       = t2.get("vix_jump_pct", 0.40)
        self._vix_persist        = t2.get("vix_persistence_periods", 2)
        self._regime_flip        = t2.get("regime_flip", True)

        # Tier 3 thresholds
        self._leadership_window  = t3.get("sector_leadership_window", 4)
        self._leadership_churn   = t3.get("sector_leadership_churn", 0.50)
        self._reward_shift_std   = t3.get("reward_shift_std", 1.5)
        self._entropy_increase   = t3.get("rl_entropy_increase_pct", 0.30)
        self._drift_min_buf      = t3.get("min_buffer_for_drift", 20)

        # Rolling state
        self._period_returns:    Deque[float] = deque(maxlen=max(self._hit_window, 20))
        self._rewards:           Deque[float] = deque(maxlen=50)
        self._vix_high_periods:  int = 0
        self._prev_regime:       str = "neutral"
        self._sector_rank_history: Deque[list] = deque(maxlen=self._leadership_window + 1)
        self._reward_baseline_mean: float | None = None
        self._reward_baseline_std:  float | None = None
        self._training_step_count:  int = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, period_outcome: dict) -> list[RetrainEvent]:
        """
        Call after each rebalance period. Returns triggered events (may be empty).

        period_outcome keys expected:
          portfolio_return    — float, period return
          max_drawdown_episode — float, current drawdown magnitude
          reward              — float, RL reward this period (optional)
          macro_state         — dict with vix_level, india_vix* keys
          risk_regime         — str: bull/bear/neutral/stressed
          sector_scores       — dict {sector: float score}
          rl_action_entropy   — float (optional, from PPO policy)
        """
        if not self._enabled:
            return []

        ret    = float(period_outcome.get("portfolio_return", 0.0))
        dd     = abs(float(period_outcome.get("max_drawdown_episode", 0.0)))
        reward = float(period_outcome.get("reward", ret))
        macro  = period_outcome.get("macro_state", {})
        regime = period_outcome.get("risk_regime", "neutral")
        sector_scores = period_outcome.get("sector_scores", {})
        entropy = period_outcome.get("rl_action_entropy", None)

        self._period_returns.append(ret)
        self._rewards.append(reward)
        self._training_step_count += 1

        # Update reward baseline after warmup
        if len(self._rewards) >= 20 and self._reward_baseline_mean is None:
            arr = list(self._rewards)
            self._reward_baseline_mean = float(np.mean(arr))
            self._reward_baseline_std  = float(np.std(arr) + 1e-6)

        # Top-3 sector momentum rank (for drift detection)
        if sector_scores:
            top3 = [s for s, _ in sorted(sector_scores.items(),
                                          key=lambda x: -x[1])[:3]]
            self._sector_rank_history.append(top3)

        events: list[RetrainEvent] = []
        events.extend(self._check_tier1(ret, dd))
        events.extend(self._check_tier2(macro, regime))
        events.extend(self._check_tier3(sector_scores, reward, entropy))

        self._prev_regime = regime

        if events:
            for e in events:
                logger.info(
                    "RetrainTrigger [Tier %d | %s] severity=%.2f  %s",
                    e.tier, e.name, e.severity, e.reason,
                )
        return events

    def notify_retrained(self) -> None:
        """Call after a retrain so drift baselines reset."""
        if len(self._rewards) >= 10:
            arr = list(self._rewards)
            self._reward_baseline_mean = float(np.mean(arr))
            self._reward_baseline_std  = float(np.std(arr) + 1e-6)
        self._vix_high_periods = 0

    # ── Tier 1 — Portfolio/PnL ────────────────────────────────────────────────

    def _check_tier1(self, ret: float, dd: float) -> list[RetrainEvent]:
        events = []

        # Drawdown breach
        if dd > self._dd_threshold:
            events.append(RetrainEvent(
                tier=1, name="drawdown_breach",
                reason=f"Portfolio drawdown {dd:.1%} > threshold {self._dd_threshold:.1%}",
                severity=min(1.0, dd / (self._dd_threshold * 2)),
            ))

        # Single-period shock
        if ret < self._shock_threshold:
            events.append(RetrainEvent(
                tier=1, name="period_shock",
                reason=f"Period return {ret:.1%} < shock threshold {self._shock_threshold:.1%}",
                severity=min(1.0, abs(ret) / abs(self._shock_threshold) * 0.8),
            ))

        # Consecutive losses
        if len(self._period_returns) >= self._consec_loss_n:
            recent = list(self._period_returns)[-self._consec_loss_n:]
            if all(r < 0 for r in recent):
                events.append(RetrainEvent(
                    tier=1, name="consecutive_losses",
                    reason=f"{self._consec_loss_n} consecutive losing periods "
                           f"(returns: {[f'{r:.1%}' for r in recent]})",
                    severity=0.7,
                ))

        # Hit rate collapse
        if len(self._period_returns) >= self._hit_window:
            recent = list(self._period_returns)[-self._hit_window:]
            hit_rate = sum(1 for r in recent if r > 0) / len(recent)
            if hit_rate < self._hit_min:
                events.append(RetrainEvent(
                    tier=1, name="hit_rate_collapse",
                    reason=f"Realised hit rate {hit_rate:.1%} < minimum {self._hit_min:.1%} "
                           f"over last {self._hit_window} periods",
                    severity=min(1.0, (self._hit_min - hit_rate) / self._hit_min + 0.5),
                ))

        return events

    # ── Tier 2 — Macro regime ─────────────────────────────────────────────────

    def _check_tier2(self, macro: dict, regime: str) -> list[RetrainEvent]:
        events = []

        # VIX spike (India VIX preferred, fall back to US VIX)
        vix_now = macro.get("india_vix_pctile_1y") or macro.get("vix_pctile_1y")
        if vix_now is not None:
            # Use percentile: >0.90 = top-decile fear = spike
            if float(vix_now) > 0.90:
                self._vix_high_periods += 1
            else:
                self._vix_high_periods = max(0, self._vix_high_periods - 1)

            if self._vix_high_periods >= self._vix_persist:
                events.append(RetrainEvent(
                    tier=2, name="vix_spike",
                    reason=f"VIX in top decile for {self._vix_high_periods} consecutive periods "
                           f"(pctile={float(vix_now):.2f})",
                    severity=0.75,
                ))

        # Regime flip
        if self._regime_flip and self._prev_regime != regime:
            # Only trigger on meaningful flips — not neutral↔neutral noise
            meaningful = {
                ("bull", "bear"), ("bear", "bull"),
                ("bull", "stressed"), ("stressed", "bull"),
                ("neutral", "stressed"), ("stressed", "neutral"),
            }
            pair = (self._prev_regime, regime)
            if pair in meaningful:
                events.append(RetrainEvent(
                    tier=2, name="regime_flip",
                    reason=f"Risk regime flipped: {self._prev_regime} → {regime}",
                    severity=0.65,
                ))

        return events

    # ── Tier 3 — Model drift ─────────────────────────────────────────────────

    def _check_tier3(
        self,
        sector_scores: dict,
        reward: float,
        entropy: float | None,
    ) -> list[RetrainEvent]:
        if self._training_step_count < self._drift_min_buf:
            return []

        events = []

        # Sector leadership churn
        if len(self._sector_rank_history) >= self._leadership_window + 1:
            oldest = set(self._sector_rank_history[0])
            newest = set(self._sector_rank_history[-1])
            if oldest:
                churn = len(oldest.symmetric_difference(newest)) / (len(oldest) + len(newest)) * 2
                if churn > self._leadership_churn:
                    events.append(RetrainEvent(
                        tier=3, name="sector_leadership_change",
                        reason=f"Top-3 sector leadership changed {churn:.0%} over "
                               f"{self._leadership_window} periods "
                               f"(was {sorted(oldest)} → now {sorted(newest)})",
                        severity=min(1.0, churn * 0.8),
                    ))

        # Reward distribution shift
        if self._reward_baseline_mean is not None:
            recent_rewards = list(self._rewards)[-10:]
            if len(recent_rewards) >= 5:
                recent_mean = float(np.mean(recent_rewards))
                drop = (self._reward_baseline_mean - recent_mean) / (self._reward_baseline_std + 1e-6)
                if drop > self._reward_shift_std:
                    events.append(RetrainEvent(
                        tier=3, name="reward_distribution_shift",
                        reason=f"Recent reward mean {recent_mean:.3f} dropped {drop:.1f} std "
                               f"from baseline {self._reward_baseline_mean:.3f}",
                        severity=min(1.0, drop / (self._reward_shift_std * 2)),
                    ))

        # RL confidence / entropy increase
        if entropy is not None and hasattr(self, "_baseline_entropy"):
            increase = (entropy - self._baseline_entropy) / (self._baseline_entropy + 1e-6)
            if increase > self._entropy_increase:
                events.append(RetrainEvent(
                    tier=3, name="rl_confidence_drop",
                    reason=f"RL action entropy increased {increase:.0%} "
                           f"(baseline={self._baseline_entropy:.3f}, now={entropy:.3f})",
                    severity=min(1.0, increase * 0.6),
                ))
        elif entropy is not None and not hasattr(self, "_baseline_entropy"):
            self._baseline_entropy = entropy  # type: ignore[attr-defined]

        return events
