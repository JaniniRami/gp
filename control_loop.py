"""MAD actuator model and budget-aware advance/hold/retract policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MadAction(Enum):
    HOLD = 0
    ADVANCE = 1
    RETRACT = 2


class MadPosition(Enum):
    RETRACTED = 0
    ADVANCING = 1
    ADVANCED = 2
    RETRACTING = 3


@dataclass
class ActuatorConfig:
    advance_sec: float = 10.0
    retract_sec: float = 10.0
    refractory_sec: float = 60.0
    # With state-based coverage, holding through bursts is credited - keep jaw
    # advanced a bit longer after quiet to cover cluster tails.
    quiet_retract_sec: float = 120.0
    hold_through_burst: bool = True
    burst_gap_sec: float = 60.0


@dataclass
class BudgetConfig:
    """Budget PI that adapts the fire threshold toward a target advance rate.

    Default ``enabled=False`` so ``simulate_night(threshold=...)`` genuinely
    controls the operating point during coverage-curve sweeps. Enable only when
    running a live adaptive-budget policy.
    """

    enabled: bool = False
    target_advances_per_hour: float = 5.0
    kp: float = 0.15
    ki: float = 0.02
    thresh_min: float = 0.15
    thresh_max: float = 0.95
    init_threshold: float = 0.5


@dataclass
class MadController:
    act: ActuatorConfig = field(default_factory=ActuatorConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    position: MadPosition = MadPosition.RETRACTED
    pos_timer: float = 0.0
    last_advance_t: float = -1e9
    quiet_timer: float = 0.0
    threshold: float = 0.5
    integral: float = 0.0
    n_advances: int = 0
    advanced_sec: float = 0.0
    t: float = 0.0
    night_start: float = 0.0

    # trajectory log
    advanced_mask: list[bool] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)

    def reset(self, night_start: float = 0.0) -> None:
        self.position = MadPosition.RETRACTED
        self.pos_timer = 0.0
        self.last_advance_t = -1e9
        self.quiet_timer = 0.0
        self.threshold = self.budget.init_threshold
        self.integral = 0.0
        self.n_advances = 0
        self.advanced_sec = 0.0
        self.t = night_start
        self.night_start = night_start
        self.advanced_mask.clear()
        self.actions.clear()
        self.thresholds.clear()

    def _elapsed_hours(self) -> float:
        return max((self.t - self.night_start) / 3600.0, 1.0 / 3600.0)

    def _update_budget(self) -> None:
        if not self.budget.enabled:
            return
        rate = self.n_advances / self._elapsed_hours()
        err = self.budget.target_advances_per_hour - rate
        # Candidate threshold before clip (for anti-windup).
        adj = -self.budget.kp * err - self.budget.ki * self.integral
        raw = self.budget.init_threshold + adj
        clipped = float(np.clip(raw, self.budget.thresh_min, self.budget.thresh_max))
        # Anti-windup: only integrate when not saturating against the error sign.
        saturating_low = clipped <= self.budget.thresh_min + 1e-12 and err > 0
        saturating_high = clipped >= self.budget.thresh_max - 1e-12 and err < 0
        if not (saturating_low or saturating_high):
            self.integral = float(np.clip(self.integral + err, -50.0, 50.0))
            adj = -self.budget.kp * err - self.budget.ki * self.integral
            clipped = float(
                np.clip(
                    self.budget.init_threshold + adj,
                    self.budget.thresh_min,
                    self.budget.thresh_max,
                )
            )
        self.threshold = clipped

    def step(
        self,
        t_sec: float,
        prob: float,
        *,
        wake: bool = False,
        in_burst: bool = False,
    ) -> MadAction:
        """One 1 Hz control step. Returns action taken this second."""
        dt = 1.0 if not self.advanced_mask else max(0.0, t_sec - self.t)
        self.t = t_sec
        action = MadAction.HOLD

        # finish transitions
        if self.position == MadPosition.ADVANCING:
            self.pos_timer += dt
            if self.pos_timer >= self.act.advance_sec:
                self.position = MadPosition.ADVANCED
                self.pos_timer = 0.0
        elif self.position == MadPosition.RETRACTING:
            self.pos_timer += dt
            if self.pos_timer >= self.act.retract_sec:
                self.position = MadPosition.RETRACTED
                self.pos_timer = 0.0

        is_advanced = self.position in (MadPosition.ADVANCED, MadPosition.ADVANCING)
        if is_advanced:
            self.advanced_sec += dt

        # risk / quiet tracking
        high_risk = (not wake) and (prob >= self.threshold)
        if high_risk or in_burst:
            self.quiet_timer = 0.0
        else:
            self.quiet_timer += dt

        can_advance = (
            self.position == MadPosition.RETRACTED
            and (t_sec - self.last_advance_t) >= self.act.refractory_sec
            and not wake
        )

        if can_advance and high_risk:
            self.position = MadPosition.ADVANCING
            self.pos_timer = 0.0
            self.last_advance_t = t_sec
            self.n_advances += 1
            action = MadAction.ADVANCE
        elif (
            self.position == MadPosition.ADVANCED
            and self.quiet_timer >= self.act.quiet_retract_sec
            and not (self.act.hold_through_burst and in_burst)
        ):
            self.position = MadPosition.RETRACTING
            self.pos_timer = 0.0
            action = MadAction.RETRACT

        self._update_budget()

        is_advanced = self.position in (MadPosition.ADVANCED, MadPosition.ADVANCING)
        self.advanced_mask.append(is_advanced)
        self.actions.append(action.value)
        self.thresholds.append(self.threshold)
        return action

    def fraction_advanced(self) -> float:
        if not self.advanced_mask:
            return 0.0
        return float(np.mean(self.advanced_mask))

    def advances_per_hour(self) -> float:
        return float(self.n_advances / self._elapsed_hours())
