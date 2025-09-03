# Memory.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Union
import numpy as np

Number = float

# =============================== Metric =====================================

@dataclass
class MemoryMetric:
    """
    Scale-free change metric (single policy, no modes).

    Design
    ------
    - EMA baseline with half-life via beta in (0,1).
      (If desired, set beta from a half-life t_half via beta = 2**(-1/t_half).)
    - Ratio r = |Δf| / EMA_hat  →  x = r/(1+r) ∈ (0,1).
    - Final magnitude M_t = σ( 4·(x-0.5) ) ∈ (0,1), slope at x=0.5 equals 1.

    Invariants
    ----------
    - M_t ∈ [0,1), finite.
    - First call uses Δf=0 → M_1 = 0 (safe warm-up).
    """
    beta: float = 0.9          # recommended to set via half-life
    min_ema: float = 1e-12
    debias: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must be in (0,1)")
        if not (self.min_ema > 0.0 and np.isfinite(self.min_ema)):
            raise ValueError("min_ema must be finite and > 0")
        self.reset()

    def reset(self) -> None:
        self._prev_f: Optional[Number] = None
        self._ema: Number = float(self.min_ema)
        self._t: int = 0  # for bias correction

    @staticmethod
    def _ema_update(prev_ema: float, u: float, beta: float) -> float:
        return beta * prev_ema + (1.0 - beta) * u

    def _ema_hat(self) -> float:
        """Bias-corrected EMA (Adam-style) if debias=True, else raw EMA."""
        if not self.debias:
            return self._ema
        corr = 1.0 - (self.beta ** max(1, self._t))  # t>=1 after first update
        if corr <= 0.0:
            return self._ema
        return self._ema / corr

    @staticmethod
    def _sigmoid(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-z)))

    def update(self, cur_f: Number) -> Number:
        """
        Update EMA(|Δf|) and return M_t (scale-free magnitude in (0,1)).
        Raises on non-finite input to prevent state corruption.
        """
        cur_f = float(cur_f)
        if not np.isfinite(cur_f):
            raise ValueError(f"cur_f must be finite, got {cur_f}")

        # |Δf|
        u = 0.0 if self._prev_f is None else abs(cur_f - self._prev_f)

        # EMA state
        self._ema = float(max(self.min_ema, self._ema_update(self._ema, u, self.beta)))
        self._t += 1
        self._prev_f = cur_f

        # First-call warm-up
        if self._t == 1:
            return 0.0

        # Scale-free mapping with bounded slope (slope=1 at x=0.5)
        ema_hat = max(self.min_ema, float(self._ema_hat()))
        r = u / ema_hat
        x = r / (1.0 + r) if np.isfinite(r) else 1.0
        x = float(np.clip(x, 1e-12, 1.0 - 1e-12))
        M = self._sigmoid(4.0 * (x - 0.5))
        return M


# =============================== Event ======================================

@dataclass
class MemoryEvent:
    """
    Event scheduler and survival tracking.

    Policy (calm-gated, with intuitive τ)
    -------------------------------------
    - Survival (per-accepted-step decay):  s_t = exp(-α · M_t),  S_t = exp(-∑ α·M_t).
    - "Opportunities" accrue whenever cumulative hazard H crosses the positive
      barrier ΔH = ln(1 + τ). (τ ≥ 0 ensures there is no sign ambiguity.)
    - Only one opportunity fires on a calm step (gate: M_t <= m_gate, default 0.5).
      Otherwise it is queued and released later, one per calm step.

    Parameters
    ----------
    alpha : >0, overall timescale (faster decay and faster opportunity accrual)
    tau   : ≥0, resolution (larger ⇒ bigger barrier ln(1+τ) ⇒ rarer opportunities)
    m_gate: in (0,1), calmness gate threshold on M_t (default 0.5)

    Runtime control (for Bridge)
    ----------------------------
    - set_speed(scale) : alpha ← alpha * scale          (scale>0)
    - set_alpha(a)     : alpha ← a                      (a>0)
    - set_tau(tau)     : tau ← tau; barrier ← ln(1+τ)   (τ≥0)
    - set_gate(g)      : m_gate ← g                     (g∈(0,1))
    """
    alpha: float = 1.0
    tau: float = 0.9
    m_gate: float = 0.5

    def __post_init__(self) -> None:
        if not (self.alpha > 0.0 and np.isfinite(self.alpha)):
            raise ValueError("alpha must be finite and > 0")
        if not (self.tau >= 0.0 and np.isfinite(self.tau)):
            raise ValueError("tau must be finite and ≥ 0")
        if not (0.0 < self.m_gate < 1.0):
            raise ValueError("m_gate must be in (0,1)")
        self._barrier = float(np.log1p(self.tau))  # ΔH = ln(1+τ) > 0 if τ>0; =0 ⇒ every step opportunity
        if self._barrier <= 0.0:
            # τ=0 ⇒ barrier=0 ⇒ treat as tiny positive to avoid storms
            self._barrier = 1e-12
        self.reset()

    def reset(self) -> None:
        self._H: float = 0.0          # cumulative hazard (for survival & opportunities)
        self._pending: int = 0        # queued opportunities not yet fired due to gate

    # --- runtime control for Bridge ---
    def set_speed(self, scale: float) -> None:
        """Multiply alpha by scale (>0)."""
        s = float(scale)
        if np.isfinite(s) and s > 0.0:
            self.alpha = float(self.alpha * s)

    def set_alpha(self, new_alpha: float) -> None:
        a = float(new_alpha)
        if np.isfinite(a) and a > 0.0:
            self.alpha = a

    def set_tau(self, new_tau: float) -> None:
        t = float(new_tau)
        if np.isfinite(t) and t >= 0.0:
            self.tau = t
            self._barrier = float(np.log1p(self.tau))
            if self._barrier <= 0.0:
                self._barrier = 1e-12

    def set_gate(self, new_gate: float) -> None:
        g = float(new_gate)
        if np.isfinite(g) and 0.0 < g < 1.0:
            self.m_gate = g

    # --- compatibility API (used by stepper) ---
    def update(self, M_t: Number) -> Tuple[bool, int]:
        """
        Backward-compatible: updates internal state and returns (calib_event, evicted_fired).
        Does NOT return survival scalar here (engine provides it).
        """
        fired, fired_count, _ = self.step_gated(M_t)
        return fired, fired_count

    # --- new detailed step with survival scalar (used by engine) ---
    def step_gated(self, M_t: Number) -> Tuple[bool, int, float]:
        """
        Update with M_t and return:
          - calib_event (bool): whether an event actually fired (after gate)
          - evicted_fired (int): 0 or 1 (fires at most one per calm step)
          - survival_mul (float): s_t = exp(-alpha * M_t)
        """
        mt = float(M_t)
        if not np.isfinite(mt) or mt < 0.0:
            mt = 0.0

        # Per-turn survival multiplier and hazard increment
        dH = self.alpha * mt
        s_t = float(np.exp(-dH))
        H_prev = self._H
        self._H = H_prev + dH

        # Opportunity accounting (ideal schedule ignoring gate)
        k_prev = int(np.floor(H_prev / self._barrier))
        k_now  = int(np.floor(self._H   / self._barrier))
        new_opps = max(0, k_now - k_prev)
        self._pending += new_opps

        # Calm-gated firing: only one per calm step for regularity
        fired = False
        fired_count = 0
        if self._pending > 0 and mt <= self.m_gate:
            self._pending -= 1
            fired = True
            fired_count = 1

        return fired, fired_count, s_t

    @property
    def S_t(self) -> float:
        """Current survival S_t = exp(-H_t)."""
        return float(np.exp(-self._H))

    @property
    def H_t(self) -> float:
        """Cumulative hazard value (monotone non-decreasing)."""
        return float(self._H)

    @property
    def xi_mem(self) -> float:
        """
        Idealized spacing (ignoring gate): xi ≈ barrier / alpha.
        Actual spacing is >= this due to calm gating.
        """
        return self._barrier / self.alpha

    @property
    def pending(self) -> int:
        """Number of queued opportunities awaiting a calm step."""
        return int(self._pending)

    @property
    def barrier(self) -> float:
        """Per-opportunity hazard barrier ΔH = ln(1+τ)."""
        return float(self._barrier)


# =============================== Engine =====================================

@dataclass
class MemoryEngine:
    """
    Lightweight ingestor combining MemoryMetric and MemoryEvent.
    - Call once per *accepted* evaluation.
    - Returns current magnitude, survival scalar (global decay), and event flags.
    - Deletion policy: caller applies `thres` at *event time only* for regularity.

    Returned dict keys
    ------------------
    M_t           : float in (0,1)
    survival_mul  : float in (0,1]  # multiply all memory weights by this each call
    S_t           : float in (0,1]  # cumulative survival (monitoring)
    calib_event   : bool            # True iff one event actually fired (after gate)
    evicted       : int             # 0 or 1 this step
    pending       : int             # queued opportunities
    xi_mem        : float           # ideal spacing (barrier/alpha), gate makes it longer
    barrier       : float           # ln(1+tau)
    """
    metric: MemoryMetric = field(default_factory=MemoryMetric)
    event:  MemoryEvent  = field(default_factory=MemoryEvent)

    def reset(self) -> None:
        self.metric.reset()
        self.event.reset()

    def ingest_f(self, cur_f: Number) -> Dict[str, Union[float, int, bool]]:
        M_t = self.metric.update(cur_f)
        fired, evicted, s = self.event.step_gated(M_t)
        return {
            "M_t": M_t,
            "survival_mul": s,
            "S_t": self.event.S_t,
            "calib_event": fired,
            "evicted": evicted,
            "pending": self.event.pending,
            "xi_mem": self.event.xi_mem,
            "barrier": self.event.barrier,
        }

    # Backward compatibility alias (name only; no loop ownership)
    step_with_f = ingest_f
