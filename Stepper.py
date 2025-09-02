# Stepper.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Callable, Dict, Tuple
import numpy as np

Array = np.ndarray

# ─────────────────────────────────────────────────────────────────────────────
# Minimal protocols (kept local to avoid import cycles)
# - grad_source must implement:
#       grad(f_y, y, A, rng) -> (g:(d,), info:dict, fevals:int)
#   and may expose .scheme == "central-2p" for detection
# - MemoryMetric:
#       update(cur_f: float) -> float   # returns M_t (scale-free magnitude)
# - MemoryEvent:
#       update(M_t: float) -> (calib_event: bool, evicted: int)
#       .xi_mem (float > 0), .S_t (float in (0,1])
# - AdjointBlend:
#       calibrate(y, g_2p, adjoint_fn or vector,
#                 grad_space='y'|'x', normalizer=None, sign=+1) -> dict
# Notes:
# * The stepper does NOT own x↔y transformation or preconditioner construction.
#   It only consumes A (if provided) and hands off delayed secant information.
# ─────────────────────────────────────────────────────────────────────────────


# ============================== errors =======================================

class StepperEvalError(RuntimeError):
    """Raised when f_y returns a non-finite value at base or trial points."""
    pass


# ============================== Adam state ===================================

@dataclass
class AdamState:
    """Optimizer state carried across steps."""
    t: int
    m: Array
    v: Array

    @staticmethod
    def init_like(y: Array) -> "AdamState":
        d = int(np.asarray(y).size)
        return AdamState(t=0, m=np.zeros(d, dtype=float), v=np.zeros(d, dtype=float))


# ============================ Stepper config =================================

@dataclass
class StepperConfig:
    # AdamW hyperparameters
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0  # decoupled (AdamW), applied to y directly

    # Trust region & backtracking
    delta: float = 0.1          # initial trust radius (in y-space L2)
    delta_grow: float = 1.5
    delta_shrink: float = 0.5
    delta_min: float = 1e-6
    delta_max: float = 10.0
    backtrack_mult: float = 0.5
    max_backtracks: int = 12

    # Burst-tolerant TR (OFF by default). Uses memory signals only.
    burst_tolerant: bool = False
    m_lo: float = 0.30      # speed-up threshold when accepted and M_t > m_lo
    m_hi: float = 0.80      # brake threshold when rejected and M_t > m_hi
    k_up: float = 1.0       # multiplicative growth strength
    k_dn: float = 1.0       # multiplicative brake strength

    # Event gating (min-gap in accepted steps; 0 disables)
    event_min_gap: int = 0

    # Adjoint calibration (cosine only). When grad_source is not central-2p,
    # we compute a tiny 2P just for calibration on event.
    calib_K2p: int = 6
    calib_eps_rel: float = 1e-4

    # Optional Muon hook: user-supplied function to adjust the raw Adam step.
    # Signature: muon_hook(y, g, step_raw) -> step_adjusted
    muon_hook: Optional[Callable[[Array, Array, Array], Array]] = None


# =============================== The Stepper =================================

class AdamMuonStepper:
    """
    Single owner of policies/decisions:
      - AdamW update (+ optional Muon hook)
      - Trust-region radius control + backtracking (improve-only acceptance)
      - Event min-gap gating and calling 'AdjointBlend' on memory events
      - Exact feval accounting: fevals_total = 1 + fe_grad + fe_line_evals
      - Handoff of delayed secant for an external Preconditioner (start→start)

    Must NOT:
      - Do x↔y transformation (Normalizer owns it)
      - Construct/update the Preconditioner (we only pass 'A' if given)
    """

    def __init__(self,
                 cfg: StepperConfig,
                 memory_metric=None,
                 memory_event=None,
                 adjoint_blend=None,
                 normalizer=None,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.metric = memory_metric          # MemoryMetric or None
        self.event = memory_event            # MemoryEvent or None
        self.blend = adjoint_blend           # AdjointBlend or None
        self.normalizer = normalizer         # for x-space adjoint conversion if needed
        self.rng = np.random.default_rng() if rng is None else rng

        # Internal acceptance counters for event min-gap
        self._accepted_steps = 0
        self._last_event_at = -10**9  # very negative sentinel

    # ───────────────────────────────── core API ──────────────────────────────

    def step(self,
             f_y: Callable[[Array], float],
             y: Array,
             A: Optional[Array],
             adam_state: AdamState,
             grad_source,                    # must expose .grad(...)
             adjoint_fn: Optional[Callable] = None,
             logger: Optional[Callable[[Dict], None]] = None,
             *,
             minimize: bool = True
            ) -> Tuple[Array, float, AdamState, Dict]:
        """
        Execute one optimization policy step.
        Returns (y_new, f_new, adam_state_new, info).

        Contracts
        ---------
        - If rejected: (y,f) unchanged; trust radius reduces (delta_shrink).
        - If accepted: (y_best,f_best) becomes the next 'start point'.
          Memory receives ONLY accepted f (scale-free Δf); Preconditioner
          will be updated OUTSIDE this function using 'info["secant"]'.
        - Exact feval accounting: fevals_total = 1 + fe_grad + fe_line_evals.
          (Adjoint calibration fevals are reported but excluded by contract.)
        - Delayed secant handoff carries (start→start) y/g for an external
          preconditioner to consume once g_new is available at the next iter.
        """
        y = np.asarray(y, float).reshape(-1)
        d = y.size
        cfg = self.cfg

        # 1) Evaluate f(y) at current start point (no policy here)
        f0 = float(f_y(y))
        if not np.isfinite(f0):
            raise StepperEvalError(f"Non-finite f(y) at start: {f0}")
        fevals_base = 1

        # 2) Measure gradient via provided source (A may be used internally)
        g, ginfo, fe_grad = grad_source.grad(f_y, y, A, self.rng)
        if not minimize:
            # maximize: descend on -f → flip gradient sign only
            g = -g

        # 3) AdamW update (raw, before TR/backtracking)
        st = adam_state
        if (st.m.size != d) or (st.v.size != d):
            st = AdamState.init_like(y)  # safety on first call / shape change

        t = st.t + 1
        beta1, beta2 = cfg.beta1, cfg.beta2

        m = beta1 * st.m + (1.0 - beta1) * g
        v = beta2 * st.v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        # Adam preconditioning only (A is not applied here by design)
        adam_step_vec = m_hat / (np.sqrt(v_hat) + cfg.eps)
        step_raw = -cfg.lr * adam_step_vec

        # Decoupled weight decay (AdamW)
        if cfg.weight_decay != 0.0:
            step_raw += -cfg.lr * cfg.weight_decay * y

        # Optional Muon hook for step shaping (user-defined)
        if cfg.muon_hook is not None:
            step_raw = np.asarray(cfg.muon_hook(y, g, step_raw), float).reshape(-1)

        # Trust-region projection (L2 ball of radius delta)
        step = self._project_to_trust_region(step_raw, cfg.delta)

        # 4) Backtracking (improve-only acceptance; maximize flips inequality)
        backtracks = 0
        line_evals = 0  # number of f evaluations along the trial line
        y_best = y
        f_best = f0
        accepted = False

        while True:
            y_try = y + step
            f_try = float(f_y(y_try))
            if not np.isfinite(f_try):
                raise StepperEvalError(
                    f"Non-finite f(y_try) during backtracking: {f_try} "
                    f"(backtracks={backtracks}, |step|={np.linalg.norm(step):.3e})"
                )
            line_evals += 1

            improved = (f_try < f_best) if minimize else (f_try > f_best)
            if improved:
                y_best, f_best = y_try, f_try
                accepted = True
                break

            backtracks += 1
            if backtracks > cfg.max_backtracks:
                break

            # shrink and retry
            step *= cfg.backtrack_mult

        # 5) Memory update (signals only; accepted transition governs Δf)
        mem_info: Dict = {}
        if self.metric is not None and self.event is not None:
            # First feed f0 (Δf=0 on first call; keeps EMA warmed and stable),
            # then feed f_best only on acceptance to accumulate true |Δf|.
            Mt = self.metric.update(f0)  # typically 0 since f0 equals last accepted
            Mt = self.metric.update(f_best) if accepted else Mt
            calib_event, evicted = self.event.update(Mt) if accepted else (False, 0)
            mem_info = {
                "M_t": Mt,
                "calib_event": calib_event,
                "evicted": int(evicted),
                "xi_mem": float(self.event.xi_mem),
                "S_t": float(self.event.S_t),
            }
        else:
            Mt = 0.0  # memory disabled

        # 6) Trust-region radius update (+ optional burst tolerance using mem signals)
        if accepted:
            new_delta = min(cfg.delta_max, cfg.delta * cfg.delta_grow)
        else:
            new_delta = max(cfg.delta_min, cfg.delta * cfg.delta_shrink)

        if cfg.burst_tolerant:
            m_signal = float(mem_info.get("M_t", 0.0))
            S_signal = float(mem_info.get("S_t", 1.0))
            if accepted:
                # Large change & low survival → expand faster
                growth = 1.0 + cfg.k_up * max(0.0, m_signal - cfg.m_lo) * (1.0 - S_signal)
                new_delta = min(cfg.delta_max, new_delta * growth)
            else:
                # Rejection with large change → brake
                brake = 1.0 + cfg.k_dn * max(0.0, m_signal - cfg.m_hi)
                new_delta = max(cfg.delta_min, new_delta / brake)

        # 7) Event-triggered cosine calibration (no policy switching here)
        calib: Dict = {}
        do_calib = False
        if accepted and self.blend is not None and adjoint_fn is not None and self.event is not None:
            # event min-gap measured in accepted steps
            if cfg.event_min_gap <= 0 or (self._accepted_steps - self._last_event_at) >= cfg.event_min_gap:
                do_calib = mem_info.get("calib_event", False)

        fe_calib = 0
        if do_calib:
            # Ensure we have a 2P reference; reuse if grad_source is already 2P
            if getattr(grad_source, "scheme", "") == "central-2p":
                g_ref = g
            else:
                from Gradient import Central2P
                K2 = min(cfg.calib_K2p, max(2, d))
                g_ref, _, fe_c2 = Central2P(K=K2, eps_rel=cfg.calib_eps_rel).grad(
                    f_y, y_best, A=None, rng=self.rng
                )
                fe_calib += fe_c2
                if not minimize:
                    g_ref = -g_ref

            # Compare in y-space (AdjointBlend handles x->y if requested)
            calib = self.blend.calibrate(
                y_best, g_ref, adjoint_fn, grad_space="y", normalizer=self.normalizer, sign=+1
            )
            # Stamp last event only when an actual calibration is performed
            self._last_event_at = self._accepted_steps

        # 8) Assemble outputs
        fevals_total = fevals_base + fe_grad + line_evals  # includes accepted attempt

        info: Dict = {
            "accepted": bool(accepted),
            "scheme": getattr(grad_source, "scheme", "unknown"),
            "K": int(ginfo.get("K", 0)),
            "eps": float(ginfo.get("eps", np.nan)),
            "fe_grad": int(fe_grad),
            "fe_line_evals": int(line_evals),  # number of f-evals along backtracking line
            "fe_backtracks": int(backtracks),
            "fevals_total": int(fevals_total),    # excludes fe_calib by contract
            "delta_old": float(cfg.delta),
            "delta_new": float(new_delta),
            "delta_stuck": bool(np.isclose(new_delta, cfg.delta_min)),
            "step_norm": float(np.linalg.norm(step_raw)),
            "step_used_norm": float(np.linalg.norm(y_best - y)),
            "g_norm": float(np.linalg.norm(g)),
            "adam_m_norm": float(np.linalg.norm(m)),
            "adam_v_mean": float(np.mean(v)),
            "mem": mem_info,                      # memory signals only
            "calib": calib,                       # calibration diagnostics (if any)
            "fe_calib": int(fe_calib),            # reported but excluded
            # Delayed secant handoff (start→start). Consumer (preconditioner) will
            # push (y_old,y_new,g_old,g_new) once g_new is available at the next start.
            "secant": {
                "kind": "start_to_start",
                "y_old": y.copy(),
                "y_new": y_best.copy() if accepted else y.copy(),
                "g_old": g.copy(),
                "ready": bool(accepted),  # True means a new start was declared
            },
        }

        # 9) State & TR update
        new_state = AdamState(t=t, m=m, v=v)
        self.cfg = replace(self.cfg, delta=new_delta)

        if accepted:
            self._accepted_steps += 1
            if logger is not None:
                logger({**info, "f0": f0, "f_new": f_best})
            return y_best, f_best, new_state, info
        else:
            if logger is not None:
                logger({**info, "f0": f0, "f_new": f0})
            return y, f0, new_state, info

    # ───────────────────────────────── helpers ───────────────────────────────

    @staticmethod
    def _project_to_trust_region(step: Array, delta: float) -> Array:
        """
        Scale 'step' to lie within the L2 ball of radius 'delta'.
        No-op if already inside. Deterministic and side-effect free.
        """
        s = np.asarray(step, float).reshape(-1)
        n = np.linalg.norm(s)
        if n <= 0.0:
            return s
        scale = 1.0 if n <= delta else (delta / n)
        return s * scale
