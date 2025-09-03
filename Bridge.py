# bridge.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Tuple
import numpy as np

from Normalize import Normalizer
from Stepper import AdamMuonStepper, StepperConfig, AdamState
from Gradient import Forward1P, Central2P, AdjointBlend
from preconditioner import Preconditioner, PreconditionerConfig
from Memory import MemoryMetric, MemoryEvent

Array = np.ndarray


# ----------------------------- Bridge config ---------------------------------

@dataclass
class BridgeConfig:
    """
    Lightweight knobs for the bridge (no mutable shared defaults).
    """
    seed: Optional[int] = None                  # RNG seed (reproducibility)
    use_precond: bool = True                    # only on/off (no policy here)
    bootstrap_K1p: int = 2                      # first call only (Forward-1P)
    central_K2p: int = 6                        # subsequent calls (Central-2P)
    eps_rel_1p: float = 1e-3
    eps_rel_2p: float = 1e-4
    stepper_cfg: StepperConfig = field(default_factory=StepperConfig)

    # --- NEW: Adjoint pullback policy ---
    # 'diag'   : safe baseline (width-only), never uses mixing
    # 'full'   : always J^T g_x (if available); internally clamps near squash edges or when ill-conditioned
    # 'hybrid' : (default) auto-switch full↔diag based on conditioning and boundary safety
    jac_pullback: str = "hybrid"               # 'diag' | 'full' | 'hybrid'
    jac_cond_cap: float = 1e3                  # maximum allowed cond(J)
    jac_squash_eps: float = 1e-6               # clip squash axes to y∈[eps, 1-eps]
    jac_sigma_lo: float = 1e-3                 # lower bound for SVD singular values (pullback clipping)
    jac_sigma_hi: float = 1e3                  # upper bound for SVD singular values (pullback clipping)
    jac_use_svd_clip: bool = False             # if True, clip J via SVD before pullback


# ----------------------------- The Bridge ------------------------------------

class OnePointOptimizer:
    """
    Orchestrator/bridge that connects Normalizer, Stepper, Preconditioner, and Memory.

    Owns / Decides
    --------------
    - f_y = f_x ∘ decode
    - adjoint in y-space:
        * 'diag'   : adj_y(y) = (grad_x(decode(y))) ⊙ width
        * 'full'   : adj_y(y) = J(y)^T · grad_x(decode(y))  (J=∂x/∂y)
        * 'hybrid' : automatically switches full↔diag based on conditioning and boundary safety
    - Exactly one bootstrap step with Forward-1P, then always Central-2P
    - Preconditioner ON/OFF toggle only
    - RNG/seed (reproducibility)

    Public API
    ----------
    step(y, logger=None) -> (y_new, f_new, info)
    """

    def __init__(
        self,
        f_x: Callable[[Array], float],
        *,
        normalizer: Normalizer,
        adjoint_x: Optional[Callable[[Array], Array]] = None,
        bridge_cfg: Optional[BridgeConfig] = None,
        precond_cfg: Optional[PreconditionerConfig] = None,
        memory_metric: Optional[MemoryMetric] = None,
        memory_event: Optional[MemoryEvent] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # Wiring
        self.N = normalizer
        self.f_x = f_x
        self.adj_x = adjoint_x
        self.cfg = bridge_cfg or BridgeConfig()

        # Deterministic RNG
        self.rng = np.random.default_rng(self.cfg.seed) if rng is None else rng

        # Stepper deps
        self.metric = memory_metric or MemoryMetric()
        self.event = memory_event or MemoryEvent()
        self.blend = AdjointBlend()
        self.stepper = AdamMuonStepper(
            cfg=self.cfg.stepper_cfg,
            memory_metric=self.metric,
            memory_event=self.event,
            adjoint_blend=self.blend,
            normalizer=self.N,
            rng=self.rng,
        )

        # Preconditioner
        self.precond = Preconditioner(
            dim=self.N.dim,
            periodic_mask=self.N.periodic_mask,
            config=precond_cfg or PreconditionerConfig()
        )

        # Gradient sources
        self.g1p = Forward1P(K=self.cfg.bootstrap_K1p, eps_rel=self.cfg.eps_rel_1p)
        self.g2p = Central2P(K=self.cfg.central_K2p, eps_rel=self.cfg.eps_rel_2p)

        # State
        self._state = AdamState.init_like(np.zeros(self.N.dim, float))
        self._did_bootstrap = False
        self._pending: Optional[Dict[str, Array]] = None  # {"y_old","y_new","g_old"}

        # Cache for adjoint pullback
        self._W = getattr(self.N, "_W", np.ones(self.N.dim, float)).astype(float).reshape(-1)
        self._mode = getattr(self.N, "_mode", np.array(["wrap"] * self.N.dim, dtype=object))
        self._squash_s = getattr(self.N, "_squash_s", np.ones(self.N.dim, float)).astype(float).reshape(-1)
        self._has_squash = bool(np.any(self._mode == "squash"))

        # Try to precompute constant J if available (no squash + affine axes)
        self._J_const: Optional[np.ndarray] = None
        self._J_const_cond: Optional[float] = None
        try:
            if not self._has_squash:
                J = self.N.jac_y_to_x()  # ∂x/∂y, shape (d,d)
                if isinstance(J, np.ndarray) and J.shape == (self.N.dim, self.N.dim):
                    self._J_const = J.astype(float, copy=True)
                    # cond estimate
                    self._J_const_cond = float(np.linalg.cond(self._J_const))
        except Exception:
            self._J_const = None
            self._J_const_cond = None

    # ------------------------------- f_y --------------------------------------

    def _f_y(self, y: Array) -> float:
        y = np.asarray(y, float).reshape(-1)
        if y.size != self.N.dim:
            raise ValueError(f"y dimension {y.size} != normalizer.dim {self.N.dim}")
        x = self.N.cap_decode(y, input='y')
        return float(self.f_x(x))

    # ------------------------------- J(y) -------------------------------------

    def _jac_y_to_x_dynamic(self, y: Array) -> np.ndarray:
        """
        Build J=∂x/∂y at y.
        - If precomputed constant J exists, return it.
        - If squash present: diagonal with d x_j / d y_j = s_j / (y_j*(1-y_j)),
          while non-squash axes use width W_j (mixing is disallowed with squash).
        """
        if self._J_const is not None:
            return self._J_const

        d = self.N.dim
        y = np.asarray(y, float).reshape(-1)
        if y.size != d:
            raise ValueError("y has wrong dimension")

        if not self._has_squash:
            # Affine case but constant J fetch failed: fall back to diag(W)
            return np.diag(self._W.copy())

        # Squash present: mixing is disallowed by Normalizer factory, J is diagonal
        y_clip = np.clip(y, self.cfg.jac_squash_eps, 1.0 - self.cfg.jac_squash_eps)
        deriv = np.empty(d, float)
        for j in range(d):
            if self._mode[j] == "squash":
                # x_j = c + s * logit(y_j) ⇒ dx/dy = s / (y*(1-y))
                deriv[j] = self._squash_s[j] / (y_clip[j] * (1.0 - y_clip[j]))
            else:
                deriv[j] = self._W[j]
        return np.diag(deriv)

    # ------------------------------- adj_y ------------------------------------

    def _adjoint_y(self) -> Optional[Callable[[Array], Array]]:
        """
        y-space adjoint with robust Jacobian pullback policy per BridgeConfig.
        """
        if self.adj_x is None:
            return None

        policy = self.cfg.jac_pullback.lower()
        W = self._W
        N = self.N
        ax = self.adj_x
        cond_cap = float(self.cfg.jac_cond_cap)
        use_clip = bool(self.cfg.jac_use_svd_clip)
        sig_lo = float(self.cfg.jac_sigma_lo)
        sig_hi = float(self.cfg.jac_sigma_hi)

        def pull_diag(gx: Array) -> Array:
            return (W * gx).reshape(-1)

        def pull_full(y: Array, gx: Array) -> Array:
            J = self._jac_y_to_x_dynamic(y)
            if use_clip:
                # SVD clip singular values to [sig_lo, sig_hi] for stability
                U, S, Vt = np.linalg.svd(J, full_matrices=False)
                S_clipped = np.clip(S, sig_lo, sig_hi)
                J = (U * S_clipped) @ Vt
            return (J.T @ gx).reshape(-1)

        def _adj_y(y: Array) -> Array:
            y = np.asarray(y, float).reshape(-1)
            x = N.cap_decode(y, input='y')
            gx = np.asarray(ax(x), float).reshape(-1)
            if gx.size != y.size:
                raise ValueError("adjoint_x returned wrong dimension")

            if policy == "diag":
                return pull_diag(gx)

            # Compute condition estimate & boundary proximity when needed
            if self._J_const is not None:
                condJ = self._J_const_cond if (self._J_const_cond is not None) else float(np.linalg.cond(self._J_const))
                safe = np.isfinite(condJ) and (condJ <= cond_cap)
            else:
                # dynamic: cheap cond via ratio of norms (proxy), fall back to np.linalg.cond
                J = self._jac_y_to_x_dynamic(y)
                try:
                    condJ = float(np.linalg.cond(J))
                except Exception:
                    condJ = np.inf
                # squash boundary check
                if self._has_squash:
                    yb = np.clip(y, self.cfg.jac_squash_eps, 1.0 - self.cfg.jac_squash_eps)
                    # if any component was clipped strongly, mark unsafe
                    near_edge = np.any((y <= self.cfg.jac_squash_eps) | (y >= 1.0 - self.cfg.jac_squash_eps))
                else:
                    near_edge = False
                safe = (not near_edge) and np.isfinite(condJ) and (condJ <= cond_cap)

            if policy == "full":
                # Always try full; if unsafe, apply SVD clip (if enabled), else fall back diag
                if safe:
                    return pull_full(y, gx)
                return pull_full(y, gx) if use_clip else pull_diag(gx)

            # 'hybrid' (default)
            if safe:
                return pull_full(y, gx)
            else:
                # unsafe → diag fallback (or clipped full if explicit)
                return pull_full(y, gx) if use_clip else pull_diag(gx)

        return _adj_y

    # -------------------------------- step ------------------------------------

    def step(self, y: Array, logger: Optional[Callable[[Dict], None]] = None
             ) -> Tuple[Array, float, Dict]:
        """
        Run one bridged step in y-space.

        Returns
        -------
        y_new : (d,) next start (or unchanged if rejected)
        f_new : float objective at y_new
        info  : dict from stepper, augmented with {"A_info","bridge"}
        """
        y = np.asarray(y, float).reshape(-1)
        if y.size != self.N.dim:
            raise ValueError(f"y dimension {y.size} != normalizer.dim {self.N.dim}")

        # (1) Compose A from accepted history
        if self.cfg.use_precond:
            A, A_info = self.precond.compute_A()
        else:
            A, A_info = None, {"disabled": True, "Anorm": 0.0, "n_steps": 0, "n_secants": 0}

        # (2) Gradient source
        grad_source = self.g1p if not self._did_bootstrap else self.g2p

        # (3) y-space adjoint (per policy)
        adj_y = self._adjoint_y()

        # (4) One step (Stepper owns policy/acceptance/calibration)
        y_new, f_new, self._state, info = self.stepper.step(
            f_y=self._f_y, y=y, A=A, adam_state=self._state,
            grad_source=grad_source, adjoint_fn=adj_y, logger=logger, minimize=True,
        )

        # (5) finalize previous pending secant using g @ current start
        if self._pending is not None:
            g_new_prev = np.asarray(info["secant"]["g_old"], float).reshape(-1)
            y_old_prev = self._pending["y_old"]
            y_new_prev = self._pending["y_new"]
            g_old_prev = self._pending["g_old"]
            if (g_new_prev.size != g_old_prev.size) or (y_old_prev.size != y_new_prev.size):
                raise ValueError("secant shapes mismatch while finalizing")
            self.precond.push_secant(y_old_prev, y_new_prev, g_old_prev, g_new_prev)
            self._pending = None

        # (6) on acceptance, update whitening & open new pending secant
        sec = info.get("secant", {})
        if bool(sec.get("ready", False)):
            y_old = np.asarray(sec["y_old"], float).reshape(-1)
            y_acc = np.asarray(sec["y_new"], float).reshape(-1)
            g_old = np.asarray(sec["g_old"], float).reshape(-1)
            self.precond.push_step_only(y_old, y_acc)
            self._pending = {"y_old": y_old, "y_new": y_acc, "g_old": g_old}

        # (7) one-time bootstrap flip
        if not self._did_bootstrap:
            self._did_bootstrap = True

        # (8) augment info
        info_out = dict(info)
        info_out["A_info"] = A_info
        info_out["bridge"] = {
            "used_precond": bool(self.cfg.use_precond),
            "grad_scheme": grad_source.scheme,
            "bootstrap_used": (grad_source is self.g1p),
            "jac_pullback": self.cfg.jac_pullback,
        }
        return y_new, float(f_new), info_out
