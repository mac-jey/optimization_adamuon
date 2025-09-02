# preconditioner.py — Whitening + Diagonal-Scaling Preconditioner
# A = P · W with spectral cap  ||A||_2 ≤ tau_cap.
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class PreconditionerConfig:
    max_steps: int = 64            # window size for Δy covariance (whitening)
    max_secants: int = 64          # window size for delayed secants (P)
    eps: float = 1e-10             # jitter added to Σ eigenvalues
    g_floor: float = 1e-12         # floor on |Δg_i| to avoid blow-ups
    p_min: float = 1e-3            # lower cap for diagonal scaling
    p_max: float = 1e+3            # upper cap for diagonal scaling
    tau_cap: float = 10.0          # spectral norm cap for A
    use_power_iter: bool = True    # prefer power iteration for ||A||_2 on large d
    power_iter_steps: int = 20
    svd_on_small: bool = True      # use SVD directly when d is small
    svd_small_d: int = 80

__all__ = ["Preconditioner", "PreconditionerConfig"]

class Preconditioner:
    """Whitening + robust diagonal secant preconditioner (A = P·W).

    State:
      - steps_dy : accepted Δy window for covariance → W = (Σ + eps I)^(-1/2)
      - sec_dy, sec_dg : delayed secants for diagonal scaling → P = diag(median |Δy|/max(|Δg|, g_floor))

    Invariants:
      - P_diag ∈ [p_min, p_max]
      - Spectral cap: ||A||_2 ≤ tau_cap (uniform scale-down if needed)
    """

    def __init__(
        self,
        dim: int,
        periodic_mask: Optional[Array] = None,
        config: Optional[PreconditionerConfig] = None,
    ) -> None:
        self._d = int(dim)
        if self._d <= 0:
            raise ValueError("dim must be positive")
        if periodic_mask is not None:
            pm = np.asarray(periodic_mask, dtype=bool).reshape(-1)
            if pm.shape != (self._d,):
                raise ValueError(f"periodic_mask shape {pm.shape} != (dim,)={self._d}")
            self._periodic = pm
        else:
            self._periodic = None
        self._cfg = config or PreconditionerConfig()

        # Rolling windows (accepted only)
        self._steps_dy: deque[Array] = deque(maxlen=self._cfg.max_steps)
        self._sec_dy:   deque[Array] = deque(maxlen=self._cfg.max_secants)
        self._sec_dg:   deque[Array] = deque(maxlen=self._cfg.max_secants)

    # --------------------------- helpers ----------------------------------
    def _delta(self, y_new: Array, y_old: Array) -> Array:
        """Compute Δy with optional torus wrapping on periodic axes.

        Inputs
        ------
        y_new, y_old : shape (d,)

        Returns
        -------
        dy : shape (d,)
        """
        dy = np.asarray(y_new, float).reshape(-1) - np.asarray(y_old, float).reshape(-1)
        if dy.shape != (self._d,):
            raise ValueError("y shapes incompatible with dim")
        if self._periodic is not None and self._periodic.any():
            # wrap only on periodic axes to [-0.5, 0.5)
            w = dy[self._periodic]
            w = (w + 0.5) % 1.0 - 0.5
            dy = dy.copy()
            dy[self._periodic] = w
        return dy

    @staticmethod
    def _median_ratio(abs_dy: Array, abs_dg: Array, floor: float) -> Array:
        """Component-wise median of |Δy|/max(|Δg|, floor) over secant window.
        abs_dy, abs_dg : shape (n, d)  → returns shape (d,)
        """
        denom = np.maximum(abs_dg, floor)
        ratios = abs_dy / denom
        return np.median(ratios, axis=0)

    @staticmethod
    def _sym_inv_sqrt(Sigma: Array, eps: float) -> Tuple[Array, Array, Array]:
        """Return W=(Sigma+eps I)^(-1/2) via symmetric eigendecomposition (PSD-stable)."""
        # Sigma is symmetric PSD; eigh is numerically appropriate and efficient.
        s, U = np.linalg.eigh(Sigma)  # s ascending, s>=0 ideally
        lam = s + eps                 # jitter for numerical stability
        W = (U * (lam ** -0.5)) @ U.T
        return W, U, lam

    @staticmethod
    def _op_norm_2(A: Array, use_power: bool, iters: int) -> float:
        """Spectral norm ||A||_2 via power iteration without explicitly forming A^T A.
        Deterministic init; early-stop when Rayleigh quotient (on A^T A) converges.
        """
        if not use_power:
            return float(np.linalg.svd(A, compute_uv=False)[0])
        d = A.shape[1]
        v = np.ones(d, dtype=float) / np.sqrt(d)
        ray_prev = 0.0
        for _ in range(iters):
            Av = A @ v
            w  = A.T @ Av
            nrm = np.linalg.norm(w)
            if nrm == 0.0:
                return 0.0
            v = w / nrm
            # Rayleigh quotient for A^T A at current v
            ray = float(v @ (A.T @ (A @ v)))
            if abs(ray - ray_prev) <= 1e-12 * max(1.0, ray):
                return float(np.sqrt(ray if ray > 0.0 else 0.0))
            ray_prev = ray
        return float(np.sqrt(ray_prev if ray_prev > 0.0 else 0.0))

    # ----------------------------- API ------------------------------------
    def reset(self) -> None:
        """Clear internal windows and statistics."""
        self._steps_dy.clear()
        self._sec_dy.clear()
        self._sec_dg.clear()

    def push_step_only(self, y_old: Array, y_new: Array) -> None:
        """Record an **accepted** move for whitening (prev_start → curr_start).

        Inputs
        ------
        y_old, y_new : shape (d,)
        """
        dy = self._delta(y_new, y_old)
        self._steps_dy.append(dy)

    def push_secant(self, y_old: Array, y_new: Array, g_old: Array, g_new: Array) -> None:
        """Record a **delayed secant** (prev_start → curr_start) for P.

        Inputs
        ------
        y_old, y_new : shape (d,)  (start points)
        g_old, g_new : shape (d,)  (grad at those start points)
        """
        dy = self._delta(y_new, y_old)
        dg = np.asarray(g_new, float).reshape(-1) - np.asarray(g_old, float).reshape(-1)
        if dy.shape != (self._d,) or dg.shape != (self._d,):
            raise ValueError("secant inputs have wrong shapes")
        self._sec_dy.append(np.abs(dy))
        self._sec_dg.append(np.abs(dg))

    def compute_A(self) -> Tuple[Array, Dict]:
        """Compute A = P·W, apply spectral cap, and return (A, info).

        Returns
        -------
        A : (d,d) ndarray
        info : dict with keys
            - "Anorm": float, spectral norm after capping
            - "A_scale": float, scaling applied due to cap (≤1)
            - "P_diag": (d,) ndarray, diagonal of P
            - "cond_Sigma": float, cond number of (Σ+eps I) via eigenvalues
            - "whiten_mse": float, ||W Σ W^T - I||_F / sqrt(d)
            - "n_steps": int, samples in whitening window
            - "n_secants": int, samples in secant window
        """
        d, cfg = self._d, self._cfg

        # Whitening W
        if len(self._steps_dy) >= 2:
            Y = np.vstack(self._steps_dy)  # (n, d)
            Y0 = Y - Y.mean(axis=0, keepdims=True)
            denom = max(1, (Y0.shape[0] - 1))
            Sigma = (Y0.T @ Y0) / denom
            W, U, lam = self._sym_inv_sqrt(Sigma, cfg.eps)
            cond_Sigma = float(np.max(lam) / np.min(lam))
            I_approx = W @ Sigma @ W.T
            whiten_mse = float(np.linalg.norm(I_approx - np.eye(d), ord='fro') / np.sqrt(d))
        else:
            W = np.eye(d, dtype=float)
            cond_Sigma = 1.0
            whiten_mse = 0.0

        # Diagonal P from delayed secants
        if len(self._sec_dy) >= 1:
            DY = np.vstack(self._sec_dy)  # (n, d)
            DG = np.vstack(self._sec_dg)  # (n, d)
            p_diag = self._median_ratio(DY, DG, cfg.g_floor)
            p_diag = np.clip(p_diag, cfg.p_min, cfg.p_max)
        else:
            p_diag = np.ones(d, dtype=float)
        P = np.diag(p_diag)

        # Compose & spectral cap
        A = P @ W
        use_svd = (not cfg.use_power_iter) or (cfg.svd_on_small and d <= cfg.svd_small_d)
        Anorm = float(np.linalg.svd(A, compute_uv=False)[0]) if use_svd \
            else self._op_norm_2(A, True, cfg.power_iter_steps)
        scale = 1.0
        if Anorm > cfg.tau_cap and Anorm > 0.0:
            scale = cfg.tau_cap / Anorm
            A = A * scale
            Anorm = Anorm * scale

        info = {
            "Anorm": Anorm,
            "A_scale": scale,
            "P_diag": p_diag.copy(),
            "cond_Sigma": cond_Sigma,
            "whiten_mse": whiten_mse,
            "n_steps": len(self._steps_dy),
            "n_secants": len(self._sec_dy),
        }
        return A, info
