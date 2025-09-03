# Gradient.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict

Array = np.ndarray

# ============================== errors =======================================

class GradientEvalError(RuntimeError):
    """Raised when f_y returns a non-finite value at a probe point."""
    pass

# ------------------------------- helpers ------------------------------------

def _as_row_vec(y: Array) -> Array:
    """Ensure (d,) 1-D vector view for y (no copy if possible)."""
    y = np.asarray(y, float)
    if y.ndim != 1:
        raise ValueError("y must be shape (d,)")
    return y

def _check_A(A: Optional[Array], d: int) -> Optional[Array]:
    """Validate external direction matrix A with shape (d,K)."""
    if A is None:
        return None
    A = np.asarray(A, float)
    if A.ndim != 2:
        raise ValueError("A must be 2D with shape (d, K)")
    if A.shape[0] != d:
        raise ValueError(f"A.shape[0] must equal d={d}, got {A.shape[0]}")
    if A.shape[1] < 1:
        raise ValueError("A must have at least one direction (K>=1)")
    return A

def _norm_cols(U: Array, eps: float = 1e-12) -> Array:
    """Normalize columns to unit L2; raise if a column is ~zero."""
    U = np.asarray(U, float)
    nrm = np.linalg.norm(U, axis=0)
    if np.any(nrm <= eps):
        bad = np.where(nrm <= eps)[0].tolist()
        raise ValueError(f"Direction columns with ~zero norm at indices {bad}")
    return U / nrm

def _gram_schmidt(u: Array, basis: Array, tol: float = 1e-12) -> Optional[Array]:
    """
    Project u against columns of 'basis' and return a normalized residual.
    If residual is too small, returns None.
    """
    v = u.astype(float, copy=True)
    if basis.size:
        proj = basis @ (basis.T @ v)
        v -= proj
    n = np.linalg.norm(v)
    if n <= tol:
        return None
    return v / n

def _gen_directions(d: int, K: int, rng: np.random.Generator, antithetic: bool) -> Array:
    """
    Generate K directions with good conditioning.
    - If antithetic: build ~K//2 base directions and mirror them.
    - If K <= d: use QR to obtain orthonormal columns.
    - If K > d: use QR for a d-column orthonormal basis, then append
      Gram–Schmidt re-orthogonalized near-orthogonal columns.
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    if antithetic and K >= 2:
        half = K // 2
        # 1) QR-based basis
        qcols = max(1, min(half, d))
        G = rng.standard_normal((d, qcols))
        Q, _ = np.linalg.qr(G, mode='reduced')  # d x q, q<=min(d,half)
        basis = Q.copy()
        # 2) Fill up to 'half' with Gram–Schmidt re-orthogonalized vectors
        while basis.shape[1] < half:
            v = rng.standard_normal(d)
            v = v / np.linalg.norm(v)
            v_ortho = _gram_schmidt(v, basis)
            if v_ortho is None:
                continue  # retry if too close to the span
            basis = np.concatenate([basis, v_ortho.reshape(d, 1)], axis=1)
        U_base = basis[:, :half]                        # d x half
        U = np.concatenate([U_base, -U_base], axis=1)   # ± pairs
        if U.shape[1] < K:
            # odd K: add the remaining one column via re-orthogonalization
            while U.shape[1] < K:
                v = rng.standard_normal(d)
                v = v / np.linalg.norm(v)
                v_ortho = _gram_schmidt(v, U)
                if v_ortho is None:
                    continue
                U = np.concatenate([U, v_ortho.reshape(d, 1)], axis=1)
        return U[:, :K]

    # Non-antithetic branch
    if K <= d:
        G = rng.standard_normal((d, K))
        Q, _ = np.linalg.qr(G, mode='reduced')  # d x K
        return Q

    # K > d (d orthonormal columns + additional near-orthogonal via Gram–Schmidt)
    G = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(G, mode='reduced')      # d x d
    U = Q.copy()
    while U.shape[1] < K:
        v = rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        v_ortho = _gram_schmidt(v, U)
        if v_ortho is None:
            continue
        U = np.concatenate([U, v_ortho.reshape(d, 1)], axis=1)
    return U[:, :K]

def _ls_solve(U: Array, s: Array) -> Tuple[Array, float, float]:
    """
    Solve U^T g ≈ s for g in least-squares sense.
    Returns (g, resid_norm, cond_est).
    """
    A_lsq = U.T
    g, resid, _, S = np.linalg.lstsq(A_lsq, s, rcond=None)
    resid_norm = float(np.sqrt(resid[0])) if resid.size > 0 else 0.0
    if S.size > 0 and S.min() > 0:
        cond = float(S.max() / S.min())
    else:
        cond = float('inf')
    return g, resid_norm, cond

def _compute_eps(eps_rel: float, eps_abs: float, A: Optional[Array]) -> float:
    """
    Scalar step size. Columns are normalized internally; still adapt to ||A||
    if user passed large columns by shrinking eps when norms are large.
    """
    scale = 1.0
    if A is not None:
        scale = max(1.0, float(np.linalg.norm(A, axis=0).max()))
    return max(float(eps_abs), float(eps_rel) / scale)

def _u_quality_metrics(U: Array) -> Dict[str, float | int | Tuple[int, int]]:
    """
    Compute basic quality metrics for direction matrix U (d x K):
    - U_norm_min/max: min/max column norms (should be ~1 after normalization)
    - U_coherence_max: max |u_i·u_j|, i != j
    - U_rank: numeric rank of U
    - U_shape: (d, K)
    """
    U = np.asarray(U, float)
    d, K = U.shape
    nrm = np.linalg.norm(U, axis=0)
    gram = U.T @ U                     # K x K
    gram_off = gram - np.diag(np.diag(gram))  # pairwise only
    coh = 0.0 if K == 1 else float(np.max(np.abs(gram_off)))
    rank = int(np.linalg.matrix_rank(U))
    return {
        "U_shape": (d, K),
        "U_norm_min": float(nrm.min()),
        "U_norm_max": float(nrm.max()),
        "U_coherence_max": coh,
        "U_rank": rank,
    }

# ----------------------------- base interface -------------------------------

class BaseGradientSource:
    """
    Base class: 'grad' returns (g, info, fevals).

    Contract
    --------
    - f_y : callable(y: (d,)) -> float
    - y   : (d,)
    - A   : (d,K) or None. If None, directions are generated with rng.
    - rng : numpy.random.Generator (used only when A is None)

    Must NOT do line search / step policy / boundary wrapping.
    """
    scheme: str = "base"

    def __init__(self, K: int = 0, eps_rel: float = 1e-4, eps_abs: float = 0.0, antithetic: bool = True):
        self.K = int(K) if K is not None else 0
        self.eps_rel = float(eps_rel)
        self.eps_abs = float(eps_abs)
        self.antithetic = bool(antithetic)

    def grad(self, f_y, y: Array, A: Optional[Array], rng: Optional[np.random.Generator]) -> tuple[Array, dict, int]:
        raise NotImplementedError

# ------------------------------- Forward 1P ----------------------------------

class Forward1P(BaseGradientSource):
    """
    One-sided multi-direction gradient via directional slopes:
        s_k ≈ ( f(y + ε u_k) - f(y) ) / ε
        solve U^T g ≈ s  (LS)

    Defaults
    --------
    - eps_rel: 1e-3  (good starting point for smooth f on y∈[0,1))
    - fevals : 1 + K (planned)
    - antithetic: True (encouraged; set K even to include ± pairs)
    """
    scheme = "forward-1p"

    def __init__(self, K: int = 2, eps_rel: float = 1e-3, eps_abs: float = 0.0, antithetic: bool = True):
        super().__init__(K=K, eps_rel=eps_rel, eps_abs=eps_abs, antithetic=antithetic)

    def grad(self, f_y, y: Array, A: Optional[Array] = None, rng: Optional[np.random.Generator] = None):
        y = _as_row_vec(y)
        d = y.size

        # Directions
        if A is None:
            if rng is None:
                rng = np.random.default_rng()
            U = _gen_directions(d, self.K, rng, self.antithetic)
            U = _norm_cols(U)  # safety
            u_src = "random"
        else:
            A = _check_A(A, d)
            U = _norm_cols(A)
            u_src = "provided"

        K = U.shape[1]
        eps = _compute_eps(self.eps_rel, self.eps_abs, A)

        # Base evaluation
        f0 = float(f_y(y))
        if not np.isfinite(f0):
            raise GradientEvalError(f"Non-finite f(y) at base point: f0={f0}")

        s = np.empty(K, float)
        fevals = 1

        for k in range(K):
            yk = y + eps * U[:, k]
            fk = float(f_y(yk))
            if not np.isfinite(fk):
                raise GradientEvalError(
                    f"Non-finite f(y+eps*u[{k}]) at k={k}, eps={eps:.3e}, ||u_k||={np.linalg.norm(U[:,k]):.3e}"
                )
            s[k] = (fk - f0) / eps
            fevals += 1

        g, resid, cond = _ls_solve(U, s)
        info = {
            "scheme": self.scheme,
            "K": K,
            "eps": eps,
            "eps_strategy": {"rel": self.eps_rel, "abs": self.eps_abs, "used": eps},
            "ls_resid": resid,
            "cond": cond,
            "fevals_planned": 1 + K,
            "U_source": u_src,
            **_u_quality_metrics(U),
        }
        return g, info, fevals

# ------------------------------- Central 2P ----------------------------------

class Central2P(BaseGradientSource):
    """
    Two-sided multi-direction gradient via symmetric differences:
        s_k ≈ ( f(y + ε u_k) - f(y - ε u_k) ) / (2ε)
        solve U^T g ≈ s  (LS)

    Defaults
    --------
    - eps_rel: 1e-4  (central is lower bias; can use smaller ε)
    - fevals : 2K (planned)
    - antithetic arg is ignored (2P already uses ± displacements)
    """
    scheme = "central-2p"

    def __init__(self, K: int = 2, eps_rel: float = 1e-4, eps_abs: float = 0.0):
        super().__init__(K=K, eps_rel=eps_rel, eps_abs=eps_abs, antithetic=False)

    def grad(self, f_y, y: Array, A: Optional[Array] = None, rng: Optional[np.random.Generator] = None):
        y = _as_row_vec(y)
        d = y.size

        # Directions
        if A is None:
            if rng is None:
                rng = np.random.default_rng()
            U = _gen_directions(d, self.K, rng, antithetic=False)  # symmetric in evaluation
            U = _norm_cols(U)
            u_src = "random"
        else:
            A = _check_A(A, d)
            U = _norm_cols(A)
            u_src = "provided"

        K = U.shape[1]
        eps = _compute_eps(self.eps_rel, self.eps_abs, A)

        s = np.empty(K, float)
        fevals = 0

        for k in range(K):
            y_plus  = y + eps * U[:, k]
            y_minus = y - eps * U[:, k]
            f_plus  = float(f_y(y_plus))
            f_minus = float(f_y(y_minus))
            if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
                raise GradientEvalError(
                    f"Non-finite f at central pair k={k}, eps={eps:.3e}, ||u_k||={np.linalg.norm(U[:,k]):.3e}, "
                    f"f+={f_plus}, f-={f_minus}"
                )
            s[k] = (f_plus - f_minus) / (2.0 * eps)
            fevals += 2

        g, resid, cond = _ls_solve(U, s)
        info = {
            "scheme": self.scheme,
            "K": K,
            "eps": eps,
            "eps_strategy": {"rel": self.eps_rel, "abs": self.eps_abs, "used": eps},
            "ls_resid": resid,
            "cond": cond,
            "fevals_planned": 2 * K,
            "U_source": u_src,
            **_u_quality_metrics(U),
        }
        return g, info, fevals

# ------------------------------ Adjoint Blend -------------------------------

class AdjointBlend:
    """
    Pure measurement helper (no policy).
    Cosine is computed in the SAME coordinate space as g_2p (y-space).

    Parameters (calibrate)
    ----------------------
    y : (d,)
        Current normalized coordinate (y-space).
    g_2p : (d,)
        Reference gradient in y-space (e.g., Central2P result).
    adjoint : array-like | callable
        - If array-like: gradient vector in either y- or x-space.
        - If callable: adjoint(y) if grad_space='y', or adjoint(x) if grad_space='x'.
    grad_space : {'y','x'}
        The space where 'adjoint' lives. If 'x', 'normalizer' is required.
    normalizer : object or None
        Must provide jac_y_to_x() returning J=∂x/∂y when grad_space='x'.
    sign : {+1, -1}
        Use -1 if your adjoint returns gradient of a *maximization* F but you
        compare against ∂(-F)/∂y for minimization.

    Returns
    -------
    dict: {"adj_used": False, "cos", "norm_2p", "norm_adj_y", "dot", "space"}
    """
    def calibrate(self,
                  y: np.ndarray,
                  g_2p: np.ndarray,
                  adjoint,
                  *,
                  grad_space: str = "y",
                  normalizer=None,
                  sign: int = +1) -> dict:
        y = np.asarray(y, float).reshape(-1)
        g_2p = np.asarray(g_2p, float).reshape(-1)
        if g_2p.shape != y.shape:
            raise ValueError("g_2p shape must match y")

        # Obtain adjoint gradient vector (mapped to y-space if needed)
        if callable(adjoint):
            if grad_space == "y":
                g_adj = np.asarray(adjoint(y), float).reshape(-1)
            elif grad_space == "x":
                if normalizer is None:
                    raise ValueError("normalizer is required when grad_space='x'")
                x = normalizer.cap_decode(y, input='y')
                g_adj_x = np.asarray(adjoint(x), float).reshape(-1)
                try:
                    J = normalizer.jac_y_to_x()  # ∂x/∂y
                except Exception as e:
                    raise ValueError(
                        "Failed to obtain Jacobian ∂x/∂y from normalizer. "
                        "If any axis uses mode='squash', jac_y_to_x is undefined."
                    ) from e
                g_adj = (J.T @ g_adj_x).reshape(-1)
            else:
                raise ValueError("grad_space must be 'y' or 'x'")
        else:
            g_adj = np.asarray(adjoint, float).reshape(-1)
            if grad_space == "x":
                if normalizer is None:
                    raise ValueError("normalizer is required when grad_space='x'")
                try:
                    J = normalizer.jac_y_to_x()
                except Exception as e:
                    raise ValueError(
                        "Failed to obtain Jacobian ∂x/∂y from normalizer. "
                        "If any axis uses mode='squash', jac_y_to_x is undefined."
                    ) from e
                g_adj = (J.T @ g_adj).reshape(-1)

        if g_adj.shape != y.shape:
            raise ValueError("adjoint gradient dimensionality mismatch")

        # Optional sign fix (max vs min conventions)
        g_adj *= float(sign)

        # Cosine in y-space
        n1 = float(np.linalg.norm(g_2p))
        n2 = float(np.linalg.norm(g_adj))
        if n1 == 0.0 or n2 == 0.0:
            cos = 0.0; dot = 0.0
        else:
            dot = float(np.dot(g_2p, g_adj))
            cos = float(np.clip(dot / (n1 * n2), -1.0, 1.0))

        return {
            "adj_used": False,
            "cos": cos,
            "dot": dot,
            "norm_2p": n1,
            "norm_adj_y": n2,
            "space": "y",  # comparison space
        }
