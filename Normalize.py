# Normalize.py
# -----------------------------------------------------------------------------
# Boundary-aware Normalizer with unified CAP (Complex Amplitude–Phase) interface
# -----------------------------------------------------------------------------
# Overview
#   - Maps real-space variables x ∈ ℝ^d to internal [0,1)^d coordinates y with
#     per-axis boundary "modes":
#       'wrap'    : periodic → y = ((x-L)/W) mod 1
#       'clip'    : finite bounds → y ∈ [0,1) half-open (1.0→nextafter(1,0))
#       'reflect' : finite bounds → triangular reflection in [0,1]
#       'squash'  : unbounded → y = σ((x-c)/s), inverse via logit (c=center, s>0)
#   - CAP embedding (optional):
#       amplitude from aperiodic axis, phase from periodic axis → z = r·e^{iφ}
#     Unpaired periodic axes → unit circle; unpaired aperiodic axes → radius-only.
#   - Factory:
#       N = boundary(bounds=[(L0,U0), (L1,U1), ...], mode=..., period=..., M=...)
#     Default mode is 'wrap' (periodic) for all axes.
# Public API
#   - N.cap_encode(x, out='complex'|'reim'|'y')   # default: 'complex'
#   - N.cap_decode(z,  input='complex'|'reim'|'y')
#   - N.cap_channel_count(pairs)
#   - boundary(bounds, ...)  → Normalizer  (module-level ergonomic constructor)
# Invariants
#   - decode(encode(x)) ≈ x for interior points (machine precision).
#   - All wrap/clip/reflect/squash is performed only inside this class.
#   - Mixing matrix M (if provided) applies only to affine axes
#     ('wrap'/'clip'/'reflect'); disallowed with any 'squash' axis.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple, Literal

Array = np.ndarray


# ------------------------ small utility ------------------------
def _as_row(a: Array | Sequence[float]) -> Array:
    """Ensure shape (n,d) from (d,) or (n,d)."""
    A = np.asarray(a, float)
    if A.ndim == 1:
        return A[None, :]
    if A.ndim == 2:
        return A
    raise ValueError("input must be (d,) or (n,d)")


# ------------------------ core class ---------------------------
class Normalizer:
    """
    Boundary-aware normalizer with unified CAP (Complex Amplitude–Phase) interface.

    Notes
    -----
    - Affine axes (wrap/clip/reflect) can be mixed by M (invertible d×d).
    - If any axis uses 'squash', M is not allowed (to avoid non-affine coupling).
    - Half-open handling: y==1.0 is clamped to nextafter(1.0, 0.0).
    """

    def __init__(self, bounds: Sequence[Sequence[float]], periodic_mask: Sequence[bool],
                 M: Optional[Array] = None):
        # Basic bounds & masks
        b = np.asarray(bounds, float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be shape (d,2)")
        L = b[:, 0]
        U = b[:, 1]
        if not np.all(U > L):
            bad = np.where(~(U > L))[0]
            raise ValueError(f"U>L violated at dims {bad.tolist()}")
        self._L: Array = L.copy()
        self._W: Array = (U - L).astype(float)
        self._Winv: Array = 1.0 / self._W
        self._pmask: Array = np.asarray(periodic_mask, bool)
        if self._pmask.size != self._W.size:
            raise ValueError("periodic_mask length must equal dim")

        d = self.dim
        # Mixing matrix (affine axes only)
        if M is None:
            self._M = np.eye(d)
        else:
            M = np.asarray(M, float)
            if M.shape != (d, d):
                raise ValueError(f"M must be ({d},{d})")
            # Invertibility check with a clearer message
            try:
                self._Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                raise ValueError("M must be invertible (singular matrix).")
            self._M = M
        # If we get here without exception and M not provided, compute inverse now.
        if not hasattr(self, "_Minv"):
            self._Minv = np.linalg.inv(self._M)

        self._D_Winv = np.diag(self._Winv)
        self._D_W = np.diag(self._W)

        # Half-open upper bound 1-ulp
        self._hi = np.nextafter(1.0, 0.0)

        # Set by factory
        self._mode = np.array(['wrap'] * d, dtype=object)
        self._period = np.array([None] * d, dtype=object)
        self._squash_c = np.zeros(d)
        self._squash_s = np.ones(d)
        self._has_squash: bool = False

    # ------------------ FACTORY (tuple-based) -------------------
    @classmethod
    def boundary_tuples(
        cls,
        bounds: Sequence[Tuple[float, float]],
        *,
        mode: Optional[Sequence[str]] = None,       # 'wrap' | 'clip' | 'reflect' | 'squash'
        period: Optional[Sequence[Optional[float]]] = None,
        M: Optional[Array] = None,
        squash_center: Optional[Sequence[float]] = None,
        squash_scale: Optional[Sequence[float]] = None
    ) -> "Normalizer":
        """
        Create a normalizer from per-dimension (low, high) tuples.

        Parameters
        ----------
        bounds : sequence of (L, U)
            If mode='wrap' and (L,U) not finite, provide positive 'period'.
        mode : sequence[str] or None
            Defaults to 'wrap' for all axes.
        period : sequence[float|None] or None
            Used only for 'wrap' when (L,U) are not both finite. Must be >0.
        M : ndarray or None
            Mixing matrix for affine axes. Disallowed if any axis is 'squash'.
        squash_center : sequence[float] or None
            Per-axis centers c_j for 'squash' axes (ignored elsewhere).
        squash_scale : sequence[float] or None
            Per-axis positive scales s_j (>0) for 'squash' axes (ignored elsewhere).
        """
        b = np.asarray(bounds, float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be shape (d,2) as list of (L,U)")
        L, U = b[:, 0], b[:, 1]
        d = L.size

        mode_arr = np.array(mode if mode is not None else ['wrap'] * d, dtype=object)
        period_arr = np.array(period if period is not None else [None] * d, dtype=object)

        pmask = np.zeros(d, bool)
        L_out = np.empty(d, float)
        U_out = np.empty(d, float)
        any_squash = False

        for j in range(d):
            m = mode_arr[j]
            if m not in ('wrap', 'clip', 'reflect', 'squash'):
                raise ValueError(f"mode[{j}] must be one of wrap|clip|reflect|squash")
            if m == 'wrap':
                if np.isfinite(L[j]) and np.isfinite(U[j]) and (U[j] > L[j]):
                    L_out[j], U_out[j] = L[j], U[j]
                else:
                    P = period_arr[j]
                    if not (P is not None and np.isfinite(P) and P > 0):
                        raise ValueError(f"wrap axis {j} needs finite (L,U) with U>L or positive period[{j}]")
                    L_out[j], U_out[j] = 0.0, float(P)
                pmask[j] = True
            elif m in ('clip', 'reflect'):
                if not (np.isfinite(L[j]) and np.isfinite(U[j]) and (U[j] > L[j])):
                    raise ValueError(f"{m} axis {j} requires finite bounds with U>L")
                L_out[j], U_out[j] = L[j], U[j]
                pmask[j] = False
            elif m == 'squash':
                any_squash = True
                # For squash, the internal y is (0,1), but we keep [0,1) protocol.
                L_out[j], U_out[j] = 0.0, 1.0
                pmask[j] = False

        if any_squash and M is not None:
            raise ValueError("Mixing matrix M is not allowed when any axis uses mode='squash'.")

        N = cls(bounds=np.stack([L_out, U_out], 1), periodic_mask=pmask.tolist(), M=M)
        N._mode = mode_arr
        N._period = period_arr
        N._has_squash = bool(any_squash)

        # Initialize squash center/scale
        if any_squash:
            c = np.zeros(d)
            s = np.ones(d)
            if squash_center is not None:
                c_in = np.asarray(squash_center, float).reshape(-1)
                if c_in.size not in (1, d):
                    raise ValueError("squash_center must be scalar-like or length d")
                c[:] = c_in[0] if c_in.size == 1 else c_in
            if squash_scale is not None:
                s_in = np.asarray(squash_scale, float).reshape(-1)
                if s_in.size not in (1, d):
                    raise ValueError("squash_scale must be scalar-like or length d")
                s[:] = s_in[0] if s_in.size == 1 else s_in
            if not np.all(s > 0):
                raise ValueError("All squash_scale entries must be > 0")
            N._squash_c = c
            N._squash_s = s

        return N

    # ---------------- properties & getters ----------------
    @property
    def dim(self) -> int:
        return self._W.size

    @property
    def periodic_mask(self) -> np.ndarray:
        """Boolean mask of periodic axes (True=periodic)."""
        return self._pmask.copy()

    @property
    def mode(self) -> Tuple[str, ...]:
        """Tuple of per-axis boundary modes."""
        return tuple(self._mode.tolist())

    @property
    def period(self) -> Tuple[Optional[float], ...]:
        """Tuple of per-axis periods (for 'wrap' axes; None otherwise)."""
        return tuple(self._period.tolist())

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """(L, U) arrays actually used internally (after mode resolution)."""
        return self._L.copy(), (self._L + self._W).copy()

    # ---------------- internal helpers (mode application) ----------------
    def _apply_boundary_encode(self, Y: Array) -> Array:
        """
        Apply wrap/clip/reflect so each component lies in [0,1) (half-open).
        'squash' axes are already in (0,1) from the logistic map.
        """
        for j in range(self.dim):
            m = self._mode[j]
            if m == 'wrap':
                Y[:, j] = np.remainder(Y[:, j], 1.0)
            elif m == 'clip':
                Y[:, j] = np.minimum(np.maximum(Y[:, j], 0.0), self._hi)
            elif m == 'reflect':
                # Triangular wave with period 2
                r = np.remainder(Y[:, j], 2.0)
                Y[:, j] = np.minimum(1.0 - np.abs(r - 1.0), self._hi)
            elif m == 'squash':
                # Already in (0,1); keep half-open safety on upper end
                Y[:, j] = np.minimum(np.maximum(Y[:, j], 0.0), self._hi)
        return Y

    # ---------------- y normalization path (with optional mixing) ----------------
    def _encode_y(self, x: Array | Sequence[float]) -> Array:
        """x → y normalization, honoring modes and (if applicable) mixing."""
        xa = np.asarray(x, float)
        is_row = xa.ndim == 1
        X = _as_row(xa)
        n, d = X.shape
        U = np.empty((n, d), float)

        if self._has_squash:
            # Element-wise handling to avoid applying M across non-affine axes
            for j in range(d):
                m = self._mode[j]
                if m == 'squash':
                    c = self._squash_c[j]
                    s = self._squash_s[j]
                    U[:, j] = 1.0 / (1.0 + np.exp(-(X[:, j] - c) / s))  # (0,1)
                else:
                    U[:, j] = (X[:, j] - self._L[j]) * self._Winv[j]     # linear
            # No mixing when any squash axis exists (enforced in factory)
        else:
            # All affine → vectorize then mix
            U = (X - self._L) * self._Winv
            U = U @ self._M.T

        Y = self._apply_boundary_encode(U)
        return Y[0] if is_row else Y

    def _decode_y(self, y: Array | Sequence[float]) -> Array:
        """y → x inverse normalization, honoring modes and mixing."""
        Ya = np.asarray(y, float)
        is_row = Ya.ndim == 1
        Y = _as_row(Ya).copy()
        n, d = Y.shape
        Y = self._apply_boundary_encode(Y)

        if self._has_squash:
            U = Y  # no mixing allowed/used
        else:
            U = Y @ self._Minv.T

        X = np.empty((n, d), float)
        for j in range(d):
            m = self._mode[j]
            if m == 'squash':
                c = self._squash_c[j]
                s = self._squash_s[j]
                t = np.clip(Y[:, j], np.finfo(float).eps, 1.0 - np.finfo(float).eps)
                X[:, j] = c + s * (np.log(t) - np.log1p(-t))  # logit
            else:
                X[:, j] = self._L[j] + U[:, j] * self._W[j]
        return X[0] if is_row else X

    # ---------------- CAP plan & channel count ----------------
    def _cap_validate_and_plan(self, pairs: Optional[Sequence[Tuple[int, int]]]
                               ) -> tuple[list[Tuple[int, int]], list[int], list[int]]:
        """Validate CAP pairs and return (pairs, remaining_periodic, remaining_aperiodic)."""
        d = self.dim
        P: list[Tuple[int, int]] = [] if pairs is None else list(pairs)
        used_amp, used_phi = set(), set()

        for a, p in P:
            if not (0 <= a < d and 0 <= p < d):
                raise IndexError(f"pair index out of range: (a={a}, p={p}), dim={d}")
            if not (self._mode[a] in ('clip', 'reflect')):  # amplitude from aperiodic
                raise ValueError(f"amp_idx {a} must be aperiodic (clip/reflect), got mode '{self._mode[a]}'")
            if self._mode[p] != 'wrap':                     # phase from periodic
                raise ValueError(f"phase_idx {p} must be periodic (wrap), got mode '{self._mode[p]}'")
            if a in used_amp or p in used_phi:
                raise ValueError("pairs contain repeated indices")
            used_amp.add(a)
            used_phi.add(p)

        idx = np.arange(d)
        rem_p = [j for j in idx if self._mode[j] == 'wrap' and j not in used_phi]
        rem_a = [j for j in idx if self._mode[j] in ('clip', 'reflect') and j not in used_amp]
        return P, rem_p, rem_a

    def cap_channel_count(self, pairs: Optional[Sequence[Tuple[int, int]]] = None) -> int:
        """
        Number of complex CAP channels produced by cap_encode(..., out='complex').
        Note: if you request out='reim', the real-valued channel count is 2× this number.
        """
        P, rem_p, rem_a = self._cap_validate_and_plan(pairs)
        return len(P) + len(rem_p) + len(rem_a)

    # ---------------- unified CAP encode/decode ----------------
    def cap_encode(
        self,
        x,
        *,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        r_min: float = 1e-6,
        r_max: float = 1.0,
        phase_for_aperiodic: float = 0.0,
        out: Literal['complex', 'reim', 'y'] = 'complex'
    ):
        """
        x → (CAP embedding) or y.

        Parameters
        ----------
        x : (d,) or (n,d)
        pairs : list[(a_idx, p_idx)] or None
            amplitude from aperiodic index a_idx, phase from periodic index p_idx.
        r_min, r_max : float
            Radius mapping for aperiodic axes: r = r_min + (r_max - r_min) * y_a.
        phase_for_aperiodic : float
            Phase (radians) used for unpaired aperiodic axes (radius-only if 0.0).
        out : 'complex'|'reim'|'y'
            Output format. 'reim' stacks [Re, Im]; total channels = 2 * cap_channel_count().
        """
        if not (np.isfinite(r_min) and np.isfinite(r_max) and 0.0 <= r_min < r_max):
            raise ValueError("Require finite 0 <= r_min < r_max")

        xa = np.asarray(x, float)
        is_row = xa.ndim == 1
        X = _as_row(xa)
        if X.shape[1] != self.dim:
            raise ValueError(f"bad last-dim: got {X.shape[1]}, expected {self.dim}")

        Y = self._encode_y(X)
        if out == 'y':
            return Y[0] if is_row else Y

        P, rem_p, rem_a = self._cap_validate_and_plan(pairs)
        chans = []

        # (1) pairs: r(y_a) · exp(i·2π·y_p)
        for a, p in P:
            ra = r_min + (r_max - r_min) * Y[:, a]
            phi = 2.0 * np.pi * Y[:, p]
            chans.append(ra * np.exp(1j * phi))
        # (2) remaining periodic: unit circle
        if rem_p:
            phi = 2.0 * np.pi * Y[:, rem_p]
            chans.append(np.exp(1j * phi))
        # (3) remaining aperiodic: radius-only (or fixed-phase)
        if rem_a:
            ra = r_min + (r_max - r_min) * Y[:, rem_a]
            if phase_for_aperiodic == 0.0:
                chans.append(ra.astype(np.complex128))
            else:
                chans.append(ra * np.exp(1j * phase_for_aperiodic))

        if not chans:
            raise ValueError("No axes to encode; check modes and pairs.")
        Z = np.column_stack(chans).astype(np.complex128, copy=False)

        if out == 'complex':
            return Z[0] if is_row else Z
        elif out == 'reim':
            Zr = np.column_stack([np.real(Z), np.imag(Z)]).astype(float, copy=False)
            return Zr[0] if is_row else Zr
        else:
            raise ValueError("out must be 'complex'|'reim'|'y'")

    def cap_decode(
        self,
        z,
        *,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        r_min: float = 1e-6,
        r_max: float = 1.0,
        input: Literal['complex', 'reim', 'y'] = 'complex'
    ):
        """
        (CAP embedding) or y → x (phase-robust).

        Parameters
        ----------
        z : (...,) CAP tensor or y
            If input='reim', the last dim must be even and equal to 2 * cap_channel_count().
        """
        if input == 'y':
            return self._decode_y(z)

        if not (np.isfinite(r_min) and np.isfinite(r_max) and 0.0 <= r_min < r_max):
            raise ValueError("Require finite 0 <= r_min < r_max")

        if input == 'complex':
            za = np.asarray(z, np.complex128)
            is_row = za.ndim == 1
            Z = _as_row(za)
        elif input == 'reim':
            za = np.asarray(z, float)
            is_row = za.ndim == 1
            Zr = _as_row(za)
            if Zr.shape[1] % 2 != 0:
                raise ValueError("reim input must have an even number of channels")
            m = Zr.shape[1] // 2
            Z = (Zr[:, :m] + 1j * Zr[:, m:]).astype(np.complex128, copy=False)
        else:
            raise ValueError("input must be 'complex'|'reim'|'y'")

        P, rem_p, rem_a = self._cap_validate_and_plan(pairs)
        m_expected = len(P) + len(rem_p) + len(rem_a)
        if Z.shape[1] != m_expected:
            raise ValueError(f"z has {Z.shape[1]} channels but expected {m_expected}")

        Y = np.zeros((Z.shape[0], self.dim), float)
        col = 0
        inv_span = 1.0 / (r_max - r_min)
        hi = self._hi

        # helper: robust phase→[0,1)
        def _phase_to_unit(phi: Array) -> Array:
            # Map to [0, 2π) robustly, then divide; clamp half-open.
            twopi = 2.0 * np.pi
            phi2 = np.remainder(phi + twopi, twopi)  # [0,2π)
            yp = phi2 / twopi
            yp = np.where(yp >= 1.0, 0.0, yp)
            yp = np.where(yp < 0.0, 0.0, yp)
            return yp

        # (1) pairs
        for a, p in P:
            zc = Z[:, col]
            col += 1
            r = np.abs(zc)
            phi = np.angle(zc)
            Y[:, a] = np.clip((r - r_min) * inv_span, 0.0, hi)
            Y[:, p] = _phase_to_unit(phi)
        # (2) remaining periodic
        if rem_p:
            zc = Z[:, col:col + len(rem_p)]
            col += len(rem_p)
            phi = np.angle(zc)
            Y[:, rem_p] = _phase_to_unit(phi)
        # (3) remaining aperiodic
        if rem_a:
            zc = Z[:, col:col + len(rem_a)]
            col += len(rem_a)
            r = np.abs(zc)
            Y[:, rem_a] = np.clip((r - r_min) * inv_span, 0.0, hi)

        return self._decode_y(Y)

    # ------------- y-only wrappers (back-compat) ----------------
    def encode(self, x):
        """Back-compat: x → y (same as cap_encode(..., out='y'))."""
        return self.cap_encode(x, out='y')

    def decode(self, y):
        """Back-compat: y → x (same as cap_decode(..., input='y'))."""
        return self.cap_decode(y, input='y')

    # ------------- Jacobians for affine axes -------------------
    def jac_x_to_y(self) -> Array:
        """∂y/∂x ≈ M @ diag(1/W) for affine axes. Undefined with 'squash'."""
        if self._has_squash:
            raise ValueError("jac_x_to_y is undefined with 'squash' axes.")
        return self._M @ self._D_Winv

    def jac_y_to_x(self) -> Array:
        """∂x/∂y ≈ diag(W) @ M^{-1} for affine axes. Undefined with 'squash'."""
        if self._has_squash:
            raise ValueError("jac_y_to_x is undefined with 'squash' axes.")
        return self._D_W @ self._Minv

    # ------------- convenience -------------------
    def roundtrip_assert(self, x: Array, atol: float = 1e-12):
        """Quick check: x → CAP(complex-by-default) → x."""
        z = self.cap_encode(x)  # default 'complex'
        xr = self.cap_decode(z)  # default 'complex'
        if not np.allclose(np.asarray(x, float), xr, atol=atol):
            raise AssertionError("roundtrip mismatch")


# ---------------- module-level ergonomic factory ----------------
def boundary(
    bounds: Sequence[Tuple[float, float]],
    *,
    mode: Optional[Sequence[str]] = None,
    period: Optional[Sequence[Optional[float]]] = None,
    M: Optional[Array] = None,
    squash_center: Optional[Sequence[float]] = None,
    squash_scale: Optional[Sequence[float]] = None
) -> Normalizer:
    """
    Ergonomic constructor. Accepts a classic list of (low, high) tuples.

    Notes
    -----
    - If any axis is 'squash', provide optional squash_center/scale; mixing matrix M is disallowed.
    - For 'wrap' with non-finite (L,U), provide a positive period for that axis.
    """
    return Normalizer.boundary_tuples(
        bounds,
        mode=mode,
        period=period,
        M=M,
        squash_center=squash_center,
        squash_scale=squash_scale,
    )