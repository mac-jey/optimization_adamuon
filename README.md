# optimization_adamuon

## Python Modules

### Normalize.py
*Domain: coordinate transforms between real-space variables `x` and normalized coordinates `y` using CAP embedding.*

- `_as_row(a)`
  - Input: `a` (array-like) vector or matrix.
  - Output: 2‑D array view of shape `(1,d)` or `(d,1)`.
  - **Description**: ensures row orientation for subsequent linear algebra operations.

- `Normalizer`
  - `boundary_tuples(bounds, mode=None, period=None, M=None, squash_center=None, squash_scale=None)`
    - Input: `bounds` (sequence of `(lo, hi)` for each axis); `mode` (list per axis: `'wrap'`, `'reflect'`, or `'squash'`); `period` (float periods for wrapped axes); `M` (scaling factors); `squash_center` and `squash_scale` (floats controlling logit squash).
    - Output: `Normalizer` instance configured per-axis.
    - **Description**: factory constructing transformations so real `x` values map into `[0,1)` coordinates with optional periodic or squash behaviour.

  - `cap_encode(x, pairs=None, r_min=1e-6, r_max=1.0, phase_for_aperiodic=0.0, out='complex')`
    - Input: `x` (`(d,)` real vector); `pairs` (axis pairing for CAP embedding); `r_min`, `r_max` (radius limits); `phase_for_aperiodic` (float); `out` (`'complex'` or `'y'`).
    - Output: CAP complex representation or normalized vector `y` depending on `out`.
    - **Description**: encodes real variables into complex unit disk or directly into normalized coordinates.

  - `cap_decode(z, pairs=None, r_min=1e-6, r_max=1.0, input='complex')`
    - Input: `z` (complex CAP array or normalized `y`); `pairs`; `r_min`, `r_max`; `input` (`'complex'` or `'y'`).
    - Output: reconstructed real-space vector `x`.
    - **Description**: reverses `cap_encode` to recover physical variables.

  - `encode(x)` / `decode(y)`
    - Input: `x` or `y` one-dimensional vectors.
    - Output: mapped `y` or `x`.
    - **Description**: light wrappers around `cap_encode`/`cap_decode` for pure coordinate transforms.

  - `jac_x_to_y()` / `jac_y_to_x()`
    - Output: `(d,d)` Jacobian matrices for affine axes.
    - **Description**: provide constant derivatives when no squash axes exist.

  - `roundtrip_assert(x, atol=1e-12)`
    - Input: `x` test vector; `atol` tolerance.
    - Output: raises `AssertionError` if decode(encode(x)) deviates beyond tolerance.
    - **Description**: sanity check for exact encode–decode mapping.

  - `boundary(bounds, ...)`
    - Input: same parameters as `boundary_tuples` via ergonomic keywords.
    - Output: convenience function returning a `Normalizer`.
    - **Description**: high-level entry point for typical normalization setups.

### Gradient.py
*Domain: gradient estimation and adjoint comparison in normalized space.*

- `GradientEvalError`
  - **Description**: raised when objective evaluation returns a non-finite value.

- `BaseGradientSource.grad(f_y, y, A, rng)`
  - Input: `f_y` callable `(y)->float`; `y` (`(d,)` vector); `A` (optional `(d,K)` direction matrix); `rng` (`np.random.Generator`) for sampling directions.
  - Output: `(g, info, fevals)` where `g` is gradient estimate `(d,)`; `info` diagnostic dictionary; `fevals` integer function evaluations.
  - **Description**: interface contract for gradient providers; implementations must not perform line search or boundary handling.

- `Forward1P.grad(...)`
  - Input: same parameters as `BaseGradientSource.grad` with step count `K` and relative step `eps_rel`.
  - Output: `(g, info, fevals)` one-sided finite-difference estimate.
  - **Description**: probes forward points `y + ε u_k` and solves a least-squares system to recover gradient.

- `Central2P.grad(...)`
  - Input: same as `Forward1P.grad`.
  - Output: `(g, info, fevals)` central-difference estimate.
  - **Description**: evaluates symmetric pairs `y ± ε u_k` to reduce bias.

- `AdjointBlend.calibrate(y, g_2p, adjoint, grad_space='y', normalizer=None, sign=+1)`
  - Input: `y` point; `g_2p` reference gradient `(d,)`; `adjoint` external gradient (callable or vector); `grad_space` (`'y'` or `'x'`); `normalizer` for conversions; `sign` (+1 or -1).
  - Output: dictionary with cosine similarity, norm ratios and evaluation counts.
  - **Description**: compares external gradients against finite differences for monitoring and adjusts sign if needed.

### preconditioner.py
*Domain: builds whitening and diagonal scaling matrices to precondition gradients.*

- `PreconditionerConfig`
  - Input: `max_steps`, `max_secants`, `eps`, `g_floor`, `p_min`, `p_max`, `tau_cap`, `use_power_iter`, `power_iter_steps`, `svd_on_small`, `svd_small_d`.
  - Output: immutable configuration dataclass.
  - **Description**: sets window sizes and numeric caps governing preconditioner behaviour.

- `Preconditioner.__init__(dim, periodic_mask=None, config=None)`
  - Input: `dim` problem dimension; `periodic_mask` (`(d,)` bool array); `config` (`PreconditionerConfig`).
  - Output: initialized preconditioner with empty windows.
  - **Description**: prepares storage for recent steps and secants and records periodic axes for torus wrapping.

- `reset()`
  - Output: clears internal step and secant buffers.
  - **Description**: forgets all accumulated statistics.

- `push_step_only(y_old, y_new)`
  - Input: `y_old`, `y_new` start points (`(d,)`).
  - Output: None.
  - **Description**: stores accepted move Δy for covariance used in whitening.

- `push_secant(y_old, y_new, g_old, g_new)`
  - Input: starting points `y_old`, `y_new`, gradients `g_old`, `g_new` (all `(d,)`).
  - Output: None.
  - **Description**: records absolute secants |Δy| and |Δg| for diagonal scaling.

- `compute_A()`
  - Output: `(A, info)` where `A` is `(d,d)` whitening matrix; `info` diagnostic dictionary.
  - **Description**: forms `A = P·W`, clips spectral norm to `tau_cap` and reports conditioning metrics.

### Stepper.py
*Domain: applies AdamW update with trust-region and event gating in normalized space.*

- `StepperEvalError`
  - **Description**: raised when objective evaluations at base or trial points are invalid.

- `AdamState.init_like(y)`
  - Input: `y` (`(d,)` reference shape).
  - Output: `AdamState(t=0, m=zeros(d), v=zeros(d))`.
  - **Description**: creates zeroed optimizer state matching problem dimension.

- `StepperConfig`
  - Input: `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `delta`, `delta_grow`, `delta_shrink`, `delta_min`, `delta_max`, `backtrack_mult`, `max_backtracks`, `burst_tolerant`, `m_lo`, `m_hi`, `k_up`, `k_dn`, `event_min_gap`, `calib_K2p`, `calib_eps_rel`, `muon_hook`.
  - Output: configuration object.
  - **Description**: controls AdamW update, trust-region policy, memory gating and optional Muon hook.

- `AdamMuonStepper.__init__(cfg, memory_metric=None, memory_event=None, adjoint_blend=None, normalizer=None, rng=None)`
  - Input: `cfg` (`StepperConfig`); `memory_metric` (`MemoryMetric`); `memory_event` (`MemoryEvent`); `adjoint_blend` (`AdjointBlend`); `normalizer` (`Normalizer`); `rng` (`np.random.Generator`).
  - Output: stepper object managing optimization policy.
  - **Description**: wires optional hooks and initializes counters for event gating.

- `step(f_y, y, A, adam_state, grad_source, adjoint_fn=None, logger=None, minimize=True)`
  - Input: `f_y` callable objective; `y` current point (`(d,)`); `A` preconditioner `(d,d)` or `None`; `adam_state` (`AdamState`); `grad_source` (implements `.grad`); `adjoint_fn` optional callable; `logger` function; `minimize` bool.
  - Output: `(y_new, f_new, adam_state_new, info)` with updated state and diagnostics.
  - **Description**: evaluates objective and gradient, produces AdamW step, projects into trust region, performs backtracking, updates memory/event hooks and returns the accepted point.

- `_project_to_trust_region(step, delta)`
  - Input: `step` (`(d,)` vector); `delta` radius.
  - Output: scaled step within L2 ball.
  - **Description**: shrinks candidate step when its norm exceeds the trust-region radius.

### Memory.py
*Domain: tracks objective change magnitude and schedules hazard-based calibration events.*

- `MemoryMetric.update(cur_f)`
  - Input: `cur_f` (float objective value).
  - Output: `M_t` float in `[0,1)` representing scale-free change magnitude.
  - **Description**: maintains EMA of absolute differences and maps them through a bounded sigmoid.

- `MemoryEvent.update(M_t)`
  - Input: `M_t` change magnitude.
  - Output: `(calib_event, evicted)` booleans/integers indicating event triggers and evictions.
  - **Description**: legacy compatibility wrapper around `step_gated`.

- `MemoryEvent.step_gated(M_t)`
  - Input: `M_t` change magnitude.
  - Output: `(fired, fired_count, survival_mul)` with hazard-based opportunity scheduling.
  - **Description**: updates cumulative hazard, applies calmness gate and computes survival multiplier.

- `MemoryEngine.ingest_f(cur_f)`
  - Input: `cur_f` accepted objective value.
  - Output: dict with keys `M_t`, `survival_mul`, `S_t`, `calib_event`, `evicted`, `pending`, `xi_mem`, `barrier`.
  - **Description**: combines metric and event logic to produce per-step signals for other modules.

### Bridge.py
*Domain: high-level orchestrator connecting normalization, stepping, preconditioning and memory.*

- `BridgeConfig`
  - Input: `seed` (int); `use_precond` (bool); `bootstrap_K1p` (int); `central_K2p` (int); `eps_rel_1p` (float); `eps_rel_2p` (float); `stepper_cfg` (`StepperConfig`); `jac_pullback` (`'diag'|'full'|'hybrid'`); `jac_cond_cap` (float); `jac_squash_eps` (float); `jac_sigma_lo` (float); `jac_sigma_hi` (float); `jac_use_svd_clip` (bool).
  - Output: configuration dictating bridge policies.
  - **Description**: sets RNG seed, gradient sampling counts and Jacobian pullback strategy for adjoint gradients.

- `OnePointOptimizer.__init__(f_x, *, normalizer, adjoint_x=None, bridge_cfg=None, precond_cfg=None, memory_metric=None, memory_event=None, rng=None)`
  - Input: `f_x` callable mapping `x` to scalar; `normalizer` (`Normalizer`); `adjoint_x` optional callable returning x-space gradients; `bridge_cfg` (`BridgeConfig`); `precond_cfg` (`PreconditionerConfig`); `memory_metric` (`MemoryMetric`); `memory_event` (`MemoryEvent`); `rng` (`np.random.Generator`).
  - Output: optimizer instance wiring all components.
  - **Description**: creates deterministic RNG, instantiates stepper and preconditioner, and prepares gradient sources for bootstrap and steady phases.

- `_adjoint_y()`
  - Output: callable `(y) -> g` returning y-space gradients or `None`.
  - **Description**: wraps external x-space adjoint using policy-controlled Jacobian pullback to produce safe y-space gradients.

- `step(y, logger=None)`
  - Input: starting point `y` (`(d,)`); `logger` optional callable.
  - Output: `(y_new, f_new, info)` after one optimization iteration.
  - **Description**: pulls objective into normalized space, selects gradient source, applies preconditioner and stepper policy, updates memory, and returns the next point.

