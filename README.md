# optimization_adamuon

## Python Modules

### Bridge.py
- `BridgeConfig`
  - Input = seed, preconditioner toggle, gradient sampling counts.
  - Output = configuration for how the bridge orchestrates components.
- `OnePointOptimizer`
  - `__init__(f_x, normalizer, adjoint_x=None, bridge_cfg=None, precond_cfg=None, memory_metric=None, memory_event=None, rng=None)`
    - Input = objective function in x-space and helper objects.
    - Output = optimizer instance that wires normalizer, stepper and preconditioner.
  - `_adjoint_y()`
    - Output = function pulling back x-space gradients to y-space using diag/full/hybrid policy.
  - `step(y, logger=None)`
    - Input = starting point in normalized coordinates.
    - Output = `(y_new, f_new, info)` after one optimization step.

### Gradient.py
- `GradientEvalError`
  - Raised when objective evaluations are non-finite.
- `BaseGradientSource`
  - `grad(f_y, y, A, rng)`
    - Input = objective in y-space, point `y`, optional direction matrix `A` and RNG.
    - Output = `(g, info, fevals)` gradient estimate via provided scheme.
- `Forward1P`
  - `grad(...)`
    - Input = `f_y`, `y`, optional `A`.
    - Output = one-sided finite difference gradient and diagnostics.
- `Central2P`
  - `grad(...)`
    - Input = `f_y`, `y`, optional `A`.
    - Output = two-sided central difference gradient and diagnostics.
- `AdjointBlend`
  - `calibrate(y, g_2p, adjoint, grad_space='y', normalizer=None, sign=+1)`
    - Input = reference gradient and external adjoint.
    - Output = cosine similarity and norm statistics in y-space.

### Memory.py
- `MemoryMetric`
  - `update(cur_f)`
    - Input = latest objective value.
    - Output = `M_t` scale-free change magnitude in `[0,1)`.
- `MemoryEvent`
  - `update(M_t)`
    - Input = change magnitude.
    - Output = `(calib_event, evicted)` event flags for compatibility.
  - `step_gated(M_t)`
    - Input = change magnitude.
    - Output = `(fired, fired_count, survival_mul)` with hazard scheduling.
- `MemoryEngine`
  - `ingest_f(cur_f)`
    - Input = accepted objective value.
    - Output = dictionary with magnitude, survival multiplier and event info.

### Normalize.py
- `_as_row(a)`
  - Input = vector or matrix.
  - Output = 2-D array view.
- `Normalizer`
  - `boundary_tuples(bounds, mode=None, period=None, M=None, squash_center=None, squash_scale=None)`
    - Input = per-axis bounds and options.
    - Output = normalizer instance.
  - `cap_encode(x, pairs=None, r_min=1e-6, r_max=1.0, phase_for_aperiodic=0.0, out='complex')`
    - Input = real-space variable.
    - Output = CAP embedding or normalized `y` coordinates.
  - `cap_decode(z, pairs=None, r_min=1e-6, r_max=1.0, input='complex')`
    - Input = CAP embedding or `y`.
    - Output = reconstructed real-space variable.
  - `encode(x)` / `decode(y)`
    - Convenience wrappers for y-only mapping.
  - `jac_x_to_y()` / `jac_y_to_x()`
    - Output = affine-axis Jacobian matrices for coordinate transforms.
  - `roundtrip_assert(x, atol=1e-12)`
    - Input = sample point.
    - Output = assertion if encode→decode mismatch.
  - `boundary(bounds, ...)`
    - Function returning a `Normalizer` with ergonomic parameters.

### Stepper.py
- `StepperEvalError`
  - Raised when evaluations at base or trial points are invalid.
- `AdamState`
  - `init_like(y)`
    - Input = reference shape.
    - Output = zeroed Adam state tuple `(t, m, v)`.
- `StepperConfig`
  - Hyperparameters for AdamW, trust region, memory gating and events.
- `AdamMuonStepper`
  - `__init__(cfg, memory_metric=None, memory_event=None, adjoint_blend=None, normalizer=None, rng=None)`
    - Input = configuration and optional hooks.
    - Output = stepper managing AdamW, trust region and memory events.
  - `step(f_y, y, A, adam_state, grad_source, adjoint_fn=None, logger=None, minimize=True)`
    - Input = objective in y-space and current state.
    - Output = `(y_new, f_new, new_state, info)` after applying policy.
  - `_project_to_trust_region(step, delta)`
    - Input = proposed step and radius.
    - Output = step scaled to fit L2 ball.

### preconditioner.py
- `PreconditionerConfig`
  - Window sizes, scaling caps and spectral norm limit for the preconditioner.
- `Preconditioner`
  - `__init__(dim, periodic_mask=None, config=None)`
    - Input = problem dimension and options.
    - Output = preconditioner with empty statistics.
  - `reset()`
    - Clears stored steps and secants.
  - `push_step_only(y_old, y_new)`
    - Input = consecutive accepted points.
    - Output = update of whitening statistics.
  - `push_secant(y_old, y_new, g_old, g_new)`
    - Input = start points and gradients.
    - Output = stores absolute secant information.
  - `compute_A()`
    - Output = `(A, info)` whitening matrix with diagonal scaling and spectral cap.
