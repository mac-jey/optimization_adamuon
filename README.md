# Optimization Adamuon

This repository contains a modular implementation of a derivative-free optimizer
that combines finite-difference gradients, trust-region Adam updates and a
memory-based preconditioner.  The optimizer works in a normalized coordinate
system and can blend external adjoint information when available.

## Overview

The optimizer operates in an internal unit cube `y ∈ [0,1)^d`.  A
`Normalizer` maps user-facing variables `x ∈ ℝ^d` to `y` and back.  Each step
of the optimization performs the sequence:

1. **Preconditioning** – compose a whitening matrix `W` from past accepted
   moves and a diagonal secant matrix `P`; the final matrix is `A = P·W`.
2. **Gradient estimation** – call either forward one-point or central two-point
   finite differences to approximate `∇_y f`.
3. **Adam trust-region step** – update with an AdamW step projected onto a
   trust region and accepted only if it reduces the objective.
4. **Memory update** – feed the function value to a change metric and event
   scheduler used for burst tolerance and adjoint calibration.

## Modules

### Normalizer
Maps physical variables to the unit cube with several boundary modes:

- **Wrap**: `y = ((x - L) / W) mod 1`
- **Clip**: `y = clip((x - L)/W, 0, 1)`
- **Reflect**: triangular reflection into `[0,1]`
- **Squash**: `y = σ((x - c)/s)` for unbounded axes.

Affine axes may be mixed with a matrix `M`; the Jacobians used for adjoint
pullback are
`∂y/∂x = M · diag(1/W)` and `∂x/∂y = diag(W) · M^{-1}`.
The class also exposes a Complex Amplitude–Phase (CAP) interface which
represents pairs of periodic and aperiodic axes as complex numbers `z = r·e^{iφ}`.

### Gradient
Implements finite-difference gradient approximations in `y` space.

- **Forward1P** – single evaluation per direction:
  `g ≈ (f(y + εu) - f(y)) / ε`
- **Central2P** – symmetric evaluation improving accuracy:
  `g ≈ (f(y + εu) - f(y - εu)) / (2ε)`
  and now automatically tunes the finite-difference step by probing a
  logarithmic range and selecting the largest step that still exhibits the
  expected second-order convergence.

Directions `u_k` are generated to be well conditioned using QR and
Gram–Schmidt procedures.  The sampled directional derivatives are solved in a
least-squares sense to recover the gradient.  `AdjointBlend` measures cosine
alignment between a reference gradient and an external adjoint vector.

### Preconditioner
Constructs the matrix `A = P·W` applied to the gradient before the Adam step.

- `W` whitens accepted steps `Δy` with
  `W = (Σ + εI)^{-1/2}`, `Σ` being the covariance of past steps.
- `P` is diagonal; each element is the median of
  `|Δy_i| / max(|Δg_i|, g_floor)` over a secant window and is clipped to
  `[p_min, p_max]`.
- The spectral norm of `A` is capped to `τ_cap` for stability.

### Stepper (AdamMuonStepper)
Carries out the optimization update:

1. Build an AdamW step using moments `m_t` and `v_t`.
2. Project the step to the trust-region radius `δ` and apply optional weight
   decay and user hook.
3. Perform backtracking line search until the objective decreases.
4. Update `δ` using grow/shrink factors and optional burst tolerance driven by
   memory signals `M_t` and `S_t`.
5. On accepted steps, hand off secant information for the preconditioner and
   optionally trigger `AdjointBlend` calibration events.

### Memory
Provides two components:

- **MemoryMetric** – maintains an EMA of `|Δf|` and computes
  a scale-free magnitude
  `M_t = σ(4(x - 0.5))` with `x = r/(1+r)` and `r = |Δf|/EMA`.
- **MemoryEvent** – hazard based event scheduler where survival is
  `S_t = exp(-∑ α·M_t)` and a new event fires when cumulative hazard crosses
  `ln(1+τ)` but only on "calm" steps with `M_t ≤ m_gate`.

### Bridge (OnePointOptimizer)
Coordinates all modules.  It normalizes user input, chooses the gradient scheme
(bootstrap with Forward1P then switch to Central2P) and performs adjoint
pullback.  Pullback policy can be:

- `diag`: scale `∇_x` by widths `W` (safe baseline)
- `full`: multiply by Jacobian `J^T` with optional SVD clipping
- `hybrid`: default; switch between the two based on condition number and
  boundary proximity.

The bridge feeds gradients to the stepper, updates the preconditioner with
accepted secants, and exposes a high-level `step` function returning
`(y_new, f_new, info)`.

## Framework Integration
`OnePointOptimizer` works on NumPy arrays but can be adapted to common
optimization libraries:

- **SciPy** – wrap `step` inside a loop and supply the objective and gradient
  through the `f_x`/`adjoint_x` callables. The interface is similar to
  `scipy.optimize.minimize` and can be embedded via its `callback`.
- **PyTorch** – create a `torch.optim.Optimizer` subclass that forwards
  parameter tensors to and from NumPy, letting the bridge drive updates on CPU
  or GPU. Adjoint gradients from autograd can be passed through `adjoint_x`.

Placing any statistical heuristics for Stepper or Memory inside the bridge
allows a single implementation to serve both ecosystems.

## Running
Only standard NumPy is required.  All modules compile under Python and can be
checked with:

```bash
python -m py_compile Bridge.py Gradient.py Memory.py Normalize.py Stepper.py preconditioner.py
```

