# Revision Notes

- Preconditioning uses full covariance recomputation each step.  Incremental
  updates could reduce cost for high dimensional problems.
- The Jacobian pullback in `Bridge._adjoint_y` falls back to full SVD even when
  the Jacobian is constant.  Caching the SVD or using condition estimates could
  avoid repeated decompositions.
- Memory scheduling lacks unit tests; edge cases such as extremely large
  function jumps or long pending queues may need additional verification.
- `Stepper` assumes the objective decreases along the backtracking line but does
  not guard against non-monotonic behavior of noisy functions.  Incorporating
  stochastic acceptance criteria could improve robustness.
- Adapters for PyTorch and SciPy could live in the bridge layer so the same
  statistical tooling for Stepper/Memory serves both CPU and GPU backends.
- The central difference step-size tuner currently searches a fixed logarithmic
  grid.  Adapting the sweep range or using noise estimates could yield more
  reliable step selection in very noisy objectives.
- Logging presently records acceptance and backtrack counts but omits wall-clock
  timing and environment hashes, which would aid reproducibility.
