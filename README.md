# Residual-driven basis and data selection for high-dimensional HJB equations

MATLAB code for the numerical experiments in

> Y. Huang, D. Kalise and Y. Zhu, *Residual-Driven Basis and Data Selection for Polynomial Approximation of High-Dimensional Hamilton–Jacobi–Bellman Equations*.

The code implements the two greedy algorithms of the paper:

- **Algorithm 1 (model-based greedy algorithm, Section 4.5).** The time-dependent Galerkin scheme of Section 3.2.1, whose basis is enlarged one function at a time using the stationary HJB residual-decay indicator e_HJB, eq. (24).
- **Algorithm 2 (data-driven greedy algorithm, Section 5.3).** Gradient-augmented regression (Section 3.3, least squares or Lasso) that adds one training sample and one basis function per iteration. Basis functions are selected with the supervised-loss indicator e_SL, eq. (26), or the empirical residual-decay indicator ẽ_PDE, eq. (27). Training samples are selected with the data score, eq. (28).

Equation, section and figure numbers refer to the current version of the manuscript.
