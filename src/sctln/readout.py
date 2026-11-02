import jax
import jax.numpy as jnp
import numpy as np

def fit_readout(xs: jax.Array, targets: jax.Array, ridge: float) -> tuple[jax.Array, jax.Array]:
    """(T,n), (T,m) -> R (m,n), b (m,). Must be jittable + vmappable
    (closed form, no Python branching), because it runs inside fitness per candidate."""
    T, n = xs.shape
    x_mean, y_mean = xs.mean(axis=0), targets.mean(axis=0)
    Xc, Yc = xs - x_mean, targets - y_mean
    G = Xc.T @ Xc
    theta = jnp.linalg.solve(G + ridge_eff * jnp.eye(n), Xc.T @ Yc) # (n, m)
    R = theta.T
    b = y_mean - R @ x_mean

    return R, b

def apply_readout(xs: jax.Array, R: jax.Array, b: jax.Array) -> jax.Array:
    """(T,n) -> (T,m)."""
    y = xs @ R.T + b   # (T, n) @ (n, m) -> (T, m)

    return y
