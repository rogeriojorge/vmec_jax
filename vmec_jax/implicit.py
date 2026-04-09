"""Implicit differentiation utilities.

This module provides *custom VJP* wrappers for equilibrium sub-solves so that
outer objectives can differentiate through equilibrium states without
backpropagating through many optimization iterations.

Initial scope
=============

Implicit differentiation for the **lambda-only** fixed-geometry solve.

Future work
===========

Extend to the full fixed-boundary solve over (R,Z,lambda) using the same
implicit-function machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Tuple

import numpy as np

from ._compat import has_jax, jax, jnp

try:
    import lineax as lx
except Exception:  # pragma: no cover - optional dependency
    lx = None

from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda, signgs_from_sqrtg
from .energy import flux_profiles_from_indata
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .grids import angle_steps
from .solve import (
    _WoutLikeVmecForces,
    _enforce_fixed_boundary_and_axis,
    _half_mesh_from_full_mesh,
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
    _enforce_lambda_gauge,
    _mask_grad_for_constraints,
    _mode00_index,
    _pressure_half_mesh_from_indata,
    _vmec_force_flux_profiles,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_residual_iter,
    solve_lambda_gd,
)
from .state import VMECState, pack_state, unpack_state


def _vmec_backward_profile_enabled() -> bool:
    value = os.environ.get("VMEC_JAX_PROFILE_BACKWARD", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_backward_profile_log(stage: str, start: float | None = None, **extra) -> None:
    if not _vmec_backward_profile_enabled():
        return
    payload = {"stage": stage}
    if start is not None:
        payload["elapsed_s"] = time.perf_counter() - start
    payload.update(extra)
    print(f"[vmec_jax backward] {payload}", flush=True)


def _vmec_residual_profile_enabled() -> bool:
    value = os.environ.get("VMEC_JAX_PROFILE_RESIDUAL", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_residual_profile_log(stage: str, start: float | None = None, **extra) -> None:
    if not _vmec_residual_profile_enabled():
        return
    payload = {"stage": stage}
    if start is not None:
        payload["elapsed_s"] = time.perf_counter() - start
    payload.update(extra)
    print(f"[vmec_jax residual] {payload}", flush=True)


def _vmec_keep_all_active_enabled() -> bool:
    value = os.environ.get("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_disable_reduced_active_enabled() -> bool:
    value = os.environ.get("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_jac_chunk_size_override() -> int | None:
    value = os.environ.get("VMEC_JAX_IMPLICIT_JAC_CHUNK_SIZE", "").strip()
    if value == "":
        return None
    chunk_size = int(value)
    if chunk_size <= 0:
        raise ValueError(f"VMEC_JAX_IMPLICIT_JAC_CHUNK_SIZE must be positive, got {chunk_size}")
    return chunk_size


def _default_vmec_jac_chunk_size(input_size: int) -> int:
    input_size = int(input_size)
    # For the reduced stellarator-symmetric adjoint systems that dominate the
    # QH workload, a single batched Jacobian build is materially faster than
    # many small chunks up to at least ~1.5k active coordinates. Keep the
    # legacy small-chunk fallback for larger systems to avoid a sudden memory
    # jump on bigger configurations.
    if input_size <= 2048:
        return input_size
    return 64


def _dense_transpose_lstsq_host(J, b, damping):
    """Host-side least-squares solve for J^T lam ~= b with optional Tikhonov damping."""
    J_host = np.asarray(J)
    b_host = np.asarray(b)
    damping_host = float(np.asarray(damping))
    A_host = J_host.T
    if damping_host > 0.0:
        eye = np.eye(int(A_host.shape[1]), dtype=A_host.dtype)
        A_host = np.concatenate(
            [A_host, np.sqrt(damping_host) * eye],
            axis=0,
        )
        b_host = np.concatenate(
            [b_host, np.zeros((int(eye.shape[0]),), dtype=b_host.dtype)],
            axis=0,
        )
    lam_host, *_ = np.linalg.lstsq(A_host, b_host, rcond=None)
    return np.asarray(lam_host, dtype=J_host.dtype)


def _dense_transpose_lstsq_jax(J, b, damping: float):
    """Device-side least-squares solve for J^T lam ~= b with optional Tikhonov damping."""
    J_arr = jnp.asarray(J)
    b_arr = jnp.asarray(b)
    A_arr = jnp.swapaxes(J_arr, 0, 1)
    damping_f = float(damping)
    if damping_f > 0.0:
        eye = jnp.eye(int(A_arr.shape[1]), dtype=A_arr.dtype)
        A_arr = jnp.concatenate(
            [A_arr, jnp.sqrt(jnp.asarray(damping_f, dtype=A_arr.dtype)) * eye],
            axis=0,
        )
        b_arr = jnp.concatenate(
            [b_arr, jnp.zeros((int(eye.shape[0]),), dtype=b_arr.dtype)],
            axis=0,
        )
    lam_arr, *_ = jnp.linalg.lstsq(A_arr, b_arr, rcond=None)
    return jnp.asarray(lam_arr, dtype=J_arr.dtype)


def _dense_lstsq_host(J, b, damping):
    """Host-side least-norm solve for J x ~= b with optional Tikhonov damping."""
    J_host = np.asarray(J)
    b_host = np.asarray(b)
    damping_host = float(np.asarray(damping))
    A_host = J_host
    if damping_host > 0.0:
        eye = np.eye(int(A_host.shape[1]), dtype=A_host.dtype)
        A_host = np.concatenate(
            [A_host, np.sqrt(damping_host) * eye],
            axis=0,
        )
        b_host = np.concatenate(
            [b_host, np.zeros((int(eye.shape[0]),), dtype=b_host.dtype)],
            axis=0,
        )
    x_host, *_ = np.linalg.lstsq(A_host, b_host, rcond=None)
    return np.asarray(x_host, dtype=J_host.dtype)


def _dense_lstsq_jax(J, b, damping: float):
    """Device-side least-norm solve for J x ~= b with optional Tikhonov damping."""
    J_arr = jnp.asarray(J)
    b_arr = jnp.asarray(b)
    A_arr = J_arr
    damping_f = float(damping)
    if damping_f > 0.0:
        eye = jnp.eye(int(A_arr.shape[1]), dtype=A_arr.dtype)
        A_arr = jnp.concatenate(
            [A_arr, jnp.sqrt(jnp.asarray(damping_f, dtype=A_arr.dtype)) * eye],
            axis=0,
        )
        b_arr = jnp.concatenate(
            [b_arr, jnp.zeros((int(eye.shape[0]),), dtype=b_arr.dtype)],
            axis=0,
        )
    x_arr, *_ = jnp.linalg.lstsq(A_arr, b_arr, rcond=None)
    return jnp.asarray(x_arr, dtype=J_arr.dtype)


def _weighted_dense_lstsq_jax(J, b, column_weights, damping: float = 0.0):
    """Least-norm solve for J x ~= b with diagonal weights on x."""
    J_arr = jnp.asarray(J)
    weights_arr = jnp.asarray(column_weights, dtype=J_arr.dtype)
    inv_weights = 1.0 / weights_arr
    x_scaled = _dense_lstsq_jax(J_arr * inv_weights[None, :], b, damping)
    return jnp.asarray(inv_weights * x_scaled, dtype=J_arr.dtype)


@dataclass(frozen=True)
class ImplicitLambdaOptions:
    """Controls for the implicit backward pass."""

    cg_max_iter: int = 80
    cg_tol: float = 1e-10
    damping: float = 1e-6


@dataclass(frozen=True)
class ImplicitFixedBoundaryOptions:
    """Controls for the implicit backward pass (fixed-boundary solve)."""

    cg_max_iter: int = 80
    cg_tol: float = 1e-10
    damping: float = 1e-6
    residual_adjoint_mode: str = "auto"
    residual_tangent_mode: str = "opaque"
    residual_forward_mode: str = "callback"
    jac_chunk_size: int | None = None


def _stop_gradient_tree(x):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, x)


def _zero_state_like(state: VMECState) -> VMECState:
    return VMECState(
        layout=state.layout,
        Rcos=jnp.zeros_like(jnp.asarray(state.Rcos)),
        Rsin=jnp.zeros_like(jnp.asarray(state.Rsin)),
        Zcos=jnp.zeros_like(jnp.asarray(state.Zcos)),
        Zsin=jnp.zeros_like(jnp.asarray(state.Zsin)),
        Lcos=jnp.zeros_like(jnp.asarray(state.Lcos)),
        Lsin=jnp.zeros_like(jnp.asarray(state.Lsin)),
    )


def _flatten_L(Lcos, Lsin) -> Any:
    return jnp.concatenate([jnp.ravel(Lcos), jnp.ravel(Lsin)], axis=0)


def _unflatten_L(x, *, shape: Tuple[int, int]):
    ns, K = shape
    n = ns * K
    x = jnp.asarray(x)
    Lcos = jnp.reshape(x[:n], (ns, K))
    Lsin = jnp.reshape(x[n:], (ns, K))
    return Lcos, Lsin


def _cg_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    x0: Any | None = None,
    tol: float,
    max_iter: int,
) -> Any:
    """Conjugate gradients for SPD systems (JAX-friendly)."""
    b = jnp.asarray(b)
    x = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    r = b - matvec(x)
    p = r
    rs = jnp.dot(r, r)

    tol2 = jnp.asarray(float(tol) ** 2, dtype=b.dtype)
    max_iter = int(max_iter)

    def cond_fun(carry):
        i, x, r, p, rs = carry
        return jnp.logical_and(i < max_iter, rs > tol2)

    def body_fun(carry):
        i, x, r, p, rs = carry
        Ap = matvec(p)
        alpha = rs / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs
        p_new = r_new + beta * p
        return i + 1, x_new, r_new, p_new, rs_new

    _i, x, _r, _p, _rs = jax.lax.while_loop(cond_fun, body_fun, (0, x, r, p, rs))
    return x


def _lineax_bicgstab_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    x0: Any | None = None,
    tol: float,
    max_iter: int,
):
    """Solve a square linear system with lineax when available.

    This is kept as an optional feature-gated path so we can benchmark whether
    lineax materially improves the residual adjoint solve before depending on it
    more broadly.
    """
    if lx is None:
        return None, False, {}

    b = jnp.asarray(b)
    input_structure = jax.ShapeDtypeStruct(tuple(b.shape), b.dtype)
    operator = lx.FunctionLinearOperator(matvec, input_structure)
    options = {}
    if x0 is not None:
        options["y0"] = jnp.asarray(x0)
    solution = lx.linear_solve(
        operator,
        b,
        solver=lx.BiCGStab(rtol=float(tol), atol=0.0, max_steps=int(max_iter)),
        options=options,
        throw=False,
    )
    value = jnp.asarray(solution.value)
    try:
        success = bool(np.all(np.isfinite(np.asarray(jax.device_get(value)))))
    except Exception:
        success = False
    return value, success, getattr(solution, "stats", {})


def _linear_map_jacobian_columns(
    linear_map: Callable[[Any], Any],
    *,
    input_size: int,
    output_size: int,
    dtype: Any,
    chunk_size: int,
):
    """Build a dense Jacobian by batching JVP columns in chunks.

    This mirrors DESC's chunked Jacobian construction pattern: apply the same
    linearized map to blocks of basis directions instead of tracing a separate
    Jacobian transform for every column.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    eye_idx = jnp.arange(int(input_size), dtype=jnp.int32)
    chunks = []
    for start in range(0, int(input_size), int(chunk_size)):
        stop = min(start + int(chunk_size), int(input_size))
        rows = jnp.arange(start, stop, dtype=jnp.int32)
        basis = (rows[:, None] == eye_idx[None, :]).astype(dtype)
        chunk = jax.vmap(linear_map)(basis).T
        chunk = jnp.reshape(chunk, (int(output_size), stop - start))
        chunks.append(chunk)
    return jnp.concatenate(chunks, axis=1)


def _stellsym_feasible_indices_np(static, *, idx00: int | None, mask_lambda_axis: bool = True):
    """NumPy flat indices for feasible lasym=False coefficients."""
    ns = int(static.cfg.ns)
    K = int(static.modes.m.shape[0])
    m = np.asarray(static.modes.m)

    rz_mask = np.ones((ns, K), dtype=bool)
    rz_mask[-1, :] = False
    rz_mask[0, :] = (m == 0)

    lam_mask = np.ones((ns, K), dtype=bool)
    if bool(mask_lambda_axis):
        lam_mask[0, :] = False
    if idx00 is not None:
        lam_mask[:, int(idx00)] = False

    return np.flatnonzero(rz_mask.reshape(-1)), np.flatnonzero(lam_mask.reshape(-1)), ns, K


def _stellsym_feasible_indices(static, *, idx00: int | None, mask_lambda_axis: bool = True):
    """Flat indices for feasible lasym=False coefficients."""
    rz_idx, lam_idx, ns, K = _stellsym_feasible_indices_np(static, idx00=idx00, mask_lambda_axis=mask_lambda_axis)
    return jnp.asarray(rz_idx, dtype=jnp.int32), jnp.asarray(lam_idx, dtype=jnp.int32), ns, K


def _pack_stellsym_feasible_state(state: VMECState, *, rz_idx, lam_idx):
    """Pack only feasible stellarator-symmetric coefficients."""
    return jnp.concatenate(
        [
            jnp.take(jnp.ravel(jnp.asarray(state.Rcos)), rz_idx),
            jnp.take(jnp.ravel(jnp.asarray(state.Zsin)), rz_idx),
            jnp.take(jnp.ravel(jnp.asarray(state.Lsin)), lam_idx),
        ],
        axis=0,
    )


def _update_stellsym_feasible_state(state: VMECState, x, *, rz_idx, lam_idx, ns: int, K: int):
    """Update feasible lasym=False coefficients, leaving constrained entries unchanged."""
    x = jnp.asarray(x)
    n_rz = int(rz_idx.shape[0])
    n_l = int(lam_idx.shape[0])

    Rcos = (
        jnp.ravel(jnp.asarray(state.Rcos))
        .at[rz_idx]
        .set(x[:n_rz], indices_are_sorted=True, unique_indices=True)
        .reshape((ns, K))
    )
    Zsin = (
        jnp.ravel(jnp.asarray(state.Zsin))
        .at[rz_idx]
        .set(x[n_rz : 2 * n_rz], indices_are_sorted=True, unique_indices=True)
        .reshape((ns, K))
    )
    Lsin = (
        jnp.ravel(jnp.asarray(state.Lsin))
        .at[lam_idx]
        .set(x[2 * n_rz : 2 * n_rz + n_l], indices_are_sorted=True, unique_indices=True)
        .reshape((ns, K))
    )
    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=jnp.asarray(state.Rsin),
        Zcos=jnp.asarray(state.Zcos),
        Zsin=Zsin,
        Lcos=jnp.asarray(state.Lcos),
        Lsin=Lsin,
    )


def _stellsym_reduced_z_indices(*, rz_idx, K: int, idx00: int | None):
    """Flat Zsin indices that remain active after dropping dead (m,n)=(0,0) rows."""
    rz_idx_np = np.asarray(rz_idx, dtype=np.int32)
    z_idx_np = np.array(rz_idx_np, copy=True)
    if idx00 is not None:
        z_idx_np = z_idx_np[(rz_idx_np % int(K)) != int(idx00)]
    return jnp.asarray(z_idx_np, dtype=jnp.int32)


def _stellsym_lambda_mn_indices(static, *, idx00: int | None, mask_lambda_axis: bool = True):
    """Independent stellarator-symmetric lambda coordinates in VMEC (m,n>=0) sin storage."""
    from .vmec_parity import signed_maps_from_modes

    ns = int(static.cfg.ns)
    maps = signed_maps_from_modes(static.modes)

    sc_mask = np.ones((ns, maps.mpol, maps.nrange), dtype=bool)
    cs_mask = np.ones((ns, maps.mpol, maps.nrange), dtype=bool)
    if bool(mask_lambda_axis):
        sc_mask[0, :, :] = False
        cs_mask[0, :, :] = False

    # For stellarator-symmetric lambda the m=0,n>0 branch lives in cs only,
    # while n=0 lives in sc only.
    sc_mask[:, 0, 1:] = False
    cs_mask[:, :, 0] = False
    if idx00 is not None:
        sc_mask[:, 0, 0] = False

    return (
        jnp.asarray(np.flatnonzero(sc_mask.reshape(-1)), dtype=jnp.int32),
        jnp.asarray(np.flatnonzero(cs_mask.reshape(-1)), dtype=jnp.int32),
        maps,
    )


def _pack_stellsym_reduced_state(
    state: VMECState,
    *,
    rz_idx,
    z_idx,
    lam_sc_idx,
    lam_cs_idx,
    lam_maps,
):
    """Pack reduced lasym=False coordinates using VMEC lambda mn-sin storage."""
    from .vmec_parity import _signed_to_mn_sin_cached

    lam_sc, lam_cs = _signed_to_mn_sin_cached(jnp.asarray(state.Lsin), maps=lam_maps)
    return jnp.concatenate(
        [
            jnp.take(jnp.ravel(jnp.asarray(state.Rcos)), rz_idx),
            jnp.take(jnp.ravel(jnp.asarray(state.Zsin)), z_idx),
            jnp.take(jnp.ravel(jnp.asarray(lam_sc)), lam_sc_idx),
            jnp.take(jnp.ravel(jnp.asarray(lam_cs)), lam_cs_idx),
        ],
        axis=0,
    )


def _update_stellsym_reduced_state(
    state: VMECState,
    x,
    *,
    rz_idx,
    z_idx,
    lam_sc_idx,
    lam_cs_idx,
    lam_maps,
    ns: int,
    K: int,
):
    """Update reduced lasym=False coordinates in VMEC lambda mn-sin storage."""
    from .vmec_parity import _mn_sin_to_signed_cached, _signed_to_mn_sin_cached

    x = jnp.asarray(x)
    n_rz = int(rz_idx.shape[0])
    n_z = int(z_idx.shape[0])
    n_sc = int(lam_sc_idx.shape[0])
    n_cs = int(lam_cs_idx.shape[0])

    Rcos = (
        jnp.ravel(jnp.asarray(state.Rcos))
        .at[rz_idx]
        .set(x[:n_rz], indices_are_sorted=True, unique_indices=True)
        .reshape((ns, K))
    )
    Zsin = (
        jnp.ravel(jnp.asarray(state.Zsin))
        .at[z_idx]
        .set(x[n_rz : n_rz + n_z], indices_are_sorted=True, unique_indices=True)
        .reshape((ns, K))
    )

    lam_sc0, lam_cs0 = _signed_to_mn_sin_cached(jnp.asarray(state.Lsin), maps=lam_maps)
    lam_sc = (
        jnp.ravel(jnp.asarray(lam_sc0))
        .at[lam_sc_idx]
        .set(x[n_rz + n_z : n_rz + n_z + n_sc], indices_are_sorted=True, unique_indices=True)
        .reshape((ns, lam_maps.mpol, lam_maps.nrange))
    )
    lam_cs = (
        jnp.ravel(jnp.asarray(lam_cs0))
        .at[lam_cs_idx]
        .set(
            x[n_rz + n_z + n_sc : n_rz + n_z + n_sc + n_cs],
            indices_are_sorted=True,
            unique_indices=True,
        )
        .reshape((ns, lam_maps.mpol, lam_maps.nrange))
    )
    Lsin = _mn_sin_to_signed_cached(lam_sc, lam_cs, maps=lam_maps, ncoeff=K)
    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=jnp.asarray(state.Rsin),
        Zcos=jnp.asarray(state.Zcos),
        Zsin=Zsin,
        Lcos=jnp.asarray(state.Lcos),
        Lsin=Lsin,
    )


def _reduced_lsin_sc_m1n0_weights(x_active, *, active_lambda_start: int, lam_sc_idx, lam_maps):
    """Optional branch-selection weights for the reduced Lsin_sc (m=1,n=0) block."""
    weight_env = os.getenv("VMEC_JAX_REDUCED_LSIN_SC_M1N0_WEIGHT", "").strip()
    if weight_env == "":
        return None
    weight = float(weight_env)
    if weight <= 0.0:
        raise ValueError(f"VMEC_JAX_REDUCED_LSIN_SC_M1N0_WEIGHT must be positive, got {weight}")

    per_surface = int(lam_maps.mpol) * int(lam_maps.nrange)
    lam_sc_idx_arr = jnp.asarray(lam_sc_idx, dtype=jnp.int32)
    rem = lam_sc_idx_arr % per_surface
    m_local = rem // int(lam_maps.nrange)
    n_local = rem % int(lam_maps.nrange)
    weights = jnp.ones_like(jnp.asarray(x_active))
    n_sc = int(lam_sc_idx_arr.shape[0])
    sc_weights = jnp.where(
        (m_local == 1) & (n_local == 0),
        jnp.asarray(weight, dtype=weights.dtype),
        jnp.asarray(1.0, dtype=weights.dtype),
    )
    return weights.at[active_lambda_start : active_lambda_start + n_sc].set(sc_weights)


def _stellsym_structural_active_keep_indices(*, rz_idx, lam_idx, K: int, idx00: int | None):
    """Packed active-column keep indices for the reduced lasym=False adjoint.

    After projecting the residual onto structurally nonzero Tomnsps rows, the
    only remaining dead active columns in the simplified fixed-boundary
    lasym=False path are the Zsin (m,n)=(0,0) coefficients across radius.
    Dropping them makes the reduced system square on the QH case.
    """
    rz_idx = np.asarray(rz_idx, dtype=np.int32)
    lam_idx = np.asarray(lam_idx, dtype=np.int32)
    n_rz = int(rz_idx.shape[0])
    n_l = int(lam_idx.shape[0])
    rz_keep = np.arange(n_rz, dtype=np.int32)
    z_keep = np.arange(n_rz, dtype=np.int32)
    if idx00 is not None:
        rz_mod = rz_idx % int(K)
        z_keep = z_keep[rz_mod != int(idx00)]
    keep = np.concatenate(
        [
            rz_keep,
            n_rz + z_keep,
            2 * n_rz + np.arange(n_l, dtype=np.int32),
        ]
    )
    return jnp.asarray(keep, dtype=jnp.int32)


def solve_lambda_state_implicit(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 60,
    step_size: float = 0.05,
    grad_tol: float | None = None,
    bt_factor: float = 0.5,
    max_backtracks: int = 12,
    implicit: ImplicitLambdaOptions | None = None,
) -> VMECState:
    """Solve lambda with a custom VJP that uses implicit differentiation.

    Notes
    -----
    - This wrapper intentionally treats ``state0`` and ``static`` as *constants*
      for differentiation purposes. Gradients are provided w.r.t. ``phipf``,
      ``chipf``, and ``lamscale``.
    - The backward pass solves a damped linear system involving the Hessian of
      ``wb(L)`` w.r.t. lambda coefficients using conjugate gradients and Hessian-
      vector products computed via ``jax.jvp``.
    """
    if not has_jax():
        raise ImportError("solve_lambda_state_implicit requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitLambdaOptions()

    state0_c = _stop_gradient_tree(state0)

    idx00 = _mode00_index(static.modes)
    nfp = int(static.cfg.nfp)

    # Metric depends only on R/Z, so compute it once from the fixed geometry.
    g0 = eval_geom(state0_c, static)
    gtt = jnp.asarray(g0.g_tt)
    gtp = jnp.asarray(g0.g_tp)
    gpp = jnp.asarray(g0.g_pp)
    sqrtg_use = jnp.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    signgs_i = int(signgs)

    def _wb_from_L(Lcos, Lsin, phipf, chipf, lamscale):
        lam_u = eval_fourier_dtheta(Lcos, Lsin, static.basis, coeffs_internal=True)
        lam_v = eval_fourier_dzeta_phys(Lcos, Lsin, static.basis, coeffs_internal=True) / nfp
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs_i,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs_i * sqrtg_use
        E_total = jnp.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    def _solve(phipf, chipf, lamscale):
        # Forward solve uses the existing robust optimizer (not differentiated through).
        res = solve_lambda_gd(
            state0_c,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs_i,
            lamscale=lamscale,
            sqrtg=sqrtg_use,
            max_iter=int(max_iter),
            step_size=float(step_size),
            grad_tol=None if grad_tol is None else float(grad_tol),
            bt_factor=float(bt_factor),
            max_backtracks=int(max_backtracks),
        )
        return res.state

    @jax.custom_vjp
    def _solve_cust(phipf, chipf, lamscale):
        return _solve(phipf, chipf, lamscale)

    def fwd(phipf, chipf, lamscale):
        st = _solve(phipf, chipf, lamscale)
        return st, (jnp.asarray(st.Lcos), jnp.asarray(st.Lsin), jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(lamscale))

    def bwd(residual, ct_state):
        Lcos_star, Lsin_star, phipf_star, chipf_star, lamscale_star = residual
        ns, K = Lcos_star.shape

        # Gauge mask: exclude (m,n)=(0,0) everywhere.
        mask = jnp.ones((ns, K), dtype=Lcos_star.dtype)
        if idx00 is not None:
            mask = mask.at[:, int(idx00)].set(0.0)

        def grad_L_flat(Lcos, Lsin, phipf, chipf, lamscale):
            gcos, gsin = jax.grad(_wb_from_L, argnums=(0, 1))(Lcos, Lsin, phipf, chipf, lamscale)
            gcos = gcos * mask
            gsin = gsin * mask
            return _flatten_L(gcos, gsin)

        x_star = _flatten_L(Lcos_star, Lsin_star)

        def Hvp(u_flat):
            ucos, usin = _unflatten_L(u_flat, shape=(ns, K))
            ucos = ucos * mask
            usin = usin * mask

            def grad_pair(Lcos, Lsin):
                gcos, gsin = jax.grad(_wb_from_L, argnums=(0, 1))(Lcos, Lsin, phipf_star, chipf_star, lamscale_star)
                gcos = gcos * mask
                gsin = gsin * mask
                return gcos, gsin

            (_gcos, _gsin), (tcos, tsin) = jax.jvp(grad_pair, (Lcos_star, Lsin_star), (ucos, usin))
            t = _flatten_L(tcos * mask, tsin * mask)
            return t + jnp.asarray(float(implicit.damping), dtype=t.dtype) * u_flat

        # Right-hand side is cotangent w.r.t. the output state's lambda coefficients.
        ct_Lcos = jnp.asarray(ct_state.Lcos) * mask
        ct_Lsin = jnp.asarray(ct_state.Lsin) * mask
        b = _flatten_L(ct_Lcos, ct_Lsin)

        v = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))

        # Gradient w.r.t. parameters: dL/dp = - v^T (∂/∂p grad_L).
        def F_params(phipf, chipf, lamscale):
            return grad_L_flat(Lcos_star, Lsin_star, phipf, chipf, lamscale)

        (_out, vjp_fun) = jax.vjp(F_params, phipf_star, chipf_star, lamscale_star)
        dphipf, dchipf, dlamscale = vjp_fun(v)
        return (-dphipf, -dchipf, -dlamscale)

    _solve_cust.defvjp(fwd, bwd)

    return _solve_cust(jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(lamscale))


def solve_fixed_boundary_state_implicit(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    pressure,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    solver: str = "lbfgs",
    max_iter: int = 25,
    step_size: float = 5e-3,
    history_size: int = 10,
    grad_tol: float | None = None,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    implicit_converge_tol: float | None = None,
    implicit_zero_unconverged: bool = True,
    implicit: ImplicitFixedBoundaryOptions | None = None,
) -> VMECState:
    """Fixed-boundary solve with a custom VJP using implicit differentiation.

    This is a building block: it returns an equilibrium state while
    exposing *implicit* gradients w.r.t. the 1D profiles/fluxes.

    Differentiable inputs (by design)
    ---------------------------------
    - ``phipf(s)``, ``chipf(s)``, ``pressure(s)``, and ``lamscale``.
    - Optional boundary edge coefficients (``edge_Rcos``/``edge_Rsin``/``edge_Zcos``/``edge_Zsin``).

    Notes
    -----
    - ``state0`` and ``static`` are treated as constants for differentiation.
    - The backward pass solves a damped linear system involving the Hessian of
      the total objective w.r.t. the (masked) Fourier coefficients using CG and
      Hessian-vector products computed via ``jax.jvp``.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_state_implicit requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitFixedBoundaryOptions()

    solver = str(solver).strip().lower()
    if solver not in ("gd", "lbfgs"):
        raise ValueError(f"solver must be 'gd' or 'lbfgs', got {solver!r}")

    state0_c = _stop_gradient_tree(state0)
    idx00 = _mode00_index(static.modes)

    signgs_i = int(signgs)
    nfp = int(static.cfg.nfp)
    gamma = float(gamma)
    jacobian_penalty = float(jacobian_penalty)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    edge_any = any(x is not None for x in (edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin))
    if edge_any and not all(x is not None for x in (edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)):
        raise ValueError("edge_Rcos/edge_Rsin/edge_Zcos/edge_Zsin must be provided together")

    edge_Rcos_use = jnp.asarray(edge_Rcos) if edge_any else jnp.asarray(state0_c.Rcos)[-1, :]
    edge_Rsin_use = jnp.asarray(edge_Rsin) if edge_any else jnp.asarray(state0_c.Rsin)[-1, :]
    edge_Zcos_use = jnp.asarray(edge_Zcos) if edge_any else jnp.asarray(state0_c.Zcos)[-1, :]
    edge_Zsin_use = jnp.asarray(edge_Zsin) if edge_any else jnp.asarray(state0_c.Zsin)[-1, :]

    def _objective(state: VMECState, phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        state_use = _enforce_fixed_boundary_and_axis(
            state,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_axis=True,
            enforce_edge=True,
            enforce_lambda_axis=True,
            idx00=idx00,
        )
        g = eval_geom(state_use, static)
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs_i, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs_i * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        w = wb + wp / (gamma - 1.0)
        # Softly enforce a consistent Jacobian sign away from the axis.
        jac2 = jac.at[0, :, :].set(0.0)
        neg = jnp.minimum(jac2, 0.0)
        penalty = jacobian_penalty * jnp.mean(neg * neg)
        return w + penalty

    def _grad_flat(state: VMECState, phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        g = jax.grad(_objective)(state, phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)
        g = _mask_grad_for_constraints(g, static, idx00=idx00)
        return pack_state(g)

    def _solve(phipf, chipf, pressure, lamscale, *, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        def _is_traced(x):
            return isinstance(x, jax.core.Tracer)

        traced = (
            _is_traced(phipf)
            or _is_traced(chipf)
            or _is_traced(pressure)
            or _is_traced(lamscale)
            or _is_traced(edge_Rcos)
            or _is_traced(edge_Rsin)
            or _is_traced(edge_Zcos)
            or _is_traced(edge_Zsin)
        )
        solver_use = solver
        if traced and solver_use != "gd":
            solver_use = "gd"

        if solver_use == "gd":
            res = solve_fixed_boundary_gd(
                state0_c,
                static,
                phipf=phipf,
                chipf=chipf,
                signgs=signgs_i,
                lamscale=lamscale,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                pressure=pressure,
                gamma=gamma,
                jacobian_penalty=jacobian_penalty,
                max_iter=int(max_iter),
                step_size=float(step_size),
                grad_tol=None if grad_tol is None else float(grad_tol),
                max_backtracks=int(max_backtracks),
                bt_factor=float(bt_factor),
                preconditioner=str(preconditioner),
                precond_exponent=float(precond_exponent),
                precond_radial_alpha=float(precond_radial_alpha),
                differentiable=traced,
                jit_grad=traced,
                verbose=False,
            )
        else:
            res = solve_fixed_boundary_lbfgs(
                state0_c,
                static,
                phipf=phipf,
                chipf=chipf,
                signgs=signgs_i,
                lamscale=lamscale,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                pressure=pressure,
                gamma=gamma,
                history_size=int(history_size),
                max_iter=int(max_iter),
                step_size=float(step_size),
                grad_tol=None if grad_tol is None else float(grad_tol),
                max_backtracks=int(max_backtracks),
                bt_factor=float(bt_factor),
                preconditioner=str(preconditioner),
                precond_exponent=float(precond_exponent),
                precond_radial_alpha=float(precond_radial_alpha),
                verbose=False,
            )
            if int(getattr(res, "n_iter", 0)) <= 0:
                # L-BFGS failed to find a decreasing step; fall back to a more
                # conservative GD run to keep implicit gradients meaningful.
                max_iter_fb = max(50, int(max_iter))
                step_size_fb = min(0.2, float(step_size) * 0.2)
                res = solve_fixed_boundary_gd(
                    state0_c,
                    static,
                    phipf=phipf,
                    chipf=chipf,
                    signgs=signgs_i,
                    lamscale=lamscale,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    pressure=pressure,
                    gamma=gamma,
                    jacobian_penalty=jacobian_penalty,
                    max_iter=max_iter_fb,
                    step_size=step_size_fb,
                    grad_tol=None if grad_tol is None else float(grad_tol),
                    max_backtracks=int(max_backtracks),
                    bt_factor=float(bt_factor),
                    preconditioner=str(preconditioner),
                    precond_exponent=float(precond_exponent),
                    precond_radial_alpha=float(precond_radial_alpha),
                    verbose=False,
                )
        grad_hist = getattr(res, "grad_rms_history", None)
        converged = False
        if grad_hist is not None and len(grad_hist) > 0:
            try:
                if implicit_converge_tol is not None:
                    tol_check = float(implicit_converge_tol)
                else:
                    tol_check = getattr(getattr(res, "diagnostics", {}), "get", lambda *_args, **_kwargs: None)("grad_tol")
                    tol_check = None if tol_check is None else float(tol_check)
                converged = bool((tol_check is not None) and (float(grad_hist[-1]) < float(tol_check)))
            except Exception:
                converged = False
        return res.state, converged

    @jax.custom_vjp
    def _solve_cust(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        return _solve(
            phipf,
            chipf,
            pressure,
            lamscale,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )[0]

    def fwd(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        st, converged = _solve(
            phipf,
            chipf,
            pressure,
            lamscale,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )
        return st, (
            _stop_gradient_tree(st),
            jnp.asarray(phipf),
            jnp.asarray(chipf),
            jnp.asarray(pressure),
            jnp.asarray(lamscale),
            jnp.asarray(edge_Rcos),
            jnp.asarray(edge_Rsin),
            jnp.asarray(edge_Zcos),
            jnp.asarray(edge_Zsin),
            bool(edge_any),
            bool(converged),
        )

    def bwd(residual, ct_state):
        (
            st_star,
            phipf_star,
            chipf_star,
            pressure_star,
            lamscale_star,
            edge_Rcos_star,
            edge_Rsin_star,
            edge_Zcos_star,
            edge_Zsin_star,
            edge_active,
            converged,
        ) = residual
        if (not bool(converged)) and bool(implicit_zero_unconverged):
            z = jnp.zeros_like(jnp.asarray(phipf_star))
            zc = jnp.zeros_like(jnp.asarray(chipf_star))
            zp = jnp.zeros_like(jnp.asarray(pressure_star))
            zl = jnp.zeros_like(jnp.asarray(lamscale_star))
            zr = jnp.zeros_like(jnp.asarray(edge_Rcos_star))
            zs = jnp.zeros_like(jnp.asarray(edge_Rsin_star))
            zc2 = jnp.zeros_like(jnp.asarray(edge_Zcos_star))
            zz = jnp.zeros_like(jnp.asarray(edge_Zsin_star))
            return (z, zc, zp, zl, zr, zs, zc2, zz)
        layout = st_star.layout

        ct_state_full = ct_state
        ct_state = _mask_grad_for_constraints(ct_state, static, idx00=idx00)
        b = pack_state(ct_state)

        def Hvp(u_flat):
            u_state = unpack_state(u_flat, layout)
            u_state = _mask_grad_for_constraints(u_state, static, idx00=idx00)
            _, hvp = jax.jvp(
                lambda st: _grad_flat(
                    st,
                    phipf_star,
                    chipf_star,
                    pressure_star,
                    lamscale_star,
                    edge_Rcos_star,
                    edge_Rsin_star,
                    edge_Zcos_star,
                    edge_Zsin_star,
                ),
                (st_star,),
                (u_state,),
            )
            return hvp + jnp.asarray(float(implicit.damping), dtype=hvp.dtype) * u_flat

        v = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))

        def F_params(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
            return _grad_flat(
                st_star,
                phipf,
                chipf,
                pressure,
                lamscale,
                edge_Rcos,
                edge_Rsin,
                edge_Zcos,
                edge_Zsin,
            )

        (_out, vjp_fun) = jax.vjp(
            F_params,
            phipf_star,
            chipf_star,
            pressure_star,
            lamscale_star,
            edge_Rcos_star,
            edge_Rsin_star,
            edge_Zcos_star,
            edge_Zsin_star,
        )
        dphipf, dchipf, dpressure, dlamscale, dRcos, dRsin, dZcos, dZsin = vjp_fun(v)
        # Direct dependence of the output state on boundary parameters.
        ct_edge_Rcos = jnp.asarray(ct_state_full.Rcos)[-1, :]
        ct_edge_Rsin = jnp.asarray(ct_state_full.Rsin)[-1, :]
        ct_edge_Zcos = jnp.asarray(ct_state_full.Zcos)[-1, :]
        ct_edge_Zsin = jnp.asarray(ct_state_full.Zsin)[-1, :]
        dRcos = dRcos + ct_edge_Rcos
        dRsin = dRsin + ct_edge_Rsin
        dZcos = dZcos + ct_edge_Zcos
        dZsin = dZsin + ct_edge_Zsin
        if not bool(edge_active):
            dRcos = jnp.zeros_like(dRcos)
            dRsin = jnp.zeros_like(dRsin)
            dZcos = jnp.zeros_like(dZcos)
            dZsin = jnp.zeros_like(dZsin)
        return (-dphipf, -dchipf, -dpressure, -dlamscale, -dRcos, -dRsin, -dZcos, -dZsin)

    _solve_cust.defvjp(fwd, bwd)

    return _solve_cust(
        jnp.asarray(phipf),
        jnp.asarray(chipf),
        jnp.asarray(pressure),
        jnp.asarray(lamscale),
        jnp.asarray(edge_Rcos_use),
        jnp.asarray(edge_Rsin_use),
        jnp.asarray(edge_Zcos_use),
        jnp.asarray(edge_Zsin_use),
    )


def solve_fixed_boundary_state_implicit_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    state0_host: VMECState | None = None,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit: ImplicitFixedBoundaryOptions | None = None,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
):
    """Implicitly differentiate a VMEC residual fixed-point solve.

    The forward solve uses ``solve_fixed_boundary_residual_iter`` with the
    VMEC2000-style control path. The backward pass differentiates the VMEC
    residual vector itself rather than an auxiliary energy objective.

    ``implicit.residual_adjoint_mode='auto'`` selects a cheaper reduced-
    coordinate adjoint for the common ``lasym=False`` path when the residual
    dimension matches the active state dimension, falling back to the legacy
    normal-equation CG solve otherwise.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_state_implicit_vmec_residual requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitFixedBoundaryOptions()
    state0_c = _stop_gradient_tree(state0)
    if state0_host is None:
        state0_host = VMECState(
            layout=state0_c.layout,
            Rcos=np.asarray(jax.device_get(state0_c.Rcos)),
            Rsin=np.asarray(jax.device_get(state0_c.Rsin)),
            Zcos=np.asarray(jax.device_get(state0_c.Zcos)),
            Zsin=np.asarray(jax.device_get(state0_c.Zsin)),
            Lcos=np.asarray(jax.device_get(state0_c.Lcos)),
            Lsin=np.asarray(jax.device_get(state0_c.Lsin)),
        )
    idx00 = _mode00_index(static.modes)
    signgs_i = int(signgs)
    scan_forward_env = os.getenv("VMEC_JAX_ENABLE_EXPERIMENTAL_SCAN_FORWARD", "").strip().lower()
    enable_experimental_scan_forward = scan_forward_env not in ("", "0", "false", "no")

    edge_Rcos_use = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0_c.Rcos)[-1, :]
    edge_Rsin_use = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0_c.Rsin)[-1, :]
    edge_Zcos_use = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0_c.Zcos)[-1, :]
    edge_Zsin_use = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0_c.Zsin)[-1, :]

    from .boundary import BoundaryCoeffs, boundary_from_indata
    from .init_guess import initial_guess_from_boundary
    from .preconditioner_1d_jax import (
        lambda_preconditioner_cached,
    )
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import TomnspsRZL, vmec_trig_tables

    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs_i)
    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)
    chipf_wout = jnp.asarray(flux.chipf)

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )
    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs_i)
    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=jnp.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=signgs_i,
        flux_is_internal=True,
    )
    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs_i,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
        phipf_internal=phipf_internal,
        chipf_internal=chipf_internal,
        chips_eff=chips_eff,
    )
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    apply_lforbal = bool(indata.get_bool("LFORBAL", False))
    mask_pack = getattr(static, "tomnsps_masks", None)
    stellsym_residual_projector = None
    stellsym_active_keep_idx = None
    if (not bool(static.cfg.lasym)) and (mask_pack is not None):
        ns_mask = int(static.cfg.ns)
        mpol_mask = int(static.cfg.mpol)
        ntor1_mask = int(static.cfg.ntor) + 1
        mask_rz_np = np.broadcast_to(np.asarray(mask_pack.mask_rz) > 0, (ns_mask, mpol_mask, ntor1_mask))
        mask_l_np = np.broadcast_to(np.asarray(mask_pack.mask_l) > 0, (ns_mask, mpol_mask, ntor1_mask))
        m0_mask = np.broadcast_to((np.arange(mpol_mask)[None, :, None] == 0), (ns_mask, mpol_mask, ntor1_mask))
        n0_mask = np.broadcast_to((np.arange(ntor1_mask)[None, None, :] == 0), (ns_mask, mpol_mask, ntor1_mask))
        stellsym_residual_projector = {
            "frcc": jnp.asarray(np.flatnonzero(mask_rz_np.reshape(-1)), dtype=jnp.int32),
            "fzsc": jnp.asarray(np.flatnonzero((mask_rz_np & ~m0_mask).reshape(-1)), dtype=jnp.int32),
            "flsc": jnp.asarray(np.flatnonzero((mask_l_np & ~m0_mask).reshape(-1)), dtype=jnp.int32),
            "frss": jnp.asarray(np.flatnonzero((mask_rz_np & ~m0_mask & ~n0_mask).reshape(-1)), dtype=jnp.int32),
            "fzcs": jnp.asarray(np.flatnonzero((mask_rz_np & ~n0_mask).reshape(-1)), dtype=jnp.int32),
            "flcs": jnp.asarray(np.flatnonzero((mask_l_np & ~n0_mask).reshape(-1)), dtype=jnp.int32),
        }
        rz_idx_np, lam_idx_np, _ns_active_tmp, K_active_tmp = _stellsym_feasible_indices_np(
            static,
            idx00=idx00,
            mask_lambda_axis=True,
        )
        stellsym_active_keep_idx = _stellsym_structural_active_keep_indices(
            rz_idx=np.asarray(rz_idx_np),
            lam_idx=np.asarray(lam_idx_np),
            K=int(K_active_tmp),
            idx00=idx00,
        )

    def _pack_residual_parts(parts, projector=None):
        packed = []
        for name, arr in parts:
            flat = jnp.ravel(jnp.asarray(arr))
            if projector is not None:
                keep = projector.get(name)
                if keep is not None:
                    flat = jnp.take(flat, keep)
            packed.append(flat)
        return jnp.concatenate(packed, axis=0)

    def _boundary_state_edge_rows(eRcos, eRsin, eZcos, eZsin):
        boundary_state = initial_guess_from_boundary(
            static,
            BoundaryCoeffs(
                R_cos=jnp.asarray(eRcos),
                R_sin=jnp.asarray(eRsin),
                Z_cos=jnp.asarray(eZcos),
                Z_sin=jnp.asarray(eZsin),
            ),
            indata,
            dtype=jnp.asarray(state0_c.Rcos).dtype,
            vmec_project=True,
        )
        return (
            jnp.asarray(boundary_state.Rcos)[-1, :],
            jnp.asarray(boundary_state.Rsin)[-1, :],
            jnp.asarray(boundary_state.Zcos)[-1, :],
            jnp.asarray(boundary_state.Zsin)[-1, :],
        )

    def _enforce_state(st, eRcos, eRsin, eZcos, eZsin):
        sRcos, sRsin, sZcos, sZsin = _boundary_state_edge_rows(eRcos, eRsin, eZcos, eZsin)
        return _enforce_fixed_boundary_and_axis(
            st,
            static,
            edge_Rcos=sRcos,
            edge_Rsin=sRsin,
            edge_Zcos=sZcos,
            edge_Zsin=sZsin,
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    def _project_state(st):
        return _mask_grad_for_constraints(st, static, idx00=idx00, mask_lambda_axis=True)

    def _zero_edge_rz(a):
        a = jnp.asarray(a)
        if a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    def _residual_vec(state, zero_m1_zforce, eRcos, eRsin, eZcos, eZsin, *, project_stellsym: bool = False):
        residual_start = time.perf_counter()
        state = _enforce_state(state, eRcos, eRsin, eZcos, eZsin)
        forces_start = time.perf_counter()
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=float(indata.get_float("TCON0", 1.0)),
            use_vmec_synthesis=True,
            trig=trig,
        )
        _vmec_residual_profile_log("forces_done", forces_start)
        tomnsps_start = time.perf_counter()
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        _vmec_residual_profile_log("tomnsps_done", tomnsps_start)
        post_start = time.perf_counter()
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1_zforce)
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
        frzl = TomnspsRZL(
            frcc=_zero_edge_rz(frzl.frcc),
            frss=_zero_edge_rz(frzl.frss) if frzl.frss is not None else None,
            fzsc=_zero_edge_rz(frzl.fzsc),
            fzcs=_zero_edge_rz(frzl.fzcs) if frzl.fzcs is not None else None,
            flsc=frzl.flsc,
            flcs=frzl.flcs,
            frsc=_zero_edge_rz(getattr(frzl, "frsc", None)) if getattr(frzl, "frsc", None) is not None else None,
            frcs=_zero_edge_rz(getattr(frzl, "frcs", None)) if getattr(frzl, "frcs", None) is not None else None,
            fzcc=_zero_edge_rz(getattr(frzl, "fzcc", None)) if getattr(frzl, "fzcc", None) is not None else None,
            fzss=_zero_edge_rz(getattr(frzl, "fzss", None)) if getattr(frzl, "fzss", None) is not None else None,
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs_i)
        scale_rz = jnp.sqrt(norms.r1 * norms.fnorm)
        scale_l = jnp.sqrt(norms.fnormL)
        lam_prec = lambda_preconditioner_cached(
            bc=k.bc,
            trig=trig,
            s=s,
            cfg=static.cfg,
        )
        parts = [
            ("frcc", scale_rz * frzl.frcc),
            ("fzsc", scale_rz * frzl.fzsc),
            ("flsc", scale_l * jnp.asarray(frzl.flsc) * jnp.asarray(lam_prec)),
        ]
        if frzl.frss is not None:
            parts.append(("frss", scale_rz * frzl.frss))
        if frzl.fzcs is not None:
            parts.append(("fzcs", scale_rz * frzl.fzcs))
        if frzl.flcs is not None:
            parts.append(("flcs", scale_l * jnp.asarray(frzl.flcs) * jnp.asarray(lam_prec)))
        for name in ["frsc", "fzcc", "flcc", "frcs", "fzss", "flss"]:
            arr = getattr(frzl, name, None)
            if arr is not None:
                scale = scale_l if name.startswith("fl") else scale_rz
                if name.startswith("fl"):
                    arr = jnp.asarray(arr) * jnp.asarray(lam_prec)
                parts.append((name, scale * arr))
        projector = stellsym_residual_projector if bool(project_stellsym) else None
        packed = _pack_residual_parts(parts, projector=projector)
        _vmec_residual_profile_log("postprocess_done", post_start)
        _vmec_residual_profile_log(
            "residual_done",
            residual_start,
            projected=bool(project_stellsym),
            output_size=int(np.prod(np.shape(packed))),
        )
        return packed

    def _stationarity_state(state, zero_m1_zforce, eRcos, eRsin, eZcos, eZsin, *, project_stellsym: bool = False):
        def _objective_from_state(st):
            residual = _residual_vec(
                st,
                zero_m1_zforce,
                eRcos,
                eRsin,
                eZcos,
                eZsin,
                project_stellsym=project_stellsym,
            )
            residual = jnp.asarray(residual)
            return 0.5 * jnp.sum(residual * residual)

        return _project_state(jax.grad(_objective_from_state)(state))

    resume_v_shape = (int(state0_c.Rcos.shape[0]), int(static.cfg.mpol), int(static.cfg.ntor) + 1)

    def _resume_output_shape():
        dtype = jnp.asarray(state0_c.Rcos).dtype
        return {
            "state_current_flat": jax.ShapeDtypeStruct((int(state0_c.layout.size),), dtype),
            "state_checkpoint_flat": jax.ShapeDtypeStruct((int(state0_c.layout.size),), dtype),
            "time_step": jax.ShapeDtypeStruct((), dtype),
            "inv_tau": jax.ShapeDtypeStruct((10,), dtype),
            "fsq_prev": jax.ShapeDtypeStruct((), dtype),
            "fsq0_prev": jax.ShapeDtypeStruct((), dtype),
            "flip_sign": jax.ShapeDtypeStruct((), dtype),
            "res0": jax.ShapeDtypeStruct((), dtype),
            "res1": jax.ShapeDtypeStruct((), dtype),
            "fsqz_prev": jax.ShapeDtypeStruct((), dtype),
            "vRcc": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vRss": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vZsc": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vZcs": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vLsc": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vLcs": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vRsc": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vRcs": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vZcc": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vZss": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vLcc": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "vLss": jax.ShapeDtypeStruct(resume_v_shape, dtype),
            "iter1": jax.ShapeDtypeStruct((), jnp.int32),
            "iter_offset": jax.ShapeDtypeStruct((), jnp.int32),
            "ijacob": jax.ShapeDtypeStruct((), jnp.int32),
            "bad_resets": jax.ShapeDtypeStruct((), jnp.int32),
            "bad_growth_streak": jax.ShapeDtypeStruct((), jnp.int32),
            "vmec2000_cache_valid": jax.ShapeDtypeStruct((), jnp.bool_),
            "force_bcovar_update": jax.ShapeDtypeStruct((), jnp.bool_),
        }

    def _pack_resume_for_output(res):
        resume = dict(getattr(res, "diagnostics", {}).get("resume_state", {}) or {})
        state_current = resume.get("state_current", res.state)
        state_checkpoint = resume.get("state_checkpoint", res.state)

        def _as_v(name):
            arr = resume.get(name, None)
            if arr is None:
                return jnp.zeros(resume_v_shape, dtype=jnp.asarray(state0_c.Rcos).dtype)
            return jnp.asarray(arr)

        def _as_i(name, default):
            return jnp.asarray(resume.get(name, default), dtype=jnp.int32)

        def _as_b(name, default):
            return jnp.asarray(resume.get(name, default), dtype=bool)

        out_dtype = jnp.asarray(state0_c.Rcos).dtype
        return {
            "state_current_flat": jnp.asarray(pack_state(state_current), dtype=out_dtype),
            "state_checkpoint_flat": jnp.asarray(pack_state(state_checkpoint), dtype=out_dtype),
            "time_step": jnp.asarray(resume.get("time_step", float(step_size)), dtype=out_dtype),
            "inv_tau": jnp.asarray(resume.get("inv_tau", np.zeros((10,), dtype=np.asarray(state0_host.Rcos).dtype)), dtype=out_dtype),
            "fsq_prev": jnp.asarray(resume.get("fsq_prev", 0.0), dtype=out_dtype),
            "fsq0_prev": jnp.asarray(resume.get("fsq0_prev", 0.0), dtype=out_dtype),
            "flip_sign": jnp.asarray(resume.get("flip_sign", 1.0), dtype=out_dtype),
            "res0": jnp.asarray(resume.get("res0", -1.0), dtype=out_dtype),
            "res1": jnp.asarray(resume.get("res1", -1.0), dtype=out_dtype),
            "fsqz_prev": jnp.asarray(resume.get("fsqz_prev", 1.0), dtype=out_dtype),
            "vRcc": _as_v("vRcc"),
            "vRss": _as_v("vRss"),
            "vZsc": _as_v("vZsc"),
            "vZcs": _as_v("vZcs"),
            "vLsc": _as_v("vLsc"),
            "vLcs": _as_v("vLcs"),
            "vRsc": _as_v("vRsc"),
            "vRcs": _as_v("vRcs"),
            "vZcc": _as_v("vZcc"),
            "vZss": _as_v("vZss"),
            "vLcc": _as_v("vLcc"),
            "vLss": _as_v("vLss"),
            "iter1": _as_i("iter1", 1),
            "iter_offset": _as_i("iter_offset", 0),
            "ijacob": _as_i("ijacob", 0),
            "bad_resets": _as_i("bad_resets", 0),
            "bad_growth_streak": _as_i("bad_growth_streak", 0),
            "vmec2000_cache_valid": _as_b("vmec2000_cache_valid", False),
            "force_bcovar_update": _as_b("force_bcovar_update", False),
        }

    def _resume_float_fields(resume_out):
        return {
            "state_current_flat": jnp.asarray(resume_out["state_current_flat"]),
            "state_checkpoint_flat": jnp.asarray(resume_out["state_checkpoint_flat"]),
            "time_step": jnp.asarray(resume_out["time_step"]),
            "inv_tau": jnp.asarray(resume_out["inv_tau"]),
            "fsq_prev": jnp.asarray(resume_out["fsq_prev"]),
            "fsq0_prev": jnp.asarray(resume_out["fsq0_prev"]),
            "flip_sign": jnp.asarray(resume_out["flip_sign"]),
            "res0": jnp.asarray(resume_out["res0"]),
            "res1": jnp.asarray(resume_out["res1"]),
            "fsqz_prev": jnp.asarray(resume_out["fsqz_prev"]),
            "vRcc": jnp.asarray(resume_out["vRcc"]),
            "vRss": jnp.asarray(resume_out["vRss"]),
            "vZsc": jnp.asarray(resume_out["vZsc"]),
            "vZcs": jnp.asarray(resume_out["vZcs"]),
            "vLsc": jnp.asarray(resume_out["vLsc"]),
            "vLcs": jnp.asarray(resume_out["vLcs"]),
            "vRsc": jnp.asarray(resume_out["vRsc"]),
            "vRcs": jnp.asarray(resume_out["vRcs"]),
            "vZcc": jnp.asarray(resume_out["vZcc"]),
            "vZss": jnp.asarray(resume_out["vZss"]),
            "vLcc": jnp.asarray(resume_out["vLcc"]),
            "vLss": jnp.asarray(resume_out["vLss"]),
        }

    def _resume_const_fields(resume_out):
        return {
            "iter1": jnp.asarray(resume_out["iter1"], dtype=jnp.int32),
            "iter_offset": jnp.asarray(resume_out["iter_offset"], dtype=jnp.int32),
            "ijacob": jnp.asarray(resume_out["ijacob"], dtype=jnp.int32),
            "bad_resets": jnp.asarray(resume_out["bad_resets"], dtype=jnp.int32),
            "bad_growth_streak": jnp.asarray(resume_out["bad_growth_streak"], dtype=jnp.int32),
            "vmec2000_cache_valid": jnp.asarray(resume_out["vmec2000_cache_valid"], dtype=bool),
            "force_bcovar_update": jnp.asarray(resume_out["force_bcovar_update"], dtype=bool),
        }

    def _resume_state_one_step(resume_float, resume_const, eRcos, eRsin, eZcos, eZsin):
        state_current = _enforce_state(unpack_state(jnp.asarray(resume_float["state_current_flat"]), state0_c.layout), eRcos, eRsin, eZcos, eZsin)
        state_checkpoint = _enforce_state(
            unpack_state(jnp.asarray(resume_float["state_checkpoint_flat"]), state0_c.layout),
            eRcos,
            eRsin,
            eZcos,
            eZsin,
        )
        resume_state = {
            "time_step": jnp.asarray(resume_float["time_step"]),
            "inv_tau": jnp.asarray(resume_float["inv_tau"]),
            "fsq_prev": jnp.asarray(resume_float["fsq_prev"]),
            "fsq0_prev": jnp.asarray(resume_float["fsq0_prev"]),
            "flip_sign": jnp.asarray(resume_float["flip_sign"]),
            "iter1": jnp.asarray(resume_const["iter1"], dtype=jnp.int32),
            "iter_offset": jnp.asarray(resume_const["iter_offset"], dtype=jnp.int32),
            "res0": jnp.asarray(resume_float["res0"]),
            "res1": jnp.asarray(resume_float["res1"]),
            "ijacob": jnp.asarray(resume_const["ijacob"], dtype=jnp.int32),
            "bad_resets": jnp.asarray(resume_const["bad_resets"], dtype=jnp.int32),
            "bad_growth_streak": jnp.asarray(resume_const["bad_growth_streak"], dtype=jnp.int32),
            "fsqz_prev": jnp.asarray(resume_float["fsqz_prev"]),
            "state_checkpoint": state_checkpoint,
            "vRcc": jnp.asarray(resume_float["vRcc"]),
            "vRss": jnp.asarray(resume_float["vRss"]),
            "vZsc": jnp.asarray(resume_float["vZsc"]),
            "vZcs": jnp.asarray(resume_float["vZcs"]),
            "vLsc": jnp.asarray(resume_float["vLsc"]),
            "vLcs": jnp.asarray(resume_float["vLcs"]),
            "vRsc": jnp.asarray(resume_float["vRsc"]),
            "vRcs": jnp.asarray(resume_float["vRcs"]),
            "vZcc": jnp.asarray(resume_float["vZcc"]),
            "vZss": jnp.asarray(resume_float["vZss"]),
            "vLcc": jnp.asarray(resume_float["vLcc"]),
            "vLss": jnp.asarray(resume_float["vLss"]),
            "vmec2000_cache_valid": jnp.asarray(resume_const["vmec2000_cache_valid"], dtype=bool),
            "force_bcovar_update": jnp.asarray(resume_const["force_bcovar_update"], dtype=bool),
        }
        res = solve_fixed_boundary_residual_iter(
            state_current,
            static,
            indata=indata,
            signgs=signgs_i,
            ftol=ftol,
            max_iter=1,
            step_size=float(step_size),
            vmec2000_control=True,
            reference_mode=False,
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=True,
            resume_state=resume_state,
            resume_state_mode="minimal",
        )
        return _resume_float_fields(_pack_resume_for_output(res))

    def _solve_host(eRcos, eRsin, eZcos, eZsin):
        eRcos_np = np.asarray(eRcos)
        eRsin_np = np.asarray(eRsin)
        eZcos_np = np.asarray(eZcos)
        eZsin_np = np.asarray(eZsin)
        boundary_host = BoundaryCoeffs(
            R_cos=np.array(eRcos_np, copy=True),
            R_sin=np.array(eRsin_np, copy=True),
            Z_cos=np.array(eZcos_np, copy=True),
            Z_sin=np.array(eZsin_np, copy=True),
        )
        state_init = initial_guess_from_boundary(
            static,
            boundary_host,
            indata,
            dtype=np.asarray(state0_host.Rcos).dtype,
            vmec_project=True,
        )
        geom0 = eval_geom(state_init, static)
        signgs0 = signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1)
        state_init = VMECState(
            layout=state_init.layout,
            Rcos=np.asarray(jax.device_get(state_init.Rcos)),
            Rsin=np.asarray(jax.device_get(state_init.Rsin)),
            Zcos=np.asarray(jax.device_get(state_init.Zcos)),
            Zsin=np.asarray(jax.device_get(state_init.Zsin)),
            Lcos=np.asarray(jax.device_get(state_init.Lcos)),
            Lsin=np.asarray(jax.device_get(state_init.Lsin)),
        )
        res = solve_fixed_boundary_residual_iter(
            state_init,
            static,
            indata=indata,
            signgs=int(signgs0),
            ftol=ftol,
            max_iter=int(max_iter),
            step_size=float(step_size),
            vmec2000_control=True,
            reference_mode=False,
            # Keep the callback/host primal path on the same strict VMEC2000
            # branch as the differentiable scan path. The extra backtracking and
            # update limiters leave the host solve materially off the residual
            # fixed point on the QH workload, which then corrupts the implicit
            # derivatives even when the local linearization is correct.
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=False,
        )
        fsqz_hist = np.asarray(getattr(res, "fsqz2_history", []), dtype=float)
        zero_m1 = 1.0 if (int(getattr(res, "n_iter", 0)) < 2 or (fsqz_hist.size > 0 and float(fsqz_hist[-1]) < 1.0e-6)) else 0.0
        return np.asarray(pack_state(res.state)), np.asarray(zero_m1, dtype=state0_host.Rcos.dtype), _pack_resume_for_output(res)

    def _solve_scan(eRcos, eRsin, eZcos, eZsin):
        boundary = BoundaryCoeffs(
            R_cos=jnp.asarray(eRcos),
            R_sin=jnp.asarray(eRsin),
            Z_cos=jnp.asarray(eZcos),
            Z_sin=jnp.asarray(eZsin),
        )
        state_init = initial_guess_from_boundary(
            static,
            boundary,
            indata,
            dtype=jnp.asarray(state0_c.Rcos).dtype,
            vmec_project=True,
        )
        res = solve_fixed_boundary_residual_iter(
            state_init,
            static,
            indata=indata,
            signgs=signgs_i,
            ftol=ftol,
            max_iter=int(max_iter),
            step_size=float(step_size),
            vmec2000_control=True,
            reference_mode=False,
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=True,
        )
        resume_state = getattr(res, "diagnostics", {}).get("resume_state", {})
        fsqz_prev = jnp.asarray(resume_state.get("fsqz_prev", 1.0), dtype=jnp.asarray(state0_c.Rcos).dtype)
        zero_m1 = jnp.where(
            jnp.logical_or(jnp.asarray(int(max_iter) < 2), fsqz_prev < 1.0e-6),
            jnp.asarray(1.0, dtype=jnp.asarray(state0_c.Rcos).dtype),
            jnp.asarray(0.0, dtype=jnp.asarray(state0_c.Rcos).dtype),
        )
        return res.state, zero_m1, _pack_resume_for_output(res)

    def _is_traced(*xs):
        return any(isinstance(x, jax.core.Tracer) for x in xs)

    def _solve(eRcos, eRsin, eZcos, eZsin):
        traced = _is_traced(eRcos, eRsin, eZcos, eZsin)
        forward_mode = str(getattr(implicit, "residual_forward_mode", "auto")).strip().lower()
        use_scan_forward = (forward_mode == "scan") and bool(enable_experimental_scan_forward)
        if use_scan_forward:
            return _solve_scan(eRcos, eRsin, eZcos, eZsin)
        if traced:
            out_shape = (
                jax.ShapeDtypeStruct((int(state0_c.layout.size),), jnp.asarray(state0_c.Rcos).dtype),
                jax.ShapeDtypeStruct((), jnp.asarray(state0_c.Rcos).dtype),
                _resume_output_shape(),
            )
            x_flat, zero_m1, resume_out = jax.pure_callback(_solve_host, out_shape, eRcos, eRsin, eZcos, eZsin)
            return unpack_state(x_flat, state0_c.layout), zero_m1, resume_out
        x_flat, zero_m1, resume_out = _solve_host(eRcos, eRsin, eZcos, eZsin)
        return unpack_state(jnp.asarray(x_flat), state0_c.layout), jnp.asarray(zero_m1), resume_out

    if not _is_traced(edge_Rcos_use, edge_Rsin_use, edge_Zcos_use, edge_Zsin_use):
        x_flat, _zero_m1, _resume_out = _solve_host(edge_Rcos_use, edge_Rsin_use, edge_Zcos_use, edge_Zsin_use)
        return unpack_state(jnp.asarray(x_flat), state0_c.layout)

    residual_tangent_mode = str(getattr(implicit, "residual_tangent_mode", "opaque")).strip().lower()

    def _state_tangent_from_boundary_tangent(
        st_star,
        zero_m1_star,
        resume_star,
        eRcos_star,
        eRsin_star,
        eZcos_star,
        eZsin_star,
        deRcos,
        deRsin,
        deZcos,
        deZsin,
    ):
        if bool(static.cfg.lasym):
            raise NotImplementedError(
                "residual_tangent_mode='linearize' currently supports only lasym=False residual solves"
            )

        tangent_mode = str(getattr(implicit, "residual_tangent_mode", "auto")).strip().lower()
        solver_map_env = os.getenv("VMEC_JAX_USE_SOLVER_MAP_TANGENT", "0").strip().lower()
        use_solver_map_tangent = solver_map_env not in ("", "0", "false", "no")
        solver_map_reference_state = None
        if use_solver_map_tangent:
            from jax.flatten_util import ravel_pytree
            from jax.scipy.sparse.linalg import bicgstab

            solver_map_steps_env = os.getenv("VMEC_JAX_SOLVER_MAP_STEPS", "").strip()
            solver_map_steps = max(1, int(solver_map_steps_env)) if solver_map_steps_env else 1

            resume_float_star = _resume_float_fields(resume_star)
            resume_const_star = _resume_const_fields(resume_star)

            def _resume_state_map(y_float):
                y_out = y_float
                for _ in range(int(solver_map_steps)):
                    y_out = _resume_state_one_step(
                        y_out,
                        resume_const_star,
                        eRcos_star,
                        eRsin_star,
                        eZcos_star,
                        eZsin_star,
                    )
                return y_out

            _, resume_jvp = jax.linearize(_resume_state_map, resume_float_star)
            rhs_resume = jax.jvp(
                lambda a, b, c, d: _resume_state_one_step(resume_float_star, resume_const_star, a, b, c, d),
                (eRcos_star, eRsin_star, eZcos_star, eZsin_star),
                (deRcos, deRsin, deZcos, deZsin),
            )[1]

            rhs_flat, unravel_resume = ravel_pytree(rhs_resume)

            def _resume_linear_op(v_flat):
                v_tree = unravel_resume(v_flat)
                gv_tree = resume_jvp(v_tree)
                gv_flat, _ = ravel_pytree(gv_tree)
                return v_flat - gv_flat

            def _resume_linear_solve(matvec, b_flat):
                sol_flat, info = bicgstab(
                    matvec,
                    b_flat,
                    tol=float(implicit.cg_tol),
                    atol=0.0,
                    maxiter=max(10, int(getattr(implicit, "cg_max_iter", 200))),
                )
                return jnp.where(jnp.isfinite(sol_flat), sol_flat, jnp.zeros_like(sol_flat))

            dy_flat = jax.lax.custom_linear_solve(
                _resume_linear_op,
                rhs_flat,
                solve=_resume_linear_solve,
                transpose_solve=_resume_linear_solve,
                symmetric=False,
            )
            dy_resume = unravel_resume(dy_flat)
            solver_map_reference_state = unpack_state(
                jnp.asarray(dy_resume["state_current_flat"]),
                state0_c.layout,
            )

        rz_idx_np, lam_idx_np, ns_active, K_active = _stellsym_feasible_indices_np(
            static,
            idx00=idx00,
            mask_lambda_axis=True,
        )
        rz_idx = jnp.asarray(rz_idx_np, dtype=jnp.int32)
        lam_idx = jnp.asarray(lam_idx_np, dtype=jnp.int32)
        st_active_ref = _stop_gradient_tree(st_star)
        if _vmec_keep_all_active_enabled():
            x_active_star_full = _pack_stellsym_feasible_state(st_active_ref, rz_idx=rz_idx, lam_idx=lam_idx)
            active_keep_idx = jnp.arange(int(x_active_star_full.shape[0]), dtype=jnp.int32)
            x_active_star = jnp.take(x_active_star_full, active_keep_idx)

            def stationarity_fun_active(x_active):
                x_active_full = x_active_star_full.at[active_keep_idx].set(
                    x_active,
                    indices_are_sorted=True,
                    unique_indices=True,
                )
                st_active = _update_stellsym_feasible_state(
                    st_active_ref,
                    x_active_full,
                    rz_idx=rz_idx,
                    lam_idx=lam_idx,
                    ns=ns_active,
                    K=K_active,
                )
                grad_state = _stationarity_state(
                    st_active,
                    zero_m1_star,
                    eRcos_star,
                    eRsin_star,
                    eZcos_star,
                    eZsin_star,
                )
                grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                return jnp.take(grad_active_full, active_keep_idx)

            def stationarity_params_active(a, b, c, d):
                grad_state = _stationarity_state(
                    st_star,
                    zero_m1_star,
                    a,
                    b,
                    c,
                    d,
                )
                grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                return jnp.take(grad_active_full, active_keep_idx)
        else:
            z_idx = _stellsym_reduced_z_indices(rz_idx=rz_idx_np, K=int(K_active), idx00=idx00)
            lam_sc_idx, lam_cs_idx, lam_maps = _stellsym_lambda_mn_indices(
                static,
                idx00=idx00,
                mask_lambda_axis=True,
            )
            active_lambda_start = int(rz_idx.shape[0]) + int(z_idx.shape[0])
            active_cs_start = active_lambda_start + int(lam_sc_idx.shape[0])
            x_active_star = _pack_stellsym_reduced_state(
                st_active_ref,
                rz_idx=rz_idx,
                z_idx=z_idx,
                lam_sc_idx=lam_sc_idx,
                lam_cs_idx=lam_cs_idx,
                lam_maps=lam_maps,
            )

            def stationarity_fun_active(x_active):
                st_active = _update_stellsym_reduced_state(
                    st_active_ref,
                    x_active,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                    ns=ns_active,
                    K=K_active,
                )
                grad_state = _stationarity_state(
                    st_active,
                    zero_m1_star,
                    eRcos_star,
                    eRsin_star,
                    eZcos_star,
                    eZsin_star,
                )
                return _pack_stellsym_reduced_state(
                    grad_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )

            def stationarity_params_active(a, b, c, d):
                grad_state = _stationarity_state(
                    st_star,
                    zero_m1_star,
                    a,
                    b,
                    c,
                    d,
                )
                return _pack_stellsym_reduced_state(
                    grad_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )

        stationarity_star_active, stationarity_jvp_active = jax.linearize(stationarity_fun_active, x_active_star)
        stationarity_vjp_active = jax.linear_transpose(stationarity_jvp_active, x_active_star)
        damping = jnp.asarray(float(implicit.damping), dtype=jnp.asarray(x_active_star).dtype)
        active_is_square = tuple(stationarity_star_active.shape) == tuple(x_active_star.shape)
        use_explicit_reduced_tangent = not _vmec_keep_all_active_enabled()

        def stationarity_jvp_active_damped(u_active):
            return stationarity_jvp_active(u_active) + damping * u_active

        boundary_tangent = jax.jvp(
            stationarity_params_active,
            (eRcos_star, eRsin_star, eZcos_star, eZsin_star),
            (deRcos, deRsin, deZcos, deZsin),
        )[1]
        rhs = jnp.asarray(boundary_tangent)
        dx_reference_active = None
        reference_keep_idx = None
        active_weights = None
        if solver_map_reference_state is not None:
            if _vmec_keep_all_active_enabled():
                dx_reference_active_full = -_pack_stellsym_feasible_state(
                    solver_map_reference_state,
                    rz_idx=rz_idx,
                    lam_idx=lam_idx,
                )
                dx_reference_active = jnp.take(dx_reference_active_full, active_keep_idx)
            else:
                dx_reference_active = -_pack_stellsym_reduced_state(
                    solver_map_reference_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )
                reference_mode = os.getenv("VMEC_JAX_SOLVER_MAP_REFERENCE_MODE", "full").strip().lower()
                if reference_mode == "lambda-only":
                    dx_reference_active = dx_reference_active.at[:active_lambda_start].set(
                        jnp.zeros((active_lambda_start,), dtype=dx_reference_active.dtype)
                    )
                elif reference_mode == "lambda-cs-m0-lock":
                    lam_cs_idx_np = np.asarray(lam_cs_idx, dtype=np.int32)
                    nrange = int(lam_maps.nrange)
                    mpol = int(lam_maps.mpol)
                    flat_per_surface = mpol * nrange
                    rem = lam_cs_idx_np % flat_per_surface
                    m_local = rem // nrange
                    n_local = rem % nrange
                    keep_local = np.flatnonzero((m_local == 0) & (n_local > 0))
                    reference_keep_idx = jnp.asarray(active_cs_start + keep_local, dtype=jnp.int32)
                    keep_mask = np.zeros((int(dx_reference_active.shape[0]),), dtype=bool)
                    keep_mask[np.asarray(reference_keep_idx, dtype=np.int32)] = True
                    dx_reference_active = jnp.where(
                        jnp.asarray(keep_mask),
                        dx_reference_active,
                        jnp.zeros_like(dx_reference_active),
                    )
        if use_explicit_reduced_tangent:
            active_weights = _reduced_lsin_sc_m1n0_weights(
                x_active_star,
                active_lambda_start=active_lambda_start,
                lam_sc_idx=lam_sc_idx,
                lam_maps=lam_maps,
            )

        dx_active = None
        if use_explicit_reduced_tangent:
            chunk_size = getattr(implicit, "jac_chunk_size", None)
            if chunk_size is None:
                chunk_size = _vmec_jac_chunk_size_override()
            if chunk_size is None:
                chunk_size = _default_vmec_jac_chunk_size(int(x_active_star.shape[0]))
            J_active = _linear_map_jacobian_columns(
                stationarity_jvp_active,
                input_size=int(x_active_star.shape[0]),
                output_size=int(np.prod(np.shape(stationarity_star_active))),
                dtype=jnp.asarray(x_active_star).dtype,
                chunk_size=int(chunk_size),
            )
            # The reduced stellarator-symmetric system is nearly singular on the
            # QH workload. A least-norm solve matches the primal solver branch
            # far better than a plain inverse or normal-equation solve.
            if dx_reference_active is not None:
                if reference_keep_idx is not None and int(reference_keep_idx.shape[0]) > 0:
                    fixed_vals = jnp.take(dx_reference_active, reference_keep_idx)
                    all_idx = np.arange(int(dx_reference_active.shape[0]), dtype=np.int32)
                    free_mask = np.ones((int(dx_reference_active.shape[0]),), dtype=bool)
                    free_mask[np.asarray(reference_keep_idx, dtype=np.int32)] = False
                    free_idx = jnp.asarray(all_idx[free_mask], dtype=jnp.int32)
                    rhs_correction = rhs - jnp.take(J_active, reference_keep_idx, axis=1) @ fixed_vals
                    dx_active = jnp.zeros_like(dx_reference_active)
                    dx_active = dx_active.at[reference_keep_idx].set(
                        fixed_vals,
                        indices_are_sorted=True,
                        unique_indices=True,
                    )
                    if int(free_idx.shape[0]) > 0:
                        free_weights = None if active_weights is None else jnp.take(active_weights, free_idx)
                        if free_weights is not None:
                            free_sol = _weighted_dense_lstsq_jax(
                                jnp.take(J_active, free_idx, axis=1),
                                rhs_correction,
                                free_weights,
                                0.0,
                            )
                        else:
                            free_sol = _dense_lstsq_jax(jnp.take(J_active, free_idx, axis=1), rhs_correction, 0.0)
                        dx_active = dx_active.at[free_idx].set(
                            free_sol,
                            indices_are_sorted=True,
                            unique_indices=True,
                        )
                else:
                    lambda_weight_env = os.getenv("VMEC_JAX_SOLVER_MAP_LAMBDA_WEIGHT", "").strip()
                    lambda_weight = float(lambda_weight_env) if lambda_weight_env else 1.0
                    rhs_correction = rhs - J_active @ dx_reference_active
                    correction_weights = active_weights
                    if lambda_weight > 1.0 and (not _vmec_keep_all_active_enabled()):
                        if correction_weights is None:
                            correction_weights = jnp.ones_like(dx_reference_active)
                        correction_weights = correction_weights.at[active_lambda_start:].set(
                            correction_weights[active_lambda_start:]
                            * jnp.asarray(lambda_weight, dtype=correction_weights.dtype)
                        )
                    if correction_weights is not None:
                        dx_active = dx_reference_active + _weighted_dense_lstsq_jax(
                            J_active,
                            rhs_correction,
                            correction_weights,
                            0.0,
                        )
                    else:
                        dx_active = dx_reference_active + _dense_lstsq_jax(J_active, rhs_correction, 0.0)
            else:
                if active_weights is not None:
                    dx_active = _weighted_dense_lstsq_jax(J_active, rhs, active_weights, 0.0)
                else:
                    dx_active = _dense_lstsq_jax(J_active, rhs, 0.0)
        if dx_active is None and active_is_square and tangent_mode in ("auto", "lineax"):
            dx_lineax, success, _stats = _lineax_bicgstab_solve(
                stationarity_jvp_active_damped,
                rhs,
                tol=float(implicit.cg_tol),
                max_iter=int(implicit.cg_max_iter),
            )
            if bool(success):
                dx_active = dx_lineax
        if dx_active is None and active_is_square and tangent_mode in ("direct", "bicgstab"):
            from jax.scipy.sparse.linalg import bicgstab

            dx_bicgstab, info = bicgstab(
                stationarity_jvp_active_damped,
                rhs,
                tol=float(implicit.cg_tol),
                atol=0.0,
                maxiter=int(implicit.cg_max_iter),
            )
            if info is None:
                dx_active = dx_bicgstab
        if dx_active is None and tangent_mode == "chunked":
            chunk_size = getattr(implicit, "jac_chunk_size", None)
            if chunk_size is None:
                chunk_size = min(int(x_active_star.shape[0]), 64)
            J_active = _linear_map_jacobian_columns(
                stationarity_jvp_active,
                input_size=int(x_active_star.shape[0]),
                output_size=int(np.prod(np.shape(stationarity_star_active))),
                dtype=jnp.asarray(x_active_star).dtype,
                chunk_size=int(chunk_size),
            )
            if active_is_square:
                dx_active = jnp.linalg.solve(
                    J_active + damping * jnp.eye(int(J_active.shape[0]), dtype=J_active.dtype),
                    rhs,
                )
            else:
                dx_active = jnp.linalg.solve(
                    J_active.T @ J_active + damping * jnp.eye(int(J_active.shape[1]), dtype=J_active.dtype),
                    J_active.T @ rhs,
                )
        if dx_active is None:
            def Hvp_active(u_active):
                jv = stationarity_jvp_active(u_active)
                jt_jv = stationarity_vjp_active(jv)[0]
                return jt_jv + damping * u_active

            rhs_normal = stationarity_vjp_active(rhs)[0]
            dx_active = _cg_solve(
                Hvp_active,
                rhs_normal,
                tol=float(implicit.cg_tol),
                max_iter=int(implicit.cg_max_iter),
            )

        if _vmec_keep_all_active_enabled():
            dx_active_full = jnp.zeros_like(x_active_star_full).at[active_keep_idx].set(
                dx_active,
                indices_are_sorted=True,
                unique_indices=True,
            )
            tangent_state = _update_stellsym_feasible_state(
                _zero_state_like(st_star),
                dx_active_full,
                rz_idx=rz_idx,
                lam_idx=lam_idx,
                ns=ns_active,
                K=K_active,
            )
        else:
            tangent_state = _update_stellsym_reduced_state(
                _zero_state_like(st_star),
                dx_active,
                rz_idx=rz_idx,
                z_idx=z_idx,
                lam_sc_idx=lam_sc_idx,
                lam_cs_idx=lam_cs_idx,
                lam_maps=lam_maps,
                ns=ns_active,
                K=K_active,
            )
        dsRcos, dsRsin, dsZcos, dsZsin = jax.jvp(
            _boundary_state_edge_rows,
            (eRcos_star, eRsin_star, eZcos_star, eZsin_star),
            (deRcos, deRsin, deZcos, deZsin),
        )[1]
        return VMECState(
            layout=tangent_state.layout,
            Rcos=(-jnp.asarray(tangent_state.Rcos)).at[-1].set(jnp.asarray(dsRcos)),
            Rsin=(-jnp.asarray(tangent_state.Rsin)).at[-1].set(jnp.asarray(dsRsin)),
            Zcos=(-jnp.asarray(tangent_state.Zcos)).at[-1].set(jnp.asarray(dsZcos)),
            Zsin=(-jnp.asarray(tangent_state.Zsin)).at[-1].set(jnp.asarray(dsZsin)),
            Lcos=jnp.asarray(tangent_state.Lcos),
            Lsin=-jnp.asarray(tangent_state.Lsin),
        )

    if residual_tangent_mode not in ("", "0", "false", "no", "opaque"):
        @jax.custom_jvp
        def _solve_cust_jvp(eRcos, eRsin, eZcos, eZsin):
            return _solve(eRcos, eRsin, eZcos, eZsin)[0]

        @_solve_cust_jvp.defjvp
        def _solve_cust_jvp_rule(primals, tangents):
            eRcos, eRsin, eZcos, eZsin = primals
            deRcos, deRsin, deZcos, deZsin = tangents
            st, zero_m1, resume_out = _solve(eRcos, eRsin, eZcos, eZsin)
            tangent_state = _state_tangent_from_boundary_tangent(
                st,
                zero_m1,
                resume_out,
                eRcos,
                eRsin,
                eZcos,
                eZsin,
                jnp.asarray(deRcos),
                jnp.asarray(deRsin),
                jnp.asarray(deZcos),
                jnp.asarray(deZsin),
            )
            return st, tangent_state

        return _solve_cust_jvp(
            jnp.asarray(edge_Rcos_use),
            jnp.asarray(edge_Rsin_use),
            jnp.asarray(edge_Zcos_use),
            jnp.asarray(edge_Zsin_use),
        )

    @jax.custom_vjp
    def _solve_cust(eRcos, eRsin, eZcos, eZsin):
        return _solve(eRcos, eRsin, eZcos, eZsin)[0]

    def fwd(eRcos, eRsin, eZcos, eZsin):
        st, zero_m1, _resume_out = _solve(eRcos, eRsin, eZcos, eZsin)
        return st, (
            _stop_gradient_tree(st),
            jnp.asarray(zero_m1),
            jnp.asarray(eRcos),
            jnp.asarray(eRsin),
            jnp.asarray(eZcos),
            jnp.asarray(eZsin),
        )

    def bwd(residual, ct_state):
        bwd_start = time.perf_counter()
        _vmec_backward_profile_log("bwd_start")
        st_star, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star = residual
        ct_state_full = ct_state
        ct_state = _project_state(ct_state)
        b = pack_state(ct_state)
        ct_edge = (
            jnp.asarray(ct_state_full.Rcos)[-1, :],
            jnp.asarray(ct_state_full.Rsin)[-1, :],
            jnp.asarray(ct_state_full.Zcos)[-1, :],
            jnp.asarray(ct_state_full.Zsin)[-1, :],
        )

        stationarity_fun = lambda st: pack_state(
            _stationarity_state(st, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
        )

        def _boundary_param_vjp_from_fun(param_fun, lam):
            vjp_start = time.perf_counter()

            def combined_params(eRcos, eRsin, eZcos, eZsin):
                return (
                    param_fun(eRcos, eRsin, eZcos, eZsin),
                    _boundary_state_edge_rows(eRcos, eRsin, eZcos, eZsin),
                )

            _, vjp_fun = jax.vjp(combined_params, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
            dRcos, dRsin, dZcos, dZsin = vjp_fun((-jnp.asarray(lam), ct_edge))
            _vmec_backward_profile_log("boundary_param_vjp_done", vjp_start)
            return dRcos, dRsin, dZcos, dZsin

        residual_adjoint_mode = str(getattr(implicit, "residual_adjoint_mode", "auto")).strip().lower()
        if (not bool(static.cfg.lasym)) and (not _vmec_disable_reduced_active_enabled()):
            active_setup_start = time.perf_counter()
            rz_idx_np, lam_idx_np, ns_active, K_active = _stellsym_feasible_indices_np(
                static,
                idx00=idx00,
                mask_lambda_axis=True,
            )
            rz_idx = jnp.asarray(rz_idx_np, dtype=jnp.int32)
            lam_idx = jnp.asarray(lam_idx_np, dtype=jnp.int32)
            if _vmec_keep_all_active_enabled():
                b_active_full = _pack_stellsym_feasible_state(ct_state, rz_idx=rz_idx, lam_idx=lam_idx)
                active_keep_idx = jnp.arange(int(b_active_full.shape[0]), dtype=jnp.int32)
                st_active_ref = _stop_gradient_tree(st_star)
                x_active_star_full = _pack_stellsym_feasible_state(st_star, rz_idx=rz_idx, lam_idx=lam_idx)
                x_active_star = jnp.take(x_active_star_full, active_keep_idx)
                b_active = jnp.take(b_active_full, active_keep_idx)

                _vmec_backward_profile_log(
                    "active_setup_done",
                    active_setup_start,
                    active_size=int(np.shape(b_active)[0]),
                    active_full_size=int(np.shape(b_active_full)[0]),
                    residual_mode=residual_adjoint_mode,
                )

                def stationarity_fun_active(x_active):
                    x_active_full = x_active_star_full.at[active_keep_idx].set(
                        x_active,
                        indices_are_sorted=True,
                        unique_indices=True,
                    )
                    st_active = _update_stellsym_feasible_state(
                        st_active_ref,
                        x_active_full,
                        rz_idx=rz_idx,
                        lam_idx=lam_idx,
                        ns=ns_active,
                        K=K_active,
                    )
                    grad_state = _stationarity_state(
                        st_active,
                        zero_m1_star,
                        eRcos_star,
                        eRsin_star,
                        eZcos_star,
                        eZsin_star,
                    )
                    grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                    return jnp.take(grad_active_full, active_keep_idx)

                def _boundary_param_vjp_active(lam):
                    def G_params(eRcos, eRsin, eZcos, eZsin):
                        grad_state = _stationarity_state(
                            st_star,
                            zero_m1_star,
                            eRcos,
                            eRsin,
                            eZcos,
                            eZsin,
                        )
                        grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                        return jnp.take(grad_active_full, active_keep_idx)

                    return _boundary_param_vjp_from_fun(G_params, lam)
            else:
                z_idx = _stellsym_reduced_z_indices(rz_idx=rz_idx_np, K=int(K_active), idx00=idx00)
                lam_sc_idx, lam_cs_idx, lam_maps = _stellsym_lambda_mn_indices(
                    static,
                    idx00=idx00,
                    mask_lambda_axis=True,
                )
                st_active_ref = _stop_gradient_tree(st_star)
                x_active_star = _pack_stellsym_reduced_state(
                    st_active_ref,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )
                b_active = _pack_stellsym_reduced_state(
                    ct_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )

                _vmec_backward_profile_log(
                    "active_setup_done",
                    active_setup_start,
                    active_size=int(np.shape(b_active)[0]),
                    active_full_size=int(np.shape(b_active)[0]),
                    residual_mode=residual_adjoint_mode,
                )

                def stationarity_fun_active(x_active):
                    st_active = _update_stellsym_reduced_state(
                        st_active_ref,
                        x_active,
                        rz_idx=rz_idx,
                        z_idx=z_idx,
                        lam_sc_idx=lam_sc_idx,
                        lam_cs_idx=lam_cs_idx,
                        lam_maps=lam_maps,
                        ns=ns_active,
                        K=K_active,
                    )
                    grad_state = _stationarity_state(
                        st_active,
                        zero_m1_star,
                        eRcos_star,
                        eRsin_star,
                        eZcos_star,
                        eZsin_star,
                    )
                    return _pack_stellsym_reduced_state(
                        grad_state,
                        rz_idx=rz_idx,
                        z_idx=z_idx,
                        lam_sc_idx=lam_sc_idx,
                        lam_cs_idx=lam_cs_idx,
                        lam_maps=lam_maps,
                    )

                def _boundary_param_vjp_active(lam):
                    def G_params(eRcos, eRsin, eZcos, eZsin):
                        grad_state = _stationarity_state(
                            st_star,
                            zero_m1_star,
                            eRcos,
                            eRsin,
                            eZcos,
                            eZsin,
                        )
                        return _pack_stellsym_reduced_state(
                            grad_state,
                            rz_idx=rz_idx,
                            z_idx=z_idx,
                            lam_sc_idx=lam_sc_idx,
                            lam_cs_idx=lam_cs_idx,
                            lam_maps=lam_maps,
                        )

                    return _boundary_param_vjp_from_fun(G_params, lam)

            active_linearize_start = time.perf_counter()
            residual_star_active, residual_jvp_active = jax.linearize(stationarity_fun_active, x_active_star)
            active_is_square = tuple(residual_star_active.shape) == tuple(b_active.shape)
            # On the reduced stellarator-symmetric active coordinates, the
            # explicit chunked Jacobian is the only path that consistently
            # matches the dense reference solve. Matrix-free reduced adjoints
            # can under-converge badly enough to spoil outer descent, so route
            # the experimental modes through the explicit Jacobian path here.
            use_chunked_active = residual_adjoint_mode in (
                "auto",
                "chunked",
                "dense",
                "lineax",
                "direct",
                "bicgstab",
            )
            use_lineax_active = False
            use_direct_stellsym = False
            _vmec_backward_profile_log(
                "active_linearize_done",
                active_linearize_start,
                residual_size=int(np.prod(np.shape(residual_star_active))),
                chunked=bool(use_chunked_active),
            )

            residual_vjp_active = None
            rhs_active = None

            def Hvp_active(lam):
                nonlocal residual_vjp_active
                if residual_vjp_active is None:
                    residual_vjp_active = jax.linear_transpose(residual_jvp_active, x_active_star)
                jt_lam = residual_vjp_active(lam)[0]
                j_jt_lam = residual_jvp_active(jt_lam)
                return j_jt_lam + jnp.asarray(float(implicit.damping), dtype=jnp.asarray(lam).dtype) * lam

            if use_chunked_active:
                chunk_size = getattr(implicit, "jac_chunk_size", None)
                if chunk_size is None:
                    chunk_size = _vmec_jac_chunk_size_override()
                if chunk_size is None:
                    chunk_size = _default_vmec_jac_chunk_size(int(x_active_star.shape[0]))
                chunk_size = int(chunk_size)
                dense_start = time.perf_counter()
                J_active = _linear_map_jacobian_columns(
                    residual_jvp_active,
                    input_size=int(x_active_star.shape[0]),
                    output_size=int(np.prod(np.shape(residual_star_active))),
                    dtype=jnp.asarray(x_active_star).dtype,
                    chunk_size=chunk_size,
                )
                _vmec_backward_profile_log(
                    "active_dense_jacobian_done",
                    dense_start,
                    chunk_size=chunk_size,
                    jac_shape=tuple(int(x) for x in J_active.shape),
                )
                solve_start = time.perf_counter()
                effective_damping = 0.0
                damping = jnp.asarray(float(effective_damping), dtype=J_active.dtype)
                if residual_adjoint_mode == "dense":
                    if _is_traced(J_active, b_active, damping):
                        out_shape = jax.ShapeDtypeStruct((int(J_active.shape[0]),), J_active.dtype)
                        lam = jax.pure_callback(
                            _dense_transpose_lstsq_host,
                            out_shape,
                            J_active,
                            b_active,
                            damping,
                        )
                    else:
                        lam = jnp.asarray(
                            _dense_transpose_lstsq_host(J_active, b_active, damping),
                            dtype=J_active.dtype,
                        )
                else:
                    # Keep the explicit chunked Jacobian path, but solve the
                    # same transpose least-squares problem as the dense
                    # reference. On the reduced stellarator-symmetric branch,
                    # additional Tikhonov damping biases the nullspace choice
                    # away from the primal solver branch, so keep it at zero.
                    lam = _dense_transpose_lstsq_jax(
                        J_active,
                        b_active,
                        effective_damping,
                    )
                _vmec_backward_profile_log("active_dense_solve_done", solve_start)
                result = _boundary_param_vjp_active(lam)
                _vmec_backward_profile_log("bwd_done_chunked", bwd_start)
                return result

            rhs_active = residual_jvp_active(b_active)

            if use_direct_stellsym:
                direct_solve_start = time.perf_counter()
                lam = _cg_solve(
                    Hvp_active,
                    rhs_active,
                    tol=float(implicit.cg_tol),
                    max_iter=int(implicit.cg_max_iter),
                )
                _vmec_backward_profile_log("direct_cg_done", direct_solve_start)
                result = _boundary_param_vjp_active(lam)
                _vmec_backward_profile_log("bwd_done_direct", bwd_start)
                return result

            if use_lineax_active:
                direct_solve_start = time.perf_counter()

                lam, success, stats = _lineax_cg_solve(
                    Hvp_active,
                    rhs_active,
                    tol=float(implicit.cg_tol),
                    max_iter=int(implicit.cg_max_iter),
                )
                num_steps = stats.get("num_steps") if isinstance(stats, dict) else None
                if num_steps is not None:
                    try:
                        num_steps = int(np.asarray(jax.device_get(num_steps)))
                    except Exception:
                        num_steps = None
                _vmec_backward_profile_log(
                    "direct_lineax_done",
                    direct_solve_start,
                    success=bool(success),
                    num_steps=num_steps,
                )
                if bool(success):
                    result = _boundary_param_vjp_active(lam)
                    _vmec_backward_profile_log("bwd_done_lineax", bwd_start)
                    return result

            # Solve the same damped least-squares adjoint system as the dense
            # reference path, but matrix-free:
            #   argmin_lam ||J^T lam - b||^2 + damping ||lam||^2
            # whose normal equations are
            #   (J J^T + damping I) lam = J b.
            cg_start = time.perf_counter()
            lam = _cg_solve(Hvp_active, rhs_active, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))
            _vmec_backward_profile_log("active_cg_done", cg_start)
            result = _boundary_param_vjp_active(lam)
            _vmec_backward_profile_log("bwd_done_active", bwd_start)
            return result

        def _boundary_param_vjp_full(lam):
            def G_params(eRcos, eRsin, eZcos, eZsin):
                return pack_state(
                    _stationarity_state(st_star, zero_m1_star, eRcos, eRsin, eZcos, eZsin)
                )

            return _boundary_param_vjp_from_fun(G_params, lam)

        linearize_start = time.perf_counter()
        residual_star, residual_jvp = jax.linearize(stationarity_fun, st_star)
        residual_vjp = jax.linear_transpose(residual_jvp, st_star)
        _vmec_backward_profile_log("full_linearize_done", linearize_start, residual_size=int(np.prod(np.shape(residual_star))))

        def Hvp(u_flat):
            u_state = _project_state(unpack_state(u_flat, st_star.layout))
            jv = residual_jvp(u_state)
            jt_jv = residual_vjp(jv)[0]
            jt_jv = _project_state(jt_jv)
            return pack_state(jt_jv) + jnp.asarray(float(implicit.damping), dtype=jnp.asarray(jv).dtype) * u_flat

        cg_start = time.perf_counter()
        u = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))
        _vmec_backward_profile_log("full_cg_done", cg_start)
        u_state = _project_state(unpack_state(u, st_star.layout))
        jvp_start = time.perf_counter()
        lam = residual_jvp(u_state)
        _vmec_backward_profile_log("full_jvp_done", jvp_start)
        result = _boundary_param_vjp_full(lam)
        _vmec_backward_profile_log("bwd_done_full", bwd_start)
        return result

    _solve_cust.defvjp(fwd, bwd)
    return _solve_cust(
        jnp.asarray(edge_Rcos_use),
        jnp.asarray(edge_Rsin_use),
        jnp.asarray(edge_Zcos_use),
        jnp.asarray(edge_Zsin_use),
    )
