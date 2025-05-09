import jax.numpy as jnp
from jax import vmap
from ValveFit.periodic_splines_funcs import compute_jacobian


def uniform_between_min_max(ctrl_pts):
    n, m, _ = ctrl_pts.shape
    min_points = ctrl_pts.min(axis=0)
    max_points = ctrl_pts.max(axis=0)
    alphas = jnp.linspace(0, 1, n).reshape(n, 1, 1)
    new_ctrl_pts = (1 - alphas) * min_points + alphas * max_points
    return new_ctrl_pts


def compute_tangents_normals(
    params, ctrl_pts, knotvectors, degrees, CP_indices, unit_vectors=True
):
    jac = compute_jacobian(params, ctrl_pts, knotvectors, degrees, CP_indices)
    t1 = jac[:, :, 0]
    t2 = jac[:, :, 1]
    if unit_vectors:
        t1_norm = jnp.linalg.norm(t1, axis=-1, keepdims=True) + 1e-8
        t2_norm = jnp.linalg.norm(t2, axis=-1, keepdims=True) + 1e-8
        t1 = t1 / t1_norm
        t2 = t2 / t2_norm
    normals = jnp.cross(t1, t2)
    return t1, t2, normals


def compute_R_orth(t1, t2):
    return jnp.mean(jnp.abs(vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(t1, t2)))


def compute_z_normal_var(normals):
    return jnp.mean(jnp.var(normals[:, -1], axis=0))


def compute_R_norm(normals):
    return jnp.abs(normals[:, -1] - normals[:, -1].mean()).max()


def compute_squared_tangent_lengths(t1, t2):
    R_t1_norm = jnp.mean(t1**2)
    R_t2_norm = jnp.mean(t2**2)

    R_t_norm = 0.5 * (R_t1_norm + R_t2_norm)
    return R_t_norm
