from ValveFit.bspline_funcs import (
    find_span_array_jax,
    basis_fns_vectorized,
    generate_parametric_coordinates,
)
import jax.numpy as jnp
from jax import jacfwd, jit, value_and_grad
import optax
from tqdm import tqdm
from functools import partial


def compute_tensor_product(params, knotvectors, degrees):
    b_fns = [
        basis_fns_vectorized(params[:, dim], knotvectors[dim], degrees[dim]).squeeze()
        for dim in range(2)
    ]
    tp = jnp.einsum("ij, ik -> ijk", *b_fns)
    return tp


def get_CP_indices(bsplines, params):
    spans = jnp.stack(
        [
            find_span_array_jax(params[:, dim], bspline.knotvector, bspline.degree)
            for dim, bspline in enumerate(bsplines)
        ]
    ).T
    p, q = tuple(bspline.degree for bspline in bsplines)
    x_offsets, y_offsets = jnp.meshgrid(
        jnp.arange(p + 1), jnp.arange(q + 1), indexing="ij"
    )
    x_indices = spans[:, 0, jnp.newaxis, jnp.newaxis] + x_offsets - p
    y_indices = spans[:, 1, jnp.newaxis, jnp.newaxis] + y_offsets - q
    CP_indices = [x_indices, y_indices]
    return CP_indices


def evaluate(ctrl_pts, tp, CP_indices, degrees):
    periodic_CP = ctrl_pts.at[:, -degrees[1] :].set(ctrl_pts[:, : degrees[1]])
    pts = jnp.einsum(
        "ijkl, ijk -> il", periodic_CP[CP_indices[0], CP_indices[1]].squeeze(), tp
    )
    return pts


def compute_tp_and_evaluate(params, ctrl_pts, knotvectors, degrees, CP_indices):
    tp = compute_tensor_product(params, knotvectors, degrees)
    pts = evaluate(ctrl_pts, tp, CP_indices, degrees)
    return pts


def compute_jacobian(params, ctrl_pts, knotvectors, degrees, CP_indices):
    jac = jacfwd(compute_tp_and_evaluate, argnums=0)(
        params, ctrl_pts, knotvectors, degrees, CP_indices
    )
    indices = jnp.arange(params.shape[0])
    jac_sliced = jac[indices, :, indices, :]
    return jac_sliced


def linear_transform_valve(valve, ctrl_pts, target_pc, n_iter=1000):
    degrees = valve.degrees
    knotvectors = valve.knotvectors

    K = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    transform_params = {
        "angle": jnp.zeros(1),
        "translation": jnp.zeros(3),
        "log_scale": jnp.zeros(1),
        "log_z_scale": jnp.zeros(1),
    }

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(transform_params)

    params = generate_parametric_coordinates((50, 50))
    fns = compute_tensor_product(params, knotvectors, degrees)
    CP_indices = get_CP_indices(valve.bsplines, params)

    def apply_transformation(transform_params, ctrl_pts):
        angle = transform_params["angle"][0]
        translation = transform_params["translation"]
        log_scale = transform_params["log_scale"][0]
        log_z_scale = transform_params["log_z_scale"][0]

        scale = jnp.exp(log_scale)
        z_scale = jnp.exp(log_z_scale)

        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)

        S_vector = jnp.array([scale, scale, scale * z_scale])
        ctrl_pts_transformed = (ctrl_pts @ R) * S_vector + translation

        return ctrl_pts_transformed

    def compute_loss(transform_params, ctrl_pts, fns, CP_indices, degrees, target_pc):
        ctrl_pts_transformed = apply_transformation(transform_params, ctrl_pts)
        pts = evaluate(ctrl_pts_transformed, fns, CP_indices, degrees)
        dists = jnp.sqrt(
            jnp.sum((pts[:, None, :] - target_pc[None, :, :]) ** 2 + 1e-8, axis=-1)
        )
        loss = jnp.mean(jnp.min(dists, axis=0)) + jnp.mean(jnp.min(dists, axis=1))
        return loss

    compute_loss_partial = partial(
        compute_loss,
        ctrl_pts=ctrl_pts,
        fns=fns,
        CP_indices=CP_indices,
        degrees=degrees,
        target_pc=target_pc,
    )

    @jit
    def step(transform_params, opt_state):
        loss, grads = value_and_grad(compute_loss_partial)(transform_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        transform_params = optax.apply_updates(transform_params, updates)
        return transform_params, opt_state, loss

    pbar = tqdm(range(n_iter))
    for _ in pbar:
        transform_params, opt_state, loss = step(transform_params, opt_state)
        pbar.set_description(f"Iteration {_}: Loss = {loss}")

    transformed_ctrl_pts = apply_transformation(transform_params, ctrl_pts)

    return transformed_ctrl_pts, transform_params


def linear_transform_valve_non_uniform_scaling(valve, ctrl_pts, target_pc, n_iter=1000):
    degrees = valve.degrees
    knotvectors = valve.knotvectors

    K = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    transform_params = {
        "angle": jnp.zeros(1),
        "translation": jnp.zeros(3),
        "log_scale": jnp.zeros(3),
    }

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(transform_params)

    params = generate_parametric_coordinates((50, 50))
    fns = compute_tensor_product(params, knotvectors, degrees)
    CP_indices = get_CP_indices(valve.bsplines, params)

    def apply_transformation(transform_params, ctrl_pts):
        angle = transform_params["angle"][0]
        translation = transform_params["translation"]
        log_scale = transform_params["log_scale"]

        scale_vec = jnp.exp(log_scale)

        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)

        ctrl_pts_transformed = (ctrl_pts @ R) * scale_vec + translation

        return ctrl_pts_transformed

    def compute_loss(transform_params, ctrl_pts, fns, CP_indices, degrees, target_pc):
        ctrl_pts_transformed = apply_transformation(transform_params, ctrl_pts)
        pts = evaluate(ctrl_pts_transformed, fns, CP_indices, degrees)
        dists = jnp.sqrt(
            jnp.sum((pts[:, None, :] - target_pc[None, :, :]) ** 2 + 1e-8, axis=-1)
        )
        loss = jnp.mean(jnp.min(dists, axis=0)) + jnp.mean(jnp.min(dists, axis=1))
        return loss

    compute_loss_partial = partial(
        compute_loss,
        ctrl_pts=ctrl_pts,
        fns=fns,
        CP_indices=CP_indices,
        degrees=degrees,
        target_pc=target_pc,
    )

    @jit
    def step(transform_params, opt_state):
        loss, grads = value_and_grad(compute_loss_partial)(transform_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        transform_params = optax.apply_updates(transform_params, updates)
        return transform_params, opt_state, loss

    pbar = tqdm(range(n_iter))
    for _ in pbar:
        transform_params, opt_state, loss = step(transform_params, opt_state)
        pbar.set_description(f"Iteration {_}: Loss = {loss}")

    transformed_ctrl_pts = apply_transformation(transform_params, ctrl_pts)

    return transformed_ctrl_pts, transform_params


def rotate_valve(valve, ctrl_pts, target_pc, n_iter=1000):
    degrees = valve.degrees
    knotvectors = valve.knotvectors

    transform_params = {
        "scale": jnp.ones(1),
        "angle": jnp.zeros(1),
    }

    opt = optax.adam(learning_rate=0.001)
    opt_state = opt.init(transform_params)

    params = generate_parametric_coordinates((30, 30))
    fns = compute_tensor_product(params, knotvectors, degrees)
    CP_indices = get_CP_indices(valve.bsplines, params)
    K = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    def compute_loss(
        transform_params, ctrl_pts, fns, CP_indices, degrees, target_pc, K
    ):
        angle = transform_params["angle"]
        scale = transform_params["scale"]

        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)

        ctrl_pts_transformed = (ctrl_pts @ R) * scale

        # Evaluate transformed points
        pts = evaluate(ctrl_pts_transformed, fns, CP_indices, degrees)

        # Compute loss with small constant
        dists = jnp.sqrt(
            jnp.sum((pts[:, None, :] - target_pc[None, :, :]) ** 2 + 1e-8, axis=-1)
        )
        loss = jnp.mean(jnp.min(dists, axis=0)) + jnp.mean(jnp.min(dists, axis=1))
        return loss

    compute_loss_partial = partial(
        compute_loss,
        ctrl_pts=ctrl_pts,
        fns=fns,
        CP_indices=CP_indices,
        degrees=degrees,
        target_pc=target_pc,
        K=K,
    )

    @jit
    def step(transform_params, opt_state):
        loss, grads = value_and_grad(compute_loss_partial)(transform_params)
        updates, opt_state = opt.update(grads, opt_state)
        transform_params = optax.apply_updates(transform_params, updates)
        return transform_params, opt_state, loss

    pbar = tqdm(range(n_iter))
    for _ in pbar:
        transform_params, opt_state, loss = step(transform_params, opt_state)
        pbar.set_description(f"Iteration {_}: Loss = {loss}")

    angle = transform_params["angle"]
    scale = transform_params["scale"]

    R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)

    transformed_ctrl_pts = ctrl_pts @ R * scale

    return transformed_ctrl_pts
