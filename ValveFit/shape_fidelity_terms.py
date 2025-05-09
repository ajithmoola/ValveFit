import jax.numpy as jnp


def pairwise_distances(x, y):
    """Compute the pairwise distances between two sets of points."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


def chamfer_distance(dists, one_sided=None, return_distances=False):
    """Compute Chamfer distances between two sets of points.

    Args:
        dists (ndarray): pairwise distance matrix between set of points x and y.
        one_sided (int, optional): If 0, compute only one-sided distance from x to y.
                                   If 1, compute only one-sided distance from y to x.
                                   If None, compute full Chamfer distance.
        return_distances (bool, optional): If True, return the minimum distances along with the sum.

    Returns:
        float or tuple: The computed Chamfer distance. If return_distances is True,
                        returns a tuple (sum of distances, minimum distances from x to y,
                        minimum distances from y to x).
    """

    if one_sided is not None:
        if one_sided == 0:
            # One-sided Chamfer distance from x to y
            x_closest_points = jnp.min(dists, axis=1)
            if return_distances:
                return jnp.mean(x_closest_points), x_closest_points
            return jnp.mean(x_closest_points)
        elif one_sided == 1:
            # One-sided Chamfer distance from y to x
            y_closest_points = jnp.min(dists, axis=0)
            if return_distances:
                return jnp.mean(y_closest_points), y_closest_points
            return jnp.mean(y_closest_points)
        else:
            raise ValueError("one_sided parameter must be None, 0, or 1")

    # Full Chamfer distance: sum of both one-sided distances
    y_closest_points = jnp.min(dists, axis=0)
    x_closest_points = jnp.min(dists, axis=1)

    if return_distances:
        return (
            jnp.mean(x_closest_points) + jnp.mean(y_closest_points),
            x_closest_points,
            y_closest_points,
        )

    return jnp.mean(x_closest_points) + jnp.mean(y_closest_points)


def hausdorff_distance(dists):
    """Compute Chamfer distances between two sets of points.

    Args:
        dists (ndarray): pairwise distance matrix between set of points x and y.
    Returns:
        float: The computed hausdorff distance.
    """
    h_xy = jnp.min(dists, axis=1).max()
    h_yx = jnp.min(dists, axis=1).max()
    h = max(h_xy, h_yx)
    return h
