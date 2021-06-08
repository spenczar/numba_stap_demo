from numba_stap import create_probe, fire_probe
import numpy as np
import numba


def find_clusters_hotspots_2d(points, eps, min_samples):
    """
    This algorithm works by quantizing points into a grid, with a resolution of
    2*eps. It searches the quantized points to find values that appear at least
    min_samples times in the dataset.

    The search of the grid is done by sorting the points, and then stepping
    through the points looking for consecutive runs of the same (x, y) value
    repeated at least min_samples times.

    This is repeated, with the X values offset by +eps, which addresses edge
    effects at the right boundaries of the 2*eps-sized grid windows.

    Similarly, it is repeated with the Y value offset by +eps (dealing with the
    bottom boundary) and with both X and Y offset (dealing with the corner
    boundary).
    """
    points = points.copy()
    points = _enforce_shape(points)

    clusters = []
    cluster_labels = _hotspot_multilabel(points, eps, min_samples)
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        clusters.append(cluster_indices)

    return clusters


probe_hotspot_2d_inner = create_probe("hotspot_2d_inner")
@numba.njit
def _hotspot_2d_inner(points, eps, min_samples):
    """
    This function holds the core work of the hotspot2d algorithm: quantize the
    points, sort them, find runs in the sorted list, and label each point with
    an ID from the runs that have been found.
    """
    fire_probe(probe_hotspot_2d_inner)
    points_quantized = _quantize_points(_make_points_nonzero(points), 2*eps)
    sort_order = _sort_order_2d(points_quantized)
    sorted_points = points_quantized[:, sort_order]
    runs = _find_runs(sorted_points, min_samples)
    cluster_labels = _label_clusters(runs, points_quantized)
    return cluster_labels


probe_hotspot_multilabel = create_probe("hotspot_multilabel")
@numba.njit
def _hotspot_multilabel(points, eps, min_samples):
    """
    Run the hotspot2d algorithm 4 times. Each time, the input points are
    adjusted a bit, offsetting them by 'eps' in the X, then Y, then X and Y
    directions. This helps deal with edge effects in the binning of datapoints.

    This code is wordy and repetitive, just in order to make things simple for
    numba's compiler, which has a big impact on performance.
    """
    fire_probe(probe_hotspot_multilabel)
    n = points.shape[1]

    # Find and label runs in the dataset.
    labels1 = _hotspot_2d_inner(points, eps, min_samples)

    # Repeat, but with X+eps, Y
    for i in range(n):
        points[0, i] = points[0, i] + eps
    labels2 = _hotspot_2d_inner(points, eps, min_samples)
    # Adjust labels so they don't collide with those of labels1.
    _adjust_labels(labels2, labels1.max() + 1)

    # Repeat, but with X+eps, Y+eps.
    for i in range(n):
        points[1, i] = points[1, i] + eps
    labels3 = _hotspot_2d_inner(points, eps, min_samples)
    _adjust_labels(labels3, labels2.max() + 1)

    # Repeat, but with X, Y+eps
    for i in range(n):
        points[0, i] = points[0, i] - eps
    labels4 = _hotspot_2d_inner(points, eps, min_samples)
    _adjust_labels(labels4, labels3.max() + 1)

    # Make an empty array which will store the cluster IDs of each point.
    final_labels = np.full(n, -1, dtype=labels1.dtype)

    # Many of the label arrays we built will have the same clusters, but with
    # different integer labels. Build a mapping to standardize things.
    label_aliases = _build_label_aliases(labels1, labels2, labels3, labels4, n)

    # Apply labels.
    for i in range(n):
        if labels1[i] != -1:
            final_labels[i] = labels1[i]
        elif labels2[i] != -1:
            final_labels[i] = label_aliases.get(labels2[i], labels2[i])
        elif labels3[i] != -1:
            final_labels[i] = label_aliases.get(labels3[i], labels3[i])
        elif labels4[i] != -1:
            final_labels[i] = label_aliases.get(labels4[i], labels4[i])

    return final_labels


probe_adjust_labels = create_probe("adjust_labels")
@numba.njit(parallel=True)
def _adjust_labels(labels, new_minimum):
    """
    Given a bunch of integer labels, adjust the labels to start at new_minimum.
    """
    fire_probe(probe_adjust_labels)
    labels[labels != -1] = labels[labels != -1] + new_minimum


probe_build_label_aliases = create_probe("build_label_aliases")
@numba.njit
def _build_label_aliases(labels1, labels2, labels3, labels4, n):
    fire_probe(probe_build_label_aliases)
    label_aliases = {}
    for i in range(n):
        # Prefer names from labels1, then labels2, then labels3, then labels4.
        if labels1[i] != -1:
            label = labels1[i]
            if labels2[i] != -1:
                label_aliases[labels2[i]] = label
            if labels3[i] != -1:
                label_aliases[labels3[i]] = label
            if labels4[i] != -1:
                label_aliases[labels4[i]] = label
        elif labels2[i] != -1:
            label = labels2[i]
            if labels3[i] != -1:
                label_aliases[labels3[i]] = label
            if labels4[i] != -1:
                label_aliases[labels4[i]] = label
        elif labels3[i] != -1:
            label = labels3[i]
            if labels4[i] != -1:
                label_aliases[labels4[i]] = label
    return label_aliases


probe_enforce_shape = create_probe("enforce_shape")
@numba.njit
def _enforce_shape(points):
    """Ensure that datapoints are in a shape of (2, N)."""
    fire_probe(probe_enforce_shape)
    if points.shape[0] != 2:
        return points.T
    return points


probe_make_points_nonzero = create_probe("make_points_nonzero")
@numba.njit
def _make_points_nonzero(points):
    # Scale up to nonzero
    #
    # Careful: numba.njit(parallel=True) would give wrong result, since min()
    # would get re-evaluated!
    fire_probe(probe_make_points_nonzero)
    return points - points.min()


probe_quantize_points = create_probe("quantize_points")
@numba.njit(parallel=True)
def _quantize_points(points, eps):
    """Quantize points to be scaled in units of eps."""
    fire_probe(probe_quantize_points)
    return (points / eps).astype(np.int32)


probe_sort_order_2d = create_probe("sort_order_2d")
@numba.njit
def _sort_order_2d(points):
    """
    Return the indices that would sort points by x and then y. Points must be
    integers.

    This is about twice as fast as np.lexsort(points), and can be numba-d. It
    works by transforming the 2D sequence of int pairs into a 1D sequence of
    integers, and then sorting that 1D sequence.
    """
    fire_probe(probe_sort_order_2d)
    scale = points.max() + 1
    return np.argsort(points[0, :]*scale + points[1, :])


probe_find_runs = create_probe("find_runs")
@numba.njit
def _find_runs(sorted_points, min_samples, expected_n_clusters=16):
    """
    Find all subssequences of at least min_samples length with the same x, y
    values in a sorted dataset of (x, y) values.

    Returns a 2D array of the (x, y) values that match. The array may have
    duplicated values.

    expected_n_clusters is a tuning parameter: an array is preallocated
    proportional to this value. If the guess is too low, we will allocate more
    memory than necessary. If it's too high, we will have to spend time growing
    that array.
    """
    fire_probe(probe_find_runs)
    result = np.empty((2, expected_n_clusters*2), dtype="float64")
    n_hit = 0

    n_consecutive = 1
    prev_x = np.nan
    prev_y = np.nan

    for i in range(sorted_points.shape[1]):
        if sorted_points[0, i] == prev_x and sorted_points[1, i] == prev_y:
            n_consecutive += 1
            if n_consecutive == min_samples:
                if n_hit == result.shape[1]:
                    result = _extend_2d_array(result, result.shape[1]*2)
                result[0, n_hit] = sorted_points[0, i]
                result[1, n_hit] = sorted_points[1, i]
                n_hit = n_hit + 1
        else:
            prev_x = sorted_points[0, i]
            prev_y = sorted_points[1, i]
            n_consecutive = 1
    return result[:, :n_hit]


probe_extend_2d_array = create_probe("extend_2d_array")
@numba.njit
def _extend_2d_array(src, new_size):
    fire_probe(probe_extend_2d_array)
    dst = np.empty((2, new_size), dtype=src.dtype)
    for i in range(src.shape[1]):
        dst[0, i] = src[0, i]
        dst[1, i] = src[1, i]
    return dst


probe_label_clusters = create_probe("label_clusters")
@numba.njit
def _label_clusters(runs, points_quantized):
    """
    Produce a 1D array of integers which label each X-Y in points_quantized with
    a cluster.
    """
    fire_probe(probe_label_clusters)
    labels = np.full(points_quantized.shape[1], -1, np.int64)
    for i in range(points_quantized.shape[1]):
        for j in range(runs.shape[1]):
            if runs[0, j] == points_quantized[0, i] and runs[1, j] == points_quantized[1, i]:
                labels[i] = j
                break
    return labels
