from pymoo.factory import get_performance_indicator
import numpy as np


def compute_hypervolume(points, maximization=True):
    if maximization:
        ref = [1, 1]
    else:
        points = -points
        ref = [0, 0]
    hv = get_performance_indicator("hv", ref_point=np.array(ref))
    return hv.do(points)


# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs, increasing=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    compare_fn = lambda a, b: a > b
    if increasing:
        compare_fn = lambda a, b: a < b

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(compare_fn(costs[:i], c), axis=1)) and np.all(
            np.any(compare_fn(costs[i + 1 :], c), axis=1)
        )
    return is_efficient.tolist()


def circle_points(n, min_angle=0.1, max_angle=np.pi / 2 - 0.1, dim=2):
    # generate evenly distributed preference vector
    assert dim > 1
    if dim == 2:
        ang0 = 1e-6 if min_angle is None else min_angle
        ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
        angles = np.linspace(ang0, ang1, n, endpoint=True)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.c_[x, y]
    elif dim == 3:
        # Fibonacci sphere algorithm
        # https://stackoverflow.com/a/26127012
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        n = n * 8  # we are only looking at the positive part
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            if x >= 0 and y >= 0 and z >= 0:
                points.append((x, y, z))
        return np.array(points)
    else:
        # this is an unsolved problem for more than 3 dimensions
        # we just generate random points
        points = np.random.rand(n, dim)
        points /= points.sum(axis=1).reshape(n, 1)
        return points
