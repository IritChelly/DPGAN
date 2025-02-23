def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import numpy as np

def percent_good_grid(x_fake, var=0.0025, nrows=5, ncols=5):
    std = np.sqrt(var)
    x = list(range(nrows))
    y = list(range(ncols))

    threshold = 3 * std
    means = []
    for i in x:
        for j in y:
            means.append(np.array([x[i] * 2 - 4, y[j] * 2 - 4]))

    tpi = (0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04)

    return percent_good_pts(x_fake, means, threshold, tpi)


def percent_good_grid_dirichlet(x_fake, var=0.0025, nrows=5, ncols=5):
    std = np.sqrt(var)
    x = list(range(nrows))
    y = list(range(ncols))

    threshold = 3 * std
    means = []
    for i in x:
        for j in y:
            means.append(np.array([x[i] * 2 - 4, y[j] * 2 - 4]))

    # dirichlet sample number 1:
    #tpi = (0.001, 0.5, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.27, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.208)
    
    # dirichlet sample number 2:
    tpi = (0.01709201, 0.02467809, 0.00639062, 0.02088758, 0.01173375, 0.06905818, 0.02275523, 0.03648351, 0.0101484, 0.10693863, 0.13513911, 0.03862016, 0.06959448, 0.07959441, 0.02502783, 0.01997247, 0.02095415, 0.01615444, 0.02081546, 0.00152879, 0.10475761, 0.0043768, 0.03210107, 0.10157113, 0.00362608)  # np.random.dirichlet(np.ones(K))

    return percent_good_pts(x_fake, means, threshold, tpi)


def percent_good_ring(x_fake, var=0.0001, n_clusters=8, radius=2.0):
    std = np.sqrt(var)
    thetas = np.linspace(0, 2 * np.pi, n_clusters + 1)[:n_clusters]
    x, y = radius * np.sin(thetas), radius * np.cos(thetas)
    threshold = np.array([std * 3, std * 3])
    means = []
    for i in range(n_clusters):
        means.append(np.array([x[i], y[i]]))
    return percent_good_pts(x_fake, means, threshold)


def percent_good_pts(x_fake, means, threshold, tpi):
    """Calculate %good, #modes, kl

    Keyword arguments:
    x_fake -- detached generated samples
    means -- true means
    threshold -- good point if l_1 distance is within threshold
    """
    count = 0
    counts = np.zeros(len(means))
    visited = set()
    for point in x_fake:
        minimum = 0
        diff_minimum = [1e10, 1e10]
        for i, mean in enumerate(means):
            diff = np.abs(point - mean)
            if np.all(diff < threshold):
                visited.add(tuple(mean))
                count += 1
                break
        for i, mean in enumerate(means):
            diff = np.abs(point - mean)
            if np.linalg.norm(diff) < np.linalg.norm(diff_minimum):
                minimum = i
                diff_minimum = diff
        counts[minimum] += 1

    kl = 0
    counts = counts / len(x_fake)
    for j, generated in enumerate(counts):
        if generated != 0:
            kl += generated * np.log(generated / tpi[j])

    prop_real = count / len(x_fake)
    modes = len(visited)
    
    return prop_real, modes, kl