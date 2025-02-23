import numpy as np
import random

mapping = list(range(25))

def map_labels(labels):
    return np.array([mapping[label] for label in labels])


def get_data_ring(batch_size, radius=2.0, var=0.0001, n_clusters=8):
    thetas = np.linspace(0, 2 * np.pi, n_clusters + 1)[:n_clusters]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    classes = np.random.multinomial(batch_size,
                                    [1.0 / n_clusters] * n_clusters)
    labels = [i for i in range(n_clusters) for _ in range(classes[i])]
    random.shuffle(labels)
    labels = np.array(labels)
    samples = np.array([
        np.random.multivariate_normal([xs[i], ys[i]], [[var, 0], [0, var]])
        for i in labels
    ])
    return samples, labels


def get_data_grid_orig(batch_size, radius=2.0, var=0.0025, nrows=5, ncols=5):
    samples = []
    labels = []
    for _ in range(batch_size):
        i, j = random.randint(0, ncols - 1), random.randint(0, nrows - 1)
        samples.append(
            np.random.multivariate_normal([i * 2 - 4, j * 2 - 4],
                                          [[var, 0], [0, var]]))
        labels.append(5 * i + j)
    return np.array(samples), map_labels(labels)


def get_data_grid(batch_size, ftrs_dim, radius=2.0, var=0.0025, nrows=5, ncols=5):
    samples = []
    y = []
    for _ in range(batch_size):
        i, j = random.randint(0, ncols - 1), random.randint(0, nrows - 1)
        samples.append(
            np.random.multivariate_normal([i * 2 - 4, j * 2 - 4],
                                          [[var, 0], [0, var]]))

        y.append(5 * i + j)  # these y values are not used.

    random.shuffle(samples)
    
    return np.array(samples), np.array(y)


def get_data_grid_dirichlet(batch_size, ftrs_dim, radius=2.0, var=0.0025, nrows=5, ncols=5, x_dim=2, K=25):
    samples = []
    y = []

    # Get the weights: (sampled this way: np.random.dirichlet(np.ones(25)))
      
    # dirichlet sample number 1:
    #tpi = (0.001, 0.5, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.27, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.208)  # np.random.dirichlet(np.ones(K))
    
    # dirichlet sample number 2:
    tpi = (0.01709201, 0.02467809, 0.00639062, 0.02088758, 0.01173375, 0.06905818, 0.02275523, 0.03648351, 0.0101484, 0.10693863, 0.13513911, 0.03862016, 0.06959448, 0.07959441, 0.02502783, 0.01997247, 0.02095415, 0.01615444, 0.02081546, 0.00152879, 0.10475761, 0.0043768, 0.03210107, 0.10157113, 0.00362608)  # np.random.dirichlet(np.ones(K))
    
    tzn = np.random.multinomial(batch_size, tpi)
    print("Data mixture:", tzn, " , where N is: ", batch_size)

    # Create means similar to grid data (the problem with sampling random means: there are 2 calls for this function, for x_cluster and data_loader, and we won't get consistent data)
    rows = list(range(nrows))
    cols = list(range(ncols))
    means = []
    for i in rows:
        for j in cols:
            means.append(np.array([rows[i] * 2 - 4, cols[j] * 2 - 4]))

    means = np.array(means)

    ind = 0
    for i in range(len(tzn)):
        for j in range(ind, ind+tzn[i]):
            samples.append(np.random.multivariate_normal([means[i,0], means[i,1]], [[var, 0], [0, var]]))
            y.append(random.randint(1, K))  # these y values are not used.

        ind = ind + tzn[i]

    random.shuffle(samples)
    
    return np.array(samples), np.array(y)
