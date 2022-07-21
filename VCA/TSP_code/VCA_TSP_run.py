import numpy as np
import os
from VCA_TSP import *

"""
Functions
"""
def read_cities(path):
    """
    Read the x and y coordinates of the list of cities from a .txt/.SPARSE file.
    path - path of the file
    coordinates_X - x coordinates of the cities
    coordinates_Y - y coordinates of the cities
    """
    with open(path, 'r') as f:
        N = int(f.readline())
        coordinates_X = []
        coordinates_Y = []
        for _ in range(N):
            line = f.readline().split()
            x, y = float(line[0]), float(line[1])
            coordinates_X.append(x)
            coordinates_Y.append(y)

    return coordinates_X, coordinates_Y, N

"""
Main
"""
if __name__ == '__main__':
    # set the appropriate filepath to the Max-Cut graph below
    filepath = "D:\SJ Thesis\RNN-VCA-CO-main\Data\TSP Instances\coordinates_N64.txt"
    assert(os.path.exists(filepath)) # check if file exists in path
    coordinates_X, coordinates_Y, N = read_cities(filepath)
    coordinates = np.array(list(zip(coordinates_X, coordinates_Y)))
    # VCA hyperparameters
    # n_warmup - number of training steps at the initial temperature T=T0
    # n_anneal - duration of the annealing procedure
    # n_train - number of training step during backprop after every annealing step
    n_warmup = 5
    n_anneal = 5
    n_train = 2
    # VCA-Dilated
    model = vca(N=N, coordinates=coordinates, n_warmup=n_warmup, n_anneal=n_anneal, n_train=n_train, T0=2.0)
    energies, samples = model.run()
    np.save("test", energies)

