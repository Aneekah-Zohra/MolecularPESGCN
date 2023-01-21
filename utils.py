import math
import numpy as np
import torch


def update_pos(pos_arr):
    # initial distances before changing oxygen position
    distances = [math.sqrt(abs(pos_arr[0][i] ** 2 - pos_arr[j][i] ** 2)) for j in range(1, 3) for i in range(3)]

    # set oxygen to be (0, 0, 0), compute new positions
    pos_arr[0] = np.zeros(3)
    new_distances = [math.sqrt(abs(pos_arr[0][i] ** 2 - pos_arr[j][i] ** 2)) for j in range(1, 3) for i in range(3)]
    for k in range(6):
        pos_change = new_distances[k] - distances[k]
        if pos_change < 0:
            if k < 3:
                pos_arr[1][k] += pos_change
            else:
                pos_arr[2][k - 3] += pos_change
        elif pos_change > 0:
            if k < 3:
                pos_arr[1][k] -= pos_change
            else:
                pos_arr[2][k - 3] -= pos_change
    return pos_arr


def load_trajs(traj_arr):
    X = np.empty((12001, 3, 3))
    y = np.empty((12001,))
    for i in range(len(traj_arr)):
        atom = traj_arr[i]
        pos = atom.get_positions()

        # set oxygen to be reference point
        X[i] = update_pos(pos)
        y[i] = atom.get_calculator().get_potential_energy()
    return X, y


def load_data(traj_arr):
    X, y = load_trajs(traj_arr)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # build symmetric adjacency matrix
    adj = np.array([[0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0]])
    adj = torch.FloatTensor(adj + np.eye(3))
    return adj, X, y
