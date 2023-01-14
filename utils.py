import numpy as np
import torch


def load_trajs(traj_arr):
    X = np.empty((12001, 3, 3))
    y = np.empty((12001, ))
    for i in range(len(traj_arr)):
        atom = traj_arr[i]
        X[i] = (atom.get_positions())
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