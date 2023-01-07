import numpy as np
from ase.io.trajectory import TrajectoryReader as tr
import torch


def load_trajs(traj_file="dft_pyscf_ase_force.traj"):
    idx = 0
    X = np.zeros((12001, 3, 3))
    y = np.zeros(12001)

    traj = tr(traj_file)
    for i in range(len(traj_file)):
        atom = traj[i]
        positions = atom.get_positions()

        # Oxygen atom
        X[idx][0][0] = positions[0][0]
        X[idx][0][1] = positions[0][1]
        X[idx][0][2] = positions[0][2]

        # 1st Hydrogen atom
        X[idx][1][0] = positions[1][0]
        X[idx][1][1] = positions[1][1]
        X[idx][1][2] = positions[1][2]

        # 2nd Hydrogen atom
        X[idx][2][0] = positions[2][0]
        X[idx][2][1] = positions[2][1]
        X[idx][2][2] = positions[2][2]

        # calculate potential energy
        e = atom.get_calculator().get_potential_energy()
        y[idx] = e

        # update counter
        idx += 1

    return X, y


def load_data():
    features, y = load_trajs()
    y = torch.FloatTensor(y)
    features = torch.FloatTensor(features)

    # build symmetric adjacency matrix
    adj = np.array([[0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0]])
    adj = torch.FloatTensor(adj + np.eye(3))

    return adj, features, y
