import numpy as np
# import scipy.sparse as sp
# import torchcd
from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw
# from sklearn.model_selection import train_test_split
import os.path


# Todo: Fix difference in losing atom configurations caused by making more trajs than needed
# Total amount = 168014
# Each file contains 15274 atoms, 11 total trajectories
# However I split each by 12001, therefore 14 total trajectories
# Then applied a 30-70 split for testing and training
def traj_split(path="../../../Downloads/pygcn-master/pygcn-master/data/dft_pyscf_ase_force.traj"):
    images = tr(path)
    atom_config = images[-1]
    for i in range(11):
        traj = tw('./data/traj_split_{}.traj'.format(i), 'a', atom_config)
        for j in range(1091):
            traj.write(images[j])

def load_trajs(path="", dataset="h2o traj"):
    traj_split()
    print('Loading {} dataset...'.format(dataset))
    # keep track of configuration
    idx = 0
    X = np.zeros((4364, 3, 3))
    y = np.zeros(4364)
    test_X = np.zeros((45822, 3, 3))
    test_y = np.zeros(45822)
    '''
    X is features node dataset
    X : [O]
        [H]
        [H]
    where each subarray has 3 coordinates [x, y, z]
    '''
    for i in range(3):
        traj = tr("./data/traj_split_{}.traj".format(i))
        for j in range(1091):
            # access the jth configuration of the Atoms
            atom = traj[j]
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
            # calculate bond length between each
            # charge = atom.get_initial_charges()
            # angles = atom.cell.cellpar()[3:]
            idx += 1
    return X, y

def load_data(dataset="h2o traj"):
    print('Loading {} dataset...'.format(dataset))


    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test