from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split
from ase.io.trajectory import TrajectoryReader as tr

from utils import load_data
from models import GCN

np.random.seed(1)

# atoms dataset
images = tr("dft_pyscf_ase_force.traj")

# apply random shuffle to dataset
index = np.arange(len(images), dtype=int)
np.random.shuffle(index)
images_shuffled = [images[idx] for idx in index]

# load data
adj, X, y = load_data(images_shuffled)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=.7, random_state=42, shuffle=False)

# Model and optimizer
model = GCN(nfeat=3,
            nhid=16)
optimizer = optim.Adam(model.parameters(),
                       lr=.001)


def test():
    i = 0
    model.eval()
    output = model(test_X, adj)
    metric_ = torch.nn.MSELoss()
    loss_test = torch.sqrt(metric_(test_y, output))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
    i += 1


# Train model
t_total = time.time()
for epoch in range(200):
    t = time.time()
    optimizer.zero_grad()
    metric = torch.nn.MSELoss()
    pred_output = model(train_X, adj)
    loss_train = metric(train_y, pred_output)
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'training_loss: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()