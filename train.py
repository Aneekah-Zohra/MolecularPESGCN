from __future__ import division
from __future__ import print_function

import time
import numpy as np
import random

import torch
import torch.optim as optim

from utils_new import load_data
from models import GCN


# Load data
adj, features, y = load_data()
lst = list(zip(features, y))

# TODO: fix random shuffling
# random shuffling here breaks and returns array of all zeroes
random.seed(42)
random.shuffle(lst)

# TODO: add validation sets
# hardcoded numbers to be near 60-40 split for now, use scikit train, test, val split function later
train_X = torch.FloatTensor(np.array(np.zeros((7637, 3, 3))))
train_y = torch.FloatTensor(np.array(np.zeros(7637)))
test_X = torch.FloatTensor(np.array(np.zeros((4364, 3, 3))))
test_y = torch.FloatTensor(np.array(np.zeros(4364)))

for idx, (X_val, y_val) in enumerate(lst):
    if idx < 4364:
        test_X[idx] = X_val
        test_y[idx] = y_val
    else:
        train_X[idx - 4364] = X_val
        train_y[idx - 4364] = y_val

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
    loss_train = torch.sqrt(metric(train_y, pred_output))
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
