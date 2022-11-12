from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import load_data
from models import GCN

# Training settings, customize arguments passed in CLI
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')

args = parser.parse_args()

# setting pseudo-random number
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Load data
adj, train_X, train_y, test_X, test_y = load_data()

# Model and optimizer
model = GCN(nfeat=train_X.shape[1],
            nhid=args.hidden)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


def train(epoch, output=None):
    i = 0
    t = time.time()
    model.train()
    optimizer.zero_grad()
    metric = torch.nn.MSELoss()
    pred_output = model(train_X[i], adj)
    loss_train = torch.sqrt(metric(train_y[i], pred_output))
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    loss_val = torch.nn.MSELoss(train_y[i], pred_output)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    i += 1


def test():
    i = 0
    model.eval()
    output = model(test_X, adj)
    loss_test = torch.nn.MSELoss(test_X[i], output[i])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
    i += 1


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
