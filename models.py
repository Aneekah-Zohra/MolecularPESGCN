import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.linear1 = nn.Linear(nfeat, nout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.linear1(x)
        return torch.mean(x, dim=1)