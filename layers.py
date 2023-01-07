import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    # setting up the state and model
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # reset to unlearned parameters
        self.reset_parameters()

    # initializing data randomly
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    # output updated features matrix [without nonlinear activation]
    def forward(self, input, adj):
        # TODO: remove hardcoded D^-1 array
        D = torch.FloatTensor(np.array(
            [[1/3, 0, 0],
             [0, 1/2, 0],
             [0, 0, 1/2]]))
        # weighted input
        D_tilde = torch.sqrt(D)
        weighted_avg = D_tilde @ adj
        weighted_avg *= adj @ D_tilde
        support = input @ self.weight
        # weighted average
        output = weighted_avg @ support
        return output

    # object String call
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
