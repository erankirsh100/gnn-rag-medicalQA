import torch
from torch_geometric.data import Data
import pandas as pd
import pickle
import numpy as np

from torch_geometric.nn import SAGEConv
import torch
import torch_geometric
import torch.nn.functional as F
torch_geometric.set_debug(True)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, with_bn=True, seed=None):
        super(GraphSAGE, self).__init__()
        torch.cuda.manual_seed(seed)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.with_bn = with_bn
    
    def encoded_input2_graph(self, encoded_input):
        pass
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)