import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing, GCNConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.autograd import Variable, grad

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def make_packer(n, n_f):
    def pack(x):
        return x.reshape(-1, n_f*n)
    return pack

def make_unpacker(n, n_f):
    def unpack(x):
        return x.reshape(-1, n, n_f)
    return unpack

def get_edge_index(n, sim):
    if sim in ['string', 'string_ball']:
        #Should just be along it.
        top = torch.arange(0, n-1)
        bottom = torch.arange(1, n)
        edge_index = torch.cat(
            (torch.cat((top, bottom))[None],
             torch.cat((bottom, top))[None]), dim=0
        )
    else:
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = torch.from_numpy(np.array(np.where(adj)))

    return edge_index


class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim)
        )
    
    #[docs]
    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        return self.msg_fnc(torch.cat([x_i, x_j], dim=1))
    
    def update(self, aggr_out, x=None):
        return self.node_fnc(torch.cat([x, aggr_out], dim=1))

    
    
class GNEDGE(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(GNEDGE, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2*n_f+1, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim)
        )
    
    #[docs]
    def forward(self, x, edge_index, edge_attr):
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
      
    def message(self, x_i, x_j, edge_attr):
        
        tmp = torch.cat([x_i, x_j, edge_attr.view(-1, 1)], dim=1)
        return self.msg_fnc(tmp)
    
    def update(self, aggr_out, x=None):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp)
    
    

class OGN(GN):
    def __init__(self, n_f, msg_dim, ndim, dt, edge_index, aggr='add', hidden=300, nt=1):

        super(OGN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim
    
    def loss(self, actual, pred):
        return torch.sum(torch.abs(actual - pred))
    
class OGNEDGE(GNEDGE):
    def __init__(self, n_f, msg_dim, ndim, aggr='add', hidden=300, nt=1):
        super(OGNEDGE, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.nt = nt
        self.ndim = ndim
    
    def loss(self, actual, pred):
        return torch.sum(torch.abs(actual - pred))
