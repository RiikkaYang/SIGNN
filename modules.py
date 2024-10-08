import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import dgl
from dgl.nn import GraphConv
import math
import torch.nn.functional as F

class EmbedProfiles(nn.Module):
    def __init__(self, in_feats, hidden_features, out_feats):
        super(EmbedProfiles, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_features)
        self.conv2 = GraphConv(hidden_features, out_feats)
        self.dropout = nn.Dropout(0.5)
        self.linear_att = nn.Linear(5, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph, features):
        features = features.view(features.size(-1), -1)
        x = self.conv1(graph, features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(graph, x)
        x = torch.relu(x)
        
        att = self.softmax(self.linear_att(x))
        z = torch.mean(att * x, dim=1)
        return z, att
    
class MLPNet(nn.Module): 
    def __init__(self, num_users, num_hidden_layers, hidden_features, out_features=1,
                 outermost_linear='sigmoid', nonlinearity='relu', use_graph=True):
        super(MLPNet, self).__init__()

        nls = {'relu': nn.ReLU(inplace=True), 
               'sigmoid': nn.Sigmoid(), 
               'tanh': nn.Tanh(), 
               'selu': nn.SELU(inplace=True), 
               'softplus': nn.Softplus(), 
               'elu': nn.ELU(inplace=True)}

        nl = nls[nonlinearity]
        nl_outermost = nls[outermost_linear]

        self.use_graph = use_graph
        if use_graph:
            self.embed_profiles = EmbedProfiles(1, hidden_features, 5)

        self.hidden_features = hidden_features

        self.embed_users = [] 
        self.embed_users.append(nn.Sequential(
                nn.Embedding(num_users, hidden_features), nl
        ))
        for i in range(num_hidden_layers-1):
            self.embed_users.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_users = nn.Sequential(*self.embed_users)

        self.embed_times = []
        self.embed_times.append(nn.Sequential(
            nn.Linear(1, hidden_features), nl
        ))
        for i in range(num_hidden_layers-1):
            self.embed_times.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_times = nn.Sequential(*self.embed_times)

        self.net = []
        if use_graph:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features*3, hidden_features), nl
            ))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features*2, hidden_features), nl
            ))
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nl_outermost
        ))
        self.net = nn.Sequential(*self.net)

    def forward(self, times, users, profs=None, params=None, **kwargs):
        x = self.embed_times(times.float())
        y = self.embed_users(users.long())
        y = torch.squeeze(y, dim=1)

        if self.use_graph: 
            graph = profs
            features = graph.ndata['feat']
            z, att = self.embed_profiles(graph, features.float())
            z = torch.mean(att*z, axis=1)
            z = z.unsqueeze(-1)
            expanded_z = z.repeat(1,8)
            combined = torch.cat([x, y, expanded_z], dim=-1)
        else: 
            combined = torch.cat([x, y], dim=-1)
        output = self.net(combined)

        if self.use_graph: 
            return output, att
        else:
            return output