#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:21:03 2023

@author: qiang
"""

import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.uniform_(layer.bias, a=-0.1, b=0.1)

class FilterNet(nn.Module):
    def __init__(self,
                 args,
                 bias=True):

        super(FilterNet, self).__init__()
        self.max_action = args.max_action
        self.net_obs = nn.Sequential(nn.Linear(args.obs_dim, 512, bias=bias),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 512, bias=bias),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256, bias=bias),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128, bias=bias),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Linear(128, args.action_dim*2, bias=bias),
                                     )

        self.net = nn.Sequential(nn.Linear(args.action_dim*3, 128, bias=bias),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128, bias=bias),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 2, bias=bias),
                                 nn.Softmax(dim=1),
                                 )

    def forward(self, o, a):
        o = self.net_obs(o)
        x = torch.cat((o, a), 1)
        x = self.net(x)
        return x
