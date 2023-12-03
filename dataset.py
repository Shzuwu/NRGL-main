# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:28:36 2023

@author: Administrator
"""
import argparse
import torch
import pickle
import os
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pickle as pkl
from torch_geometric.utils import remove_self_loops
from utils1 import normalize_row
import dgl

class Dataset():
    def __init__(self, name, training_rate, seed):
        self.name = name
        self.training_rate = training_rate
        self.seed = seed
        
        
    def load_data(self, name):
        print('Loading {} dataset...'.format(name))
        data = pkl.load(open('dataset/{}.dat'.format(name), 'rb'))
        
        features = data.x                                       # torch
        features = normalize_row(features)
        [nnodes, nfeats] = features.shape
        
        # for unbidirection and sort the node
        edges1 = remove_self_loops(data.edge_index)[0]           # torch
        g= dgl.graph((edges1[0], edges1[1]), num_nodes=nnodes)
        g1= dgl.to_bidirected(g)
        edges = g1.edges()
        edges = torch.cat([edges[0].unsqueeze(0), edges[1].unsqueeze(0)], dim=0)
        
        idx_train, idx_val, idx_test = self.get_split(nnodes, self.training_rate, self.seed)
        
        label = data.y
        fraud_rate = torch.sum(label)/len(label)
        print("{} datast is of {} fruad ratio".format(name, fraud_rate))    
    
        return features, edges, idx_train, idx_val, idx_test, label, nnodes, nfeats, g1
        
        
    def get_split(self, nnodes, training_rate, seed):
        idx = list(range(nnodes))
        idx_train, idx_left = train_test_split(idx, train_size = training_rate,
                                               random_state = seed, shuffle = True)
        idx_val, idx_test = train_test_split(idx_left, test_size = 0.6666,
                                             random_state = seed, shuffle = True)
        return idx_train, idx_val, idx_test
    
    def get_structural_encoding(self, edges, nnodes, dim=16):
        row = edges[0, :].numpy()
        col = edges[1, :].numpy()
        data = np.ones_like(row)
        
        A = sp.csr_matrix((data, (row, col)), shape=(nnodes, nnodes))
        D = (np.array(A.sum(1)).squeeze()) ** -1.0
        
        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW
        
        SE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        
        for _ in range(dim - 1):
            M_power = M_power * M
            SE.append(torch.from_numpy(M_power.diagonal()).float())
        SE = torch.stack(SE, dim=-1)
        return SE
        
        
        
        
        
        
        
        
        
        
        
        
        