#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:41:42 2023

@author: Administrator
"""

import torch
import argparse
import dgl
from dataset import Dataset
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from utils1 import *
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        
class NN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        
        
    def forward(self, features):
        h = torch.nn.functional.dropout(features)
        h = self.linear1(h)
        self.h = self.linear2(h)
        return self.h
    
    def to_prob(self, idx):
        out = self.h[idx]
        out = torch.sigmoid(out)
        return out
    
def evaluation(net, idx_val, label, dataset):
    net.eval()
    
    labels = label[idx_val].cpu().numpy()
    gnn_prob = net.to_prob(idx_val)
    
    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1_gnn = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    
    print(dataset + f"GNN auc: {auc_gnn:.4f}")
    print(dataset + f"GNN precision_gnn: {precision_gnn:.4f}")
    print(dataset + f"GNN Recall: {recall_gnn:.4f}")
    print(dataset + f"GNN f1_score: {f1_gnn:.4f}")
    
    return auc_gnn, precision_gnn, recall_gnn, f1_gnn    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="elliptic", choices=['elliptic', 'yelp'], help='dataset')
parser.add_argument('--training_rate', type=float, default = 0.4)
parser.add_argument('--seed', type=int, default = None, help='For dataset split')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=64)   # for edge weight in Edge_Discriminator 
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--noise_ratio', type=float, default=0.1)
parser.add_argument('--random_state', type=int, default=None, help="For noise generation")
parser.add_argument('--nclass', type=int, default=2)
parser.add_argument('--w_decay', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=512)
args = parser.parse_args()


'''load data'''
name = args.dataset
training_rate = args.training_rate
seed  = args.seed
dataset = Dataset(name, training_rate, seed)
features, edges, idx_train, idx_val, idx_test, label, nnodes, nfeats, g1 = dataset.load_data(name)


'''add label noise'''
label_noise, train_label, train_label_noise, train_idx_noise, train_idx_clean = noisify_with_P2(label, idx_train, args.noise_ratio, args.random_state, args.nclass)
weight = torch.sum(1-train_label_noise)/torch.sum(train_label_noise)

g1 = dgl.add_self_loop(g1)


net = NN(nfeats, args.hidden_dim, args.nclass)
optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
features = torch.tensor(features)

best_auc = 0
result = []

for epoch in range(100):
    net.train()
    logits =  net(features)
    loss = F.cross_entropy(logits[idx_train], label_noise[idx_train], weight=torch.tensor([1.,weight]))
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch%2==0):
        auc_gnn, precision_gnn, recall_gnn, f1_gnn = evaluation(net, idx_test, label_noise, name)
        if auc_gnn > best_auc:
            best_auc = auc_gnn
            best_epoch = epoch
            result = [best_epoch, auc_gnn, precision_gnn, recall_gnn, f1_gnn]

print('*****************************************')
print('Method: NN')
print('dataset:', args.dataset)
print('noise_ratio:',args.noise_ratio)
print(name + f"GNN best_epoch: {result[0]:.4f}")
print(name + str(args.noise_ratio) + f"GNN auc: {result[1]:.4f}")
print(name + str(args.noise_ratio) +f"GNN precision_gnn: {result[2]:.4f}")
print(name + str(args.noise_ratio) +f"GNN Recall: {result[3]:.4f}")
print(name + str(args.noise_ratio) +f"GNN f1_score: {result[4]:.4f}")  

