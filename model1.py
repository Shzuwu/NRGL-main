# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:30:01 2023

@author: Administrator
"""

import torch
import os
import torch.nn as nn
import dgl
from scipy import sparse
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl.function as fn
import random
from dgl.nn import EdgeWeightNorm
from utils1 import *
import numpy as np

EOS = 1e-10
norm = EdgeWeightNorm(norm='both')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Edge_Discriminator(nn.Module):
    def __init__(self, nlayers, nnodes, in_dim, emb_dim, input_dim, hidden_dim, alpha, dropout, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()
        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(2*hidden_dim, 1)
        self.nnodes = nnodes
        self.temperature = temperature
        self.bias = bias
        self.alpha = alpha
        
        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout) 
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout)
        
    def get_embedding(self, features, adj_lp, adj_hp, source='all'):
        emb_lp = self.encoder1(features, adj_lp)
        emb_hp = self.encoder2(features, adj_hp)
        return torch.cat((emb_lp, emb_hp), dim=1), emb_lp, emb_hp

    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h
    
    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1+s2)/2
        
    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()
    
    def weight_forward(self, features, edges):
        embeddings = self.get_node_embedding(features)                             
        edges_weights_raw = self.get_edge_weight(embeddings, edges)                 
        weights_lp = self.gumbel_sampling(edges_weights_raw)                        
        weights_hp = 1 - weights_lp                                                 
        return weights_lp, weights_hp
    
    def weight_to_adj(self, edges, weights_lp, weights_hp):                        
        adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device=device)            
        adj_lp = dgl.add_self_loop(adj_lp)
        weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes).to(device))) + EOS  
        weights_lp = norm(adj_lp, weights_lp)
        # weights_lp *= self.alpha
        # weights_lp[edges.shape[1]:] = 1
        adj_lp.edata['w'] = weights_lp
        
        adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device=device)
        adj_hp = dgl.add_self_loop(adj_hp)
        weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes).to(device))) + EOS
        weights_hp = norm(adj_hp, weights_hp)
        weights_hp *= - self.alpha                                                  
        weights_hp[edges.shape[1]:] = 1                                            
        adj_hp.edata['w'] = weights_hp 
        return adj_lp, adj_hp
        
    def forward(self, features, edges):
        weights_lp, weights_hp = self.weight_forward(features, edges)
        adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp) 
        return adj_lp, adj_hp, weights_lp, weights_hp
      

class SGC(nn.Module):
    def __init__(self, nlayer, in_dim, emb_dim, dropout):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.k = nlayer
        self.linear = nn.Linear(in_dim, emb_dim)
       
    def forward(self, x, g):
        x = torch.relu(self.linear(x))
        
        with g.local_scope():
            g.ndata['h'] = x
            for _ in range(self.k):
                g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                return g.ndata['h']
                

class LabelDivision(nn.Module):
    def __init__(self, args):
        super(LabelDivision, self).__init__()
        self.args = args
        self.epochs = args.epochs
        
        self.gate1 = nn.Linear(args.emb_dim, 2).to(device)
        self.gate2 = nn.Linear(args.emb_dim, 2).to(device)
        torch.nn.init.xavier_uniform_(self.gate1.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.gate2.weight, gain=1.414)
        
        self.increment = 0.5/args.epochs
    
    def to_prob(self, idx_test):
        x_test_lp = self.emb_lp[idx_test]
        x_test_hp = self.emb_hp[idx_test]
        
        x_lp = self.gate1(x_test_lp)
        x_hp = self.gate1(x_test_hp)
        
        x_lp_score = torch.sigmoid(x_lp)
        x_hp_socre = torch.sigmoid(x_hp)
        
        return x_lp_score, x_hp_socre
        
        
    def forward(self, idx_train, emb_lp, emb_hp, label_noise, epoch):
        self.emb_lp = emb_lp
        self.emb_hp = emb_hp
        
        x_train_lp = self.emb_lp[idx_train]
        x_train_hp = self.emb_hp[idx_train]
        y_train_noise = label_noise[idx_train]
        
        num = len(y_train_noise)/torch.sum(y_train_noise)
        weight=torch.tensor([1.,num]).to(device)   #repaired 20230801
        
        x_lp = self.gate1(x_train_lp)
        x_hp = self.gate2(x_train_hp)
        
        loss_pick_lp = F.cross_entropy(x_lp, y_train_noise, reduce=False)
        loss_pick_hp = F.cross_entropy(x_hp, y_train_noise, reduce=False)
        loss_pick = loss_pick_lp + loss_pick_hp
        
        # calculate the ratio threshold of adatption
        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        forget_rate = self.increment*epoch
        remember_rate = 1 - forget_rate
        
        # calculate the ratio threshold of mean
        mean_v = loss_sorted.mean()
        idx_small = torch.where(loss_sorted < mean_v)[0]
        remember_rate_small = idx_small.shape[0]/y_train_noise.shape[0]
        # calculate the final ratio threshold
        remember_rate = max(remember_rate, remember_rate_small)
        
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        
        # calculate loss for loss_clean
        loss_clean = torch.sum(loss_pick[ind_update])
        
        ind_all = torch.arange(y_train_noise.shape[0]).long()           # KX torch.arange(y_train_noise.shape[0])
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(ind_update.detach().cpu().numpy()))).to(device)
        p_1 = F.softmax(x_lp, dim=-1)
        p_2 = F.softmax(x_hp, dim=-1)
        
        # repaired  min(0.5, 1/x_lp.shape[1])
        filter_condition = ((x_lp.max(dim=1)[1][ind_update_1] != y_train_noise[ind_update_1]) &
                            (x_lp.max(dim=1)[1][ind_update_1] == x_hp.max(dim=1)[1][ind_update_1]) &
                             (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > (1-(1-min(0.5, 1/x_lp.shape[1]))*epoch/self.args.epochs)))
        dc_idx = ind_update_1[filter_condition]
        
        adpative_weight = (p_1.max(dim=1)[0][dc_idx]*p_2.max(dim=1)[0][dc_idx])**(0.5-0.5*epoch/self.args.epochs)
        loss_fuzzy = 0.8*adpative_weight*(F.cross_entropy(x_lp[dc_idx], x_lp.max(dim=1)[1][dc_idx], weight, reduce=False) +
                                   F.cross_entropy(x_hp[dc_idx], x_hp.max(dim=1)[1][dc_idx], weight, reduce=False))
        
        loss_fuzzy = loss_fuzzy.sum()
    
        remain_idx = torch.LongTensor(list(set(ind_update_1.detach().cpu().numpy())-set(dc_idx.detach().cpu().numpy())))
        
        loss_remain = 0.5*torch.sum(loss_pick[remain_idx])
        
        # loss_division = (loss_clean)/len(ind_update)
        # loss_division = (loss_clean + loss_dc)/(len(ind_update) + len(dc_idx))
        loss_division = (loss_clean + loss_fuzzy + loss_remain)/(len(idx_train))
        
        # # for clean loss
        # loss_clean_lp = F.cross_entropy(x_lp[ind_update], y_train_noise[ind_update], weight, reduce=False)
        # loss_clean_hp = F.cross_entropy(x_hp[ind_update], y_train_noise[ind_update], weight, reduce=False)
        # loss_clean = torch.sum(loss_clean_lp +loss_clean_hp)/len(ind_update)
        
        # # for remain loss
        # loss_remain_lp = F.cross_entropy(x_lp[remain_idx], y_train_noise[remain_idx], weight, reduce=False)
        # loss_remain_hp = F.cross_entropy(x_hp[remain_idx], y_train_noise[remain_idx], weight, reduce=False)
        # loss_remain = 0.5*torch.sum(loss_remain_lp +loss_remain_hp)/len(remain_idx)
        
        # loss_division = loss_clean + loss_fuzzy + loss_remain
        
        train_clean_list = list(np.array(idx_train)[np.array(ind_update.cpu())])
        train_fuzzy_list = list(np.array(idx_train)[np.array(dc_idx.cpu())])
        train_remain_list= list(np.array(idx_train)[np.array(remain_idx.cpu())])
        
        self.adpative_weight = adpative_weight
        self.ind_update = ind_update
        
        train_weight = torch.ones(len(idx_train)).to(device)
        train_weight[dc_idx] = adpative_weight*0.8
        train_weight[remain_idx] = 0.5 * train_weight[remain_idx]*0.8
        
        # train_weight = torch.cat([torch.ones(len(self.ind_update)).to(device), self.adpative_weight, (1/x_lp.shape[1])*torch.ones(len(train_remain_list)).to(device)])
        return loss_division, train_clean_list, train_fuzzy_list, train_remain_list, train_weight, x_lp, x_hp 
    
        
    # rebuild the graph
class RebuitGraph(nn.Module):
    def __init__(self, args, idx_unlabel, nlayers, in_dim, emb_dim, dropout, nclass, edges, features:torch.Tensor, K, idx_train, nnodes):
        super(RebuitGraph, self).__init__()
        self.idx_unlabel = torch.tensor(idx_unlabel) 
        self.edges = edges
        self.features = torch.FloatTensor(features.detach().clone().cpu().numpy()).cuda()

        self.idx_train = idx_train
        self.args = args
        
        self.GCN = SGC(nlayers, 2*emb_dim, emb_dim, dropout).to(device)
        self.mlp = nn.Linear(emb_dim, nclass).to(device)
        nn.init.xavier_normal_(self.mlp.weight, gain=1.414)
        
        self.pred_edge_index = self.KNN(edges, features, K, idx_train, args.dataset, args.seed)
        self.nnodes = nnodes
        
    def newGraph(self, embedding, weights_lp):

        predictor_weights = self.get_estimated_weigths(self.pred_edge_index, embedding)
        edges_idx_add = torch.where(predictor_weights != 0)[0].detach()
        '''For edges'''
        edges_add = self.pred_edge_index[:, edges_idx_add]
        e1 = torch.arange(self.nnodes).unsqueeze(0)
        edges_add1 = torch.cat([e1,e1], dim=0).to(device)
        '''For edge weight'''
        edges_weight_add = predictor_weights[edges_idx_add]
        edges_weight_add1 = torch.ones(self.nnodes).to(device)
        
        edges_total = torch.cat([self.edges, edges_add, edges_add1], dim=1)
        edges_weight_total = torch.cat([weights_lp, edges_weight_add, edges_weight_add1]) + EOS
        self.adj = dgl.graph((edges_total[0], edges_total[1]), num_nodes=self.nnodes, device = device)
        
        weight = norm(self.adj, edges_weight_total)
        self.adj.edata['w'] = weight
        
    def forward(self,features):
        representations = self.GCN(features, self.adj)
        self.outputs = self.mlp(representations)
        
        return self.outputs

    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
#        output = torch.div(torch.sum(torch.mul(x0, x1), dim=1), x0.norm(dim=1)*x1.norm(dim=1))
        output = torch.cosine_similarity(x0,x1)
        estimated_weights = output
        
        estimated_weights[estimated_weights < self.args.tau] = 0
        return estimated_weights
        
    def loss(self, label_correct, new_train, features):
        outputs = self.forward(features)
        loss_cl = F.cross_entropy(outputs[new_train], label_correct[new_train])
        return loss_cl
    
    def to_prob(self, idx):
        outputs = self.outputs[idx]
        outputs_prob = torch.sigmoid(outputs)
        return outputs_prob
        
    
    def KNN(self, edges, features, K, idx_train, dataset, seed):
        path = './data/knn/'
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = path + '/{}_{}_{}.pt'.format(dataset, K, seed)
        # if os.path.exists(file_name):
        #     poten_edges = torch.load(file_name)
        #     return poten_edges
        
        if K == 0:
            return torch.LongTensor([])
        poten_edges = []
        if K > len(idx_train):
            print('K is larger than the size of training set')
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edges[1, edges[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in idx_train:
                # sim = torch.div(torch.matmul(features[i], features[self.idx_idx_unlabel].T), features[i].norm() * features[self.idx_unlabel].norm(dim=1))
                sim = torch.cosine_similarity(features[i].unsqueeze(dim=0), features[self.idx_unlabel])
                _, rank = sim.topk(K)
                indices = self.idx_unlabel[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
            for i in self.idx_unlabel:
                sim = torch.cosine_similarity(features[i].unsqueeze(dim=0), features[idx_train])
                _, rank = sim.topk(K)
                indices = torch.tensor(idx_train)[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        edge_index = list(edges.T)
        poten_edges = set([tuple(t) for t in poten_edges]) - set([tuple(t) for t in edge_index])
        poten_edges = [list(s) for s in poten_edges]
        poten_edges = torch.as_tensor(poten_edges).T.to(device)
        
        torch.save(poten_edges, file_name)
        return poten_edges
          

    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        