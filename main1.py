# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:39:02 2023

@author: Administrator
"""

import torch
import numpy
import argparse
from dataset import Dataset
from model1 import *
import torch.nn.functional as F
from utils1 import *
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_discriminator(discriminator, criterion, graphbuilder, optimizer_discriminator, features, edges, args):
    discriminator.train()
    criterion.eval()
    graphbuilder.eval()
    
    optimizer_discriminator.zero_grad()
    
    # adj_lp, adj_hp, weights_lp, weights_hp = discriminator(torch.cat((features, str_encodings), 1), edges)
    adj_lp, adj_hp, weights_lp, weights_hp = discriminator(features, edges)
    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1])  
    psu_label = torch.ones(edges.shape[1]).to(device)
    
    embedding, emb_lp, emb_hp = discriminator.get_embedding(features, adj_lp, adj_hp)
    # emb_lp, emb_hp = embedding, embedding
    
    edge_emb_sim_lp = F.cosine_similarity(emb_lp[edges[0]], emb_lp[edges[1]])  
    rnp_emb_sim_lp = F.cosine_similarity(emb_lp[rand_np[0]], emb_lp[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim_lp, rnp_emb_sim_lp, psu_label, margin=args.margin_hom, reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)

    edge_emb_sim_hp = F.cosine_similarity(emb_hp[edges[0]], emb_hp[edges[1]])
    rnp_emb_sim_hp = F.cosine_similarity(emb_hp[rand_np[0]], emb_hp[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim_hp, psu_label, margin=args.margin_het, reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)
    
    loss_rank = (loss_lp.mean() + loss_hp.mean()) / 2
    loss_rank.backward()
    optimizer_discriminator.step()

    return loss_rank.item(), adj_lp, adj_hp, embedding, emb_lp, emb_hp, weights_lp, weights_hp
                  

def train_division(discriminator, criterion, graphbuilder, optimizer_criterion, idx_train, features, edges, label_noise, epoch):   
    discriminator.train()
    graphbuilder.eval()
    criterion.train()
    
    adj_lp, adj_hp, weights_lp, weights_hp = discriminator(features, edges)
    embedding, emb_lp, emb_hp = discriminator.get_embedding(features, adj_lp, adj_hp)
    
    optimizer_criterion.zero_grad()
    
    loss_division, train_clean_list, train_fuzzy_list, train_remain_list, train_weight, x_lp, x_hp  = criterion(idx_train, emb_lp, emb_hp, label_noise, epoch)
        
    loss_division.backward()
    optimizer_criterion.step()
    
    return loss_division, train_clean_list, train_fuzzy_list, train_remain_list, train_weight, x_lp, x_hp

def train_graphbuild(g1, discriminator, criterion, graphbuilder, optimizer_builder, label_correct, features, edges, batch, idx_train, label_noise, epoch, train_node_prob):  
    discriminator.eval()
    criterion.eval()
    graphbuilder.train()
    
    adj_lp, adj_hp, weights_lp, weights_hp = discriminator(features, edges)
    embedding, emb_lp, emb_hp = discriminator.get_embedding(features, adj_lp, adj_hp)
    _, train_clean_list, train_fuzzy_list, train_remain_list, train_weight, x_lp, x_hp  = criterion(idx_train, emb_lp, emb_hp, label_noise, epoch)
    
    
    label_correct[train_fuzzy_list] = 1-label_correct[train_fuzzy_list]

    
    graphbuilder.newGraph(embedding, weights_lp)
    
    # train_node_prob = node_proability(g1, idx_train, label_correct)
    train_node_prob =  train_node_prob * np.array(train_weight.cpu().detach())
    train_node_prob = train_node_prob/np.sum(train_node_prob)
    num_batches = int(len(idx_train)/batch) + 1
    
    loss = 0
    
    for i in range(num_batches):
        new_train_batch = np.random.choice(a=idx_train, size = batch, replace=True, p=train_node_prob).tolist()
        
        optimizer_builder.zero_grad()
        
        loss_builder_temp = graphbuilder.loss(label_correct, new_train_batch, embedding)
        
        loss_builder_temp.backward(retain_graph=True)
        optimizer_builder.step()
        
        loss = loss + loss_builder_temp.item()
    
    loss_builder = loss / num_batches
    
    return loss_builder


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="elliptic", choices=['elliptic', 'yelp'], help='dataset')
parser.add_argument('--training_rate', type=float, default = 0.4)
parser.add_argument('--seed', type=int, default = None, help='For dataset split')
parser.add_argument('--emb_dim', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=64)   # for edge weight in Edge_Discriminator 
parser.add_argument('--nlayers', type=int, default=2)
'''noise ratio'''
parser.add_argument('--noise_ratio', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=2, help="For noise generation")
parser.add_argument('--nclass', type=int, default=2)
'''allpha'''
parser.add_argument('--alpha', type=float, default=0.1)             
parser.add_argument('--k', type=int, default=20) 
'''Top-K nearest neighbors'''     
parser.add_argument("--K", type=int, default=20)              
parser.add_argument('--maskfeat_rate_1', type=float, default=0.)
parser.add_argument('--maskfeat_rate_2', type=float, default=0.)
parser.add_argument('--dropedge_rate_1', type=float, default=0.)
parser.add_argument('--dropedge_rate_2', type=float, default=0.)
parser.add_argument('--margin_hom', type=float, default=0.5)
parser.add_argument('--margin_het', type=float, default=0.5)
parser.add_argument('--lr_gcl', type=float, default=0.01)
parser.add_argument('--lr_disc', type=float, default=0.01)
parser.add_argument('--lr_criterion', type=float, default=0.01)
parser.add_argument('--lr_builder', type=float, default=0.01)
parser.add_argument('--cl_batch_size', type=int, default=4068)          
parser.add_argument('--w_decay', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--tau', type=int, default=0.9)               
parser.add_argument('--batch', type=int, default=512)

args = parser.parse_args()

name = args.dataset
training_rate = args.training_rate
seed  = args.seed
dataset = Dataset(name, training_rate, seed)
features, edges, idx_train, idx_val, idx_test, label, nnodes, nfeats, g1 = dataset.load_data(name)
#narray   torch     tensor         list       list     list  tensor    int    int

g1=g1.to(device)

# noise label generation
label_noise, train_label, train_label_noise, train_idx_noise, train_idx_clean = noisify_with_P2(label, idx_train, args.noise_ratio, args.random_state, args.nclass)
idx_unlabel = idx_val + idx_test

features = torch.tensor(features).to(device)
edges = edges.to(device)
label = label.to(device)
label_noise = label_noise.to(device)

print('build model')
# build model
nlayers = args.nlayers
in_dim = nfeats
emb_dim = args.emb_dim
input_dim = nfeats
hidden_dim = args.hidden_dim
alpha = args.alpha
dropout = args.dropout
nclass = 2
K = args.K

discriminator = Edge_Discriminator(nlayers, nnodes, in_dim, emb_dim, input_dim, hidden_dim, alpha, dropout).to(device)
criterion = LabelDivision(args).to(device)
graphbuilder = RebuitGraph(args, idx_unlabel, nlayers, in_dim, emb_dim, dropout, nclass, edges, features, K, idx_train, nnodes)


optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)
optimizer_criterion = torch.optim.Adam(criterion.parameters(), lr=args.lr_criterion, weight_decay=args.w_decay)
optimizer_builder = torch.optim.Adam(graphbuilder.parameters(), lr=args.lr_builder, weight_decay=args.w_decay)


best_auc_recall = 0
best_auc_recall1 = 0
best_auc = 0

num_clean = []
num_fuzzy = []
num_remain = []

clean_ratio_list = []
fuzzy_ratio_list = []
auc_list = []
recall_list = []
F1_list = []

weights_lp_list, weights_hp_list = [],[]

a = torch.sum(label_noise[idx_val] != label[idx_val])
b = torch.sum(label_noise[idx_test] != label[idx_test])
if (a+b >0):
    print('information disclosure!')
else:
    print('information is ok!')


print('begin to train')
print('Training ratio: {}, Noise ratio:{}'.format(args.training_rate, args.noise_ratio))


train_node_prob = node_proability(g1, idx_train, label_noise)

for epoch in range(1, args.epochs+1):
    label_correct = label_noise.clone()
    
    loss_rank, adj_lp, adj_hp, embedding, emb_lp, emb_hp, weights_lp, weights_hp = train_discriminator(discriminator, criterion, graphbuilder, optimizer_discriminator, features, edges, args)                                                                                     
    loss_division, train_clean_list, train_fuzzy_list, train_remain_list, train_weight, x_lp, x_hp  = train_division(discriminator, criterion, graphbuilder, optimizer_criterion, idx_train, features, edges, label_noise, epoch)
    loss_builder = train_graphbuild(g1, discriminator, criterion, graphbuilder, optimizer_builder, label_correct, features, edges, args.batch, idx_train, label_noise, epoch, train_node_prob)

    num_clean.append(len(train_clean_list))
    num_fuzzy.append(len(train_fuzzy_list))
    num_remain.append(len(train_remain_list))
    
    clean_ratio = len(set(train_clean_list) & set(np.array(train_idx_clean)))/(1+len(set(train_clean_list)))
    fuzzy_ratio = len(set(train_fuzzy_list) & set(np.array(train_idx_noise)))/(1+len(set(train_fuzzy_list)))
    clean_ratio_list.append(clean_ratio)
    fuzzy_ratio_list.append(fuzzy_ratio)
    
    weights_lp_list.append(weights_lp)
    weights_hp_list.append(weights_hp)
     
    print("[TRAIN] Epoch:{:04d} |loss_rank：{:.4f} |loss_divisions:{:.4f} |loss_builder:{:.4f}".format(epoch, loss_rank, loss_division, loss_builder))
    
    if epoch % 1 == 0:
        criterion.eval()
        discriminator.eval()
        graphbuilder.eval()
        str1 = 'Val: '
        val_auc_gnn, val_precision_gnn, val_Recall, val_f1_score = test_model(graphbuilder, idx_val, label, str1)
        
        if (val_auc_gnn+val_f1_score)>best_auc_recall:
            best_auc_recall = (val_auc_gnn+val_f1_score)
            test_auc_gnn, test_precision_gnn, test_Recall, test_f1_score = test_model(graphbuilder, idx_test, label, str1)
            best_epoch = epoch
            result = [best_epoch, test_auc_gnn, test_precision_gnn, test_Recall, test_f1_score]
            
        str1 = 'Test: '
        test_auc_gnn, test_precision_gnn, test_Recall, test_f1_score = test_model(graphbuilder, idx_test, label, str1)
        auc_list.append(test_auc_gnn)
        recall_list.append(test_Recall)
        F1_list.append(test_f1_score)

best_epoch, auc_gnn, precision_gnn, recall_gnn, f1_gnn = result[0], result[1], result[2], result[3], result[4]
print('***************************************************')
print('Test: ' + f"Best epoch: {best_epoch-1}")
print('Test: ' + f"GNN auc: {auc_gnn:.4f}")
# print('Test: ' +  f"GNN precision: {precision_gnn:.4f}")
print('Test: ' + f"GNN Recall: {recall_gnn:.4f}")
print('Test: ' + f"GNN f1_score: {f1_gnn:.4f}")    
print('Test: ' + f"GNN auc: {best_auc:.4f}")      
print('Training ratio: {}, Noise ratio:{}, Knn-K: {}'.format(args.training_rate, args.noise_ratio, args.K))      
print(args)    

plt.figure(1)
plt.plot(num_clean, label='num_clean')
plt.plot(num_fuzzy, label='num_fuzzy')
plt.plot(num_remain,label='num_remain')
plt.legend()
csv1 = np.vstack((num_clean,num_fuzzy, num_remain)).T
name1 = ['num_clean', 'num_fuzzy', 'num_remain']
writerCSV1=pd.DataFrame(columns=name1,data=csv1)
writerCSV1.to_csv('./csv1_num.csv',encoding='utf-8')


plt.figure(2)
plt.plot(clean_ratio_list, label='clean_ratio_list')
plt.plot(fuzzy_ratio_list, label='fuzzy_ratio_list')
plt.legend()
csv2 = np.vstack((clean_ratio_list,fuzzy_ratio_list)).T
name2 = ['clean_ratio_list', 'fuzzy_ratio_list']
writerCSV2=pd.DataFrame(columns=name2,data=csv2)
writerCSV2.to_csv('./csv2_ratio.csv',encoding='utf-8')


plt.figure(3)
plt.plot(auc_list, label='auc_list')
# plt.plot(recall_list, label='recall_list')
plt.plot(F1_list, label='F1_list')
plt.legend()
csv3 = np.vstack((auc_list,F1_list)).T
name3 = ['auc_list', 'F1_list']
writerCSV3=pd.DataFrame(columns=name3,data=csv3)
writerCSV3.to_csv('./csv3_perfermance.csv',encoding='utf-8')


clean_ratio = len(set(train_clean_list) & set(np.array(train_idx_clean)))/(1+len(set(train_clean_list)))
fuzzy_ratio = len(set(train_fuzzy_list) & set(np.array(train_idx_noise)))/(1+len(set(train_fuzzy_list)))

print('clean_ratio:{}'.format(clean_ratio))
print('fuzzy_ratio:{}'.format(fuzzy_ratio))


print('K:', args.K)
print('alpha:', args.alpha)
print('seed:', args.seed)

#weights_lp = weights_lp_list[best_epoch]
#weights_lp = weights_lp_list[best_epoch]
#calculate(g1, weights_lp, weights_hp, label)   
    
# np.load('clean-fuzzy-remain.npz')   
# np.load('clean-fuzzy-ratio.npz')
# np.load('auc-recall-f1.npz')