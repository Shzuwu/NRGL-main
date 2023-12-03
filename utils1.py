# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:19:09 2023

@author: Administrator
"""

import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_model(graphbuilder, idx_val, label, str0):

    labels = label[idx_val].cpu().numpy()
    gnn_prob = graphbuilder.to_prob(idx_val)
    
    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1_gnn = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    
    print(str0 + f"GNN auc: {auc_gnn:.4f}")
    # print(str0 + f"GNN precision_gnn: {precision_gnn:.4f}")
    print(str0 + f"GNN Recall: {recall_gnn:.4f}")
    print(str0 + f"GNN f1_score: {f1_gnn:.4f}")
    
    return auc_gnn, precision_gnn, recall_gnn, f1_gnn

def test_model1(criterion, idx_val, label, str1):
    labels = label[idx_val].cpu().numpy()
    gnn_prob_lp, gnn_prob_hp = criterion.to_prob(idx_val)
    
    gnn_prob = gnn_prob_lp
    
    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    # precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1_gnn = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average='macro')
    
    print(str1 + f"GNN_lp auc: {auc_gnn:.4f}")
    # print(str1 + f"GNN_lp precision_gnn: {precision_gnn:.4f}")
    print(str1 + f"GNN_lp Recall: {recall_gnn:.4f}")
    print(str1 + f"GNN_lp f1_score: {f1_gnn:.4f}")
    
    gnn_prob = gnn_prob_hp
    
    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    # precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1_gnn = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average='macro')
    
    print(str1 + f"GNN_hp auc: {auc_gnn:.4f}")
    # print(str1 + f"GNN_hp precision_gnn: {precision_gnn:.4f}")
    print(str1 + f"GNN_hp Recall: {recall_gnn:.4f}")
    print(str1 + f"GNN_hp f1_score: {f1_gnn:.4f}")
    
    
def node_proability(g1, idx_train, label_noise, w1=0.5, w2=1.5):
    node_degree = g1.in_degrees()
    node_degree = torch.pow(node_degree, w1)
    node_degree = np.array(node_degree.cpu())
    node_degree = node_degree[idx_train]
    
    
    y_train = label_noise[idx_train]
    fraud_rate = torch.sum(y_train)/len(y_train)
    fraud_rate = torch.pow(fraud_rate, w2)
    print('Fraud rate is {} in noise training set'.format(fraud_rate))
    
    norm_rate = 1 -fraud_rate
    node_prob = np.zeros(len(y_train))
    
    for i in range(len(y_train)):
        if y_train[i] == 1:
            node_prob[i] = node_degree[i]/fraud_rate
        elif y_train[i] == 0:
            node_prob[i] = node_degree[i]/norm_rate
    return node_prob

def normalize_row(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx
    
def noisify_with_P1(label, idx_train, noise_ratio, random_state, nclass):
    train_label = label[idx_train]
    P = np.float64(noise_ratio) / np.float64(nclass-1) * np.ones((nclass, nclass))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise_ratio))*np.ones(nclass))
    
    train_num = len(train_label)
    flipper = np.random.RandomState(random_state)
    train_label_noise = np.array(train_label)
    
    for idx in range(train_num):
        i = train_label[idx]
        random_result = flipper.multinomial(1, P[i,:], 1)[0]
        train_label_noise[idx] = np.where(random_result == 1)[0]
        
    actual_noise = (np.array(train_label_noise) != np.array(train_label)).mean()
    print('actual_noise: {}'.format(actual_noise))
    train_idx_noise = np.array(idx_train)[np.where(train_label_noise != np.array(train_label))]
    train_idx_clean = np.array(idx_train)[np.where(train_label_noise == np.array(train_label))]
    train_label_noise = torch.tensor(train_label_noise)
    
    label_noise = np.array(label)
    label_noise[idx_train] = train_label_noise
    label_noise = torch.tensor(label_noise)
    train_idx_noise = torch.tensor(train_idx_noise)
    train_idx_clean = torch.tensor(train_idx_clean)
    
    return label_noise, train_label, train_label_noise, train_idx_noise, train_idx_clean

def noisify_with_P2(label, idx_train, noise_ratio, random_state, nclass):
    train_label = label[idx_train]
    train_num = len(idx_train)
    train_select = np.random.choice(list(range(train_num)), size=int(train_num*noise_ratio), replace=False)
    train_label_noise = np.array(train_label)
    for i in range(len(train_select)):
        idx = train_select[i]
        train_label_noise[idx] = 1-train_label_noise[idx]
    label_noise = np.array(label)
    label_noise[idx_train] = train_label_noise
    
    actual_noise_ratio = (np.array(train_label_noise) != np.array(train_label)).mean()
    print('actual_noise_ratio:{}'.format(actual_noise_ratio))
    
    train_idx_noise = np.array(idx_train)[np.where(train_label_noise != np.array(train_label))]
    train_idx_clean = np.array(idx_train)[np.where(train_label_noise == np.array(train_label))]
    
    label_noise = torch.tensor(label_noise)
    train_label_noise = torch.tensor(train_label_noise)
    train_idx_noise = torch.tensor(train_idx_noise)
    train_idx_clean = torch.tensor(train_idx_clean)
    
    return label_noise, train_label, train_label_noise, train_idx_noise, train_idx_clean
        

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list
    
def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    if torch.cuda.is_available():
        mask = mask.to(device)
    return mask, samples

def augmentation(features_1, adj_1, features_2, adj_2, args, training):
    # view 1
    mask_1, _ = get_feat_mask(features_1, args.maskfeat_rate_1)
    features_1 = features_1 * (1 - mask_1)
    adj_1.edata['w'] = F.dropout(adj_1.edata['w'], p=args.dropedge_rate_1, training=training)

    # # view 2
    mask_2, _ = get_feat_mask(features_2, args.maskfeat_rate_2)
    features_2 = features_2 * (1 - mask_2)
    adj_2.edata['w'] = F.dropout(adj_2.edata['w'], p=args.dropedge_rate_2, training=training)

    return features_1, adj_1, features_2, adj_2

def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]  
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges.to(device)
    
    
def calculate(g1, weights_lp, weights_hp, label):
    from collections import defaultdict
    
    edges_idx = torch.cat([g1.edges()[0].unsqueeze(0), g1.edges()[1].unsqueeze(0)], dim=0)
    idx_lp = torch.where(weights_lp > 0.5)
    idx_hp = torch.where(weights_hp > 0.5)
    
    edges_idx_lp = edges_idx[:,idx_lp[0]]
    edges_idx_hp = edges_idx[:,idx_hp[0]]
    
    dict_lp = defaultdict(list)
    dict_hp = defaultdict(list)
    dict_all = defaultdict(list)
    
    key_lp, value_lp = zip(edges_idx_lp)
    key_hp, value_hp = zip(edges_idx_hp)
    key_all, value_all = zip(edges_idx)
    
    key_lp, value_lp = np.array(key_lp[0].detach().cpu()), np.array(value_lp[0].detach().cpu())
    key_hp, value_hp = np.array(key_hp[0].detach().cpu()), np.array(value_hp[0].detach().cpu())
    key_all, value_all = np.array(key_all[0].detach().cpu()), np.array(value_all[0].detach().cpu())
    
    for i in range(len(key_lp)):
        dict_lp[key_lp[i]].append(value_lp[i])
        
    for i in range(len(key_hp)):
        dict_hp[key_hp[i]].append(value_hp[i])
        
    for i in range(len(key_all)):
        dict_all[key_all[i]].append(value_all[i])
    
    ho_fraud_lp, ho_normal_lp = label_similarity(dict_lp, label)
    ho_fraud_hp, ho_normal_hp = label_similarity(dict_hp, label)
    ho_fraud_all, ho_normal_all = label_similarity(dict_all, label)
    print('ho_fraud_lp:{},ho_normal_lp:{}'.format(ho_fraud_lp, ho_normal_lp))
    print('ho_fraud_hp:{},ho_normal_hp:{}'.format(ho_fraud_hp, ho_normal_hp))
    print('ho_fraud_all:{},ho_normal_all:{}'.format(ho_fraud_all, ho_normal_all))   
    
def label_similarity(dict_lp, label):
    keys = list(dict_lp.keys())
    fraud_list = []
    normal_list = []
    
    for i in range(len(keys)):
        if label[keys[i]] == 1:
            fraud_ratio = torch.sum(label[dict_lp[keys[i]]])/len(dict_lp[keys[i]])
            fraud_list.append(np.array(fraud_ratio.cpu()))
        else:
            normal_ratio = 1 - torch.sum(label[dict_lp[keys[i]]])/len(dict_lp[keys[1]])
            normal_list.append(np.array(normal_ratio.cpu()))
    
    ho_fraud = np.mean(fraud_list)
    ho_normal = np.mean(normal_list)
    return ho_fraud, ho_normal
    
    
    
    
    
    
    
    