#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:29:50 2022

@author: liuyan
"""
from __future__ import print_function
import argparse
import random
import torch
import plot_heatmap
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
from data_loader import  load_sourcedata,load_targetdata
import math
import data_loader
from model import scMDR
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
# Training settings
batch_size = 4
iteration=150000

lr = [0.0005, 0.001]
momentum = 0.9
no_cuda =False
seed=666
log_interval = 10
l2_decay = 5e-4
root_path = "/data/zhuyc/OFFICE31/"
src_name = "amazon"
tgt_name = "dslr"


cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_name = "ID"
source2_name = 'CL'
source3_name = 'DR'
target_name = "ID"
torch.manual_seed(seed)
batch_size=4


    ##################load_data##################
if target_name == "human":
    source1_data=np.load("data/mouse_ALM_data.npy")
    source1_label=np.load("data/mouse_ALM_3_label.npy")
    source2_data=np.load("data/mouse_v1_data.npy")
    source2_label=np.load("data/mouse_v1_3_label.npy")
    source_data=np.vstack((source1_data,source2_data))
    source_label=np.hstack((source1_label,source2_label))
    target_train=np.load("data/human_MTG_data.npy")
    target_label=np.load("data/human_MTG_3_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    tgt_train_loader = load_targetdata(np.load("data/human_MTG_data.npy"), np.load("data/human_MTG_3_label.npy"),batch_size)
    tgt_test_loader =load_targetdata(np.load("data/human_MTG_data.npy"), np.load("data/human_MTG_3_label.npy"),batch_size)
    
if target_name == "DR":
    source1_data=np.load("data_5/source_ID_data.npy")
    source1_label=np.load("data_5/source_ID_label.npy")
    source2_data=np.load("data_5/source_CL_data.npy")
    source2_label=np.load("data_5/source_CL_label.npy")
    source3_data=np.load("data/source_x10v2_data.npy")
    source3_label=np.load("data/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data_5/source_DR_data.npy")
    target_label=np.load("data_5/source_DR_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    tgt_test_loader =load_targetdata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    
if target_name == "10xV2":
    source1_data=np.load("data_5/source_ID_data.npy")
    source1_label=np.load("data_5/source_ID_label.npy")
    source2_data=np.load("data_5/source_CL_data.npy")
    source2_label=np.load("data_5/source_CL_label.npy")
    source3_data=np.load("data_5/source_DR_data.npy")
    source3_label=np.load("data_5/source_DR_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data_5/source_x10v2_data.npy")
    target_label=np.load("data_5/source_x10v2_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size)
  
if target_name == "ID":
    source1_data=np.load("data_5/source_x10v2_data.npy")
    source1_label=np.load("data_5/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
    source2_data=np.load("data_5/source_CL_data.npy")
    source2_label=np.load("data_5/source_CL_label.npy")
    source3_data=np.load("data_5/source_DR_data.npy")
    source3_label=np.load("data_5/source_DR_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data_5/source_ID_data.npy")
    target_label=np.load("data_5/source_ID_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size)   
if target_name == "CL":
    source1_data=np.load("data_5/source_x10v2_data.npy")
    source1_label=np.load("data_5/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
    source2_data=np.load("data_5/source_ID_data.npy")
    source2_label=np.load("data_5/source_ID_label.npy")
    source3_data=np.load("data_5/source_DR_data.npy")
    source3_label=np.load("data_5/source_DR_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data_5/source_CL_data.npy")
    target_label=np.load("data_5/source_CL_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size) 

if target_name == "xin":
    source1_data=np.load("data/baron_human_data.npy")
    source1_label=np.load("data/baron_human_label.npy")
    source2_data=np.load("data/muraro_data.npy")
    source2_label=np.load("data/muraro_label.npy")
    source3_data=np.load("data/segerstolpe_data.npy")
    source3_label=np.load("data/segerstolpe_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data/xin_data.npy")
    target_label=np.load("data/xin_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size) 
    
    
    
    
if target_name == "baron_human":
    source1_data=np.load("data/xin_data.npy")
    source1_label=np.load("data/xin_label.npy")
    source2_data=np.load("data/muraro_data.npy")
    source2_label=np.load("data/muraro_label.npy")
    source3_data=np.load("data/segerstolpe_data.npy")
    source3_label=np.load("data/segerstolpe_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data/baron_human_data.npy")
    target_label=np.load("data/baron_human_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size)
if target_name == "muraro":
    source1_data=np.load("data/xin_data.npy")
    source1_label=np.load("data/xin_label.npy")
    source2_data=np.load("data/baron_human_data.npy")
    source2_label=np.load("data/baron_human_label.npy")
    source3_data=np.load("data/segerstolpe_data.npy")
    source3_label=np.load("data/segerstolpe_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data/muraro_data.npy")
    target_label=np.load("data/muraro_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size)
if target_name == "segerstolpe":
    source1_data=np.load("data/xin_data.npy")
    source1_label=np.load("data/xin_label.npy")
    source2_data=np.load("data/baron_human_data.npy")
    source2_label=np.load("data/baron_human_label.npy")
    source3_data=np.load("data/muraro_data.npy")
    source3_label=np.load("data/muraro_label.npy")
    common_1_data=np.vstack((source1_data,source2_data))
    source_data=np.vstack((common_1_data,source3_data))
    common_1_label=np.hstack((source1_label,source2_label))
    source_label=np.hstack((common_1_label,source3_label))
    target_train=np.load("data/segerstolpe_data.npy")
    target_label=np.load("data/segerstolpe_label.npy")
    src_loader=load_sourcedata(source_data,source_label,batch_size,"scMDR")
    
    tgt_train_loader = load_targetdata(target_train,target_label,batch_size)
    tgt_test_loader = load_targetdata(target_train,target_label,batch_size)
# src_loader = data_loader.load_training(root_path, src_name, batch_size, kwargs)
# tgt_train_loader = data_loader.load_training(root_path, tgt_name, batch_size, kwargs)
# tgt_test_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs)

src_dataset_len = source2_data.shape[0]
tgt_dataset_len = target_train.shape[0]
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)

def add_noise(inputs):
    return inputs + (torch.randn(inputs.shape) * 0.1)
def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    correct = 0

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet1.parameters()},

        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration+1):
        model.train()
       
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[index] / math.pow((1 + 10 * (i - 1) / iteration), 0.75)

        try:
            src_data, src_label,_= src_iter.next()
        except Exception as err:
            src_iter=iter(src_loader)
            src_data, src_label,_ = src_iter.next()
            
        try:
            tgt_data, _,_ = tgt_iter.next()
        except Exception as err:
            tgt_iter=iter(tgt_train_loader)
            tgt_data, _,_ = tgt_iter.next()
            
        if cuda:
            # src_data=add_noise(src_data)
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_data.cuda()

        optimizer.zero_grad()
        src_pred, mmd_loss,_,_ = model(src_data, tgt_data)
        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)
        lambd = 2 / (1 + math.exp(-5 * (i) / iteration)) - 1
        # print ("lambd---------:",lambd)
        loss = cls_loss + 0.01*mmd_loss
        loss=cls_loss
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))
        if i%(log_interval*20)==0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
              src_name, tgt_name, correct, 100. * correct / tgt_dataset_len ))
        
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for tgt_test_data, tgt_test_label,_ in tgt_test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred, mmd_loss,_,_ = model(tgt_test_data, tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim = 1), tgt_test_label, reduction='sum').item() # sum up batch loss
            pred = tgt_pred.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_name, test_loss, correct, tgt_dataset_len,
        100. * correct / tgt_dataset_len))
    return correct
from vision import draw_gcn_cluster,draw_batcheffectembedding,draw_gcn_clusters
def get_cell_type(number_label,index):
    cell_type=[]
    # number_label=list(number_label)
    for i in range(len(number_label)):
        if number_label[i]==0:
            cell_type.append(str(index[0]))
        if number_label[i]==1:
            cell_type.append(str(index[1]))
        if number_label[i]==2:
            cell_type.append(str(index[2]))
        if number_label[i]==3:
            cell_type.append(str(index[3]))
        if number_label[i]==4:
            cell_type.append(str(index[4]))
    return cell_type
def obtain_batch_label(platform_name,number):
    label=[]
    for i in range (len(number)):
        for j in range(number[i]):
            label.append(platform_name[i])
    return label
from sklearn.manifold import TSNE
def vision_pbmc(target_name):
    model.eval()
    pbmc_index=np.load("cell_typeindex.npy")
    pan_index=np.load("pan_index.npy")
    if target_name == "DR":
        test_data=np.load("data_5/source_DR_data.npy")
        test_label=np.load("data_5/source_DR_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,"DR_MDR")
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
        
    if target_name == "CL":
        test_data=np.load("data_5/source_CL_data.npy")
        test_label=np.load("data_5/source_CL_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,"CL_MDR")
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
        
    if target_name == "ID":
        test_data=np.load("data_5/source_ID_data.npy")
        test_label=np.load("data_5/source_ID_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,"ID_MDR")
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
        
        
    if target_name == "10xV2":
        test_data=np.load("data_5/source_x10v2_data.npy")
        test_label=np.load("data_5/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,"10xV2_MDR")
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
        
    if target_name == "xin":
        test_data=np.load("data/xin_data.npy")
        test_label=np.load("data/xin_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name+"_MDR")
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
        
        
    if target_name == "segerstolpe":
        test_data=np.load("data/segerstolpe_data.npy")
        test_label=np.load("data/segerstolpe_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name+"_MDR") 
    
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps","segerstolpe")
    if target_name == "muraro":
        test_data=np.load("data/muraro_data.npy")
        test_label=np.load("data/muraro_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name+"_MDR")  
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
    if target_name == "baron_human":
        test_data=np.load("data/baron_human_data.npy")
        test_label=np.load("data/baron_human_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor,test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name+"_MDR")  
        
        cell_index=[]
        def get_same_element_index(ob_list, word):
            return [i for (i, v) in enumerate(ob_list) if v == word]
        for i in range(len(list(set(test_label)))):
            ss=get_same_element_index(test_label, list(set(test_label))[i])
            for j in range(25):
                cell_index.append(ss[j])
        # for i in range (len(test_label)):
        #     if test_label[i]
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_cell_type[cell_index[i]])
            cell_labels.append(test_cell_type[cell_index[i]])
        embedding_cell_index=query_data[cell_index,:]
        print (embedding_cell_index.shape)
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,target_name+".eps",target_name)
        
def obtain_batch_label(platform_name,number):
    label=[]
    for i in range (len(number)):
        for j in range(number[i]):
            label.append(platform_name[i])
    return label
from sklearn.manifold import TSNE
def draw_batch(target_name):
    model.eval()
    pbmc_index=np.load("cell_typeindex.npy")
    pan_index=np.load("pan_index.npy")
    if target_name == "DR":
        source1_data=np.load("data_5/source_ID_data.npy")
        source1_label=np.load("data_5/source_ID_label.npy")
        source2_data=np.load("data_5/source_CL_data.npy")
        source2_label=np.load("data_5/source_CL_label.npy")
        source3_data=np.load("data/source_x10v2_data.npy")
        source3_label=np.load("data/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data_5/source_DR_data.npy")
        target_label=np.load("data_5/source_DR_label.npy")
        
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))

        ######
        all_cell_type=get_cell_type(all_label,pbmc_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["ID","CL","10Xv2","DR"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()

        train=tsne.fit_transform(all_data)
        
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_")
        draw_gcn_cluster(train,all_cell_type,5,"DR_MDR_ALL")
    if target_name == "CL":
        source1_data=np.load("data_5/source_ID_data.npy")
        source1_label=np.load("data_5/source_ID_label.npy")
        source2_data=np.load("data_5/source_DR_data.npy")
        source2_label=np.load("data_5/source_DR_label.npy")
        source3_data=np.load("data/source_x10v2_data.npy")
        source3_label=np.load("data/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data_5/source_DR_data.npy")
        target_label=np.load("data_5/source_DR_label.npy")
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pbmc_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["ID","DR","10Xv2","CL"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_")
        draw_gcn_cluster(train,all_cell_type,5,"CL_MDR_ALL")   
    if target_name == "ID":
        source1_data=np.load("data_5/source_DR_data.npy")
        source1_label=np.load("data_5/source_DR_label.npy")
        source2_data=np.load("data_5/source_CL_data.npy")
        source2_label=np.load("data_5/source_CL_label.npy")
        source3_data=np.load("data/source_x10v2_data.npy")
        source3_label=np.load("data/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data_5/source_ID_data.npy")
        target_label=np.load("data_5/source_ID_label.npy")
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pbmc_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["DR","CL","10Xv2","ID"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_")
        draw_gcn_cluster(train,all_cell_type,5,"ID_MDR_ALL") 
    if target_name == "10xV2":
        source1_data=np.load("data_5/source_DR_data.npy")
        source1_label=np.load("data_5/source_DR_label.npy")
        source2_data=np.load("data_5/source_CL_data.npy")
        source2_label=np.load("data_5/source_CL_label.npy")
        source3_data=np.load("data/source_ID_data.npy")
        source3_label=np.load("data/source_ID_label.npy")
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data_5/source_x10v2_data.npy")
        target_label=np.load("data/source_x10v2_label.npy").reshape(np.load("data/source_x10v2_label.npy").shape[0])
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pbmc_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["DR","CL","ID","10Xv2"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_")
        draw_gcn_cluster(train,all_cell_type,5,"10Xv2_MDR_ALL") 
    if target_name == "muraro":
        source1_data=np.load("data/xin_data.npy")
        source1_label=np.load("data/xin_label.npy")
        source2_data=np.load("data/baron_human_data.npy")
        source2_label=np.load("data/baron_human_label.npy")
        source3_data=np.load("data/segerstolpe_data.npy")
        source3_label=np.load("data/segerstolpe_label.npy")
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data/muraro_data.npy")
        target_label=np.load("data/muraro_label.npy")
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pan_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["xin","baron_human","segerstolpe","muraro"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scMDR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_MDRall")
    if target_name == "xin":
        source1_data=np.load("data/muraro_data.npy")
        source1_label=np.load("data/muraro_label.npy")
        source2_data=np.load("data/baron_human_data.npy")
        source2_label=np.load("data/baron_human_label.npy")
        source3_data=np.load("data/segerstolpe_data.npy")
        source3_label=np.load("data/segerstolpe_label.npy")
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data/xin_data.npy")
        target_label=np.load("data/xin_label.npy")
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pan_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["muraro","baron_human","segerstolpe","xin"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scMDR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_MDRall")
    if target_name == "segerstolpe":
        source1_data=np.load("data/muraro_data.npy")
        source1_label=np.load("data/muraro_label.npy")
        source2_data=np.load("data/baron_human_data.npy")
        source2_label=np.load("data/baron_human_label.npy")
        source3_data=np.load("data/xin_data.npy")
        source3_label=np.load("data/xin_label.npy")
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data/segerstolpe_data.npy")
        target_label=np.load("data/segerstolpe_label.npy")
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pan_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["muraro","baron_human","xin","segerstolpe"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scMDR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_MDRall.png")
    if target_name == "baron_human":
        source1_data=np.load("data/muraro_data.npy")
        source1_label=np.load("data/muraro_label.npy")
        source2_data=np.load("data/segerstolpe_data.npy")
        source2_label=np.load("data/segerstolpe_label.npy")
        source3_data=np.load("data/xin_data.npy")
        source3_label=np.load("data/xin_label.npy")
        common_1_data=np.vstack((source1_data,source2_data))
        source_data=np.vstack((common_1_data,source3_data))
        common_1_label=np.hstack((source1_label,source2_label))
        source_label=np.hstack((common_1_label,source3_label))
        target_train=np.load("data/baron_human_data.npy")
        target_label=np.load("data/baron_human_label.npy")
        all_data=np.vstack((source_data,target_train))
        all_label=np.hstack((source_label,target_label))
        ######
        all_cell_type=get_cell_type(all_label,pan_index)
        all_data_tensor=torch.tensor(all_data,dtype=torch.float32).cuda()
        _,_,_,all_data=model(all_data_tensor,all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["muraro","segerstolpe","xin","baron_human"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_MDR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_MDRall")

if __name__ == '__main__':
    model = scMDR(22280,34)#22280/15642
    print(model)
    if cuda:
        model.cuda()
    # target=["DR","CL","ID","10Xv2"]
    # for i in range(len(target)):
    #     target_name=target[i]
    train(model)
    # vision_pbmc(target_name)
 