#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:27:07 2022

@author: liuyan
"""
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import plot_heatmap
import numpy as np
from model import multiR
import argparse, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from read_CellM import read_cell
from data_loader import  load_sourcedata,load_targetdata
import math
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.15)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default =0.25)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'DR')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--pretrain_or_not', default=False)
parser.add_argument('--same_batch', default=False )
parser.add_argument('--different_batch', default=False)
parser.add_argument('--original_heatmap', default=False)
parser.add_argument('--draw_embedding', default=False)
args = parser.parse_args()

if args.dataset=='10Xv2':
    platform="10Xv2"
    DataPath="paper_data/PbmcBench/10Xv2/10Xv2_pbmc1.csv"
    LabelsPath="paper_data/PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv"
    CV_RDataPath="paper_data/PbmcBench/10Xv2/10Xv2_pbmc1_folds.RData"
    #All_source_data,All_source_label,y_train,test_x,test_label,y_test_cell_name,test_name,gene_names,cell_typeindex=read_cell(DataPath, LabelsPath, CV_RDataPath, platform)
    # legth_data=[253,6444,3222,253,3222,3176]
    # datasetname=["SM2","10Xv3","CL","DR","ID","SW"]
    # ############x10v3_data
    # x10v3_data=test_x[253:3475]
    # x10v3_label=test_label[253:3475]
    # need_index=[]
    # for i in range(x10v3_label.shape[0]):
    #     if x10v3_label[i] in [0,1,2,3,4]:
    #         need_index.append(i)
    #     source_10Xv3_data=x10v3_data[need_index]
    #     source_10Xv3_label=x10v3_label[need_index]
    # np.save("./data_5/source_10Xv3_data.npy",source_10Xv3_data)
    # np.save("./data_5/source_10Xv3_label.npy",source_10Xv3_label)  
    # #############CL#########################
    # CL_data=test_x[3475:3728]
    # CL_label=test_label[3475:3728]
    # need_index=[]
    # for i in range(CL_label.shape[0]):
    #     if CL_label[i] in [0,1,2,3,4]:
    #         need_index.append(i)
    #     source_CL_data=CL_data[need_index]
    #     source_CL_label=CL_label[need_index]
    # np.save("./data_5/source_CL_data.npy",source_CL_data)
    # np.save("./data_5/source_CL_label.npy",source_CL_label)
    # #############DR#########################
    # DR_data=test_x[3728:6950]
    # DR_label=test_label[3728:6950]
    # need_index=[]
    # for i in range(DR_label.shape[0]):
    #     if DR_label[i] in [0,1,2,3,4]:
    #         need_index.append(i)
    #     source_DR_data=DR_data[need_index]
    #     source_DR_label=DR_label[need_index]
    # np.save("./data_5/source_DR_data.npy",source_DR_data)
    # np.save("./data_5/source_DR_label.npy",source_DR_label)
    # ############ID###########################
    # ID_data=test_x[6950:10172]
    # ID_label=test_label[6950:10172]
    # need_index=[]
    # for i in range(ID_label.shape[0]):
    #     if ID_label[i] in [0,1,2,3,4]:
    #         need_index.append(i)
    #     source_ID_data=ID_data[need_index]
    #     source_ID_label=ID_label[need_index]
    # np.save("./data_5/source_ID_data.npy",source_ID_data)
    # np.save("./data_5/source_ID_label.npy",source_ID_label)
    # ##############10XV2################################
    # x10v2_data=All_source_data
    # x10v2_label=All_source_label
    # need_index=[]
    # for i in range(x10v2_label.shape[0]):
    #     if x10v2_label[i] in [0,1,2,3,4]:
    #         need_index.append(i)
    #     source_x10v2_data=x10v2_data[need_index]
    #     source_x10v2_label=x10v2_label[need_index]
    # np.save("./data/source_x10v2_data.npy",source_x10v2_data)
    # np.save("./data/source_x10v2_label.npy",source_x10v2_label)
    ##################################################
    # Training settings
batch_size =4
    
iteration = 1500
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 222
log_interval = 10
l2_decay = 5e-4
class_num = 65
root_path = "./dataset/"
source1_name = "ID"
source2_name = 'CL'
source3_name = 'DR'
target_name = "10xV2"
print ("ss")
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    ##################load_data##################
if target_name == "DR":
    w1,source1_loader = load_sourcedata(np.load("data_5/source_ID_data.npy"), np.load("data_5/source_ID_label.npy"),batch_size)
    w2,source2_loader = load_sourcedata(np.load("data_5/source_CL_data.npy"), np.load("data_5/source_CL_label.npy"),batch_size)
    w3,source3_loader = load_sourcedata(np.load("data/source_x10v2_data.npy"), np.load("data/source_x10v2_label.npy"),batch_size)
    target_train_loader = load_targetdata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    target_test_loader = load_targetdata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
if target_name == "10xV2":
    w1,source1_loader = load_sourcedata(np.load("data_5/source_ID_data.npy"), np.load("data_5/source_ID_label.npy"),batch_size)
    w2,source2_loader = load_sourcedata(np.load("data_5/source_CL_data.npy"), np.load("data_5/source_CL_label.npy"),batch_size)
    w3,source3_loader = load_sourcedata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    target_train_loader = load_targetdata(np.load("data/source_x10v2_data.npy"), np.load("data/source_x10v2_label.npy"),batch_size)
    target_test_loader = load_targetdata(np.load("data/source_x10v2_data.npy"), np.load("data/source_x10v2_label.npy"),batch_size)
if target_name == "ID":
    w1,source1_loader = load_sourcedata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    w2,source2_loader = load_sourcedata(np.load("data_5/source_CL_data.npy"), np.load("data_5/source_CL_label.npy"),batch_size)
    w3,source3_loader = load_sourcedata(np.load("data/source_x10v2_data.npy"), np.load("data/source_x10v2_label.npy"),batch_size)
    target_train_loader = load_targetdata(np.load("data_5/source_ID_data.npy"), np.load("data_5/source_ID_label.npy"),batch_size)
    target_test_loader = load_targetdata(np.load("data_5/source_ID_data.npy"), np.load("data_5/source_ID_label.npy"),batch_size)   
if target_name == "CL":
    w1,source1_loader = load_sourcedata(np.load("data_5/source_ID_data.npy"), np.load("data_5/source_ID_label.npy"),batch_size)
    w2,source2_loader = load_sourcedata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    w3,source3_loader = load_sourcedata(np.load("data/source_x10v2_data.npy"), np.load("data/source_x10v2_label.npy"),batch_size)
    target_train_loader = load_targetdata(np.load("data_5/source_DR_data.npy"), np.load("data_5/source_DR_label.npy"),batch_size)
    target_test_loader = load_targetdata(np.load("data_5/source_CL_data.npy"), np.load("data_5/source_CL_label.npy"),batch_size)
if target_name=="xin":
    w1,source1_loader = load_sourcedata(np.load("data/segerstolpe_data.npy"), np.load("data/segerstolpe_label.npy"),batch_size)
    w2,source2_loader = load_sourcedata(np.load("data/muraro_data.npy"), np.load("data/muraro_label.npy"),batch_size)
    w3,source3_loader = load_sourcedata(np.load("data/baron_human_data.npy"), np.load("data/baron_human_label.npy"),batch_size)
    target_train_loader = load_targetdata(np.load("data/xin_data.npy"), np.load("data/xin_label.npy"),batch_size)
    target_test_loader = load_targetdata(np.load("data/xin_data.npy"), np.load("data/xin_label.npy"),batch_size)
if target_name=="segerstolpe":
    w1,source1_loader = load_sourcedata(np.load("data/xin_data.npy"), np.load("data/xin_label.npy"),batch_size)
    w2,source2_loader = load_sourcedata(np.load("data/muraro_data.npy"), np.load("data/muraro_label.npy"),batch_size)
    w3,source3_loader = load_sourcedata(np.load("data/baron_human_data.npy"), np.load("data/baron_human_label.npy"),batch_size)
    target_train_loader = load_targetdata(np.load("data/segerstolpe_data.npy"), np.load("data/segerstolpe_label.npy"),batch_size)
    target_test_loader = load_targetdata(np.load("data/segerstolpe_data.npy"), np.load("data/segerstolpe_label.npy"),batch_size)
    if target_name=="muraro":
        w1,source1_loader = load_sourcedata(np.load("data/xin_data.npy"), np.load("data/xin_label.npy"),batch_size)
        w2,source2_loader = load_sourcedata(np.load("data/segerstolpe_data.npy"), np.load("data/segerstolpe_label.npy"),batch_size)
        w3,source3_loader = load_sourcedata(np.load("data/baron_human_data.npy"), np.load("data/baron_human_label.npy"),batch_size)
        target_train_loader = load_targetdata(np.load("data/muraro_data.npy"), np.load("data/muraro_label.npy"),batch_size)
        target_test_loader = load_targetdata(np.load("data/muraro_data.npy"), np.load("data/muraro_label.npy"),batch_size)
    if target_name=="baron_human":
        w1,source1_loader = load_sourcedata(np.load("data/xin_data.npy"), np.load("data/xin_label.npy"),batch_size)
        w2,source2_loader = load_sourcedata(np.load("data/segerstolpe_data.npy"), np.load("data/segerstolpe_label.npy"),batch_size)
        w3,source3_loader = load_sourcedata(np.load("data/muraro_data.npy"), np.load("data/muraro_label.npy"),batch_size)
        target_train_loader = load_targetdata(np.load("data/baron_human_data.npy"), np.load("data/baron_human_label.npy"),batch_size)
        target_test_loader = load_targetdata(np.load("data/baron_human_data.npy"), np.load("data/baron_human_label.npy"),batch_size)        
def train(model,source_loader1,source_loader2,source_loader3,target_train_loader1):
    source1_iter = iter(source_loader1)
    source2_iter = iter(source_loader2)
    source3_iter = iter(source_loader3)
    target_iter = iter(target_train_loader1)
    correct = 0
    optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': lr[1]},
            {'params': model.cls_fc_son2.parameters(), 'lr': lr[1]},
            {'params': model.cls_fc_son3.parameters(), 'lr': lr[1]},
            {'params': model.sonnet1.parameters(), 'lr': lr[1]},
            {'params': model.sonnet2.parameters(), 'lr': lr[1]},
            {'params': model.sonnet3.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration + 1):
        model.train()
        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[5]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        try:
            source_data, source_label,source1_iter= source1_iter.next()
        except Exception as err:
            source1_iter = iter(source_loader1)
            source_data, source_label,source1_iter= source1_iter.next()
        try:
            target_data, _,_ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, _,_ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss,mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
        # gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss=cls_loss+0.001*mmd_loss+0.001*l1_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        try:
            source_data, source_label,source2_iter = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label,source2_iter = source2_iter.next()
        try:
            target_data, _,_ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, _,_ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
        # gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        #loss=cls_loss+0.001*mmd_loss+0.001*l1_loss
        loss=cls_loss+0.001*l1_loss

        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
        try:
            source_data, source_label,source3_iter = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label,source3_iter = source3_iter.next()
        try:
            target_data, _,_ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, _,_ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=3)
        # gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss=cls_loss+0.001*mmd_loss+0.001*l1_loss
        loss=cls_loss+0.001*l1_loss
        loss.backward()
        optimizer.step()

        # if i % log_interval == 0:
        #     print(
        #         'Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
        #             i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print(source1_name, source2_name, source3_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    with torch.no_grad():
        for data, target,s in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3,_ = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            # pred = (pred1 + pred2 + pred3) / 3
            pred=(w1/(w1+w2+w3))*pred1+(w2/(w1+w2+w3))*pred2+(w3/(w1+w2+w3))*pred3
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target.long()).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]  # get the index of the max log-probability
            correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}，source3 accnum {}'.format(correct1, correct2, correct3))
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
def vision_pbmc(target_name):
    model.eval()
    pbmc_index=np.load("cell_typeindex.npy")
    pan_index=np.load("pan_index.npy")
    if target_name == "DR":
        test_data=np.load("data_5/source_DR_data.npy")
        test_label=np.load("data_5/source_DR_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)
    if target_name == "CL":
        test_data=np.load("data_5/source_CL_data.npy")
        test_label=np.load("data_5/source_CL_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)
    if target_name == "ID":
        test_data=np.load("data_5/source_ID_data.npy")
        test_label=np.load("data_5/source_ID_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)
    if target_name == "10xV2":
        test_data=np.load("data/source_x10v2_data.npy")
        test_label=np.load("data/source_x10v2_label.npy")
        test_cell_type=get_cell_type(test_label,pbmc_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)
    if target_name == "xin":
        test_data=np.load("data/xin_data.npy")
        test_label=np.load("data/xin_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)
    if target_name == "segerstolpe":
        test_data=np.load("data/segerstolpe_data.npy")
        test_label=np.load("data/segerstolpe_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)   
    if target_name == "muraro":
        test_data=np.load("data/muraro_data.npy")
        test_label=np.load("data/muraro_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)  
    if target_name == "baron_human":
        test_data=np.load("data/baron_human_data.npy")
        test_label=np.load("data/baron_human_label.npy")
        test_cell_type=get_cell_type(test_label,pan_index)
        test_data_tensor=torch.tensor(test_data,dtype=torch.float32).cuda()
        _,_,_,query_data=model(test_data_tensor)
        query_data=query_data.cpu().detach().numpy()
        print (query_data.shape)
        print (len(test_cell_type))
        draw_gcn_clusters(query_data,test_cell_type,5,target_name)  
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["ID","CL","10Xv2","DR"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()

        train=tsne.fit_transform(all_data)
        
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,5,target_name+"_all")
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["ID","DR","10Xv2","CL"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,5,target_name+"_all")   
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["DR","CL","10Xv2","ID"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,5,target_name+"_all") 
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["DR","CL","ID","10Xv2"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,5,target_name+"_all")
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["xin","baron_human","segerstolpe","muraro"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_all")
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["muraro","baron_human","segerstolpe","xin"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_all")
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["muraro","baron_human","xin","segerstolpe"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_all")
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
        _,_,_,all_data=model(all_data_tensor)
        all_data=all_data.cpu().detach().numpy()
        number_platform=[source1_data.shape[0],source2_data.shape[0],source3_data.shape[0],target_train.shape[0]]
        platform_name=["muraro","segerstolpe","xin","baron_human"]
        batch_label=obtain_batch_label(platform_name,number_platform)
        tsne = TSNE()
        train=tsne.fit_transform(all_data)
        draw_batcheffectembedding(train,batch_label,4,"batch_effect_",target_name,"_scmultiR")
        draw_gcn_cluster(train,all_cell_type,4,target_name+"_all")
if __name__ == '__main__':
    model = multiR(22280,5)#22280
    print(model)
    # if cuda:
    #     model.cuda()
    #train(model.cuda())

    train(model.cuda(),source1_loader,source2_loader,source3_loader,target_train_loader)
    vision_pbmc(target_name)
    draw_batch(target_name)

