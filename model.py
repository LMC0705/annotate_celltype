#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:11:49 2022

@author: liuyan
"""
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
import torch

class multiR(nn.Module):

    def __init__(self,input_dim=2000,num_classes=5):
        super(multiR, self).__init__()
        self.sharedNet = nn.Linear(input_dim, 256)
        self.sonnet1 = nn.Linear(256, 32)
        self.sonnet2 = nn.Linear(256, 32)
        self.sonnet3 = nn.Linear(256,32)
        self.cls_fc_son1 = nn.Linear(32, num_classes)
        self.cls_fc_son2 = nn.Linear(32, num_classes)
        self.cls_fc_son3 = nn.Linear(32, num_classes)
  
    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1):
        mmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_son1 = self.sonnet1(data_tgt)

            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

            data_tgt_son2 = self.sonnet2(data_tgt)

            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

            data_tgt_son3 = self.sonnet3(data_tgt)

            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)

            if mark == 1:

                data_src = self.sonnet1(data_src)

                # mmd_loss += mmd.mmd(data_src, data_tgt_son1)
                mmd_loss = mmd.mmd(data_src, data_tgt_son1)
                

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)) )
                pred_src = self.cls_fc_son1(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss / 2

            if mark == 2:

                data_src = self.sonnet2(data_src)
                mmd_loss += mmd.mmd(data_src, data_tgt_son2)

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)) )
                pred_src = self.cls_fc_son2(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss / 2

            if mark == 3:

                data_src = self.sonnet3(data_src)

                mmd_loss += mmd.mmd(data_src, data_tgt_son3)

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)) )
                pred_src = self.cls_fc_son3(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss / 2

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)


            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)

            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data)

            pred3 = self.cls_fc_son3(fea_son3)

            return pred1, pred2, pred3,fea_son1


        
class scMDR(nn.Module):
    def __init__(self, in_features,num_classes=31):
        super(scMDR, self).__init__()
        self.sharedNet1 = nn.Linear(in_features, 256)
        self.sharedNet2 = nn.Linear(256, 32)
        self.cls_fc = nn.Linear(32, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet1(source)
        source = F.relu(source)
        # source = F.dropout(source,p=0.5)
        source1 = self.sharedNet2(source)
        if self.training == True:
            target = self.sharedNet1(target)
            target=F.relu(target)
            target = F.dropout(target,p=0.5)
            target = self.sharedNet2(target)
            target=F.relu(target)
            #loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd(source1, target)
        source1 = F.relu(source1)
        # source1 = F.dropout(source1,p=0.5)
        source = self.cls_fc(source1)
        #target = self.cls_fc(target)

        return source, loss,source1,source1