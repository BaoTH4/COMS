# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import re
import torch
import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp
from torch.nn import functional as F


class AutoscaleFocalLoss:
    ##Credit: Lê Duy Khánh - khanhld (team Speech - Phòng VAT)
    def __init__(self, threshold, weights = None):
        self.threshold = threshold
    
    def gamma(self, logits):
        return self.threshold/2 * (torch.cos(np.pi*(logits+1)) + 1)

    def __call__(self, logits, labels):
        labels = F.one_hot(labels, 5)
        assert logits.shape == labels.shape, \
                "Mismatch in shape, logits.shape: {} - labels.shape: {}".format(logits.shape, labels.shape)
        logits =  F.softmax(logits, dim=-1)
        CE = - labels * torch.log(logits)
        loss = ((1 - logits)**self.gamma(logits)) * CE
        loss = torch.sum(loss, dim=-1).mean()
        return loss


def is_one_exist(labels,ignore_index):
    '''
        Hàm giúp kiểm tra nếu có nhãn 1 trong labels hay không giúp quyết định bước xây dựng query tiếp theo
    '''
    if 1 not in labels:
        return False
    else:
        count=0
        one_index=(labels==1).nonzero(as_tuple=True)[0]
        for idx in one_index:
            if idx.item() in ignore_index:
                count+=1
        if count==len(one_index):
            return False
    return True


def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor


def calculate_preference_loss(logits, targets):
    targets = targets.squeeze(-1)

    if torch.cuda.is_available():
        weight = torch.tensor([1., 3.2093, 5.75, 7.2632, 4.3125]).cuda()
        # weight = torch.tensor([1., 1., 1., 1., 1.]).cuda()
        targets = targets.to(torch.long).cuda()
        logits = logits.cuda()
    else:
        weight = torch.tensor([1., 3.2093, 5.75, 7.2632, 4.3125])
        # weight = torch.tensor([1., 1., 1., 1., 1.])

    loss = F.cross_entropy(logits, targets, size_average = False ,weight = weight)

    return loss


def calculate_preference_loss_2(logits, targets):

    ##Define loss function
    # weights = torch.tensor([1., 3., 7., 6., 4.])
    loss_function = AutoscaleFocalLoss(threshold = 2)
    
    #targets shape: [batch_size,]
    targets = targets.squeeze(-1)

    if torch.cuda.is_available():
        targets = targets.to(torch.long).cuda()
        logits = logits.cuda()

    loss = loss_function(logits = logits, labels = targets)

    return loss