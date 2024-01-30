# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as Data
import os
import time
import math
from torch.autograd import Variable
from utils import *
from Transformer_pre_model import *

from dataset.CMAPSS.CMAPSSDataset_pre import *

# def pre_train(args):
if __name__ == '__main__':
    SEED = 529
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    sub_dataset = 'FD001'  # FD001, FD002
    # Hyperparameters
    lr = 0.001
    batch_size = 256
    num_hidden = 16
    ffn_hidden = 32
    mlp_size = 32
    n_heads = 2
    seq_len = 45

    if sub_dataset in ['FD001', 'FD003']:
        mask_p = 0.6
        random_p = 0.3
        dropout = 0.1
        epochs = 20
        encoder_layers = 1
        pre_model = 'FD0013'
    if sub_dataset in ['FD002', 'FD004']:
        mask_p = 0.3
        random_p = 0.6
        epochs = 100
        dropout = 0
        encoder_layers = 2
        pre_model = 'FD0024'

    input_size = 17
    max_rul = 125
    n_type = 'z-score'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'batch_size': batch_size}
    if use_cuda:
        kwargs.update({
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        })

    # Dataloader
    train_dataset, test_dataset = CMAPSSDataset_pre.get_datasets(
        dataset_root='./dataset/CMAPSS',
        sub_dataset=sub_dataset,
        mask_p=mask_p,
        random_p=random_p,
        max_rul=max_rul,
        sequence_len=seq_len,
        norm_type=n_type,
        use_max_rul_on_test=True,
    )

    train_loader = DataLoader(train_dataset, **kwargs)  # 对于双星号的理解  表示传入参数为字典
    test_loader = DataLoader(test_dataset, **kwargs)


    # Initialize model parameters
    model = Transformer_pre(input_size, num_hidden, seq_len, ffn_hidden, mlp_size, encoder_layers, n_heads, dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()


    # Training
    loss_list = []
    train_loss_list = []


    test_loss_list = []  ########
    test_score_list = []  ########

    for epoch in range(epochs):
        model.train()
        start1 = time.time()
        for i, (X, Y, Y2) in enumerate(train_loader):
            X, Y, Y2 = X.to(device), Y.to(device), Y2.to(device)
            optimizer.zero_grad()
            out, classout = model(X)
            loss = torch.sqrt(criterion(out, Y)) + criterion2(classout, Y2.squeeze(1))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())  ######

        end1 = time.time()
        loss_epoch = np.mean(np.array(loss_list))
        train_loss_list.append(loss_epoch)
        print('epoch %d ,train_loss = %.2f, spend %.1f s' % (epoch + 1, loss_epoch.item(), end1 - start1))

    # show_train_loss(np.array(train_loss_list), sub_dataset, lr)


    # save model
    ###################
    PATH = './pre_{}.pt'.format(pre_model)
    torch.save(model.encoder.state_dict(), PATH)
    ###################











