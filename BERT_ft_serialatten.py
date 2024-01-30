import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as Data
import os
import time
from torch.autograd import Variable
from utils import *
from BERT_ft_serialatten_model import *
from torch.utils.data import TensorDataset, DataLoader
from dataset.CMAPSS.CMAPSSDataset_ft import *


if __name__ == '__main__':
    SEED = 529
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    sub_dataset = 'FD002'
    if sub_dataset in ['FD001', 'FD003']:
        lr = 0.001
        encoder_layers = 1
        dropout = 0.1
        pre_model = 'FD0013'
        epochs = 200
    if sub_dataset in ['FD002', 'FD004']:
        lr = 0.0001
        encoder_layers = 2
        dropout = 0
        pre_model = 'FD0024'
        epochs = 600
    batch_size = 256
    num_hidden = 16
    ffn_hidden = 32
    mlp_size = 32
    n_heads = 2
    seq_len = 45
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
    train_dataset, test_dataset = CMAPSSDataset_ft.get_datasets(
        dataset_root='./dataset/CMAPSS',
        sub_dataset=sub_dataset,
        max_rul=max_rul,
        sequence_len=seq_len,
        norm_type=n_type,
        use_max_rul_on_test=True,
    )

    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    # valid_loader = DataLoader(valid_dataset, **kwargs)
    # Initialize model parameters
    model = BERT_ft(input_size, num_hidden, seq_len, ffn_hidden, mlp_size, encoder_layers, n_heads, dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # load model
    ###############
    PATH = './pre_{}.pt'.format(pre_model)
    # PATH = 'FD002_600.pt'
    model.encoder.load_state_dict(torch.load(PATH), strict=False)
    ###############
    for param in model.encoder.parameters():
        param.requires_grad = False
    # Training
    loss_list = []
    train_loss_list = []
    train_time = []
    out_list = []
    Y_list = []
    for epoch in range(epochs):
        model.train()
        start1 = time.time()
        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            # out_list.append(out)
            # Y_list.append(Y)
            loss = torch.sqrt(criterion(out, Y))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        end1 = time.time()
        loss_epoch = np.mean(np.array(loss_list))
        train_loss_list.append(loss_epoch)
        print('epoch %d ,train_loss = %.2f, spend %.1f s' % (epoch + 1, loss_epoch.item(), end1 - start1))


    # testing
    model.eval()
    with torch.no_grad():
        prediction_list = []
        Y_list = []
        for i, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            prediction = model(X)
            prediction[prediction < 0] = 0
            prediction_list.append(prediction)
            Y_list.append(Y)

        prediction_tensor = torch.cat(prediction_list)
        Y_tensor = torch.cat(Y_list)
        test_loss = torch.sqrt(criterion(prediction_tensor, Y_tensor))

        prediction_numpy = prediction_tensor.cpu().detach().numpy()
        Y_numpy = Y_tensor.cpu().detach().numpy()
        test_score = myScore(Y_numpy, prediction_numpy)

        print('test_loss = %.2f   score = %.2f' % (test_loss.item(), test_score))

        # visualize
        result = np.concatenate((Y_numpy, prediction_numpy), axis=1)
        result_df = pd.DataFrame(result, columns=['True RUL', 'Predict RUL'])
        result_df = result_df.sort_values('True RUL', ascending=False)
        save_dir = './result'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        result_df.to_csv(save_dir + '/result_{}.csv'.format(sub_dataset))
        # visualize(result_df, test_loss.item(), test_score, sub_dataset)







