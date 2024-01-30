import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


# 这里的dropout就没用过
class EncoderLayer(torch.nn.Module):
    def __init__(self, num_hidden, ffn_hidden, heads=1, dropout=0.5):
        super(EncoderLayer, self).__init__()
        self.query = nn.Linear(num_hidden, num_hidden)
        self.key = nn.Linear(num_hidden, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.attn = nn.MultiheadAttention(embed_dim=num_hidden, num_heads=heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=45, num_heads=5,
                                           dropout=dropout)  # num_hidden必须是heads的倍数
        self.norm1 = nn.LayerNorm(num_hidden)
        self.fc1 = nn.Linear(num_hidden, ffn_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_hidden, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(num_hidden)

    def forward(self, X):
        # Q = self.query(X)
        # K = self.key(X)
        # V = self.value(X)
        # Y, _ = self.attn(Q,K,V)
        X = X.permute(1, 0, 2)
        Y, attn1 = self.attn(X, X, X)
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)
        # x_texts = [str(x) for x in range(1, 46)]
        # for i in range(100):
        #     attention_plot(attn1[i].cpu().detach().numpy(), x_texts, x_texts, figsize=(45, 45), figure_path='./figures',
        #                figure_name='Unit {} seq_attention_weight.png'.format(i+1))
        Y = Y.permute(2, 0, 1)
        Y, attn2 = self.attn2(Y, Y, Y)
        Y = Y.permute(1, 2, 0)
        # x_texts = [str(x) for x in range(1, 17)]
        # for i in range(100):
        #     attention_plot(attn2[i].cpu().detach().numpy(), x_texts, x_texts, annot=False, figsize=(16, 16), figure_path='./figures_fea',
        #                figure_name='Unit {} fea_attention_weight.png'.format(i+1))

        X = self.norm1(X + self.dropout(Y))
        Y = self.fc2(self.relu(self.fc1(X)))
        X = self.norm2(X + self.dropout(Y))
        return X


class MLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, 1))

    def forward(self, X):
        return self.mlp(X)


class Transformer_ft(torch.nn.Module):
    def __init__(self, input_size, num_hidden, seq_len, ffn_hidden, mlp_size, encoder_layers=1, heads=1, dropout=0.5):

        super(Transformer_ft, self).__init__()
        self.linear = nn.Linear(input_size, num_hidden)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, num_hidden))

        # Initiate Time_step encoder
        self.encoder = nn.Sequential()
        for i in range(encoder_layers):
            self.encoder.add_module(f"{i}", EncoderLayer(num_hidden, ffn_hidden, heads, dropout))

        self.out = MLP(num_hidden, mlp_size)
        # self.out_fc = nn.Linear(num_hidden, 1)
        # self.norm1 = nn.LayerNorm(num_hidden)

    def forward(self, X):
        # X的形状train:256X64X17  test:100X64X17
        # input embedding and positional encoding
        X = self.linear(X) + self.pos_embedding  # ((batch_size,seq_len,num_hidden))

        # time step encoder
        for enc_layer in self.encoder:
            X = enc_layer(X)

        Y = self.out(X[:, -1, :])
        return Y

def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()








