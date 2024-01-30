import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


# 这里的dropout就没用过
class EncoderLayer(torch.nn.Module):
    def __init__(self, num_hidden, ffn_hidden, heads=1, dropout=0.5):
        super(EncoderLayer, self).__init__()
        # self.attn = TimeStepMultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.query = nn.Linear(num_hidden, num_hidden)
        self.key = nn.Linear(num_hidden, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.attn = nn.MultiheadAttention(embed_dim=num_hidden, num_heads=heads,
                                          dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=45, num_heads=5,
                                          dropout=dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.fc1 = nn.Linear(num_hidden, ffn_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_hidden, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(num_hidden)

    def forward(self, X):
        X = X.permute(1, 0, 2)
        Y, _ = self.attn(X, X, X)
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)


        Y = Y.permute(2, 0, 1)
        Y, _ = self.attn2(Y, Y, Y)
        Y = Y.permute(1, 2, 0)

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

class CLASSMLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(CLASSMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, 2),
                                 nn.ReLU(),
                                 nn.Softmax(-1))

    def forward(self, X):
        return self.mlp(X)


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Transformer_pre(torch.nn.Module):
    def __init__(self, input_size, num_hidden, seq_len, ffn_hidden, mlp_size, encoder_layers=1, heads=1, dropout=0.5):

        super(Transformer_pre, self).__init__()
        self.linear = nn.Linear(input_size, num_hidden)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, num_hidden))

        # Initiate Time_step encoder
        self.encoder = nn.Sequential()
        for i in range(encoder_layers):
            self.encoder.add_module(f"{i}", EncoderLayer(num_hidden, ffn_hidden, heads, dropout))

        self.out = MLP(num_hidden, mlp_size)
        self.classout = CLASSMLP(num_hidden, mlp_size)
        # self.out_fc = nn.Linear(num_hidden, 1)
        # self.norm1 = nn.LayerNorm(num_hidden)

    def forward(self, X):
        # input embedding and positional encoding

        X = self.linear(X) + self.pos_embedding  # ((batch_size,seq_len,num_hidden))

        # time step encoder
        for enc_layer in self.encoder:
            X = enc_layer(X)

        Y = self.out(X[:, -1, :])
        X = GradReverse.apply(X, 1)
        CLASSY = self.classout(X[:, -1, :])
        return Y, CLASSY







