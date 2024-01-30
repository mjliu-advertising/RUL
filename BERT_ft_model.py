import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#这里的dropout就没用过
class EncoderLayer(torch.nn.Module):
    def __init__(self, num_hidden, ffn_hidden, heads = 1,dropout=0.5):
        super(EncoderLayer, self).__init__()
        self.query = nn.Linear(num_hidden, num_hidden)
        self.key = nn.Linear(num_hidden, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.attn = nn.MultiheadAttention(embed_dim = num_hidden,num_heads = heads,dropout = dropout)
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
        Y, _ = self.attn(X,X,X)
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)
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
    def forward(self,X):
        return self.mlp(X)


class BERT_ft(torch.nn.Module):
    def __init__(self, input_size, num_hidden, seq_len,ffn_hidden , mlp_size, encoder_layers=1, heads=1, dropout=0.5):
        
        super(BERT_ft, self).__init__()
        self.linear = nn.Linear(input_size, num_hidden)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, num_hidden))
        
        #Initiate Time_step encoder
        self.encoder = nn.Sequential()
        for i in range(encoder_layers):
            self.encoder.add_module(f"{i}", EncoderLayer(num_hidden, ffn_hidden, heads, dropout))
 
        self.out = MLP(num_hidden, mlp_size)
        # self.out_fc = nn.Linear(num_hidden, 1)
        # self.norm1 = nn.LayerNorm(num_hidden)
    
    def forward(self, X):
        #X的形状train:256X64X17  test:100X64X17
        #input embedding and positional encoding
        X = self.linear(X) + self.pos_embedding   #((batch_size,seq_len,num_hidden))

        #time step encoder 
        for enc_layer in self.encoder:
            X = enc_layer(X)

        Y = self.out(X[:,-1,:])
        return Y
    
    
    


    
    
   