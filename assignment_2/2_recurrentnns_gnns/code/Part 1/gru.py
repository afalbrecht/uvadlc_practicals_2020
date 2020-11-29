"""
This module implements a GRU in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(GRU, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = device

        embed_dim = int(hidden_dim/4)
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #z_t
        self.W_z = nn.Parameter(torch.Tensor(hidden_dim + embed_dim, hidden_dim))
        
        #r_t
        self.W_r = nn.Parameter(torch.Tensor(hidden_dim + embed_dim, hidden_dim))

        #h_tilde
        self.W = nn.Parameter(torch.Tensor(hidden_dim + embed_dim, hidden_dim))

        #p_t
        self.W_ph = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_p = nn.Parameter(torch.Tensor(num_classes))

        self.init_params() 

        self.embedding = nn.Embedding(input_dim, embed_dim)

        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def init_params(self):

        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)


    def forward(self, x):

        batch_size, _, seq_size = x.size()

        h_t, c_t = (torch.zeros(batch_size, self.hidden_dim).to(x.device),
                    torch.zeros(batch_size, self.hidden_dim).to(x.device))
        
        for t in range(seq_size):

            x_t = self.embedding(x[:, 0, t].long()).to(x.device)

            x_th_t = torch.cat((h_t, x_t), dim=1).to(x.device)
            z_t = torch.sigmoid(x_th_t @ self.W_z).to(x.device)
            r_t = torch.sigmoid(x_th_t @ self.W_r).to(x.device)
            r_th_tx_t = torch.cat((r_t * h_t, x_t), dim=1).to(x.device)
            h_tilde = torch.tanh(r_th_tx_t @ self.W).to(x.device)
            h_t = (h_t * (1 - z_t) + h_tilde * z_t).to(x.device)
        
        p_t = (h_t @ self.W_ph + self.b_p).to(x.device)
        y_hat = self.logsoftmax(p_t).to(x.device)

        return y_hat
