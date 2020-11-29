"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

# seq_length = [4,5,6]
# optimizer = Adam
# learning_rate = 0.0001
# update_iterations = 3000
# batch_size = 256
# nr_hidden = 256
# max_norm = 10
# initialization = Kaiming_normal

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        embed_dim = int(hidden_dim/4)
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #g_t
        self.W_gx = nn.Parameter(torch.Tensor(embed_dim, hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))
        
        #i_t
        self.W_ix = nn.Parameter(torch.Tensor(embed_dim, hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        #f_t
        self.W_fx = nn.Parameter(torch.Tensor(embed_dim, hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        #o_t
        self.W_ox = nn.Parameter(torch.Tensor(embed_dim, hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

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

            # print('x_t', x_t.shape)
            # print('h_t', h_t.shape)

            g_t = torch.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g)
            i_t = torch.sigmoid(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o)
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
        
        p_t = h_t @ self.W_ph + self.b_p
        y_hat = self.logsoftmax(p_t)

        return y_hat

        