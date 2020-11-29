# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        embed_dim = int(lstm_num_hidden/4)
        print('embed_dim:', embed_dim)
        self.embed_dim = embed_dim
        self.device = device
        self.num_hidden = lstm_num_hidden
        self.num_layers = lstm_num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_num_hidden,
            num_layers=lstm_num_layers
        )

        self.fc = nn.Linear(lstm_num_hidden, vocabulary_size)

        self.init_params()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embed_dim
        )

        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def init_params(self):

        for name, param in self.named_parameters():
            # print(name)
            if len(param.shape) > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)
    
    def init_state(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.num_hidden).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.num_hidden).to(device))

    # def forward(self, x, prev_state):
    #     # print(x.size())
    #     embed = self.embedding(x)
    #     # print(embed.size())
    #     output, new_state = self.lstm(embed, prev_state)
    #     output = self.fc(output)
    #     output = self.logsoftmax(output)

    #     return output, new_state
    
    def forward(self, x):
        # print(x.size())
        embed = self.embedding(x)
        # print(embed.size())
        output, _ = self.lstm(embed, self.init_state())
        output = self.fc(output)
        output = self.logsoftmax(output)

        return output




