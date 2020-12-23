################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
import torch.nn as nn
import numpy as np

def init_weights(layer, std=0.0001):
  if type(layer) == nn.Linear:
    layer.weight.data.normal_(std=std)
    layer.bias.data.fill_(0.0)


class MLPEncoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dims=[512], z_dim=20):
        """
        Encoder with an MLP network and ReLU activations (except the output layer).

        Inputs:
            input_dim - Number of input neurons/pixels. For MNIST, 28*28=784
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            z_dim - Dimensionality of latent vector.
        """
        super().__init__()

        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.

        self.z_dim = z_dim
        self.encoder = nn.ModuleList()
        for layer_size in hidden_dims:
            self.encoder.append(nn.Linear(input_dim, layer_size))
            self.encoder.append(nn.ReLU())
            input_dim = layer_size
        self.encoder.append(nn.Linear(input_dim, z_dim*2))

        self.encoder.apply(init_weights)          # TODO: init_weights

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """

        # Remark: Make sure to understand why we are predicting the log_std and not std

        img_size = x.size()[1] * x.size()[2] * x.size()[3]
        x = x.view(-1, img_size)
        for layer in self.encoder:
            x = layer(x)
        
        mean = x[:, :self.z_dim]
        log_std = x[:, self.z_dim:]

        return mean, log_std


class MLPDecoder(nn.Module):

    def __init__(self, z_dim=20, hidden_dims=[512], output_shape=[1, 28, 28]):
        """
        Decoder with an MLP network.
        Inputs:
            z_dim - Dimensionality of latent vector (input to the network).
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            output_shape - Shape of output image. The number of output neurons of the NN must be
                           the product of the shape elements.
        """
        super().__init__()
        self.output_shape = output_shape
        output_shape = output_shape[1] * output_shape[2]

        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.
        
        input_dim = z_dim
        self.decoder = nn.ModuleList()
        for layer_size in hidden_dims:
            self.decoder.append(nn.Linear(input_dim, layer_size))
            self.decoder.append(nn.ReLU())
            input_dim = layer_size
        self.decoder.append(nn.Linear(input_dim, output_shape))

        self.decoder.apply(init_weights) 

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,output_shape[0],output_shape[1],output_shape[2]]
        """

        for layer in self.decoder:
            z = layer(z)
        return z.view(-1, self.output_shape[0], self.output_shape[1], self.output_shape[2])

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
