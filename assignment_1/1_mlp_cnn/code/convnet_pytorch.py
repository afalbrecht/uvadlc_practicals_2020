"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import cifar10_utils
import torch
import torch.nn as nn

class PreActBlock(nn.Module):
  '''
  Implements PreActivation Resnet block
  '''

  def __init__(self, n_channels):
    '''
    Initializes PreActBlock object.

    ARgs:
      n_channels: number of input channels
      n_output: number of output channels
    
    '''

    super(PreActBlock, self).__init__()
    self.net = nn.Sequential(
      nn.BatchNorm2d(n_channels),
      nn.ReLU(),
      nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False)
    )

  def forward(self, x):
    input = x
    
    for layer in self.net:
      input = layer(input)
    
    return x + input

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        """

        super().__init__()
        channel_size = 64
        
        net_list = []
        net_list.append(nn.Conv2d(n_channels, channel_size, kernel_size=3, stride=1, padding=1, bias=False))
        net_list.append(PreActBlock(channel_size))
        for i in range(5):
          if channel_size != 512:
            net_list.append(nn.Conv2d(channel_size, channel_size*2, kernel_size=1, stride=1, padding=0, bias=False))
            channel_size = channel_size * 2
          net_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
          net_list.append(PreActBlock(channel_size))
          net_list.append(PreActBlock(channel_size))
        net_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        net_list.append(nn.BatchNorm2d(channel_size))
        net_list.append(nn.ReLU())
        net_list.append(nn.Linear(channel_size, 10))

        self.net = nn.Sequential(*net_list)
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        """
        
        for layer in self.net:
          if type(layer) == nn.Linear:
            x = x.reshape(x.shape[0], x.shape[1])
          x = layer(x)
        out = x

        return out
