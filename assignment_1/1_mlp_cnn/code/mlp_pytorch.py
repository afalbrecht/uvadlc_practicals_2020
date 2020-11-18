"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

def init_weights(layer, std=0.0001):
  if type(layer) == nn.Linear:
    layer.weight.data.normal_(std=std)
    layer.bias.data.fill_(0.0)


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        """

        super().__init__()
        self.layers = nn.ModuleList()
        for layer_size in n_hidden:
            self.layers.append(nn.Linear(n_inputs, layer_size))
            self.layers.append(nn.Tanh())
            n_inputs = layer_size
        self.layers.append(nn.Linear(n_inputs, n_classes))

        self.layers.apply(init_weights)


    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        """

        for layer in self.layers:
          x = layer(x)
        out = x

        return out
