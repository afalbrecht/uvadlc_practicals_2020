"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.net = []
        for layer_size in n_hidden:
            self.net.append(LinearModule(n_inputs, layer_size))
            self.net.append(ELUModule())
            n_inputs = layer_size
        self.net.append(LinearModule(n_inputs, n_classes))
        self.net.append(SoftMaxModule())

        # self.net = []
        # prev_layer = n_inputs
        # for i, layer_size in enumerate(n_hidden):
        #     self.net.append(LinearModule(prev_layer, layer_size))
        #     prev_layer = layer_size
        #     self.net.append(ELUModule())
        # self.net.append(SoftMaxModule())

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.net:
          x = layer.forward(x)
        out = x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.net[::-1]:
          dout = layer.backward(dout)
        ########################
        # END OF YOUR CODE    #
        #######################

        return
    
    def update(self, learning_rate):
      for lin_layer in self.net[::2]:
        # print(lin_layer.params['weight'].shape)
        # print(np.mean(lin_layer.grads['bias'], axis=0).shape)
        lin_layer.params['weight'] = lin_layer.params['weight'] - (lin_layer.grads['weight'] * learning_rate)
        lin_layer.params['bias'] = lin_layer.params['bias'] - (lin_layer.grads['bias'] * learning_rate)
        # print(lin_layer.params['weight'].shape)
        # print(lin_layer.params['bias'].shape)