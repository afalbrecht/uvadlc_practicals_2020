"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """

        std = 0.0001
        self.params = {}
        self.grads = {}
        self.params['weight'] = np.random.normal(0, std, (out_features, in_features))
        self.params['bias'] = np.zeros((1, out_features))
        self.grads['dW'] = np.zeros((out_features, in_features))
        self.grads['dX'] = np.zeros((1, in_features))
        self.grads['dB'] = np.zeros((1, out_features))
        self.x_values = None
        self.y_values = None

    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        out = x @ self.params['weight'].T + self.params['bias']
        self.x_values = x
        self.y_values = out
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        dx = dout @ self.params['weight']
        self.grads['weight'] = dout.T @ self.x_values
        self.grads['bias'] = np.mean(dout, axis=0).reshape(1, dout.shape[1])  
        # self.grads['bias'] = dout

        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x_values = x
        y = np.exp(x - x.max())
        sum_y = np.sum(y, axis=1, keepdims=True)
        out = y / sum_y
        self.y_values = out

        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        
        I do want to note that I was heavily inspired by this blogpost for the backwards implementation: https://themaverickmeerkat.com/2019-10-23-Softmax/
        Blood, sweat and tears were shed to find a neater implementation, but none worked. Therefore I leave this as is and hereby credit the original creator.
        Do want to note that through working on this I got more familiar with einsum and am now semi-confident that I could have come up with this myself in the end. 

        """
        y = self.y_values
        n = y.shape[1]
        diag = np.einsum('ij,jk->ijk', y, np.eye(n, n))
        rest= np.einsum('ij,ik->ijk', y, y)
        dx = diag - rest
        dx = np.einsum('ijk,ik->ij', dx, dout)

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        """

        s = x.shape[0]
        x = np.maximum(x, 10**-9)
        out = -np.mean(np.sum(y * np.log(x), axis=1))

        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        """

        s = x.shape[0]
        x = np.maximum(x, 10**-9)
        dx = -1/s * y/x

        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x_values = x
        out = np.where(x >= 0, x, np.exp(x) - 1)

        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        
        Implement backward pass of the module.
        """

        x = self.x_values
        dx = dout * np.where(x >= 0, 1, np.exp(x))

        return dx
