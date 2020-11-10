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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
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
        
        ########################
        # END OF YOUR CODE    #
        #######################
    
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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        out = x @ self.params['weight'].T + self.params['bias']
        self.x_values = x
        self.y_values = out
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout @ self.params['weight']
        self.grads['weight'] = dout.T @ self.x_values
        self.grads['bias'] = dout
        
        ########################
        # END OF YOUR CODE    #
        #######################
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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.x_values = x
        y = np.exp(x - x.max())
        out = y / y.sum(axis=1, keepdims=True)
        self.y_values = out
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        y = self.y_values
        # print('softmax s:', s.shape)
        # print('dout', dout.shape)
        # print(np.diagflat(s))
        # print('diag:', (np.diagflat(s) - np.dot(s, s.T)).shape)
        # dx = dout @ (np.diagflat(s) - np.dot(s, s.T))
        # print('softmax dx:', dx.shape)

        # s = s.reshape(-1, 1)
        # print('diag:', (np.diagflat(s) - np.matmul(s, s.T)).shape)
        # dx = dout @ (np.diagflat(s) - np.matmul(s, s.T))
        # print('softmax dx:', dx.shape)
        
        n = y.shape[1]
        # diag = np.eye(y.shape[1]) * y[:, :, np.newaxis]
        diag= np.einsum('ij,jk->ijk', y, np.eye(n, n))
        # rest = (y * y)[:, :, np.newaxis]
        rest= np.einsum('ij,ik->ijk', y, y)
        # print('1', rest.shape)
        # print('2', rest2.shape)
        dx = diag - rest
        dx = np.einsum('ijk,ik->ij', dx, dout)
        print(dx.shape)

        '''TODO: Recheck backwards pass '''

        ########################
        # END OF YOUR CODE    #
        #######################
        
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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        s = x.shape[0]
        # print('y, x:', y.shape, x.shape)
        # print('dot vector:', np.dot(y.T, x).shape)
        # print('dot scalar:', np.dot(y[0], x[0]).shape)
        out = -np.mean(np.sum(y * np.log(x), axis=1))
        # print("cross entropy out:", out.shape)
        # out = - 1/s*np.sum(np.dot(y.T, np.log(x)))
        # print("cross entropy sum out:", out.shape)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        s = x.shape[0]
        dx = -1/s * y/x
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
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
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.x_values = x
        out = np.where(x >= 0, x, np.exp(x) - 1)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
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
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        x = self.x_values
        dx = dout * np.where(x >= 0, 1, np.exp(x))

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
