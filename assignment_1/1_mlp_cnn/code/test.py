import numpy as np
from modules import *
import os
import matplotlib.pyplot as plt

print(np.isnan(np.nan))

# acc = [0.1, 0.2, 0.4, 0.6, 0.99]
# loss = [3, 2.5, 2, 1, 0.1]

# plt.plot(acc)
# plt.plot(loss)
# plt.legend(['accuracy', 'loss'])
# plt.show()

# X = np.ones((5, 100))
# Y = np.ones((5, 50))

# dic = {1: 2, 3:4, 5:6}
# one, two, three = dic.values()
# print(one, two, three)

#print(os.listdir('cifar10/cifar-10-batches-py'))

# print(np.multiply(Y, X))
# print(np.sum(np.multiply(Y, X)))
# arr = np.array([[1, 2, 3, 8, 0], [2, 9, 3, 2, 1], [1, 2, 16, 8, 5]])
# onehot = np.array([[0,0,0,1,0],[0,0,0,1,0],[0,0,1,0,0]])
# print(arr/onehot    )
# print(arr.shape)
# print(np.argmax(arr, axis=1))
# print(np.argmax(onehot, axis=1))
# argarr = np.argmax(arr, axis=1)
# arghot = np.argmax(onehot, axis=1)
# print(len(argarr[argarr == arghot])/len(arr))

# np.random.seed(42)
# lin = LinearModule(100, 50)
# print(lin.params['weight'].shape)
# out = lin.forward(X)
# print(out.shape)
# dx = lin.backward(out)
# print(dx.shape, lin.grads['weight'].shape, lin.grads['bias'].shape)
# softmax = SoftMaxModule()
# print(out.shape)
# soft_out = softmax.forward(out)
# print(soft_out.shape)
# soft_dx = softmax.backward(soft_out)
# print(soft_dx.shape)
# # cross_ent = CrossEntropyModule()
# # cross_out = crosse

#     x = np.maximum(x, 10**-9) # for numerical stability
#     L = -np.multiply(y, np.log(x))

#     out = np.mean(np.sum(L, axis=1))

#     self.s = len(x)
#     return out

# def backward(self, x, y):
#     """
#     Backward pass.
#     Args:
#         x: input to the module
#         y: labels of the input
#     Returns:
#         dx: gradient of the loss with the respect to the input x.

#     TODO:
#     Implement backward pass of the module.
#     """
#     x = np.maximum(x, 10 ** -9)  # for numerical stability
#     x = 1/x
#     dx = -np.multiply(y, x)
#     dx = -(1 / self.s) * dx
#     return dx

self.network = []
# Test if n_hidden is empty / no hidden units required
if not n_hidden:
    self.network.append(LinearModule(n_inputs, n_classes))
    self.network.append(SoftMaxModule())
    self.loss_module = CrossEntropyModule()
else:
    for n_outputs in n_hidden:
        self.network.append(LinearModule(n_inputs, n_outputs))
        self.network.append(ELUModule())
        n_inputs = n_outputs
    self.network.append(LinearModule(n_hidden[-1], n_classes))
    self.network.append(SoftMaxModule())
    self.loss_module = CrossEntropyModule()