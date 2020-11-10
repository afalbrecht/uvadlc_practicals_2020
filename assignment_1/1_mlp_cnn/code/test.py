import numpy as np
from modules import *

X = np.ones((5, 100))
Y = np.ones((5, 50))

dic = {1: 2, 3:4, 5:6}
one, two, three = dic.values()
print(one, two, three)

# print(np.multiply(Y, X))
# print(np.sum(np.multiply(Y, X)))
# arr = np.array([[1, 2, 3, 8, 5], [2, 9, 3, 2, 1], [1, 2, 16, 8, 5]])
# onehot = np.array([[0,0,0,1,0],[0,0,0,1,0],[0,0,1,0,0]])
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