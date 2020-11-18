import numpy as np
from modules import *
import os
import matplotlib.pyplot as plt
import torch
import time

print([i for i in range(0, 1401, 100)])

# for step in range(1400):
#     if step % 100 == 0:
#         print(step)
print(1e-2)
# x = torch.Tensor([[1, 2, 3, 4], [1,4,2,3], [4,1,2,3]])
# print(torch.argmax(x, dim=0))

# plt.plot([1,2], [3,4])
# plt.savefig('test.png')

# t = 10000
# v = np.ones(10000)

# for i in range(0, v.shape[0], int(v.shape[0]/20)):
#     print(v[i: i+500].shape)

# print(time.time())
# print(os.getcwd())
# print(os.listdir())
# print(time.time())

# with open('/home/lgpu0376/code/output_dir/output.out', 'a+') as f:
#     f.write('test')
#     f.write('test2')


# x = torch.ones(10, 3)
# m = torch.ones(10)
# g = torch.ones(3)
# b = torch.ones(3)
# print(g*x + b)
# var = torch.mean((x + m.view(-1, 1))**2, axis=1)
# var2 = torch.ones(10) * 2
# print((var2  - 1).view(-1, 1))

# print((x + m.view(-1, 1))**2)
# print(m[...,None])

# print(np.isnan(np.nan))

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
