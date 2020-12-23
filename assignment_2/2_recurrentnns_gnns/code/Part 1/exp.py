import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

tens = torch.tensor([[1], [2]]).float()

print(nn.functional.softmax(tens*2, dim=0))

# l1 = [1,2,3,4,5,6,7,8,9]
# l2 = [2,3,4,5,6,7,8,9,10]
# l3 = [3,4,5,6,7,8,9,10,11]
# loss = np.asarray([l1, l2, l3])
# ar2 = np.array([[1,2], [3,4], [5,6]])

# print(loss.shape)
# print(ar2.shape)

# print(np.concatenate((loss, ar2), axis=1).shape)

# print(np.asarray(loss))
# print(np.mean(loss, axis=0))
# print(np.std(loss, axis=0))

# mean = np.mean(loss, axis=0)
# std = np.std(loss, axis=0)
# seq_length = 5

# with open('test.txt', 'a+') as f:
#     f.write(f'\nFinal loss for seq_length {seq_length}: {mean[-1]}')  
#     f.write(f'\nFinal loss std for seq_length {seq_length}: {np.mean(std)}')  

# plt.plot(mean)
# plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
# # plt.plot(acc_history) # range(0, FLAGS.max_steps, FLAGS.eval_freq)
# # plt.legend(['loss', 'accuracy'])
# plt.show()