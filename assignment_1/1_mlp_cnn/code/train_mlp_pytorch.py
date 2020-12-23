 """
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from mlp_pytorch import MLP
import cifar10_utils
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 2400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OPTIMIZER_DEFAULT = 'SGD'


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

#class TorchDataset(data)

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    """
    
    argpred = torch.argmax(predictions, dim=1)
    arghot = torch.argmax(targets, dim=1)
    accuracy = len(argpred[argpred == arghot])/len(predictions)
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
  
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device", device)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    

    # DNN_HIDDEN_UNITS_DEFAULT = '100'
    # LEARNING_RATE_DEFAULT = 1e-3
    # MAX_STEPS_DEFAULT = 1400
    # BATCH_SIZE_DEFAULT = 200
    # EVAL_FREQ_DEFAULT = 100
    
    data = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)
    train = data['train']
    print(train.images.shape)
    test = data['test']
    n_inputs = train.images[0].flatten().shape[0]
    n_classes = train.labels[0].shape[0]

    mlp = MLP(n_inputs, dnn_hidden_units, n_classes)
    loss_mod = nn.CrossEntropyLoss()
    if FLAGS.optimizer == 'SGD':
        optimizer = torch.optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=FLAGS.learning_rate)
    
    mlp.to(device)

    loss_history = []
    acc_history = []
    for step in range(FLAGS.max_steps): #FLAGS.max_steps
        mlp.train()
        x, y = train.next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x.reshape(x.shape[0], n_inputs)).to(device)
        y = torch.from_numpy(np.argmax(y, axis=1)).to(device) # converts onehot to dense

        out = mlp(x)
        loss = loss_mod(out, y)
        loss_history.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % FLAGS.eval_freq == 0:
            mlp.eval()
            with torch.no_grad():
                x, y = test.images, test.labels
                x = torch.from_numpy(x.reshape(x.shape[0], n_inputs)).to(device)
                y = torch.from_numpy(y).to(device)
                test_out = mlp.forward(x)
                acc = accuracy(test_out, y)
                print('Accuracy:', acc)
                acc_history.append(acc)
    print('Final loss:', loss_history[-1])
    print('Final acc:', acc_history[-1])

    plt.plot(loss_history)
    plt.step(range(0, FLAGS.max_steps + 1, FLAGS.eval_freq), acc_history) # range(0, FLAGS.max_steps, FLAGS.eval_freq)
    plt.legend(['loss', 'accuracy'])
    plt.show()
    # plt.savefig('/home/lgpu0376/code/output_dir/loss_acc_graph_mlp.png')
    # plt.savefig('loss_acc_torchmlp.png')

    # with open('/home/lgpu0376/code/output_dir/output_mlp.out', 'w+') as f:
    #     f.write(f'Final loss: {loss_history[-1]}')  
    #     f.write(f'\nFinal acc: {acc_history[-1]}')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                    help='Optimizer to use, choice between "SGD" and "AdamW"')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
