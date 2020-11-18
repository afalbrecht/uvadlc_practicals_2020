"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms
from skimage.transform import resize

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
MODEL_DEFAULT = 'STANDARD'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


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
    Performs training and evaluation of ConvNet model.
  
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

    data = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)
    train = data['train']
    test = data['test']
    # print(train.images[0].shape)
    # n_inputs = train.images[0].flatten().shape[0]
    n_channels = train.images[0].shape[0]
    n_classes = train.labels[0].shape[0]

    print(train.images.shape)

    # transform = transforms.Compose(
    #     [transforms.Resize((224, 224)),
    #      transforms.ToTensor(),
    #      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    if FLAGS.model == 'ALEX':
        train.images = torch.Tensor([resize(img) for img in train.images])
        test.images = torch.Tensor([resize(img) for img in test.images])
        print('doneee')
        model = models.alexnet(pretrained=True)
        torch.save(model.state_dict(), 'alexnet.txt')

        # model = torch.load('alexnet.txt')

        for param in model.features.parameters():
            param.requires_grad = False
        
        loss_mod = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=FLAGS.learning_rate)
    else:
        model = ConvNet(n_channels, n_classes)
        loss_mod = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate)

    model.to(device)

    loss_history = []
    acc_history = []
    for step in range(2): #FLAGS.max_steps
        model.train()
        x, y = train.next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(np.argmax(y, axis=1)).to(device) # converts onehot to dense

        if FLAGS.model == 'ADAM': x = resizer(x)

        out = model(x)
        loss = loss_mod(out, y)
        loss_history.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % FLAGS.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                x, y = torch.from_numpy(test.images), torch.from_numpy(test.labels)
                acc = 0
                test_step = int(x.shape[0]/20)
                for i in range(0, x.shape[0], test_step):
                    batch_x = x[i:i+test_step].to(device)
                    batch_y = y[i:i+test_step].to(device)
                    if FLAGS.model == 'ADAM': batch_x = resizer(batch_x)
                    test_out = model.forward(batch_x)
                    acc += accuracy(test_out, batch_y)/20
                print('Accuracy:', acc)
                acc_history.append(acc)
    print('Final loss:', loss_history[-1])
    print('Final acc:', acc_history[-1])

    plt.plot(loss_history)
    plt.step(range(0, FLAGS.max_steps+1, FLAGS.eval_freq), acc_history) # range(0, FLAGS.max_steps, FLAGS.eval_freq)
    plt.legend(['loss', 'accuracy'])
    # plt.show()
    # plt.savefig('/home/lgpu0376/code/output_dir/loss_acc_graph.png')
    # plt.savefig('loss_acc_graph.png')

    # with open('/home/lgpu0376/code/output_dir/output.out', 'w+') as f:
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
    parser.add_argument('--model', type=str, default=MODEL_DEFAULT,
                        help='Model with choice between standard and AlexNet')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
